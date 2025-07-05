import os
import traceback
import re
import math
import torch
import numpy as np

from pathlib import Path
from typing import List
from PIL import Image
from transformers import (
    DFineForObjectDetection,
    AutoImageProcessor,
    AutoModelForImageSegmentation,
)
from torchvision import transforms
from torch.hub import load_state_dict_from_url
import trimesh

from hy3dgen.shapgen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FaceReducer,
    FloaterRemover,
    DegenerateFaceRemover,
)
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from .dfine_wrapper import run_dfine_inference

# ─── Config from .env ──────────────────────────────────────────────────────────
DFINE_ROOT      = os.getenv("DFINE_ROOT", "./D-FINE")
DFINE_CONFIG    = os.getenv("DFINE_CONFIG", "configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml")
DFINE_CHECKPT   = os.getenv("DFINE_CHECKPT", "checkpoints/dfine_x_obj365.pth")
PIPELINE_DEVICE = os.getenv("PIPELINE_DEVICE", "cuda:0")

HUNYUAN_SHAPEDIR           = os.getenv("HUNYUAN_SHAPEDIR", "tencent/Hunyuan3D-2mini")
HUNYUAN_SHAPEDIR_SUBFOLDER = os.getenv("HUNYUAN_SHAPEDIR_SUBFOLDER", "hunyuan3d-dit-v2-mini")
HUNYUAN_SHAPEDIR_VARIANT   = os.getenv("HUNYUAN_SHAPEDIR_VARIANT", "fp16")
HUNYUAN_PAINTDIR           = os.getenv("HUNYUAN_PAINTDIR", "tencent/Hunyuan3D-2")

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_shape_pipeline  = None
_paint_pipeline  = None
_depth_model     = None

# ─── BiRefNet segmentation setup ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Improve CUDA matmul precision
torch.set_float32_matmul_precision("high")

# Load BiRefNet once
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).to(DEVICE)

transform_seg = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def remove_bg_biref(image: Image.Image) -> Image.Image:
    """Run BiRefNet to get a soft mask, threshold, and compose RGBA."""
    try:
        inp = transform_seg(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask_logits = birefnet(inp)[-1]       # (1,1,H,W)
            mask = mask_logits.sigmoid()[0,0].cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        # clear cache and downsample on OOM
        torch.cuda.empty_cache()
        small = image.resize((512, 512))
        inp = transform_seg(small).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask_logits = birefnet(inp)[-1]
            mask_small = mask_logits.sigmoid()[0,0].cpu().numpy()
        # upsample mask back to original size
        mask = np.array(
            Image.fromarray((mask_small * 255).astype(np.uint8))
            .resize(image.size)
        ) / 255.0

    alpha = (mask * 255).astype(np.uint8)
    alpha_im = Image.fromarray(alpha).resize(image.size)
    out = image.convert("RGBA")
    out.putalpha(alpha_im)
    return out


def _init_hunyuan_pipelines():
    global _shape_pipeline, _paint_pipeline
    if _shape_pipeline is None:
        kwargs = {}
        if HUNYUAN_SHAPEDIR_SUBFOLDER:
            kwargs["subfolder"] = HUNYUAN_SHAPEDIR_SUBFOLDER
        if HUNYUAN_SHAPEDIR_VARIANT:
            kwargs["variant"] = HUNYUAN_SHAPEDIR_VARIANT
        _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            HUNYUAN_SHAPEDIR, **kwargs
        )
    if _paint_pipeline is None:
        _paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_PAINTDIR)


def _init_zoe_depth():
    """Lazy-load ZoeDepth_NK for depth estimation."""
    global _depth_model
    if _depth_model is None:
        repo = "isl-org/ZoeDepth"
        _depth_model = torch.hub.load(
            repo, "ZoeD_NK", pretrained=False, trust_repo=True
        ).eval().cuda()
        url = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"
        sd  = load_state_dict_from_url(url, progress=True, map_location="cpu")
        _depth_model.load_state_dict(sd, strict=False)
        for m in _depth_model.modules():
            if hasattr(m, "drop_path1") and not hasattr(m, "drop_path"):
                m.drop_path = m.drop_path1
    return _depth_model


def detect_objects(image_path: str, scene_folder: str) -> List[str]:
    crop_dir = Path(scene_folder) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    run_dfine_inference(
        dfine_root=DFINE_ROOT,
        config_path=DFINE_CONFIG,
        checkpoint_path=DFINE_CHECKPT,
        input_image=image_path,
        device=PIPELINE_DEVICE,
        output_dir=str(crop_dir),
    )

    pattern = re.compile(r"crop(\d+)")
    crops = []
    for ext in ("png","jpg","jpeg"):
        for p in crop_dir.rglob(f"input_crop*.{ext}"):
            m = pattern.search(p.name)
            if m:
                crops.append((int(m.group(1)), str(p)))
    if not crops:
        raise RuntimeError(f"No crops found in {crop_dir}")
    return [path for _, path in sorted(crops, key=lambda x: x[0])]


def build_mesh(crop_path: str, scene_folder: str) -> str:
    _init_hunyuan_pipelines()
    image = Image.open(crop_path).convert("RGB")
    base  = Path(crop_path).stem

    # background removal using BiRefNet
    image_no_bg = remove_bg_biref(image)
    no_bg_path  = Path(crop_path).with_name(f"{base}_no_bg.png")
    image_no_bg.save(no_bg_path)

    out_dir = Path(scene_folder) / "meshes" / base
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = _shape_pipeline(
        image=image_no_bg,
        num_inference_steps=50,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(12345),
        output_type="trimesh",
    )[0]
    mesh.export(out_dir / f"{base}_raw.glb")

    for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
        mesh = cleaner(mesh)

    painted = _paint_pipeline(mesh, image=image_no_bg)
    out_path = out_dir / f"{base}_textured.glb"
    painted.export(out_path)
    return str(out_path)


def full_reconstruction(image_path: str, scene_folder: str) -> str:
    try:
        crops = detect_objects(image_path, scene_folder)
    except Exception:
        traceback.print_exc()
        raise

    meshes = []
    valid_crop_indices = []
    for idx, crop in enumerate(crops):
        try:
            mesh_path = build_mesh(crop, scene_folder)
            meshes.append(mesh_path)
            valid_crop_indices.append(idx)
        except Exception:
            traceback.print_exc()

    if not meshes:
        raise RuntimeError(
            f"No meshes could be built for any of the {len(crops)} crops. "
            "Check if BiRefNet is OOM'ing or DFine crops are invalid."
        )

    return position_meshes(
        mesh_paths=meshes,
        image_path=image_path,
        scene_folder=scene_folder,
        valid_indices=valid_crop_indices
    )


def position_meshes(
    mesh_paths: List[str],
    image_path: str,
    scene_folder: str,
    valid_indices: List[int] = None,
    threshold: float = 0.6
) -> str:
    """
    Given a list of textured mesh paths, filters out any detections whose
    meshes failed, re-detects bounding boxes, estimates depth-map on CPU,
    calibrates focal length from the first mesh, sorts left→right and positions
    meshes in 3D. Exports scene_positioned.glb under scene_folder/final/.
    """
    # 1) Load image and get its center
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    cx, cy = W / 2, H / 2

    # 2) Run D-FINE object detection
    processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_obj365")
    model_det = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_obj365")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model_det(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(H, W)], threshold=threshold
    )[0]
    boxes = results["boxes"].cpu().numpy()  # (N,4)

    # 2.1) Filter out boxes whose crops failed
    if valid_indices is not None:
        boxes = boxes[valid_indices]

    # 3) Estimate depth-map on CPU to match types
    model_zoe = _init_zoe_depth()
    model_cpu = model_zoe.cpu()
    depth_map = model_cpu.infer_pil(image)
    model_zoe.cuda()

    # 4) Sort detections and mesh_paths from left to right
    order = np.argsort(boxes[:, 0])
    boxes_sorted = boxes[order]
    meshes_sorted = [mesh_paths[i] for i in order]

    # 5) Calibrate focal length using the first mesh
    mesh_ref = trimesh.load(meshes_sorted[0], force="scene")
    w_mesh = float(mesh_ref.extents[0])                # model-unit width
    x0, y0, x1, y1 = boxes_sorted[0].astype(int)
    pix_w = x1 - x0                                    # pixel width
    Z_ref = float(np.median(depth_map[y0:y1, x0:x1]))  # depth in meters
    f = pix_w * Z_ref / w_mesh                         # focal length in px

    # 6) Place each mesh in the scene
    scene = trimesh.Scene()
    for i, (mp, bb) in enumerate(zip(meshes_sorted, boxes_sorted)):
        x0, y0, x1, y1 = bb.astype(int)
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        crop = depth_map[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        Z = float(np.median(crop))
        X = (xc - cx) * Z / f
        Y = -(yc - cy) * Z / f

        mesh = trimesh.load(mp, force="scene")
        mesh.apply_translation([X, Y, Z])
        scene.add_geometry(mesh, node_name=f"obj_{i}")

    # 7) Export final positioned scene
    out_dir = Path(scene_folder) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_positioned.glb"
    scene.export(str(out_path))
    return str(out_path)


def merge_meshes(mesh_paths: List[str], scene_folder: str) -> str:
    out_dir = Path(scene_folder) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_merged.glb"

    meshes = [trimesh.load(mp, force="scene") for mp in mesh_paths]
    merged_scene = trimesh.util.concatenate(meshes)
    merged_scene.export(str(out_path))
    return str(out_path)
