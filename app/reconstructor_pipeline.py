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

from hy3dgen.shapegen import (
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
PIPELINE_DEVICE = os.getenv("PIPELINE_DEVICE", "cpu")  # Use CPU for BiRefNet segmentation

HUNYUAN_SHAPEDIR           = os.getenv("HUNYUAN_SHAPEDIR", "tencent/Hunyuan3D-2mini")
HUNYUAN_SHAPEDIR_SUBFOLDER = os.getenv("HUNYUAN_SHAPEDIR_SUBFOLDER", "hunyuan3d-dit-v2-mini")
HUNYUAN_SHAPEDIR_VARIANT   = os.getenv("HUNYUAN_SHAPEDIR_VARIANT", "fp16")
HUNYUAN_PAINTDIR           = os.getenv("HUNYUAN_PAINTDIR", "tencent/Hunyuan3D-2")

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_shape_pipeline  = None
_paint_pipeline  = None
_depth_model     = None

# ─── BiRefNet segmentation setup ────────────────────────────────────────────────
# Run BiRefNet on CPU only
device = torch.device("cpu")
o_seg = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).to(device)

def remove_bg_biref(image: Image.Image) -> Image.Image:
    """Run BiRefNet on CPU to get a soft mask and compose RGBA."""
    inp = o_seg(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_logits = birefnet(inp)[-1]
        mask = mask_logits.sigmoid()[0, 0].cpu().numpy()
    alpha = (mask * 255).astype(np.uint8)
    alpha_im = Image.fromarray(alpha).resize(image.size)
    out = image.convert("RGBA")
    out.putalpha(alpha_im)
    return out

# ─── Initialize Hunyuan pipelines ──────────────────────────────────────────────
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

# ─── Initialize ZoeDepth for depth estimation ─────────────────────────────────
def _init_zoe_depth():
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

# ─── Object detection and cropping via D-FINE ──────────────────────────────────
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
    for ext in ("png", "jpg", "jpeg"):  # gather all crop files
        for p in crop_dir.rglob(f"input_crop*.{ext}"):
            m = pattern.search(p.name)
            if m:
                crops.append((int(m.group(1)), str(p)))
    if not crops:
        raise RuntimeError(f"No crops found in {crop_dir}")
    # return sorted by crop index
    return [path for _, path in sorted(crops, key=lambda x: x[0])]

# ─── Build per-object mesh ─────────────────────────────────────────────────────
def build_mesh(crop_path: str, scene_folder: str) -> str:
    _init_hunyuan_pipelines()
    image = Image.open(crop_path).convert("RGB")
    base  = Path(crop_path).stem

    # background removal
    image_no_bg = remove_bg_biref(image)
    no_bg_path  = Path(crop_path).with_name(f"{base}_no_bg.png")
    image_no_bg.save(no_bg_path)

    # shape reconstruction
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

    # mesh cleanup
    for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
        mesh = cleaner(mesh)

    # texture painting
    painted = _paint_pipeline(mesh, image=image_no_bg)
    out_path = out_dir / f"{base}_textured.glb"
    painted.export(out_path)
    return str(out_path)

# ─── Full scene reconstruction ─────────────────────────────────────────────────
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
            f"No meshes could be built for any of the {len(crops)} crops."
            " Check background removal or crop validity."
        )

    return position_meshes(
        mesh_paths=meshes,
        image_path=image_path,
        scene_folder=scene_folder,
        valid_indices=valid_crop_indices
    )

# ─── Position meshes in scene ─────────────────────────────────────────────────
def position_meshes(
    mesh_paths: List[str],
    image_path: str,
    scene_folder: str,
    valid_indices: List[int] = None,
    threshold: float = 0.6
) -> str:
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    cx, cy = W / 2, H / 2

    # run detection for updated boxes
    processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_obj365")
    model_det = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_obj365")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model_det(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(H, W)], threshold=threshold
    )[0]
    boxes = results["boxes"].cpu().numpy()
    if valid_indices is not None:
        boxes = boxes[valid_indices]

    # estimate depth map on CPU
    model_zoe = _init_zoe_depth()
    depth_map = model_zoe.cpu().infer_pil(image)
    model_zoe.cuda()

    # sort left→right
    order = np.argsort(boxes[:, 0])
    boxes_sorted = boxes[order]
    meshes_sorted = [mesh_paths[i] for i in order]

    # calibrate scale using the largest bounding-box area
    areas = []
    f_vals = []
    for mp, bb in zip(meshes_sorted, boxes_sorted):
        x0, y0, x1, y1 = bb.astype(int)
        w_px = x1 - x0
        h_px = y1 - y0
        areas.append(w_px * h_px)
        mesh_ref = trimesh.load(mp, force="scene")
        w_mesh = float(mesh_ref.extents[0])
        Z_ref = float(np.median(depth_map[y0:y1, x0:x1]))
        f_vals.append(w_px * Z_ref / w_mesh)
    if not areas:
        raise RuntimeError("Cannot calibrate focal length: no valid detections.")
    idx_largest = int(np.argmax(areas))
    f = float(f_vals[idx_largest])

    # place each mesh
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

    out_dir = Path(scene_folder) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_positioned.glb"
    scene.export(str(out_path))
    return str(out_path)

# ─── Merge all meshes into one ─────────────────────────────────────────────────
def merge_meshes(mesh_paths: List[str], scene_folder: str) -> str:
    out_dir = Path(scene_folder) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_merged.glb"

    meshes = [trimesh.load(mp, force="scene") for mp in mesh_paths]
    merged_scene = trimesh.util.concatenate(meshes)
    merged_scene.export(str(out_path))
    return str(out_path)
