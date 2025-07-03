import os
import traceback
import re
import torch
import numpy as np
import time
from pathlib import Path
from typing import List
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import trimesh
from torch.hub import load_state_dict_from_url

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FaceReducer,
    FloaterRemover,
    DegenerateFaceRemover,
)
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from .dfine_wrapper import run_dfine_inference

# ─── Config from .env ──────────────────────────────────────────────────────────
DFINE_ROOT = os.getenv("DFINE_ROOT", "./D-FINE")
DFINE_CONFIG = os.getenv("DFINE_CONFIG", "configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml")
DFINE_CHECKPT = os.getenv("DFINE_CHECKPT", "checkpoints/dfine_x_obj365.pth")
DEVICE = os.getenv("PIPELINE_DEVICE", "cuda:0")

HUNYUAN_SHAPEDIR = os.getenv("HUNYUAN_SHAPEDIR", "tencent/Hunyuan3D-2mini")
HUNYUAN_SHAPEDIR_SUBFOLDER = os.getenv("HUNYUAN_SHAPEDIR_SUBFOLDER", "hunyuan3d-dit-v2-mini")
HUNYUAN_SHAPEDIR_VARIANT = os.getenv("HUNYUAN_SHAPEDIR_VARIANT", "fp16")
HUNYUAN_PAINTDIR = os.getenv("HUNYUAN_PAINTDIR", "tencent/Hunyuan3D-2")

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_shape_pipeline = None
_paint_pipeline = None
_depth_model = None


def log(msg: str):
    print(f"[LOG] {msg}")


def _init_hunyuan_pipelines():
    global _shape_pipeline, _paint_pipeline
    if _shape_pipeline is None:
        log("Initializing shape generation pipeline...")
        kwargs = {}
        if HUNYUAN_SHAPEDIR_SUBFOLDER:
            kwargs["subfolder"] = HUNYUAN_SHAPEDIR_SUBFOLDER
        if HUNYUAN_SHAPEDIR_VARIANT:
            kwargs["variant"] = HUNYUAN_SHAPEDIR_VARIANT
        _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            HUNYUAN_SHAPEDIR, **kwargs
        )
        log("Shape generation pipeline initialized.")

    if _paint_pipeline is None:
        log("Initializing texture painting pipeline...")
        _paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_PAINTDIR)
        log("Texture painting pipeline initialized.")


def _init_zoe_depth():
    global _depth_model
    if _depth_model is None:
        log("Initializing ZoeDepth model...")
        repo = "isl-org/ZoeDepth"
        _depth_model = torch.hub.load(repo, "ZoeD_NK", pretrained=False, trust_repo=True).eval().cuda()
        url = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"
        sd = load_state_dict_from_url(url, progress=True, map_location="cpu")
        _depth_model.load_state_dict(sd, strict=False)
        log("ZoeDepth model initialized.")
    return _depth_model


def detect_objects(image_path: str, scene_folder: str, threshold: float = 0.6) -> List[str]:
    start_time = time.time()
    log("Starting object detection...")

    # Prepare output directory
    crop_dir = Path(scene_folder) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Run D-FINE inference to crop objects
    run_dfine_inference(
        dfine_root=DFINE_ROOT,
        config_path=DFINE_CONFIG,
        checkpoint_path=DFINE_CHECKPT,
        input_image=image_path,
        device=DEVICE,
        output_dir=str(crop_dir),
    )

    # Gather and sort crop file paths
    pattern = re.compile(r"crop(\d+)")
    crops = []
    for ext in ("png", "jpg", "jpeg"):
        for p in crop_dir.rglob(f"input_crop*.{ext}"):
            m = pattern.search(p.name)
            if m:
                crops.append((int(m.group(1)), str(p)))

    if not crops:
        raise RuntimeError(f"No crops found in {crop_dir}")

    crops_sorted = [path for _, path in sorted(crops, key=lambda x: x[0])]
    log(f"Detected {len(crops_sorted)} object(s) in {time.time() - start_time:.2f}s.")
    return crops_sorted


def build_mesh(crop_path: str, scene_folder: str) -> str:
    start_time = time.time()
    log(f"Building mesh from crop: {crop_path}")

    _init_hunyuan_pipelines()
    image = Image.open(crop_path).convert("RGB")
    base = Path(crop_path).stem

    # Remove background
    rembg = BackgroundRemover()
    image_no_bg = rembg(image)
    no_bg_path = Path(crop_path).with_name(f"{base}_no_bg.png")
    image_no_bg.save(no_bg_path)
    log(f"Background removed: {no_bg_path}")

    # Generate raw mesh
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
    raw_path = out_dir / f"{base}_raw.glb"
    mesh.export(raw_path)
    log(f"Raw mesh exported: {raw_path}")

    # Clean mesh
    for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
        mesh = cleaner(mesh)
    log("Mesh cleaned.")

    # Texture painting
    painted = _paint_pipeline(mesh, image=image_no_bg)
    textured_path = out_dir / f"{base}_textured.glb"
    painted.export(textured_path)
    log(f"Textured mesh exported: {textured_path}")

    log(f"Mesh building took {time.time() - start_time:.2f}s.")
    return str(textured_path)


def position_meshes(
    mesh_paths: List[str],
    image_path: str,
    scene_folder: str,
    threshold: float = 0.6
) -> str:
    start_time = time.time()
    log("Positioning meshes in scene...")

    # Load image and compute center
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    cx, cy = W / 2, H / 2

    # Object detection on full scene for bounding boxes
    processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_obj365")
    model_det = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_obj365")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model_det(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(H, W)], threshold=threshold
    )[0]
    boxes = results["boxes"].cpu().numpy()

    # Depth estimation
    model_zoe = _init_zoe_depth()
    model_cpu = model_zoe.cpu()
    depth_map = model_cpu.infer_pil(image)
    model_zoe.cuda()

    # Sort by x coordinate
    order = np.argsort(boxes[:, 0])
    boxes_sorted = boxes[order]
    meshes_sorted = [mesh_paths[i] for i in order][: len(boxes_sorted)]

    # Calibrate focal length
    mesh_ref = trimesh.load(meshes_sorted[0], force="scene")
    w_mesh = float(mesh_ref.extents[0])
    x0, y0, x1, y1 = boxes_sorted[0].astype(int)
    pix_w = x1 - x0
    Z_ref = float(np.median(depth_map[y0:y1, x0:x1]))
    f = pix_w * Z_ref / w_mesh
    log(f"Calibrated focal length: {f:.2f}")

    # Create scene and position each mesh
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
    
    # Export final positioned scene
    out_dir = Path(scene_folder) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_positioned.glb"
    scene.export(str(out_path))

    log(f"Positioning took {time.time() - start_time:.2f}s.")
    return str(out_path)


def full_reconstruction(image_path: str, scene_folder: str) -> str:
    total_start = time.time()
    log("Starting full reconstruction pipeline...")

    # Detect and crop objects
    try:
        crops = detect_objects(image_path, scene_folder)
    except Exception:
        traceback.print_exc()
        raise

    # Build meshes for each crop
    mesh_paths = []
    for crop in crops:
        try:
            mesh_paths.append(build_mesh(crop, scene_folder))
        except Exception:
            traceback.print_exc()
            continue

    # Position meshes
    final_scene = position_meshes(mesh_paths, image_path, scene_folder)
    log(f"Full reconstruction total time: {time.time() - total_start:.2f}s.")
    return final_scene


def merge_meshes(mesh_paths: List[str], scene_folder: str) -> str:
    start_time = time.time()
    log("Merging meshes...")

    out_dir = Path(scene_folder) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_merged.glb"

    # Load and concatenate scenes
    scenes = [trimesh.load(mp, force="scene") for mp in mesh_paths]
    merged = trimesh.util.concatenate(scenes)
    merged.export(str(out_path))

    log(f"Merging took {time.time() - start_time:.2f}s.")
    return str(out_path)
