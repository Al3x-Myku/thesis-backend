import os
import glob
import traceback
from pathlib import Path
from typing import List

import torch
from PIL import Image

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
DFINE_ROOT     = os.getenv("DFINE_ROOT", "./D-FINE")
DFINE_CONFIG   = os.getenv("DFINE_CONFIG", "configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml")
DFINE_CHECKPT  = os.getenv("DFINE_CHECKPT", "checkpoints/dfine_x_obj365.pth")
DEVICE         = os.getenv("PIPELINE_DEVICE", "cuda:0")

HUNYUAN_SHAPEDIR       = os.getenv("HUNYUAN_SHAPEDIR", "tencent/Hunyuan3D-2mini")
HUNYUAN_SHAPEDIR_SUBFOLDER = os.getenv("HUNYUAN_SHAPEDIR_SUBFOLDER", "hunyuan3d-dit-v2-mini")
HUNYUAN_SHAPEDIR_VARIANT   = os.getenv("HUNYUAN_SHAPEDIR_VARIANT", "fp16")
HUNYUAN_PAINTDIR      = os.getenv("HUNYUAN_PAINTDIR", "tencent/Hunyuan3D-2")

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_shape_pipeline = None
_paint_pipeline = None

def _init_hunyuan_pipelines():
    global _shape_pipeline, _paint_pipeline
    if _shape_pipeline is None:
        print("→ Loading Hunyuan shape pipeline…")
        kwargs = {}
        if HUNYUAN_SHAPEDIR_SUBFOLDER:
            kwargs["subfolder"] = HUNYUAN_SHAPEDIR_SUBFOLDER
        if HUNYUAN_SHAPEDIR_VARIANT:
            kwargs["variant"] = HUNYUAN_SHAPEDIR_VARIANT
        _shape_pipeline = (
            Hunyuan3DDiTFlowMatchingPipeline
            .from_pretrained(HUNYUAN_SHAPEDIR, **kwargs)
            .to(DEVICE)
        )
    if _paint_pipeline is None:
        print("→ Loading Hunyuan paint pipeline…")
        _paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_PAINTDIR).to(DEVICE)

def detect_objects(image_path: str, scene_folder: str) -> List[str]:
    crop_dir = Path(scene_folder) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    run_dfine_inference(
        dfine_root=DFINE_ROOT,
        config_path=DFINE_CONFIG,
        checkpoint_path=DFINE_CHECKPT,
        input_image=image_path,
        device=DEVICE,
        output_dir=str(crop_dir),
    )

    # recursively gather whatever D-FINE created under crops/
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    crops = []
    for pat in patterns:
        crops.extend(str(p) for p in crop_dir.rglob(pat))
    if not crops:
        raise RuntimeError(f"No crops found in {crop_dir}")
    return sorted(crops)

def build_mesh(crop_path: str, scene_folder: str) -> str:
    _init_hunyuan_pipelines()

    # — exactly your original background removal snippet —
    image = Image.open(crop_path).convert("RGB")
    if image.mode == 'RGB':
        print("entering background removal")
        rembg = BackgroundRemover()
        image = rembg(image)
        print("removed background")

    base = Path(crop_path).stem
    out_dir = Path(scene_folder) / "meshes" / base
    out_dir.mkdir(parents=True, exist_ok=True)

    # shape
    print(f"  • shape gen for {base}")
    mesh = _shape_pipeline(
        image=image,
        num_inference_steps=50,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(12345),
        output_type="trimesh",
    )[0]

    # cleanup
    for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
        mesh = cleaner(mesh)
    mesh_path = out_dir / f"{base}.glb"
    mesh.export(str(mesh_path))

    # paint
    print(f"  • paint gen for {base}")
    painted = _paint_pipeline(mesh, image=image)[0]
    tex_path = out_dir / f"{base}_textured.glb"
    painted.export(str(tex_path))

    return str(tex_path)

def merge_meshes(mesh_paths: List[str], scene_folder: str) -> str:
    from trimesh import load, Scene as TScene
    final_dir = Path(scene_folder) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    scene = TScene()
    for mp in mesh_paths:
        scene.add_geometry(load(mp))
    out = final_dir / "scene.glb"
    scene.export(str(out))
    return str(out)

def full_reconstruction(image_path: str, scene_folder: str) -> str:
    """
    High‐level: detect → build each mesh → merge → return final path
    """
    try:
        crops = detect_objects(image_path, scene_folder)
    except Exception:
        traceback.print_exc()
        raise

    results = []
    for c in crops:
        try:
            results.append(build_mesh(c, scene_folder))
        except Exception:
            traceback.print_exc()
            continue

    return merge_meshes(results, scene_folder)
