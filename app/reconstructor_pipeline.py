# app/reconstructor_pipeline.py
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
_shape_pipeline: Hunyuan3DDiTFlowMatchingPipeline | None = None
_paint_pipeline: Hunyuan3DPaintPipeline       | None = None
_remover = BackgroundRemover()

def _init_hunyuan_pipelines():
    global _shape_pipeline, _paint_pipeline
    if _shape_pipeline is None:
        print("→ Loading Hunyuan shape pipeline…")
        _shape_pipeline = (
            Hunyuan3DDiTFlowMatchingPipeline
            .from_pretrained(
                HUNYUAN_SHAPEDIR,
                subfolder=HUNYUAN_SHAPEDIR_SUBFOLDER,
                variant=HUNYUAN_SHAPEDIR_VARIANT,
            )
            .to(DEVICE)
        )
    if _paint_pipeline is None:
        print("→ Loading Hunyuan paint pipeline…")
        _paint_pipeline = (
            Hunyuan3DPaintPipeline
            .from_pretrained(HUNYUAN_PAINTDIR)
            .to(DEVICE)
        )

def detect_objects(image_path: str, scene_folder: str) -> List[str]:
    """
    1) Run D-FINE to crop objects from `image_path` into `scene_folder/crops/`
    2) Return a sorted list of crop paths.
    """
    crop_dir = Path(scene_folder) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    run_dfine_inference(
        dfine_root=DFINE_ROOT,
        config_path=os.path.join(DFINE_ROOT, DFINE_CONFIG),
        checkpoint_path=os.path.join(DFINE_ROOT, DFINE_CHECKPT),
        input_image=image_path,
        device=DEVICE,
        output_dir=str(crop_dir),          # ← here
    )

    # Now include any nested subfolder (e.g. "crops/input/")
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    crops: List[str] = []
    # rglob via pathlib is easiest:
    for pat in patterns:
        for path in Path(crop_dir).rglob(pat):
            crops.append(str(path))
    if not crops:
        raise RuntimeError(f"No crops found in {crop_dir} (tried recursive search)")
    return sorted(crops)

def build_mesh(crop_path: str, scene_folder: str) -> str:
    """
    For one crop:
    • remove background
    • run shape→paint
    • clean mesh
    • export `.glb` to `scene_folder/meshes/{basename}/`
    """
    _init_hunyuan_pipelines()

    # 1) load & bg‐remove
    img = Image.open(crop_path).convert("RGB")
    img = _remover(img)

    base = Path(crop_path).stem
    out_dir = Path(scene_folder) / "meshes" / base
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) shape
    print(f"  • shape gen for {base}")
    mesh = _shape_pipeline(
        image=img,
        num_inference_steps=50,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(12345),
        output_type="trimesh",
    )[0]

    # 3) optional cleanups
    for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
        mesh = cleaner(mesh)

    mesh_path = out_dir / f"{base}.glb"
    mesh.export(str(mesh_path))

    # 4) paint
    print(f"  • paint gen for {base}")
    painted = _paint_pipeline(mesh, image=img)[0]
    paint_path = out_dir / f"{base}_textured.glb"
    painted.export(str(paint_path))

    return str(paint_path)

def merge_meshes(mesh_paths: List[str], scene_folder: str) -> str:
    """
    Combines all the per-object `.glb` into one final scene.glb
    """
    from trimesh import load, Scene as TScene

    final_dir = Path(scene_folder) / "final"
    final_dir.mkdir(exist_ok=True, parents=True)
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
