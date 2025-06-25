# app/routers/debug.py
import os, uuid, shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlmodel import Session
from app.database import get_db
from app.reconstructor_pipeline import (
    detect_objects,
    build_mesh,
    merge_meshes,
    full_reconstruction,
)

router = APIRouter(prefix="/debug", tags=["debug"])
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

def _make_scene_folder(owner_id: int, scene_id: str) -> Path:
    folder = DATA_DIR / f"user_{owner_id}" / f"scene_{scene_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

@router.post("/detect/")
async def debug_detect(
    file: UploadFile = File(...),
    owner_id: int = 1,
    scene_id: str = None,
    db: Session = Depends(get_db),
):
    """
    Upload a single image and run just the detect_objects step.
    Returns the list of crop‐file paths.
    """
    scene_id = scene_id or uuid.uuid4().hex
    scene_folder = _make_scene_folder(owner_id, scene_id)

    # save the uploaded file
    input_path = scene_folder / "input.png"
    with open(input_path, "wb") as dst:
        shutil.copyfileobj(file.file, dst)

    try:
        crops = detect_objects(str(input_path), str(scene_folder))
    except Exception as e:
        raise HTTPException(500, f"detect_objects failed: {e}")
    return {"scene_folder": str(scene_folder), "crops": crops}

@router.post("/build/")
async def debug_build(
    crop_path: str,
    owner_id: int = 1,
    scene_id: str = None,
    db: Session = Depends(get_db),
):
    """
    Run build_mesh on a single crop file.
    You must pass the full path to an existing crop (e.g. from /detect/).
    """
    scene_id = scene_id or "debug"
    scene_folder = _make_scene_folder(owner_id, scene_id)

    try:
        mesh = build_mesh(crop_path, str(scene_folder))
    except Exception as e:
        raise HTTPException(500, f"build_mesh failed: {e}")
    return {"scene_folder": str(scene_folder), "mesh": mesh}

@router.post("/merge/")
async def debug_merge(
    mesh_paths: list[str],
    owner_id: int = 1,
    scene_id: str = None,
    db: Session = Depends(get_db),
):
    """
    Merge a list of .glb paths into one final scene.
    """
    scene_id = scene_id or "debug"
    scene_folder = _make_scene_folder(owner_id, scene_id)

    try:
        final = merge_meshes(mesh_paths, str(scene_folder))
    except Exception as e:
        raise HTTPException(500, f"merge_meshes failed: {e}")
    return {"scene_folder": str(scene_folder), "scene": final}

@router.post("/full/")
async def debug_full(
    file: UploadFile = File(...),
    owner_id: int = 1,
    scene_id: str = None,
    db: Session = Depends(get_db),
):
    """
    Run the entire pipeline end-to-end on one upload.
    Returns the final .glb path.
    """
    scene_id = scene_id or uuid.uuid4().hex
    scene_folder = _make_scene_folder(owner_id, scene_id)

    # save input
    input_path = scene_folder / "input.png"
    with open(input_path, "wb") as dst:
        shutil.copyfileobj(file.file, dst)

    try:
        final = full_reconstruction(str(input_path), str(scene_folder))
    except Exception as e:
        raise HTTPException(500, f"full_reconstruction failed: {e}")
    return {"scene_folder": str(scene_folder), "scene": final}
