
from pathlib import Path
from uuid import uuid4
from sqlmodel import Session
from app.repositories.scene_repo import create_scene_record, get_scene
from app.models.scene import Scene
from app.celery_app import celery_app
from app.reconstructor_pipeline import detect_objects, build_mesh, merge_meshes
from typing import List
from app.repositories.scene_repo import get_scenes_by_owner
import os
import shutil
from fastapi import HTTPException, status
from app.core.config import settings
from app.repositories.scene_repo import get_scene_by_id, delete_scene as repo_delete

DATA_ROOT = Path(__import__("os").getenv("DATA_DIR", "./data"))

def create_and_enqueue_scene(
    db: Session, owner_id: int, upload_file
) -> Scene:
    scene = create_scene_record(db, owner_id, input_path="")

    scene_folder = DATA_ROOT / f"user_{owner_id}" / f"scene_{scene.id}"
    scene_folder.mkdir(parents=True, exist_ok=True)

    input_path = scene_folder / "input.png"
    contents = upload_file.file.read()
    input_path.write_bytes(contents)

    scene.input_path = str(input_path)
    db.add(scene); db.commit(); db.refresh(scene)

    celery_app.send_task(
        "app.tasks.reconstruct_scene",
        args=(scene.id,),
    )

    return scene

def fetch_status(db: Session, scene_id: int) -> Scene:
    return get_scene(db, scene_id)


def list_scenes_for_user(db: Session, owner_id: int) -> List[Scene]:
    return get_scenes_by_owner(db, owner_id)

def delete_scene(db: Session, scene_id: int, owner_id: int) -> None:
    scene = get_scene_by_id(db, scene_id)
    if not scene or scene.owner_id != owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scene not found or not owned by you.")

    repo_delete(db, scene)

    data_dir = getattr(settings, "DATA_DIR", os.getenv("DATA_DIR", "./data"))
    scene_folder = os.path.join(
        data_dir,
        f"user_{owner_id}",
        f"scene_{scene_id}"
    )
    shutil.rmtree(scene_folder, ignore_errors=True)