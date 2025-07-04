
from pathlib import Path
from uuid import uuid4
from sqlmodel import Session
from app.repositories.scene_repo import create_scene_record, get_scene
from app.models.scene import Scene
from app.celery_app import celery_app
from app.reconstructor_pipeline import detect_objects, build_mesh, merge_meshes
from typing import List
from sqlmodel import Session
from app.repositories.scene_repo import get_scenes_by_owner

DATA_ROOT = Path(__import__("os").getenv("DATA_DIR", "./data"))

def create_and_enqueue_scene(
    db: Session, owner_id: int, upload_file
) -> Scene:
    # 1) Create DB placeholder (input_path empty for now)
    scene = create_scene_record(db, owner_id, input_path="")

    # 2) Build the directory: data/user_{owner_id}/scene_{scene.id}/
    scene_folder = DATA_ROOT / f"user_{owner_id}" / f"scene_{scene.id}"
    scene_folder.mkdir(parents=True, exist_ok=True)

    # 3) Save the uploaded file as input.png
    input_path = scene_folder / "input.png"
    contents = upload_file.file.read()
    input_path.write_bytes(contents)

    # 4) Update the scene record with the real path
    scene.input_path = str(input_path)
    db.add(scene); db.commit(); db.refresh(scene)

    # 5) Kick off the Celery task
    celery_app.send_task(
        "app.tasks.reconstruct_scene",
        args=(scene.id,),
    )

    return scene

def fetch_status(db: Session, scene_id: int) -> Scene:
    return get_scene(db, scene_id)


def list_scenes_for_user(db: Session, owner_id: int) -> List[Scene]:
    return get_scenes_by_owner(db, owner_id)