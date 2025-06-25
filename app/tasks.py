# app/tasks.py

import app.models.user    # registers User
import app.models.scene   # registers Scene

from pathlib import Path
from app.celery_app import celery_app
from app.database import engine
from sqlmodel import Session
from app.models.scene import Scene, SceneStatus
from app.reconstructor_pipeline import (
    detect_objects,
    build_mesh,
    merge_meshes,
)

@celery_app.task(bind=True)
def reconstruct_scene(self, scene_id: int) -> str:
    # 1) Load the scene, mark IN_PROGRESS, grab input_path
    with Session(engine) as session:
        scene = session.get(Scene, scene_id)
        scene.status = SceneStatus.IN_PROGRESS
        session.add(scene)
        session.commit()

        input_path = scene.input_path
    scene_folder = Path(input_path).parent

    # 2) Run the full pipeline
    try:
        # 2.a) Detect and crop with D-FINE
        crop_paths = detect_objects(input_path, str(scene_folder))
        total = len(crop_paths)
        mesh_paths: list[str] = []

        # 2.b) For each crop: build/pain mesh and update progress
        for idx, crop in enumerate(crop_paths, start=1):
            mesh = build_mesh(crop, str(scene_folder))
            mesh_paths.append(mesh)

            with Session(engine) as session:
                sc = session.get(Scene, scene_id)
                sc.progress = idx / total
                session.add(sc)
                session.commit()

        # 2.c) Merge into one final GLB
        result_path = merge_meshes(mesh_paths, str(scene_folder))

    except Exception as e:
        # mark FAILED
        with Session(engine) as session:
            sc = session.get(Scene, scene_id)
            sc.status = SceneStatus.FAILED
            session.add(sc)
            session.commit()
        # re-raise so Celery records the failure
        raise

    # 3) Mark COMPLETE
    with Session(engine) as session:
        sc = session.get(Scene, scene_id)
        sc.status      = SceneStatus.COMPLETED
        sc.progress    = 1.0
        sc.result_path = result_path
        session.add(sc)
        session.commit()

    return result_path
