# app/routers/scenes.py
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Path
from fastapi.responses import FileResponse
from typing import List
import os

from sqlmodel import Session
from app.database import get_db
from app.core.config import settings
from app.core.security import get_current_user
from app.models.scene import SceneRead
from app.services.scene_service import (
    create_and_enqueue_scene,
    fetch_status,
    list_scenes_for_user,
    delete_scene as svc_delete_scene
)

router = APIRouter(prefix="/scenes", tags=["scenes"])


@router.post(
    "/",
    response_model=SceneRead,
    status_code=status.HTTP_202_ACCEPTED
)
async def create_scene(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new Scene for the authenticated user and enqueue reconstruction.
    """
    scene = create_and_enqueue_scene(db, current_user.id, file)
    return SceneRead.from_orm(scene)


@router.get(
    "/",
    response_model=List[SceneRead],
    status_code=status.HTTP_200_OK,
    summary="List all scenes for the current user"
)
def list_my_scenes(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns all scenes (creations) belonging to the logged-in user.
    """
    scenes = list_scenes_for_user(db, current_user.id)
    return [SceneRead.from_orm(s) for s in scenes]


@router.get(
    "/{scene_id}",
    response_model=SceneRead,
    status_code=status.HTTP_200_OK
)
def get_scene_status(
    scene_id: int = Path(..., description="The ID of the scene to fetch"),
    db: Session = Depends(get_db),
):
    """
    Fetch the status (and metadata) of a specific Scene.
    """
    scene = fetch_status(db, scene_id)
    if not scene:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scene not found"
        )
    return SceneRead.from_orm(scene)


@router.delete(
    "/{scene_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a scene"
)
def delete_scene(
    scene_id: int = Path(..., description="The ID of the scene to delete"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a scene owned by the authenticated user.
    """
    scene = fetch_status(db, scene_id)
    if not scene or scene.owner_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scene not found or not owned by you.")
    svc_delete_scene(db, scene_id, current_user.id)
    return None


@router.get(
    "/{scene_id}/download",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Download the final .glb file for a completed scene"
)
def download_scene(
    scene_id: int = Path(..., description="The ID of the scene to download"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Stream the final .glb file for a completed Scene, only if owned by the current user.
    """
    scene = fetch_status(db, scene_id)
    if not scene or scene.owner_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scene not found or not owned by you.")

    # Build path to the final glb
    data_dir = getattr(settings, "DATA_DIR", os.getenv("DATA_DIR", "./data"))
    base = os.path.join(
        data_dir,
        f"user_{current_user.id}",
        f"scene_{scene_id}",
        "final"
    )
    glb_path = os.path.join(base, "scene_positioned.glb")
    if not os.path.isfile(glb_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="3D model not found"
        )

    return FileResponse(
        path=glb_path,
        media_type="model/gltf-binary",
        filename=f"scene_{scene_id}.glb"
    )


@router.get(
    "/{scene_id}/input",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Download the original input image for a scene"
)
def get_scene_input(
    scene_id: int = Path(..., description="ID of the scene"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns the raw 2D image that was uploaded for this scene, provided it belongs to the authenticated user.
    """
    scene = fetch_status(db, scene_id)
    if not scene or scene.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scene not found or not owned by you."
        )

    data_dir = getattr(settings, "DATA_DIR", os.getenv("DATA_DIR", "./data"))
    input_path = os.path.join(
        data_dir,
        f"user_{current_user.id}",
        f"scene_{scene_id}",
        "input.png"
    )

    if not os.path.isfile(input_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original input image not found on server."
        )

    return FileResponse(
        path=input_path,
        media_type="image/png",
        filename=f"scene_{scene_id}_input.png"
    )
