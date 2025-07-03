from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Path, FileResponse
import os 
from sqlmodel import Session
from app.database import get_db
from app.models.scene import SceneRead
from app.services.scene_service import create_and_enqueue_scene, fetch_status

router = APIRouter(prefix="/scenes", tags=["scenes"])

@router.post(
    "/", response_model=SceneRead, status_code=status.HTTP_202_ACCEPTED
)
async def create_scene(
    file: UploadFile = File(...),
    owner_id: int = 1,             # replace with real auth!
    db: Session = Depends(get_db),
):
    """
    - Creates a Scene DB record  
    - Makes data/user_{owner}/scene_{id}/  
    - Saves input.png there  
    - Updates record + enqueues Celery task  
    """
    scene = create_and_enqueue_scene(db, owner_id, file)
    return SceneRead.from_orm(scene)

@router.get(
    "/{scene_id}", response_model=SceneRead, status_code=status.HTTP_200_OK
)
def get_scene_status(
    scene_id: int,
    db: Session = Depends(get_db),
):
    scene = fetch_status(db, scene_id)
    if not scene:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scene not found")
    return SceneRead.from_orm(scene)


def download_scene(
    owner_id: int = Path(..., description="ID of the scene owner"),
    scene_id: int = Path(..., description="ID of the scene"),
):
    """
    Streams back the .glb file for a completed scene.
    """
    # Base data directory (adjust if your data root is elsewhere)
    base_dir = os.path.abspath("data")

    # Build path to the GLB
    glb_path = os.path.join(
        base_dir,
        f"user_{owner_id}",
        f"scene_{scene_id}",
        "final",
        "model.glb",    # ← change if your filename differs
    )

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