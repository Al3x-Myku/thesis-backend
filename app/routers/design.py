from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Path as PathParam, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session

from app.database import get_db
from app.core.security import get_current_user
from app.repositories import design_repo
from app.services import design_service

router = APIRouter(prefix="/scenes/{scene_id}/design", tags=["design"])


class FeedbackItem(BaseModel):
    element_type: str           # "palette" | "slot"
    element_key: str = ""        # "" for palette, slot id for slots
    liked: bool = True


def _latest_session_or_404(db: Session, owner_id: int, scene_id: int):
    # Ownership is enforced inside the service via _scene_or_404.
    design_service._scene_or_404(db, scene_id, owner_id)
    sess = design_repo.get_latest_session_for_scene(db, scene_id)
    if sess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No design session for this scene yet.")
    return sess


# ── Mood board + session ─────────────────────────────────────────────────────

@router.post("/moodboard", status_code=status.HTTP_202_ACCEPTED)
def upload_moodboard(
    scene_id: int = PathParam(...),
    files: List[UploadFile] = File(..., description="Mood board reference images"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sess = design_service.create_moodboard_session(db, current_user.id, scene_id, files)
    return design_service.session_summary(db, sess)


@router.get("/", status_code=status.HTTP_200_OK)
def get_design(
    scene_id: int = PathParam(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sess = _latest_session_or_404(db, current_user.id, scene_id)
    return design_service.session_summary(db, sess)


# ── Variants ─────────────────────────────────────────────────────────────────

@router.post("/variants", status_code=status.HTTP_202_ACCEPTED)
def generate_variants(
    scene_id: int = PathParam(...),
    n: int = Query(4, ge=1, le=12),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sess = _latest_session_or_404(db, current_user.id, scene_id)
    sess = design_service.request_variants(db, current_user.id, sess.id, n)
    return design_service.session_summary(db, sess)


@router.get("/variants", status_code=status.HTTP_200_OK)
def list_variants(
    scene_id: int = PathParam(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sess = _latest_session_or_404(db, current_user.id, scene_id)
    variants = design_service.list_variants(db, current_user.id, sess.id)
    return [
        {
            "id": v.id,
            "variant_index": v.variant_index,
            "seed": v.seed,
            "parent_variant_id": v.parent_variant_id,
            "locked_fields": v.locked_fields_json,
        }
        for v in variants
    ]


@router.get("/variants/{variant_id}/image", response_class=FileResponse)
def get_variant_image(
    scene_id: int = PathParam(...),
    variant_id: int = PathParam(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    variant = design_service.get_variant_owned(db, current_user.id, variant_id)
    if not Path(variant.image_path).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Variant image not found on disk.")
    return FileResponse(
        path=variant.image_path,
        media_type="image/png",
        filename=f"variant_{variant.variant_index}.png",
    )


@router.post("/variants/{variant_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
def submit_feedback(
    scene_id: int = PathParam(...),
    variant_id: int = PathParam(...),
    items: List[FeedbackItem] = ...,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    design_service.submit_feedback(
        db, current_user.id, variant_id, [it.model_dump() for it in items]
    )
    return None


@router.post("/variants/{variant_id}/regenerate", status_code=status.HTTP_202_ACCEPTED)
def regenerate(
    scene_id: int = PathParam(...),
    variant_id: int = PathParam(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sess = design_service.regenerate_variant(db, current_user.id, variant_id)
    return design_service.session_summary(db, sess)


@router.post("/variants/{variant_id}/commit", status_code=status.HTTP_202_ACCEPTED)
def commit(
    scene_id: int = PathParam(...),
    variant_id: int = PathParam(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sess = design_service.commit_variant(db, current_user.id, variant_id)
    return design_service.session_summary(db, sess)


@router.get("/commit/download", response_class=FileResponse)
def download_committed(
    scene_id: int = PathParam(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sess = _latest_session_or_404(db, current_user.id, scene_id)
    if not sess.committed_glb_path or not Path(sess.committed_glb_path).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No committed 3D model available.")
    return FileResponse(
        path=sess.committed_glb_path,
        media_type="model/gltf-binary",
        filename=f"scene_{scene_id}_design.glb",
    )
