import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from sqlmodel import Session
from fastapi import HTTPException, status

from app.celery_app import celery_app
from app.models.scene import Scene
from app.models.design import DesignSession, Variant, DesignStatus
from app.repositories.scene_repo import get_scene_by_id
from app.repositories import design_repo

DATA_ROOT = Path(os.getenv("DATA_DIR", "./data"))


def _scene_or_404(db: Session, scene_id: int, owner_id: int) -> Scene:
    scene = get_scene_by_id(db, scene_id)
    if not scene or scene.owner_id != owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scene not found or not owned by you.")
    return scene


def scene_folder(owner_id: int, scene_id: int) -> Path:
    return DATA_ROOT / f"user_{owner_id}" / f"scene_{scene_id}"


def moodboard_dir(owner_id: int, scene_id: int) -> Path:
    return scene_folder(owner_id, scene_id) / "moodboard"


def variants_dir(owner_id: int, scene_id: int) -> Path:
    return scene_folder(owner_id, scene_id) / "variants"


# ── Mood board upload + session creation ─────────────────────────────────────

def create_moodboard_session(
    db: Session, owner_id: int, scene_id: int, upload_files: List
) -> DesignSession:
    """Persist mood-board images, create a DesignSession, dispatch analysis."""
    _scene_or_404(db, scene_id, owner_id)

    if not upload_files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No mood board images provided.")

    mb_dir = moodboard_dir(owner_id, scene_id)
    mb_dir.mkdir(parents=True, exist_ok=True)

    session = design_repo.create_session(db, scene_id)

    for order, uf in enumerate(upload_files):
        suffix = Path(uf.filename or f"board_{order}.png").suffix or ".png"
        dest = mb_dir / f"board_{order}{suffix}"
        dest.write_bytes(uf.file.read())
        design_repo.add_moodboard_image(db, scene_id, str(dest), upload_order=order)

    celery_app.send_task("app.tasks.analyze_moodboard", args=(session.id,))
    return session


def get_session(db: Session, owner_id: int, session_id: int) -> DesignSession:
    sess = design_repo.get_session(db, session_id)
    if sess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Design session not found.")
    _scene_or_404(db, sess.scene_id, owner_id)
    return sess


def session_summary(db: Session, sess: DesignSession) -> Dict[str, Any]:
    """Compact, client-facing view: status + palette swatches + furniture categories."""
    spec = json.loads(sess.design_spec_json) if sess.design_spec_json else None
    palette = None
    furniture = None
    if spec:
        palette = [s.get("rgb") for s in spec.get("palette", {}).get("swatches", [])]
        furniture = [
            item.get("category_name")
            for item in spec.get("furniture", {}).get("items", [])
        ]
    return {
        "id": sess.id,
        "scene_id": sess.scene_id,
        "status": sess.status,
        "progress": sess.progress,
        "palette": palette,
        "furniture": furniture,
        "committed_glb": sess.committed_glb_path,
    }


# ── Variant generation / feedback / regeneration ─────────────────────────────

def request_variants(db: Session, owner_id: int, session_id: int, n: int) -> DesignSession:
    sess = get_session(db, owner_id, session_id)
    if sess.status not in (DesignStatus.READY, DesignStatus.COMMITTED):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Session not ready for variant generation (status={sess.status}).",
        )
    design_repo.set_session_status(db, session_id, DesignStatus.GENERATING, progress=0.0)
    celery_app.send_task("app.tasks.generate_2d_variants", args=(session_id, n))
    return design_repo.get_session(db, session_id)


def list_variants(db: Session, owner_id: int, session_id: int) -> List[Variant]:
    get_session(db, owner_id, session_id)
    return design_repo.get_variants_by_session(db, session_id)


def get_variant_owned(db: Session, owner_id: int, variant_id: int) -> Variant:
    variant = design_repo.get_variant(db, variant_id)
    if variant is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Variant not found.")
    sess = design_repo.get_session(db, variant.session_id)
    if sess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Variant session not found.")
    _scene_or_404(db, sess.scene_id, owner_id)
    return variant


def submit_feedback(
    db: Session, owner_id: int, variant_id: int, items: List[Dict[str, Any]]
) -> None:
    get_variant_owned(db, owner_id, variant_id)
    for it in items:
        element_type = it.get("element_type")
        if element_type not in ("palette", "slot"):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Invalid element_type: {element_type!r}",
            )
        design_repo.upsert_feedback(
            db,
            variant_id=variant_id,
            element_type=element_type,
            element_key=str(it.get("element_key", "")),
            liked=bool(it.get("liked", True)),
        )


def feedback_to_locked(db: Session, variant_id: int) -> Dict[str, Any]:
    """Translate stored feedback into the locked-fields dict consumed by the restyle engine.

    liked == lock (keep), disliked == unlock (resample).
    Shape: {"palette": bool, "slots": {slot_key: bool}}
    """
    locked: Dict[str, Any] = {"palette": False, "slots": {}}
    for fb in design_repo.get_feedback_for_variant(db, variant_id):
        if fb.element_type == "palette":
            locked["palette"] = fb.liked
        elif fb.element_type == "slot":
            locked["slots"][fb.element_key] = fb.liked
    return locked


def regenerate_variant(db: Session, owner_id: int, variant_id: int) -> DesignSession:
    variant = get_variant_owned(db, owner_id, variant_id)
    sess = design_repo.get_session(db, variant.session_id)
    design_repo.set_session_status(db, sess.id, DesignStatus.GENERATING, progress=0.0)
    celery_app.send_task("app.tasks.regenerate_variant", args=(variant_id,))
    return design_repo.get_session(db, sess.id)


def commit_variant(db: Session, owner_id: int, variant_id: int) -> DesignSession:
    variant = get_variant_owned(db, owner_id, variant_id)
    sess = design_repo.get_session(db, variant.session_id)
    design_repo.update_session(
        db, sess, status=DesignStatus.COMMITTING, progress=0.0, committed_variant_id=variant_id
    )
    celery_app.send_task("app.tasks.commit_3d", args=(sess.id, variant_id))
    return design_repo.get_session(db, sess.id)
