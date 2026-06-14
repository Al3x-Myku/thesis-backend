from typing import Optional, List
from sqlmodel import Session, select

from app.models.moodboard import MoodBoardImage
from app.models.design import DesignSession, Variant, ElementFeedback, DesignStatus


# ── DesignSession ────────────────────────────────────────────────────────────

def create_session(db: Session, scene_id: int) -> DesignSession:
    sess = DesignSession(scene_id=scene_id)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess

def get_session(db: Session, session_id: int) -> Optional[DesignSession]:
    return db.get(DesignSession, session_id)

def get_sessions_by_scene(db: Session, scene_id: int) -> List[DesignSession]:
    stmt = select(DesignSession).where(DesignSession.scene_id == scene_id)
    return db.exec(stmt).all()

def get_latest_session_for_scene(db: Session, scene_id: int) -> Optional[DesignSession]:
    stmt = (
        select(DesignSession)
        .where(DesignSession.scene_id == scene_id)
        .order_by(DesignSession.id.desc())
    )
    return db.exec(stmt).first()

def update_session(db: Session, sess: DesignSession, **fields) -> DesignSession:
    for k, v in fields.items():
        setattr(sess, k, v)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess

def set_session_status(
    db: Session, session_id: int, status: DesignStatus, progress: Optional[float] = None
) -> Optional[DesignSession]:
    sess = db.get(DesignSession, session_id)
    if sess is None:
        return None
    sess.status = status
    if progress is not None:
        sess.progress = progress
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess


# ── MoodBoardImage ───────────────────────────────────────────────────────────

def add_moodboard_image(
    db: Session, scene_id: int, file_path: str, upload_order: int = 0
) -> MoodBoardImage:
    img = MoodBoardImage(scene_id=scene_id, file_path=file_path, upload_order=upload_order)
    db.add(img)
    db.commit()
    db.refresh(img)
    return img

def get_moodboard_images(db: Session, scene_id: int) -> List[MoodBoardImage]:
    stmt = (
        select(MoodBoardImage)
        .where(MoodBoardImage.scene_id == scene_id)
        .order_by(MoodBoardImage.upload_order)
    )
    return db.exec(stmt).all()


# ── Variant ──────────────────────────────────────────────────────────────────

def create_variant(
    db: Session,
    session_id: int,
    variant_index: int,
    seed: int,
    image_path: str,
    locked_fields_json: str = "{}",
    spec_snapshot_json: Optional[str] = None,
    parent_variant_id: Optional[int] = None,
) -> Variant:
    variant = Variant(
        session_id=session_id,
        variant_index=variant_index,
        seed=seed,
        image_path=image_path,
        locked_fields_json=locked_fields_json,
        spec_snapshot_json=spec_snapshot_json,
        parent_variant_id=parent_variant_id,
    )
    db.add(variant)
    db.commit()
    db.refresh(variant)
    return variant

def get_variant(db: Session, variant_id: int) -> Optional[Variant]:
    return db.get(Variant, variant_id)

def get_variants_by_session(db: Session, session_id: int) -> List[Variant]:
    stmt = (
        select(Variant)
        .where(Variant.session_id == session_id)
        .order_by(Variant.variant_index)
    )
    return db.exec(stmt).all()

def next_variant_index(db: Session, session_id: int) -> int:
    existing = get_variants_by_session(db, session_id)
    return (max((v.variant_index for v in existing), default=-1)) + 1


# ── ElementFeedback ──────────────────────────────────────────────────────────

def upsert_feedback(
    db: Session, variant_id: int, element_type: str, element_key: str, liked: bool
) -> ElementFeedback:
    stmt = select(ElementFeedback).where(
        ElementFeedback.variant_id == variant_id,
        ElementFeedback.element_type == element_type,
        ElementFeedback.element_key == element_key,
    )
    fb = db.exec(stmt).first()
    if fb is None:
        fb = ElementFeedback(
            variant_id=variant_id,
            element_type=element_type,
            element_key=element_key,
            liked=liked,
        )
    else:
        fb.liked = liked
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb

def get_feedback_for_variant(db: Session, variant_id: int) -> List[ElementFeedback]:
    stmt = select(ElementFeedback).where(ElementFeedback.variant_id == variant_id)
    return db.exec(stmt).all()
