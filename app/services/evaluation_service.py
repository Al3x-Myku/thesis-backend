import csv
import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException, status
from sqlmodel import Session

from app.models.evaluation import (
    StudySession,
    EvaluatorProfile,
    Variant2DEvaluation,
    VariantSetEvaluation,
    Commit3DEvaluation,
)
from app.repositories import evaluation_repo
from app.repositories.design_repo import get_variants_by_session, get_latest_session_for_scene
from app.repositories.scene_repo import get_scene_by_id


def _token_or_404(db: Session, token: str) -> StudySession:
    ss = evaluation_repo.get_study_session_by_token(db, token)
    if ss is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Study session not found.")
    if ss.expires_at and ss.expires_at < datetime.utcnow():
        raise HTTPException(status.HTTP_410_GONE, "This study session link has expired.")
    return ss


def _profile_or_404(db: Session, study_session_id: int, participant_code: str) -> EvaluatorProfile:
    profile = evaluation_repo.get_profile_by_code(db, study_session_id, participant_code)
    if profile is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Participant profile not found. Submit /profile first.",
        )
    return profile


# ── Study session management (researcher) ────────────────────────────────────

def create_study_session(
    db: Session,
    scene_id: int,
    user_id: int,
    design_session_id: Optional[int],
    notes: Optional[str],
    generation_engine_disclosed: bool,
    expires_at: Optional[datetime],
) -> StudySession:
    scene = get_scene_by_id(db, scene_id)
    if scene is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scene not found.")

    presentation_order: List[int] = []
    if design_session_id:
        variants = get_variants_by_session(db, design_session_id)
        presentation_order = [v.id for v in variants]

    ss = StudySession(
        token=str(uuid.uuid4()),
        scene_id=scene_id,
        design_session_id=design_session_id,
        created_by_user_id=user_id,
        presentation_order_json=json.dumps(presentation_order),
        generation_engine_disclosed=generation_engine_disclosed,
        notes=notes,
        expires_at=expires_at,
    )
    return evaluation_repo.create_study_session(db, ss)


def get_session_info(db: Session, token: str) -> dict:
    """Public: return session metadata + ordered variant image paths for the viewer page."""
    ss = _token_or_404(db, token)
    order = json.loads(ss.presentation_order_json)

    variant_images = []
    if ss.design_session_id:
        variants = get_variants_by_session(db, ss.design_session_id)
        vid_map = {v.id: v for v in variants}
        for vid in order:
            v = vid_map.get(vid)
            if v:
                variant_images.append({"variant_id": v.id, "index": v.variant_index})

    scene = get_scene_by_id(db, ss.scene_id)
    committed_glb = None
    if ss.design_session_id:
        from app.repositories.design_repo import get_session as get_design_session
        ds = get_design_session(db, ss.design_session_id)
        if ds and ds.committed_glb_path:
            committed_glb = ds.committed_glb_path

    return {
        "scene_id": ss.scene_id,
        "design_session_id": ss.design_session_id,
        "generation_engine_disclosed": ss.generation_engine_disclosed,
        "variant_count": len(variant_images),
        "variant_images": variant_images,
        "has_3d_commit": committed_glb is not None,
        "notes": ss.notes,
    }


# ── Participant form submissions ──────────────────────────────────────────────

def submit_evaluator_profile(db: Session, token: str, data: dict) -> EvaluatorProfile:
    ss = _token_or_404(db, token)
    code = data.get("participant_code", "").strip()
    if not code:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "participant_code is required.")

    existing = evaluation_repo.get_profile_by_code(db, ss.id, code)
    if existing:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A profile with this participant code already exists for this session.",
        )

    profile = EvaluatorProfile(study_session_id=ss.id, **data)
    return evaluation_repo.create_evaluator_profile(db, profile)


def submit_variant_evaluation(
    db: Session,
    token: str,
    participant_code: str,
    variant_id: int,
    data: dict,
) -> Variant2DEvaluation:
    ss = _token_or_404(db, token)
    profile = _profile_or_404(db, ss.id, participant_code)

    order = json.loads(ss.presentation_order_json)
    position = (order.index(variant_id) + 1) if variant_id in order else 0

    obj = Variant2DEvaluation(
        study_session_id=ss.id,
        evaluator_profile_id=profile.id,
        variant_id=variant_id,
        presentation_position=position,
        **data,
    )
    return evaluation_repo.create_variant_eval(db, obj)


def submit_set_evaluation(
    db: Session,
    token: str,
    participant_code: str,
    data: dict,
) -> VariantSetEvaluation:
    ss = _token_or_404(db, token)
    profile = _profile_or_404(db, ss.id, participant_code)

    obj = VariantSetEvaluation(
        study_session_id=ss.id,
        evaluator_profile_id=profile.id,
        scene_id=ss.scene_id,
        **data,
    )
    return evaluation_repo.create_set_eval(db, obj)


def submit_commit3d_evaluation(
    db: Session,
    token: str,
    participant_code: str,
    data: dict,
) -> Commit3DEvaluation:
    ss = _token_or_404(db, token)
    profile = _profile_or_404(db, ss.id, participant_code)

    obj = Commit3DEvaluation(
        study_session_id=ss.id,
        evaluator_profile_id=profile.id,
        scene_id=ss.scene_id,
        **data,
    )
    return evaluation_repo.create_commit_eval(db, obj)


# ── CSV export (researcher) ───────────────────────────────────────────────────

def _row_to_dict(obj) -> dict:
    return {c: getattr(obj, c) for c in obj.__fields__}


def export_csv_all(db: Session) -> str:
    """Export all evaluation data as a single multi-sheet-equivalent CSV blob."""
    from sqlmodel import select
    all_profiles = db.exec(select(EvaluatorProfile)).all()
    all_variant = evaluation_repo.get_all_variant_evals(db)
    all_set = evaluation_repo.get_all_set_evals(db)
    all_commit = evaluation_repo.get_all_commit_evals(db)

    def to_csv(rows) -> str:
        if not rows:
            return ""
        buf = io.StringIO()
        fields = list(rows[0].__fields__.keys())
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({f: getattr(r, f) for f in fields})
        return buf.getvalue()

    sections = {
        "evaluator_profiles": to_csv(all_profiles),
        "variant_2d_evaluations": to_csv(all_variant),
        "variant_set_evaluations": to_csv(all_set),
        "commit_3d_evaluations": to_csv(all_commit),
    }
    # Combine with section headers so it's readable as a single file
    parts = []
    for name, csv_data in sections.items():
        parts.append(f"### TABLE: {name} ###\n{csv_data}")
    return "\n\n".join(parts)


def export_variant_csv(db: Session) -> str:
    rows = evaluation_repo.get_all_variant_evals(db)
    if not rows:
        return ""
    buf = io.StringIO()
    fields = list(rows[0].__fields__.keys())
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({f: getattr(r, f) for f in fields})
    return buf.getvalue()


def export_set_csv(db: Session) -> str:
    rows = evaluation_repo.get_all_set_evals(db)
    if not rows:
        return ""
    buf = io.StringIO()
    fields = list(rows[0].__fields__.keys())
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({f: getattr(r, f) for f in fields})
    return buf.getvalue()


def export_commit_csv(db: Session) -> str:
    rows = evaluation_repo.get_all_commit_evals(db)
    if not rows:
        return ""
    buf = io.StringIO()
    fields = list(rows[0].__fields__.keys())
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({f: getattr(r, f) for f in fields})
    return buf.getvalue()


def export_profiles_csv(db: Session) -> str:
    from sqlmodel import select
    rows = db.exec(select(EvaluatorProfile)).all()
    if not rows:
        return ""
    buf = io.StringIO()
    fields = list(rows[0].__fields__.keys())
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({f: getattr(r, f) for f in fields})
    return buf.getvalue()
