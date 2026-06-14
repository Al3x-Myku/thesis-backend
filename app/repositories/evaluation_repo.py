from typing import Optional, List
from sqlmodel import Session, select

from app.models.evaluation import (
    StudySession,
    EvaluatorProfile,
    Variant2DEvaluation,
    VariantSetEvaluation,
    Commit3DEvaluation,
)


# ── StudySession ──────────────────────────────────────────────────────────────

def create_study_session(db: Session, obj: StudySession) -> StudySession:
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_study_session(db: Session, session_id: int) -> Optional[StudySession]:
    return db.get(StudySession, session_id)

def get_study_session_by_token(db: Session, token: str) -> Optional[StudySession]:
    stmt = select(StudySession).where(StudySession.token == token)
    return db.exec(stmt).first()

def list_study_sessions(db: Session, scene_id: Optional[int] = None) -> List[StudySession]:
    stmt = select(StudySession)
    if scene_id is not None:
        stmt = stmt.where(StudySession.scene_id == scene_id)
    stmt = stmt.order_by(StudySession.created_at.desc())
    return db.exec(stmt).all()


# ── EvaluatorProfile ──────────────────────────────────────────────────────────

def create_evaluator_profile(db: Session, obj: EvaluatorProfile) -> EvaluatorProfile:
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_evaluator_profile(db: Session, profile_id: int) -> Optional[EvaluatorProfile]:
    return db.get(EvaluatorProfile, profile_id)

def get_profile_by_code(db: Session, study_session_id: int, participant_code: str) -> Optional[EvaluatorProfile]:
    stmt = select(EvaluatorProfile).where(
        EvaluatorProfile.study_session_id == study_session_id,
        EvaluatorProfile.participant_code == participant_code,
    )
    return db.exec(stmt).first()

def list_profiles_for_session(db: Session, study_session_id: int) -> List[EvaluatorProfile]:
    stmt = select(EvaluatorProfile).where(EvaluatorProfile.study_session_id == study_session_id)
    return db.exec(stmt).all()


# ── Variant2DEvaluation ───────────────────────────────────────────────────────

def create_variant_eval(db: Session, obj: Variant2DEvaluation) -> Variant2DEvaluation:
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_variant_evals_for_session(db: Session, study_session_id: int) -> List[Variant2DEvaluation]:
    stmt = select(Variant2DEvaluation).where(Variant2DEvaluation.study_session_id == study_session_id)
    return db.exec(stmt).all()

def get_variant_evals_by_profile(db: Session, evaluator_profile_id: int) -> List[Variant2DEvaluation]:
    stmt = select(Variant2DEvaluation).where(Variant2DEvaluation.evaluator_profile_id == evaluator_profile_id)
    return db.exec(stmt).all()

def get_all_variant_evals(db: Session) -> List[Variant2DEvaluation]:
    return db.exec(select(Variant2DEvaluation)).all()


# ── VariantSetEvaluation ──────────────────────────────────────────────────────

def create_set_eval(db: Session, obj: VariantSetEvaluation) -> VariantSetEvaluation:
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_set_evals_for_session(db: Session, study_session_id: int) -> List[VariantSetEvaluation]:
    stmt = select(VariantSetEvaluation).where(VariantSetEvaluation.study_session_id == study_session_id)
    return db.exec(stmt).all()

def get_all_set_evals(db: Session) -> List[VariantSetEvaluation]:
    return db.exec(select(VariantSetEvaluation)).all()


# ── Commit3DEvaluation ────────────────────────────────────────────────────────

def create_commit_eval(db: Session, obj: Commit3DEvaluation) -> Commit3DEvaluation:
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_commit_evals_for_session(db: Session, study_session_id: int) -> List[Commit3DEvaluation]:
    stmt = select(Commit3DEvaluation).where(Commit3DEvaluation.study_session_id == study_session_id)
    return db.exec(stmt).all()

def get_all_commit_evals(db: Session) -> List[Commit3DEvaluation]:
    return db.exec(select(Commit3DEvaluation)).all()
