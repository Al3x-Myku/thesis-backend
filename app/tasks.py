import app.models.user
import app.models.scene
import app.models.moodboard
import app.models.design
import app.models.evaluation
from pathlib import Path
from app.celery_app import celery_app
from app.database import engine
from sqlmodel import Session
from app.models.scene import Scene, SceneStatus
from app.reconstructor_pipeline import cleanup_gpu, full_reconstruction

@celery_app.task(bind=True)
def reconstruct_scene(self, scene_id: int) -> str:
    with Session(engine) as session:
        scene = session.get(Scene, scene_id)
        scene.status = SceneStatus.IN_PROGRESS
        session.add(scene)
        session.commit()
        input_path = scene.input_path

    scene_folder = Path(input_path).parent

    try:
        result_path = full_reconstruction(input_path, str(scene_folder))
    except Exception:
        with Session(engine) as session:
            sc = session.get(Scene, scene_id)
            sc.status = SceneStatus.FAILED
            session.add(sc)
            session.commit()
        raise
    finally:
        cleanup_gpu()

    with Session(engine) as session:
        sc = session.get(Scene, scene_id)
        sc.status      = SceneStatus.COMPLETED
        sc.progress    = 1.0
        sc.result_path = result_path
        session.add(sc)
        session.commit()

    return result_path


# ──────────────────────────────────────────────────────────────────────────────
# Generative Interior Design Zone (mood board → 2D iterations → 3D on commit)
#
# M1: these are no-ML SKELETON stubs that exercise the full persistence + status
# wiring by producing placeholder artifacts. M2–M5 replace the bodies with the
# real moodboard analysis / 2D restyle / 3D commit implementations.
# ──────────────────────────────────────────────────────────────────────────────

import os
import json
import shutil
from app.models.design import DesignSession, Variant, DesignStatus
from app.repositories import design_repo
from app.services import design_service


def _resolve_scene(session, scene_id: int):
    return session.get(Scene, scene_id)


def _placeholder_image(dest: Path, source_hint: Path | None = None) -> None:
    """Produce a stand-in PNG for the skeleton: copy a hint image if present,
    else write a tiny valid 1x1 PNG so downloads succeed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_hint and source_hint.is_file():
        shutil.copyfile(source_hint, dest)
        return
    # 1x1 transparent PNG
    png_1x1 = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6300010000050001a5f645400000000049454e44ae426082"
    )
    dest.write_bytes(png_1x1)


def _fail_session(session_id: int) -> None:
    with Session(engine) as session:
        sess = session.get(DesignSession, session_id)
        if sess is not None:
            sess.status = DesignStatus.FAILED
            session.add(sess)
            session.commit()


@celery_app.task(bind=True)
def analyze_moodboard(self, session_id: int) -> int:
    """Analyze mood-board images into a disentangled {palette, furniture} DesignSpec."""
    from app.moodboard import analyze_moodboard as run_analysis

    with Session(engine) as session:
        sess = session.get(DesignSession, session_id)
        scene = _resolve_scene(session, sess.scene_id)
        owner_id = scene.owner_id
        scene_id = sess.scene_id
        image_paths = [m.file_path for m in design_repo.get_moodboard_images(session, scene_id)]
        sess.status = DesignStatus.ANALYZING
        sess.progress = 0.1
        session.add(sess)
        session.commit()

    try:
        work_dir = str(design_service.moodboard_dir(owner_id, scene_id))
        spec = run_analysis(image_paths, work_dir, scene_id=str(scene_id))
        spec_json = json.dumps(spec.to_dict())
    except Exception:
        _fail_session(session_id)
        raise
    finally:
        cleanup_gpu()

    with Session(engine) as session:
        sess = session.get(DesignSession, session_id)
        sess.design_spec_json = spec_json
        sess.status = DesignStatus.READY
        sess.progress = 1.0
        session.add(sess)
        session.commit()

    return session_id


@celery_app.task(bind=True)
def generate_2d_variants(self, session_id: int, n: int = 4, mode: str = "full") -> list:
    """Render N 2D restyle variants of the room from the session's DesignSpec.

    mode: "full" — palette + moodboard furniture crops condition the diffusion.
          "palette_only" — only the colour palette is applied; furniture style from
          the moodboard is excluded so the room's own furniture is preserved cleanly.
    """
    from app.moodboard import DesignSpec
    from app.restyle_2d import generate_variants, build_variant_plans

    try:
        with Session(engine) as session:
            sess = session.get(DesignSession, session_id)
            scene = _resolve_scene(session, sess.scene_id)
            owner_id = scene.owner_id
            scene_id = sess.scene_id
            spec_json = sess.design_spec_json
            start_index = design_repo.next_variant_index(session, session_id)

        spec = DesignSpec.from_dict(json.loads(spec_json)) if spec_json else None
        if spec is None:
            raise RuntimeError("Cannot generate variants: session has no DesignSpec.")

        room_photo = str(design_service.scene_folder(owner_id, scene_id) / "input.png")
        folder = str(design_service.scene_folder(owner_id, scene_id))

        image_paths = generate_variants(
            room_photo, folder, str(scene_id), spec, n, start_index=start_index, mode=mode
        )
        plans = build_variant_plans(spec, n, start_index=start_index)

        created = []
        for plan, img_path in zip(plans, image_paths):
            with Session(engine) as session:
                variant = design_repo.create_variant(
                    session,
                    session_id=session_id,
                    variant_index=plan.variant_index,
                    seed=plan.seed,
                    image_path=img_path,
                    spec_snapshot_json=spec_json,
                )
                created.append(variant.id)

        with Session(engine) as session:
            design_repo.set_session_status(session, session_id, DesignStatus.READY, progress=1.0)
        return created
    except Exception:
        _fail_session(session_id)
        raise
    finally:
        cleanup_gpu()


@celery_app.task(bind=True)
def regenerate_variant(self, variant_id: int) -> int:
    """Read per-element feedback → locked dict, then re-render a child variant
    resampling only the unlocked fields (M4 adds pixel-exact freezing of locked
    regions; the engine already consumes the locked dict)."""
    from app.moodboard import DesignSpec
    from app.restyle_2d import generate_variants, build_variant_plans

    session_id = None
    try:
        with Session(engine) as session:
            variant = session.get(Variant, variant_id)
            session_id = variant.session_id
            sess = session.get(DesignSession, session_id)
            scene = _resolve_scene(session, sess.scene_id)
            owner_id = scene.owner_id
            scene_id = sess.scene_id
            spec_json = sess.design_spec_json
            locked = design_service.feedback_to_locked(session, variant_id)
            idx = design_repo.next_variant_index(session, session_id)

        spec = DesignSpec.from_dict(json.loads(spec_json)) if spec_json else None
        if spec is None:
            raise RuntimeError("Cannot regenerate: session has no DesignSpec.")

        room_photo = str(design_service.scene_folder(owner_id, scene_id) / "input.png")
        folder = str(design_service.scene_folder(owner_id, scene_id))

        image_paths = generate_variants(
            room_photo, folder, str(scene_id), spec, 1, start_index=idx, locked=locked,
            parent_image_path=variant.image_path,
        )
        plan = build_variant_plans(spec, 1, start_index=idx, locked=locked)[0]

        with Session(engine) as session:
            new_variant = design_repo.create_variant(
                session,
                session_id=session_id,
                variant_index=plan.variant_index,
                seed=plan.seed,
                image_path=image_paths[0],
                locked_fields_json=json.dumps(locked),
                spec_snapshot_json=spec_json,
                parent_variant_id=variant_id,
            )
            design_repo.set_session_status(session, session_id, DesignStatus.READY, progress=1.0)
            return new_variant.id
    except Exception:
        if session_id is not None:
            _fail_session(session_id)
        raise
    finally:
        cleanup_gpu()


@celery_app.task(bind=True)
def commit_3d(self, session_id: int, variant_id: int) -> str:
    """Realize the committed variant as a palette-tinted 3D .glb (M5).

    Calls commit_reconstruction which re-runs detect → build_mesh(palette) →
    position_meshes(shell_colors) and writes to committed/scene_positioned.glb."""
    from app.moodboard import DesignSpec
    from app.reconstructor_pipeline import commit_reconstruction

    try:
        with Session(engine) as session:
            sess = session.get(DesignSession, session_id)
            scene = _resolve_scene(session, sess.scene_id)
            owner_id = scene.owner_id
            scene_id = sess.scene_id
            spec_json = sess.design_spec_json
            sess.status = DesignStatus.COMMITTING
            sess.progress = 0.1
            session.add(sess)
            session.commit()

        spec = DesignSpec.from_dict(json.loads(spec_json)) if spec_json else None
        room_photo = str(design_service.scene_folder(owner_id, scene_id) / "input.png")
        folder = str(design_service.scene_folder(owner_id, scene_id))

        glb_path = commit_reconstruction(
            room_photo, folder, spec, scene_id=str(scene_id)
        )

        with Session(engine) as session:
            sess = session.get(DesignSession, session_id)
            sess.status = DesignStatus.COMMITTED
            sess.progress = 1.0
            sess.committed_variant_id = variant_id
            sess.committed_glb_path = glb_path
            session.add(sess)
            session.commit()
        return glb_path
    except Exception:
        _fail_session(session_id)
        raise
    finally:
        cleanup_gpu()
