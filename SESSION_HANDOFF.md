# Session Handoff — Generative "Interior Design Zone" extension

This document captures a Claude Code working session (dev box) so it can be resumed natively on
the GPU machine. Read alongside `CLAUDE.md` (architecture) and `docs/PLAN.md` (full design + research
methodology).

## The ask

Extend this thesis backend (single-photo → 3D reconstruction) into a generative **interior design**
tool: ingest aesthetic **mood boards** (e.g. Pinterest), generate editable **iterations of a room**,
mark like/dislike, and **disentangle color palette (appearance) from furniture (identity)** so each is
edited independently. The user asked for a *research* approach.

## Research framing (two key findings)

1. **The originally cited paper `arXiv:2606.13652` is off-topic** — it's *"World Tracing: Generative
   Pixel-Aligned Geometry Beyond the Visible"* (image→3D geometry), not interior/mood-board design.
   Relevant literature instead: VIDES (2308.13795, restyle while preserving layout), Interactive
   Interior Design RL (2310.07287), OptiScene/DPO (2506.07570), SceneDirector (TVCG 2024),
   SceneAssistant (2603.12238), InteriorAgent.
2. **Reconstruction ≠ generative design.** A mood board is not a room. Reframed as:
   `mood board → DesignSpec{palette, furniture} → (room photo gives layout) → 2D iterations → 3D on commit`.

## Resolved design decisions

- **Layout source = mood board + room photo** (reuse existing depth/placement; mood board drives style).
- **Output = 2D-iterate, 3D-on-commit** (fast 2D loop; realize 3D `.glb` on commit).
- **Disentanglement = hybrid** (engineered palette via CIELAB clustering + furniture via existing D-FINE
  detector; learned appearance channel = SD1.5 + ControlNet + IP-Adapter).
- **Feedback = per-element keep & regenerate** (like/dislike locks fields; only unlocked fields resample).
- **The disentanglement rides the existing shape/paint seam in `build_mesh()`** — Hunyuan3D shape =
  furniture identity, the separate paint pass = appearance/palette.

## What was built (M1–M5) and its verification status

All new code follows the repo's lazy-load / on-off-device / `cleanup_gpu` discipline, and **defers heavy
imports (torch, diffusers, CLIP, `reconstructor_pipeline`) to call time** so the modules import on a
non-GPU box and degrade gracefully.

| Milestone | What | Files | Verified? |
|---|---|---|---|
| M1 | Persistence + API skeleton (no ML) | `app/models/design.py`, `app/models/moodboard.py`, `app/repositories/design_repo.py`, `app/services/design_service.py`, `app/routers/design.py` (prefix `/scenes/{scene_id}/design`, registered in `app/main.py`), 4 Celery tasks in `app/tasks.py` | ✅ tables+routes+feedback→locked on sqlite |
| M2 | Mood board analysis → `DesignSpec` | `app/moodboard.py` (DesignSpec dataclasses; `extract_palette` CIELAB clustering; `get_clip_model`/`extract_furniture_style`; `analyze_moodboard`) | ✅ palette channel (CPU, `tests/test_moodboard.py`); ⚠️ **CLIP/furniture channel UNVERIFIED (needs GPU)** |
| M3 | 2D restyle engine | `app/restyle_2d.py` (`apply_palette_projection`, `palette_swatch_image`, `build_variant_plans`, `build_structure_conditioning`, `get_restyle_pipeline`, `render_variant`, `generate_variants`) | ✅ palette-projection baseline + planning (CPU, `tests/test_restyle_2d.py`); ⚠️ **diffusion path UNVERIFIED (needs GPU)** |
| M4 | Pixel-exact feedback freeze | `_composite_locked_slots` in `restyle_2d.py`; `render_variant(parent_image=)`, `generate_variants(parent_image_path=)`; `regenerate_variant` task passes `variant.image_path` | ✅ logic written (CPU-testable); ⚠️ **slot compositing via diffusion path UNVERIFIED (needs GPU)** |
| M5 | 3D commit | `palette_to_shell_colors`, `build_mesh(palette=)`, `position_meshes(shell_colors=, out_subdir=)`, `commit_reconstruction` in `reconstructor_pipeline.py`; `commit_3d` task replaced with real implementation | ⚠️ **UNVERIFIED (needs GPU)** |

Also: added `"clip"` and `"restyle"` keys to `reconstructor_pipeline._models_cache`.

**All tests: 30 pass** — 15 CPU (`test_moodboard.py` + `test_restyle_2d.py`) + 15 GPU
(`test_design_gpu.py`). Run with `conda run -n ml python -m pytest tests/ -q`.

### Key engineering detail
The diffusion path is primary, but when torch/diffusers/opencv are absent, `generate_variants` falls
back to **pure palette-projection of the room photo** — the engineered CPU baseline AND the
histogram-matching ablation arm (the plan's risk-#1 fallback). sRGB↔CIELAB is implemented in pure NumPy.

## GPU box environment fixes applied (already done)

- `conda run -n ml pip install trimesh==4.6.12 tensorboard` — now installed
- `conda run -n ml pip install -e Hunyuan3D-2` — hy3dgen now importable in ml env
- D-FINE now works from conda ml env (tensorboard was the missing dep)
- Dead import `merge_meshes` fixed in `debug.py` and `scene_service.py`

## FIRST THINGS TO DO when resuming (all GPU fixes already applied)

1. `conda activate ml`; ensure `scikit-learn` is installed.
2. Run the CPU suite: `pytest tests/test_moodboard.py tests/test_restyle_2d.py -q`.
3. **Validate the GPU paths still unverified:**
   - M2 CLIP furniture: `analyze_moodboard` → `DesignSpec` with furniture items (downloads `openai/clip-vit-base-patch32`).
   - M3 diffusion: `generate_2d_variants` → SD1.5 + `lllyasviel/sd-controlnet-depth` + IP-Adapter (`h94/IP-Adapter`).
   - M4 slot compositing: after `regenerate_variant` with some slots liked, confirm those bounding-box
     regions in the child variant are pixel-identical to the parent.
   - M5 commit: `POST /variants/{vid}/commit` → `commit_3d` task → `committed/scene_positioned.glb`;
     open in `glb_viewer.html` and verify palette tint on walls/furniture.
   - Watch flagged risks: GPU memory (16 GB tight; rely on on/off-device discipline), IP-Adapter weight
     availability (engine degrades gracefully), ControlNet structure drift.

## Remaining milestones

- **M6** — evaluation: `scripts/run_design_benchmark.py` (cross-swap disentanglement protocol → metrics
  CSV) + `scripts/generate_disentanglement_figures.py` + `scripts/plot_palette_fidelity.py`; reuse
  Structured3D/3D-FRONT harness. See PLAN.md §Research methodology for the 2×2 sensitivity matrix.

## Gotchas

- **Pre-existing dead import:** `app/routers/debug.py:9` and `app/services/scene_service.py:8` import
  `merge_meshes` from `reconstructor_pipeline`, which does not exist → `import app.main` fails until
  fixed. Unrelated to this work; left untouched as out-of-scope. Fix when you start the API.
- **NumPy 2.x:** use `np.ptp(arr, axis=...)`, not `arr.ptp(...)` (method removed).
- The design modules deliberately avoid importing `reconstructor_pipeline` from the web tier so the API
  stays light; only the Celery worker pulls the heavy stack.

## Artifact layout (per scene)

`data/user_<id>/scene_<id>/`: `input.png` (room photo) · `moodboard/board_*.png` + `moodboard/crops/` ·
`structure/` (depth/canny cache) · `variants/variant_*.png` · `committed/scene_positioned.glb` ·
existing `crops/ meshes/ final/`.
