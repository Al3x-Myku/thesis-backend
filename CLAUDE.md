# CLAUDE.md — project context for AI assistants

## What this is

Single-photo → textured 3D scene reconstruction, extended with a generative **Interior Design Zone**:
upload a room photo + mood board → editable 2D iterations (palette and furniture disentangled) → commit to 3D GLB.
Research thesis project. The disentanglement claim is the core contribution.

## Stack

- Python 3.12, FastAPI, SQLModel (MySQL via pymysql), Celery (DB as broker — no Redis)
- PyTorch 2.11 + CUDA 12.8, diffusers 0.38.0, transformers 5.11.0
- RTX 5080 16 GB GPU box; dev box is CPU-only (python3.11, no venv, no torch)
- Hunyuan3D-2 for 3D generation; D-FINE (Objects365) for detection
- SD1.5 + ControlNet(depth) + IP-Adapter for 2D restyle

## Milestones status

- M1 skeleton/persistence: done
- M2 moodboard analysis (CIELAB palette + CLIP furniture): done
- M3 2D restyle (diffusion + palette projection): done
- M4 feedback freeze (pixel-exact locked slots): done
- M5 3D commit (palette-tinted Hunyuan3D-2): done
- M6 evaluation harness: in progress (scripts/run_structured3d_benchmark.py exists)
- Human validation system: done (app/routers/evaluation.py)

All 43 GPU tests pass as of 2026-06-14 (commit fefd1e1).

## Key files

```
app/reconstructor_pipeline.py   full GPU pipeline; _models_cache dict; MESH_ENGINE env var
app/restyle_2d.py               2D restyle; generate_variants(mode="full"|"palette_only")
app/moodboard.py                DesignSpec{Palette, FurnitureSpec}; analyze_moodboard()
app/wall_pipeline.py            panoramic path; full_reconstruction_panoramic()
app/panoramic_depth.py          MTPano depth for panoramas; DepthPro fallback
app/instantmesh_wrapper.py      InstantMesh ablation arm subprocess wrapper
app/routers/evaluation.py       human validation API; 4 forms; CSV export
app/routers/design.py           design zone API; mode= query param on /variants
app/tasks.py                    Celery tasks; imports all models for init_db()
scripts/batch_run.py            batch pipeline runner (API-based, resumable)
```

## Critical gotchas

**IP-Adapter + attention slicing**: never call `enable_attention_slicing()` after `load_ip_adapter()`.
It replaces `IPAdapterAttnProcessor2_0` with `SlicedAttnProcessor` which fails on tuple `encoder_hidden_states`.

**merge_meshes**: dead import in debug.py:9 and scene_service.py:8 — do not use, do not fix by importing it elsewhere.

**D-FINE**: subprocess via dfine_wrapper.py with PYTHONPATH=D-FINE/ prepended. HF fallback (`ustc-community/dfine_x_obj365`) auto-activates on subprocess failure.

**CLIP transformers ≥5**: `get_image_features()` returns `BaseModelOutputWithPooling`, not a tensor. Unwrap via `.pooler_output`.

**Model cache discipline**: always `model.to(device)` before inference, `model.to("cpu")` + `cleanup_gpu(False)` after. Never skip this.

**New DB tables**: must be imported in app/tasks.py at module level so `init_db()` creates them.

**numpy 2.x**: use `np.ptp(arr, axis=...)` not `arr.ptp(...)` (method removed).

## Environment variables

```
SECRET_KEY, DATABASE_URL                          required
DATA_DIR=./data, PIPELINE_DEVICE=cuda:0           optional
MESH_ENGINE=hunyuan|instantmesh                   ablation switch (default: hunyuan)
INSTANTMESH_ROOT=./InstantMesh                    clone TencentARC/InstantMesh here
MTPANO_MODEL_ID=Evergreen0929/MTPano              panoramic depth model
DFINE_ROOT, DFINE_CONFIG, DFINE_CHECKPT           D-FINE paths
HUNYUAN_SHAPEDIR, HUNYUAN_PAINTDIR                Hunyuan3D-2 model dirs
```

## Two-channel disentanglement

- **Palette / appearance**: CIELAB KMeans → Palette → IP-Adapter swatch image + LAB Reinhard projection. Knob: `strength` (0–1).
- **Furniture / structure**: ControlNet depth conditioning from the room photo (not moodboard). Layout is fixed.
- **mode="full"**: IP-Adapter gets palette swatch + moodboard furniture crop collage. Furniture style bleeds in.
- **mode="palette_only"**: IP-Adapter gets only the palette swatch. Room's furniture preserved cleanly.

## Running

```bash
uvicorn app.main:app --reload
celery -A app.celery_app.celery_app worker --loglevel=info
pytest tests/ -q   # 43 tests; GPU tests need conda activate ml
python scripts/batch_run.py --pairs-dir pairs/ --api http://localhost:8000 --email x --password x
```

## Ablation arms

- **Baseline**: `apply_palette_projection` CPU-only (no diffusion) — palette-projection only.
- **InstantMesh**: `MESH_ENGINE=instantmesh` — PBR UV maps instead of Hunyuan vertex colours.
- **palette_only mode**: `?mode=palette_only` — colour-only transfer, no moodboard furniture bleed.

## Human validation

Researcher flow: `POST /evaluation/sessions` (JWT) → share `/evaluation/{token}/view` with participants → export `/evaluation/export/*.csv`. Four forms: demographics, per-variant 2D (18 Likert 1–7), cross-variant comparison (disentanglement probe), 3D model evaluation.
