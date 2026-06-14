# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A thesis backend that turns a **single interior photo** (or a 360° panorama) into a textured 3D `.glb` scene. A FastAPI web layer accepts image uploads, persists scene records, and dispatches the heavy GPU reconstruction to a Celery worker. The reconstruction itself is a multi-model deep-learning pipeline (detection → segmentation → shape/texture generation → metric geometric placement).

A second subsystem (in progress) extends this into a generative **Interior Design Zone** — mood-board-driven, editable room iterations with palette/furniture disentanglement (see its own section below).

The two large directories `D-FINE/` and `Hunyuan3D-2/` are **vendored upstream third-party repos** (the object detector and the 3D generator). Treat them as external dependencies — the thesis code lives in `app/`, `scripts/`, and `tests/`.

## Architecture

The system is split into a thin synchronous web tier and an asynchronous GPU worker, communicating through a SQL database that doubles as the Celery broker/result backend.

```
HTTP upload → app/routers/scenes.py → app/services/scene_service.py
   → writes data/user_<uid>/scene_<id>/input.png, creates Scene row (PENDING)
   → celery_app.send_task("app.tasks.reconstruct_scene", scene_id)
                              │
Celery worker ───────────────┘
   app/tasks.py → app/reconstructor_pipeline.py:full_reconstruction()
      1. detect_objects()  → D-FINE (run as subprocess via dfine_wrapper.py), writes crops + *_box.txt
      2. build_mesh()      → BiRefNet bg removal → Hunyuan3D shape gen → mesh cleanup → Hunyuan3D texture paint
      3. position_meshes() → depth estimation (Apple DepthPro / Depth-Anything-V2) + scene_geometry.py
                             places each object metrically in a reconstructed room box
      → writes final .glb, sets Scene status COMPLETED/FAILED
```

Key boundaries to understand before editing:

- **`app/reconstructor_pipeline.py`** (~1700 lines) is the heart. It owns a module-level `_models_cache` of lazily-loaded GPU models, the `ThesisProfiler` context manager (logs latency + peak VRAM to `production_metrics.csv`), all the Objects365 class-id tables (`OBJ365_NAMES`, `WALL_MOUNTED_LABELS`, `FURNITURE_SIZE_PRIORS`, etc.), and the `position_meshes()` placement solver. The geometry primitives it calls (intrinsics from vanishing points, RANSAC floor plane, Manhattan alignment, metric scaling, overlap resolution) live in **`app/scene_geometry.py`**.
- **D-FINE is invoked out-of-process.** `dfine_wrapper.run_dfine_inference()` shells out to `D-FINE/tools/inference/torch_inf.py` and reads back crop PNGs plus sidecar `*_box.txt` files (`x0,y0,x1,y1,score,label`). It does not import D-FINE as a library.
- **Panorama path** (`full_reconstruction_panoramic`) lives in `app/wall_pipeline.py`; it decomposes a 360° image into per-wall perspective views via `scripts/virtual_photo.py`, then reuses the same per-object pipeline.
- **GPU memory is the hard constraint.** Models are moved on/off device around each stage and `cleanup_gpu()` is called between objects; preserve this pattern when modifying the pipeline.

## Generative "Interior Design Zone" (in progress — thesis extension)

A second subsystem layered on top of reconstruction: given a **room photo + a mood board** (aesthetic reference images), it generates editable **2D iterations** of the room, then realizes a chosen one as a **3D `.glb` on commit**. The research claim is **disentangling color palette (appearance) from furniture (identity)**, each independently editable; per-element like/dislike locks elements and resamples only the rest. Full design + methodology: `/home/micu/.claude/plans/i-want-to-extend-clever-hammock.md`.

```
POST /scenes/{id}/design/moodboard → design_service → analyze_moodboard task
   app/moodboard.py: extract_palette (CIELAB clustering) + extract_furniture_style
   (reuses detect_objects + CLIP) → DesignSpec{Palette, FurnitureSpec} (JSON in DB)
POST .../variants → generate_2d_variants task
   app/restyle_2d.py: build_structure_conditioning (depth+Canny+boxes, reused) +
   SD1.5+ControlNet+IP-Adapter → 2D variant PNGs; apply_palette_projection snaps color
POST .../variants/{vid}/feedback + /regenerate → keep liked, resample unlocked (locked dict)
POST .../variants/{vid}/commit → commit_3d task → 3D .glb (M5, not yet built)
```

Key facts for working here:
- **The disentanglement rides the existing shape/paint seam in `build_mesh()`**: Hunyuan3D shape = furniture identity, the separate paint pass = appearance/palette. Editing palette re-textures without regenerating geometry.
- **Two channels, deliberately independent**: `app/moodboard.py` builds the `DesignSpec`; `app/restyle_2d.py` injects structure (furniture knob, ControlNet) and palette (appearance knob, IP-Adapter + `apply_palette_projection`) separately.
- **CPU-baseline fallback is built in and load-bearing.** sRGB↔CIELAB and `apply_palette_projection` are pure NumPy; when torch/diffusers/opencv are absent, `generate_variants` degrades to palette-projecting the room photo (the engineered baseline / ablation arm). Heavy imports (`reconstructor_pipeline`, torch, diffusers, CLIP) are **deferred to call time** so these modules import on a non-GPU box. New `_models_cache` keys: `"clip"`, `"restyle"`.
- **Design web tier** mirrors the existing split: `app/routers/design.py` (prefix `/scenes/{scene_id}/design`), `app/services/design_service.py`, `app/repositories/design_repo.py`, models `app/models/design.py` (`DesignSession`, `Variant`, `ElementFeedback`, `DesignStatus`) + `app/models/moodboard.py`. New artifacts: `data/user_<id>/scene_<id>/moodboard/`, `structure/`, `variants/`, `committed/`. New tables are registered via imports in `app/tasks.py`.
- **Status as of pause:** M1 (persistence/API skeleton), M2 (mood board analysis), M3 (2D restyle engine) are implemented; the **CPU-only paths are verified** (`tests/test_moodboard.py`, `tests/test_restyle_2d.py`). The **furniture/CLIP channel and the diffusion path are written but unverified — they need the GPU box.** Remaining: M4 (pixel-exact feedback freezing), M5 (`commit_reconstruction` 3D commit), M6 (evaluation harness).

### Web tier layering

Standard router → service → repository split under `app/`:
- `routers/` — HTTP only (auth dep, request/response). `scenes`, `auth`, `users`, `debug` (exposes the pipeline stages individually), `design` (the Design Zone subsystem).
- `services/` — orchestration (`scene_service`, `user_service`); file I/O under `DATA_DIR` and Celery dispatch happen here.
- `repositories/` — SQLModel DB access (`scene_repo`, `user_repo`).
- `models/` — SQLModel tables (`Scene` with `SceneStatus` enum, `User`). `database.py` builds the engine; `init_db()` runs on FastAPI startup.
- Auth is JWT (`core/security.py`, `python-jose` + `passlib[bcrypt]`); config via `core/config.py` (pydantic-settings, reads `.env`).

## Running it

There is no single launch script for the web app; run the two processes separately. Both need a populated `.env` and the GPU/ML environment installed.

```bash
# API (dev)
uvicorn app.main:app --reload

# Celery worker (must run for any reconstruction to happen)
celery -A app.celery_app.celery_app worker --loglevel=info
```

`celery[sqlalchemy]` is used: the broker and result backend are both the `DATABASE_URL` (no Redis/RabbitMQ). MySQL is the target DB (`pymysql`).

### Required environment (`.env`)

`SECRET_KEY` and `DATABASE_URL` are mandatory (app fails to start without them). Other knobs read via `os.getenv` with defaults: `DATA_DIR` (default `./data`), `PIPELINE_DEVICE`, `DFINE_ROOT` / `DFINE_CONFIG` / `DFINE_CHECKPT`, and the `HUNYUAN_*` model-path vars. Model weights are large and git-ignored — they must be downloaded/placed separately.

### Dependencies

- `requirements_backend.txt` — minimal web/worker deps (FastAPI, Celery, SQLModel, pymysql).
- `requirements_all.txt` — full ML stack.
- PyTorch + CUDA and the Hunyuan3D custom CUDA ops are environment-specific. `deploy_and_run.sh` (CUDA 12.4, RTX 5080/Blackwell) and `scripts/setup.sh` (compiles Hunyuan3D's custom rasterizer/renderer, forces PTX for forward compat) document the exact build steps used. These are Romanian-commented provisioning scripts for a specific GPU box, not portable installers.

## Tests

```bash
pytest tests/                              # full suite
pytest tests/test_scene_geometry.py        # geometry unit tests (needs trimesh/scipy)
pytest tests/test_moodboard.py tests/test_restyle_2d.py   # design-zone CPU tests (numpy/Pillow/sklearn only)
pytest tests/test_scene_geometry.py::<name> -v   # single test
```

`tests/` covers pure-CPU logic with synthetic data only — geometry primitives (`scene_geometry.py`), the palette/CIELAB channel (`test_moodboard.py`), and the palette-projection restyle baseline + variant planning (`test_restyle_2d.py`). The GPU models, the diffusion restyle path, the CLIP furniture channel, and the web tier are **not** exercised by the suite; validate those on the model box. The design-zone tests need only numpy/Pillow/scikit-learn, so they run without the full ML stack.

## Benchmarking

The `scripts/` directory is a self-contained evaluation harness for the thesis (Structured3D / 3D-FRONT datasets), separate from serving traffic. `scripts/run_structured3d_benchmark.py` imports `full_reconstruction` directly and scores it; the `generate_*.py` / `plot_*.py` scripts produce the thesis figures into `benchmark_results/`. `deploy_benchmark.sh` wires up data download + run. Use `scripts/requirements_benchmark.txt` (adds Open3D, chamfer/ICP math) for these.

## Conventions worth noting

- Pipeline log lines are namespaced `[<scene_id>]` and the logger is `"ThesisPipeline"`. Each costly stage is wrapped in `with ThesisProfiler(stage_name, scene_id):` — keep new stages consistent so metrics keep flowing to `production_metrics.csv`.
- Scene artifacts live under `data/user_<owner_id>/scene_<id>/` (input.png, crops/, meshes/, debug/, final/scene_positioned.glb). Download endpoints resolve paths from this layout, so don't change it casually.
- Object class IDs throughout the pipeline are **Objects365** ids (the D-FINE checkpoint is the obj365 model), not COCO — see `OBJ365_NAMES`.
- Many commit messages and provisioning-script comments are in Romanian; code identifiers and logs are in English.
