# thesis-backend

Single-photo → textured 3D scene reconstruction, extended with a generative **Interior Design Zone**: upload a room photo and a mood board, get editable 2D room iterations with palette and furniture disentangled, commit a chosen variant to a 3D `.glb`.

---

## Architecture overview

```
HTTP upload → app/routers/scenes.py → app/services/scene_service.py
   → writes data/user_<uid>/scene_<id>/input.png, creates Scene row (PENDING)
   → celery_app.send_task("app.tasks.reconstruct_scene", scene_id)

Celery worker → app/tasks.py → app/reconstructor_pipeline.py:full_reconstruction()
   1. detect_objects()  → D-FINE (subprocess, HF fallback) → crops + box files
   2. build_mesh()      → BiRefNet bg removal → Hunyuan3D shape → Hunyuan3D texture
   3. position_meshes() → DepthPro depth + scene_geometry.py → final .glb
```

### Interior Design Zone (design-zone branch)

```
POST /scenes/{id}/design/moodboard → analyze_moodboard task
   app/moodboard.py: extract_palette (CIELAB KMeans) + extract_furniture_style (D-FINE + CLIP)
   → DesignSpec{ Palette, FurnitureSpec }

POST .../variants → generate_2d_variants task
   app/restyle_2d.py: structure conditioning (depth + Canny + boxes)
   + SD1.5 + ControlNet(depth) + IP-Adapter (palette swatch image)
   + apply_palette_projection (LAB Reinhard transport)
   → N variant PNGs

POST .../variants/{vid}/feedback   → mark elements liked/disliked (locked dict in DB)
POST .../variants/{vid}/regenerate → regenerate with locked slots composited verbatim (M4)
POST .../variants/{vid}/commit     → commit_3d task → commit_reconstruction() → .glb (M5)
```

**Disentanglement**: CIELAB palette = appearance knob (IP-Adapter + `apply_palette_projection`). Furniture placement = structure knob (ControlNet conditioning). They are injected independently and can be edited independently. On 3D commit, palette re-textures the Hunyuan3D paint pass without regenerating geometry.

---

## Repository layout

```
app/
  reconstructor_pipeline.py   # full GPU pipeline + detect_objects + build_mesh + position_meshes
  dfine_wrapper.py             # D-FINE subprocess wrapper (HF Transformers fallback built in)
  moodboard.py                 # DesignSpec dataclasses, extract_palette, extract_furniture_style
  restyle_2d.py                # apply_palette_projection, build_structure_conditioning, generate_variants
  scene_geometry.py            # geometry primitives (intrinsics, floor RANSAC, metric scaling)
  wall_pipeline.py             # 360° panorama decomposition path
  tasks.py                     # Celery task definitions
  routers/                     # FastAPI routers: scenes, auth, users, debug, design
  services/                    # orchestration layer
  repositories/                # SQLModel DB access
  models/                      # SQLModel tables (Scene, User, DesignSession, Variant, ElementFeedback)
D-FINE/                        # vendored object detector (Objects365, run as subprocess)
Hunyuan3D-2/                   # vendored 3D generator
scripts/                       # evaluation harness (Structured3D / 3D-FRONT benchmarks)
tests/                         # test suite
docs/PLAN.md                   # full design methodology and milestone breakdown
```

---

## Environment setup

Tested on RTX 5080 16 GB, CUDA 12.8, Ubuntu 24.04.

```bash
# Create conda environment with full ML stack
conda create -n ml python=3.12
conda activate ml

# PyTorch (CUDA 12.8 / Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Backend dependencies
pip install -r requirements_backend.txt

# Full ML stack
pip install -r requirements_all.txt

# Additional runtime deps
pip install trimesh==4.6.12 calflops

# Hunyuan3D custom CUDA ops (compiles rasterizer + renderer)
pip install -e Hunyuan3D-2

# D-FINE (editable, for the import path)
# Note: the subprocess uses PYTHONPATH=D-FINE/ automatically; no pip install needed
```

### Environment variables (`.env`)

```
SECRET_KEY=<random-string>
DATABASE_URL=mysql+pymysql://user:pass@host/dbname

# Optional — defaults shown
DATA_DIR=./data
PIPELINE_DEVICE=cuda:0
DFINE_ROOT=./D-FINE
DFINE_CONFIG=configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml
DFINE_CHECKPT=weights/dfine_x_obj365.pth
HUNYUAN_SHAPEDIR=tencent/Hunyuan3D-2
HUNYUAN_PAINTDIR=tencent/Hunyuan3D-2
```

**D-FINE weights**: download `dfine_x_obj365.pth` from the [D-FINE releases](https://github.com/Peterande/D-FINE/releases) and place at `D-FINE/weights/dfine_x_obj365.pth`. If the file is absent or the subprocess fails for any reason, `detect_objects` automatically falls back to `ustc-community/dfine_x_obj365` via HuggingFace Transformers — no manual action required.

---

## Running the server

```bash
# API (dev, auto-reload)
uvicorn app.main:app --reload

# Celery worker (required for any reconstruction or design task)
celery -A app.celery_app.celery_app worker --loglevel=info
```

The broker and result backend are both `DATABASE_URL` (SQLAlchemy). No Redis or RabbitMQ needed.

---

## Running tests

The test suite is split into two tiers:

### CPU-only tests (no GPU required)

These cover pure-logic modules and run on any machine with numpy/Pillow/scikit-learn:

```bash
# Scene geometry primitives
pytest tests/test_scene_geometry.py -v

# Moodboard palette extraction (CIELAB clustering, DesignSpec serialization)
pytest tests/test_moodboard.py -v

# 2D restyle engine: palette projection, variant planning, locked-slot compositing
pytest tests/test_restyle_2d.py -v

# All CPU tests at once
pytest tests/test_scene_geometry.py tests/test_moodboard.py tests/test_restyle_2d.py -v
```

These need only:
```bash
pip install numpy pillow scikit-learn scipy trimesh pytest
```

### GPU tests (requires the full ML environment)

These validate the live pipeline paths — CLIP embeddings, diffusion restyle (SD1.5 + ControlNet + IP-Adapter), D-FINE detection, depth estimation, and the M4/M5 design zone flows:

```bash
conda activate ml
pytest tests/test_design_gpu.py -v
```

What each test class covers:

| Class | Tests | What runs on GPU |
|---|---|---|
| `TestM2MoodboardAnalysis` | 4 | `extract_palette`, `extract_furniture_style` (D-FINE + CLIP), `analyze_moodboard` roundtrip |
| `TestM3RestyLe2D` | 3 | `build_structure_conditioning` (depth + detection), `generate_variants` (diffusion), palette color-shift assertion |
| `TestM4FeedbackFreeze` | 2 | Locked-slot pixel identity, unlocked variant divergence |
| `TestM5PaletteUtils` | 5 | `palette_to_shell_colors`, `position_meshes(shell_colors=, out_subdir=)`, backward compat |
| `TestDesignFlowIntegration` | 1 | Full moodboard → spec → 3 variants end-to-end |

Run the full suite (CPU + GPU):

```bash
conda activate ml
pytest tests/ -q
# Expected: 43 passed
```

### Quick end-to-end diffusion example

Generate 4 room variants from a moodboard palette without the web server:

```python
import sys, os, numpy as np
sys.path.insert(0, '/path/to/thesis-backend')
os.chdir('/path/to/thesis-backend')

from app.moodboard import analyze_moodboard
from app.restyle_2d import generate_variants
from PIL import Image

# Point at a real moodboard image (e.g. a Pinterest screenshot)
spec = analyze_moodboard(
    ['path/to/moodboard.jpg'],
    work_dir='/tmp/example/moodboard',
    scene_id='example',
    k=6,
)
print(f'Palette: {len(spec.palette.swatches)} swatches, Furniture: {len(spec.furniture.items)} items')

paths = generate_variants(
    room_photo_path='path/to/room.jpg',
    scene_folder='/tmp/example',
    scene_id='example',
    spec=spec,
    n=4,
)
for p in paths:
    print(p, Image.open(p).size)
```

---

## Key design notes

- **GPU memory discipline**: models are moved on/off device around each stage (`model.to(device)` before, `model.to("cpu")` + `cleanup_gpu()` after). Preserve this pattern when adding stages.
- **Lazy imports**: `app/moodboard.py` and `app/restyle_2d.py` defer all heavy imports (torch, diffusers, CLIP, `reconstructor_pipeline`) to call time so the web tier stays importable on a CPU-only box.
- **Diffusion fallback chain**: SD1.5 + ControlNet + IP-Adapter → retry without IP-Adapter → pure `apply_palette_projection` (CPU baseline). The CPU path is also the histogram-matching ablation arm for evaluation.
- **Object class IDs**: the pipeline uses **Objects365** IDs throughout (D-FINE is the obj365 checkpoint). See `OBJ365_NAMES` in `reconstructor_pipeline.py`.
- **Scene artifacts**: `data/user_<id>/scene_<id>/` holds `input.png`, `crops/`, `meshes/`, `structure/`, `variants/`, `committed/`, `final/scene_positioned.glb`. Don't restructure this layout without updating the download endpoints.
