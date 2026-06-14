# Plan: Extend thesis-backend into a generative "Interior Design Zone"

## Context (why this change)

The current system is a **single-photo → 3D reconstruction** pipeline (FastAPI + Celery + a
multi-model GPU pipeline in `app/reconstructor_pipeline.py`). The user (a master's thesis) wants to
extend it into a **generative interior-design** tool: ingest aesthetic **mood boards** (e.g. Pinterest
collections), generate multiple editable **iterations of a room**, let the user mark like/dislike, and
crucially **disentangle "color palette" (appearance) from "furniture" (identity)** so each is edited
independently. The user asked for a research approach.

Two framing findings:
1. **The cited paper `arXiv:2606.13652` is off-topic** — it is *"World Tracing: Generative
   Pixel-Aligned Geometry Beyond the Visible"* (image→3D geometry completion), not interior/mood-board
   design. The relevant literature is interior scene synthesis with style control + preference
   feedback: **VIDES** (2308.13795, restyle while preserving layout), **Interactive Interior Design
   Recommendation via coarse-to-fine multimodal RL** (2310.07287), **OptiScene/DPO** (2506.07570),
   **SceneDirector** (TVCG 2024), **SceneAssistant** (2603.12238), **InteriorAgent**.
2. **Reconstruction ≠ generative design.** A mood board is not a room. We reframe as:
   `mood board → design spec {palette, furniture} → (room photo gives layout) → 2D iterations → 3D on commit`.

### Resolved design decisions (driving this plan)
- **Layout source = mood board + room photo.** Room photo provides layout/geometry via the EXISTING
  pipeline; mood board only drives palette + furniture style. Maximize reuse.
- **Output = 2D-iterate, 3D-on-commit.** Iterations are fast 2D renders; "commit" realizes a 3D `.glb`.
- **Disentanglement = hybrid.** Engineered split (palette via CIELAB clustering; furniture via the
  existing D-FINE detector) + a learned appearance channel (SD1.5 + ControlNet + IP-Adapter).
- **Feedback = per-element keep & regenerate.** Like/dislike locks fields; only unlocked fields resample.

### The architectural seam that makes this work
`build_mesh()` (`reconstructor_pipeline.py:426–460`) already runs Hunyuan3D **shape** gen and a
**separate** **paint** pass (`paint_pipe(mesh, image=image_no_bg)`) fed only an image. Shape = identity
(furniture), paint = appearance (palette). This existing seam is the physical realization of the
disentanglement claim and the central narrative of the thesis.

### Verified environment facts
- Installed: `diffusers 0.34.0`, `transformers 4.52.4`, `accelerate 1.8.1`, opencv, skimage, pillow,
  numpy → SD + ControlNet + IP-Adapter feasible. **New deps:** `scikit-learn` (clustering), CLIP (use
  HF `transformers.CLIPModel`, already installed, to avoid an `open_clip` dependency).
- Pre-existing dead import to avoid: `app/services/scene_service.py:8` imports nonexistent
  `merge_meshes`. Do not import it in new code.

---

## Architecture

```
ROOM PHOTO ─► detect_objects + depth + position  (EXISTING)  ─► layout/geometry + furniture slots
MOOD BOARD ─► moodboard.analyze ─► DesignSpec{ Palette, FurnitureSpec }   (the disentangled rep)
                                       │
            ┌──────────────── 2D ITERATION LOOP (restyle_2d) ──────────────┐
            │ structure knob = ControlNet(depth/canny of room)  ← furniture │
            │ style knob     = IP-Adapter(palette+furniture crops)+LAB proj ← palette │
            │ per-element like/dislike → lock fields → resample only unlocked │
            └────────────────────────────────────────────────────────────────┘
                                       │ commit
            COMMIT (commit_reconstruction): build_mesh(palette-projected crop) + position_meshes(shell_colors) ─► .glb
```

---

## New modules & key signatures

### `app/moodboard.py` (new — the disentangled representation)
Dataclasses (JSON-serializable for DB): `Swatch(lab,rgb,proportion)`, `Palette(swatches,k,space)`,
`FurnitureStyleItem(obj365_label,category_name,clip_embedding,source_crop_path,score)`,
`FurnitureSpec(items,style_centroid)`, `DesignSpec(palette,furniture,seed,spec_version)`.
- `extract_palette(image_paths, k=6) -> Palette` — sRGB→CIELAB (`skimage.color.rgb2lab`), pool pixels,
  KMeans (sklearn) or median-cut fallback, proportions = cluster mass, ordered desc.
- `extract_furniture_style(image_paths, work_dir, scene_id) -> FurnitureSpec` — **REUSE
  `detect_objects`** on board images, filter to `INDOOR_FURNITURE_CLASSES`, embed crops with CLIP.
- `analyze_moodboard(image_paths, work_dir, scene_id, k=6, seed=12345) -> DesignSpec`.
- `get_clip_model()` — lazy-load HF CLIP into `reconstructor_pipeline._models_cache["clip"]`, following
  the existing `.to(PIPELINE_DEVICE)`/`.to("cpu")`/`cleanup_gpu` discipline.

### `app/restyle_2d.py` (new — appearance/furniture channels in 2D)
- `build_structure_conditioning(room_photo, scene_folder, scene_id) -> {depth, canny, boxes, labels}`
  — **REUSE `detect_objects` + the depth model used in `position_meshes`**; cache to `structure/`.
- `get_restyle_pipeline()` — lazy-load SD1.5 + ControlNet(depth) into `_models_cache`; `load_ip_adapter`.
- `render_variant(room_photo, structure, spec, plan, out_path) -> str` — one deterministic render
  (`generator=torch.manual_seed(plan.seed)`); ControlNet=structure (furniture knob), IP-Adapter=style
  image from `spec.palette`+furniture crops (palette knob); then `apply_palette_projection`.
- `apply_palette_projection(img, palette, strength) -> Image` — LAB-space transport toward palette;
  `strength=1.0` doubles as the **histogram-matching baseline**.
- `generate_variants(room_photo, scene_folder, scene_id, spec, n, locked=None) -> List[str]` — builds N
  `VariantPlan`s (`seed = spec.seed + i`); when `locked` given, frozen fields reuse prior conditioning
  and untouched furniture slots are inpaint-masked so they stay pixel-identical.

**Knob ↔ lock mapping:** locked `palette` ⇒ freeze IP-Adapter style image + palette-projection target;
locked furniture slot *i* ⇒ mask box *i* out of inpainting (pixels frozen); unlocked ⇒ resample with
`seed+variant_index`.

### `app/reconstructor_pipeline.py` (additive edits only)
- `position_meshes(...)`: add optional `shell_colors: dict | None = None`; replace the hardcoded room
  vertex colors at `:1603–1607` with it when provided (backward compatible).
- `palette_to_shell_colors(palette) -> dict` (new) — map neutral swatches → floor/walls/ceiling.
- `build_mesh(...)`: add optional `palette: Palette | None = None`; when set, apply
  `apply_palette_projection` to the crop's RGB (preserve alpha) **before** the paint pass — shape stays
  from the raw crop (identity), paint inherits palette (appearance).
- `commit_reconstruction(room_photo, scene_folder, spec, committed_variant) -> str` (new orchestrator)
  — REUSE detect/depth/placement; only texture source + shell colors change; preserve
  `cleanup_gpu(aggressive=True)` at the end like `full_reconstruction`.

---

## Persistence (new SQLModel tables; Scene reused unchanged)

- `app/models/moodboard.py`: `MoodBoardImage(id, scene_id FK, file_path, created_at)`.
- `app/models/design.py`: `DesignStatus` enum; `DesignSession(id, scene_id FK, status, progress,
  design_spec_json, committed_variant_id, committed_glb_path, created_at)`; `Variant(id, session_id FK,
  variant_index, seed, image_path, locked_fields_json, spec_snapshot_json, created_at)`;
  `ElementFeedback(id, variant_id FK, element_type['palette'|'slot'], element_key, liked, created_at)`.
- Import these model modules in `app/tasks.py` so SQLModel metadata + Celery
  `database_create_tables_at_setup` create the tables.
- `app/repositories/design_repo.py` + `app/services/design_service.py` mirroring the existing
  `scene_repo.py` / `scene_service.py` patterns.

### Filesystem under `data/user_<id>/scene_<id>/`
`input.png` (room photo, existing) · `crops/ meshes/ final/` (existing) · `moodboard/board_*.png` +
`moodboard/crops/` · `structure/` (depth/canny cache) · `spec.json` · `variants/variant_*.png` ·
`committed/scene_positioned.glb`.

---

## API (`app/routers/design.py`, prefix `/scenes/{scene_id}/design`) + Celery tasks

Endpoints (auth + ownership via existing `get_current_user` / `scene.owner_id`):
`POST /moodboard` (multi-file) → `analyze_moodboard` · `GET /` (status + spec summary) ·
`POST /variants?n=4` → `generate_2d_variants` · `GET /variants`, `GET /variants/{vid}`,
`GET /variants/{vid}/image` · `POST /variants/{vid}/feedback` · `POST /variants/{vid}/regenerate` →
`regenerate_variant` · `POST /variants/{vid}/commit` → `commit_3d` · `GET /commit/download`.
Register the router in `app/main.py`.

New Celery tasks in `app/tasks.py` (mirror `reconstruct_scene`: status transitions + `finally:
cleanup_gpu()`): `analyze_moodboard(session_id)`, `generate_2d_variants(session_id, n)`,
`regenerate_variant(variant_id)`, `commit_3d(session_id, variant_id)`.

---

## Research methodology & evaluation

**Central claim:** the `{palette, furniture}` representation disentangles appearance from identity —
editing one channel leaves the other measurably invariant while each faithfully reflects the board.

- **Disentanglement (core).** Palette-swap → identity preserved: D-FINE box-IoU + label-match and CLIP
  content similarity of corresponding furniture crops stay ≈high. Furniture-swap → palette preserved:
  LAB/EMD color distance to target palette stays ≈0. Report a 2×2 sensitivity matrix
  (palette-edit / furniture-edit) × (color-metric / identity-metric); strong off-diagonal suppression =
  good disentanglement.
- **Aesthetic fidelity.** Palette EMD (output vs board, LAB); CLIP style similarity of output furniture
  to `FurnitureSpec.style_centroid`.
- **Ablations.** (a) palette-only / furniture-only / joint; (b) learned restyle vs histogram-matching
  baseline (`apply_palette_projection`, strength=1.0) — isolates what the learned channel buys;
  (c) palette `k` sweep; (d) ControlNet depth vs canny vs both.
- **Datasets.** Structured3D / 3D-FRONT via existing `scripts/download_*subset.py` +
  `run_structured3d_benchmark.py` (rooms + GT geometry); 10–15 curated mood boards across styles under
  `data/eval/moodboards/`.
- **Harness (new, mirrors existing benchmark scripts):** `scripts/run_design_benchmark.py` (cross-swap
  protocol → metrics CSV), `scripts/generate_disentanglement_figures.py`, `scripts/plot_palette_fidelity.py`.
- **Optional user study** (per interactive-design literature): forced-choice + Likert on palette match,
  furniture-style match, and edit predictability (did locking keep the element?), N≈15–25.
- **Thesis chapters:** Related Work · System Design (the disentangled spec) · Implementation
  (reuse + 2D-iterate/3D-commit + feedback locking) · Evaluation · Discussion/Limitations.

---

## Phasing (each milestone independently demoable)

- **M1 — Skeleton & persistence (no ML).** New models/repos/service/router; task stubs copy placeholder
  images. Demo: upload boards → session → list variants → feedback → status transitions, zero GPU.
- **M2 — Mood board analysis.** `app/moodboard.py` (palette + CLIP/D-FINE furniture). Demo: `GET /design`
  returns real swatches + detected furniture.
- **M3 — 2D restyle engine.** `app/restyle_2d.py` + `generate_2d_variants`. Demo: room photo + boards →
  N restyled 2D variants. The visible "wow" milestone.
- **M4 — Disentangled feedback loop.** `ElementFeedback` → locked dict → `regenerate_variant`. Demo:
  dislike a slot, regenerate, only that slot changes; palette + locked slots stay pixel-identical.
- **M5 — 3D commit.** `position_meshes` `shell_colors` param + `commit_reconstruction` + `commit_3d`.
  Demo: commit a 2D variant → downloadable `.glb` with palette-tinted shell + restyled textures.
- **M6 — Evaluation.** Benchmark + figure scripts; run ablations; assemble results.

---

## Risks & fallbacks
1. **Diffusion restyle drifts furniture identity.** → Engineered `apply_palette_projection` alone gives
   deterministic, identity-perfect restyle; learned channel becomes an *enhancement/ablation arm*, not a
   dependency.
2. **IP-Adapter weights unavailable.** → Drive palette via `apply_palette_projection` + ControlNet only;
   two-channel abstraction unchanged.
3. **GPU memory (many models).** → Strict one-model-on-device via `_models_cache` + `.to("cpu")` +
   `cleanup_gpu` (as `build_mesh` already does); SD1.5 not SDXL; enable attention slicing / CPU offload;
   2D and 3D are separate tasks so never co-resident.
4. **Palette→shell-color mapping heuristic.** → Expose shell colors as an editable committed field;
   default to current neutrals when confidence low.
5. **Paint pass is image-only (no style text).** → Palette-projected-crop trick needs no Hunyuan
   internals; if textures look wrong, fall back to post-export tinting toward the palette.

---

## Verification (end-to-end)
- **Unit:** `extract_palette` on a synthetic 3-color image returns those swatches with correct
  proportions; `apply_palette_projection` shifts a gray image toward a target swatch (assert LAB
  distance drops); extend `tests/` (currently `test_scene_geometry.py`).
- **Disentanglement check (scriptable):** generate a variant, palette-swap it, assert D-FINE box-IoU and
  label-match ≈ unchanged and LAB distance to new palette drops — the core claim, automated.
- **API integration:** `uvicorn app.main:app` + `celery -A app.celery_app.celery_app worker`; upload a
  room photo (existing flow) + a mood board, `POST /variants`, download 2D images, submit feedback,
  regenerate (confirm locked regions byte-identical), `commit`, download `.glb` and open in
  `glb_viewer.html`.
- **Benchmark:** `python scripts/run_design_benchmark.py` over rooms×boards → metrics CSV + figures.
