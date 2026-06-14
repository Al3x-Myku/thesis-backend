"""2D restyle engine — fast iteration of a room from a DesignSpec.

Two independently controllable channels (the disentanglement claim, in 2D):

* FURNITURE / structure  → ControlNet conditioning built from the room photo's
  geometry (depth + Canny edges). Holds layout / furniture placement fixed.
* PALETTE / appearance    → IP-Adapter style image (palette swatches + furniture
  crops) plus a deterministic LAB ``apply_palette_projection``.

The diffusion path (SD1.5 + ControlNet + IP-Adapter) is the primary engine. When
the diffusion stack is unavailable, ``generate_variants`` falls back to a pure
``apply_palette_projection`` of the room photo — the engineered, GPU-free baseline
(and the histogram-matching ablation arm). The pure-NumPy/Pillow pieces here are
CPU-testable; the diffusion pieces defer their heavy imports to call time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from app.moodboard import Palette, DesignSpec, rgb_to_lab, lab_to_rgb

logger = logging.getLogger("ThesisPipeline")

# Per-variant palette-projection strengths for the CPU baseline (subtle → strong).
_BASELINE_STRENGTHS = [0.35, 0.55, 0.75, 0.95]


# ──────────────────────────────────────────────────────────────────────────────
# Variant planning (pure logic) — maps locked/unlocked feedback to per-variant config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VariantPlan:
    variant_index: int
    seed: int
    strength: float                                  # palette-projection strength
    locked_fields: Dict[str, Any] = field(default_factory=dict)  # {"palette":bool,"slots":{id:bool}}


def build_variant_plans(
    spec: DesignSpec, n: int, start_index: int = 0, locked: Optional[Dict[str, Any]] = None
) -> List[VariantPlan]:
    """Deterministic per-variant plans. ``seed`` derives from the spec seed so
    re-runs reproduce pixels; ``strength`` spreads the baseline restyle intensity.
    A locked ``palette`` pins strength to the dominant baseline value so the color
    channel stays stable while other (furniture) fields resample.
    """
    locked = locked or {}
    palette_locked = bool(locked.get("palette", False))
    plans: List[VariantPlan] = []
    for i in range(n):
        idx = start_index + i
        strength = _BASELINE_STRENGTHS[-1] if palette_locked else _BASELINE_STRENGTHS[idx % len(_BASELINE_STRENGTHS)]
        plans.append(
            VariantPlan(
                variant_index=idx,
                seed=spec.seed + idx,
                strength=strength,
                locked_fields=locked,
            )
        )
    return plans


# ──────────────────────────────────────────────────────────────────────────────
# Palette projection (APPEARANCE channel) — LAB Reinhard-style transport, CPU
# ──────────────────────────────────────────────────────────────────────────────

def _palette_target_stats(palette: Palette) -> tuple[np.ndarray, np.ndarray]:
    """Proportion-weighted mean and std (per LAB channel) of the palette swatches."""
    if not palette.swatches:
        return np.array([60.0, 0.0, 0.0]), np.array([20.0, 10.0, 10.0])
    labs = np.array([s.lab for s in palette.swatches], dtype=np.float64)        # (k,3)
    w = np.array([max(s.proportion, 1e-6) for s in palette.swatches], dtype=np.float64)
    w = w / w.sum()
    mean = (w[:, None] * labs).sum(axis=0)
    var = (w[:, None] * (labs - mean) ** 2).sum(axis=0)
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean, std


def apply_palette_projection(img: Image.Image, palette: Palette, strength: float = 1.0) -> Image.Image:
    """Transport an image's colors toward the palette in CIELAB (Reinhard-style:
    match per-channel mean, gently match spread). ``strength`` in [0,1] blends
    between the original (0) and the fully projected image (1)."""
    strength = float(np.clip(strength, 0.0, 1.0))
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    lab = rgb_to_lab(rgb)                                   # (H,W,3)

    i_mean = lab.reshape(-1, 3).mean(axis=0)
    i_std = lab.reshape(-1, 3).std(axis=0)
    t_mean, t_std = _palette_target_stats(palette)

    scale = np.clip(t_std / (i_std + 1e-6), 0.25, 4.0)
    projected = (lab - i_mean) * scale + t_mean
    out_lab = (1.0 - strength) * lab + strength * projected

    out_rgb = lab_to_rgb(out_lab)
    return Image.fromarray((out_rgb * 255.0).round().astype(np.uint8), mode="RGB")


def palette_swatch_image(palette: Palette, size: int = 512) -> Image.Image:
    """Render the palette as proportion-weighted vertical color bars. Doubles as
    the IP-Adapter style image and a client-facing palette preview."""
    img = Image.new("RGB", (size, size), (128, 128, 128))
    arr = np.asarray(img).copy()
    if not palette.swatches:
        return img
    x = 0
    total = sum(max(s.proportion, 1e-6) for s in palette.swatches)
    for i, s in enumerate(palette.swatches):
        w = size - x if i == len(palette.swatches) - 1 else int(round(size * max(s.proportion, 1e-6) / total))
        w = max(1, min(w, size - x))
        arr[:, x:x + w, :] = np.array(s.rgb, dtype=np.uint8)
        x += w
        if x >= size:
            break
    return Image.fromarray(arr, mode="RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Structure conditioning (FURNITURE channel) — Canny is CPU; depth/detection GPU
# ──────────────────────────────────────────────────────────────────────────────

def canny_edges(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """Canny edge map for ControlNet structure conditioning (CPU via OpenCV)."""
    import cv2

    gray = np.asarray(img.convert("L"))
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(edges, mode="L").convert("RGB")


def build_structure_conditioning(
    room_photo_path: str, scene_folder: str, scene_id: str
) -> Dict[str, Any]:
    """Build (and cache) ControlNet conditioning from the room photo: Canny edges
    (CPU), a metric depth map (reusing the pipeline depth model), and the detected
    furniture boxes/labels (reusing D-FINE)."""
    struct_dir = Path(scene_folder) / "structure"
    struct_dir.mkdir(parents=True, exist_ok=True)

    room = Image.open(room_photo_path).convert("RGB")
    canny: Optional[Image.Image] = None
    try:
        canny = canny_edges(room)
        canny.save(struct_dir / "canny.png")
    except Exception as e:
        logger.warning(f"[restyle] Canny unavailable ({e}); structure without edges.")

    depth_img: Optional[Image.Image] = None
    boxes: List[List[int]] = []
    labels: List[int] = []
    try:
        from app.reconstructor_pipeline import detect_objects
        from app import reconstructor_pipeline as rp
        import torch

        # Depth via the shared depth model (DepthPro / DA-V2), normalized to 8-bit.
        model, processor, kind = rp.get_depth_model()
        device = rp.PIPELINE_DEVICE
        model.to(device)
        inputs = processor(images=room, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        model.to("cpu")
        depth = out.predicted_depth[0].cpu().float().numpy()
        d = depth - depth.min()
        d = d / (d.max() + 1e-6)
        depth_img = Image.fromarray((d * 255).astype(np.uint8)).resize(room.size).convert("RGB")
        depth_img.save(struct_dir / "depth.png")

        detections = detect_objects(room_photo_path, scene_folder, scene_id)
        boxes = [list(map(int, b)) for (_c, b, _s, _l) in detections]
        labels = [int(l) for (_c, _b, _s, l) in detections]
    except Exception as e:
        logger.warning(f"[restyle] structure depth/detection unavailable ({e}); Canny-only conditioning.")

    return {"canny": canny, "depth": depth_img, "boxes": boxes, "labels": labels, "room": room}


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion pipeline (GPU) — lazy-loaded into the shared model cache
# ──────────────────────────────────────────────────────────────────────────────

_SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
_CONTROLNET_DEPTH_ID = "lllyasviel/sd-controlnet-depth"


def get_restyle_pipeline():
    """Lazy-load SD1.5 + depth ControlNet (+ IP-Adapter if weights available) into
    ``reconstructor_pipeline._models_cache["restyle"]``. Returns the pipeline or
    None if the diffusion stack cannot be constructed (→ baseline fallback)."""
    from app import reconstructor_pipeline as rp

    cache = rp._models_cache
    if "restyle" not in cache:
        cache["restyle"] = None
    if cache["restyle"] is None:
        try:
            import torch
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

            logger.info("Initializing SD1.5 + ControlNet(depth) restyle pipeline...")
            controlnet = ControlNetModel.from_pretrained(_CONTROLNET_DEPTH_ID, torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                _SD_MODEL_ID, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
            )
            try:
                pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
                pipe._has_ip_adapter = True
            except Exception as e:
                logger.warning(f"[restyle] IP-Adapter unavailable ({e}); palette via projection only.")
                pipe._has_ip_adapter = False
            pipe.enable_attention_slicing()
            cache["restyle"] = pipe
        except Exception as e:
            logger.warning(f"[restyle] diffusion stack unavailable ({e}); using palette-projection baseline.")
            cache["restyle"] = None
    return cache["restyle"]


def _style_prompt(spec: DesignSpec) -> str:
    cats = [it.category_name for it in spec.furniture.items]
    uniq = list(dict.fromkeys(cats))[:6]
    furniture_str = (", with " + ", ".join(uniq)) if uniq else ""
    return f"interior room, cohesive color palette, tasteful furnishing{furniture_str}, photorealistic, high quality"


def _composite_locked_slots(
    out: Image.Image,
    parent: Image.Image,
    boxes: List[List[int]],
    locked_slots: Dict[str, Any],
) -> Image.Image:
    """Paste pixel-identical locked slot regions from the parent variant into the new render."""
    if not locked_slots or not boxes:
        return out
    parent_rs = parent.resize(out.size)
    out_arr = np.array(out).copy()
    parent_arr = np.array(parent_rs)
    for slot_key, is_locked in locked_slots.items():
        if not is_locked:
            continue
        try:
            box_idx = int(slot_key)
        except (ValueError, TypeError):
            continue
        if box_idx >= len(boxes):
            continue
        x0, y0, x1, y1 = [int(v) for v in boxes[box_idx]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(out.width, x1), min(out.height, y1)
        if x1 > x0 and y1 > y0:
            out_arr[y0:y1, x0:x1] = parent_arr[y0:y1, x0:x1]
    return Image.fromarray(out_arr)


def render_variant(
    structure: Dict[str, Any],
    spec: DesignSpec,
    plan: VariantPlan,
    out_path: str,
    parent_image: Optional[Image.Image] = None,
) -> str:
    """Render one 2D variant. Uses the diffusion pipeline when available (ControlNet
    structure + IP-Adapter style), then applies palette projection. Falls back to a
    pure palette-projected room photo when diffusion is unavailable.

    If ``parent_image`` is provided, pixel regions corresponding to locked furniture
    slots in ``plan.locked_fields`` are composited verbatim from the parent so they
    stay byte-identical (M4 pixel-exact freezing)."""
    room: Image.Image = structure["room"]

    pipe = None
    try:
        pipe = get_restyle_pipeline()       # may ImportError on a CPU-only box
    except Exception as e:
        logger.warning(f"[restyle] restyle pipeline unavailable ({e}); palette-projection baseline.")
        pipe = None

    if pipe is None:
        # Engineered baseline: deterministic palette projection of the room photo.
        result = apply_palette_projection(room, spec.palette, strength=plan.strength)
        result = _composite_locked_slots(
            result, parent_image, structure.get("boxes", []),
            plan.locked_fields.get("slots", {}) if plan.locked_fields else {},
        ) if parent_image is not None else result
        result.save(out_path)
        return out_path

    import torch
    from app import reconstructor_pipeline as rp

    control = structure.get("depth") or structure.get("canny")
    device = rp.PIPELINE_DEVICE
    generator = torch.manual_seed(plan.seed)
    kwargs: Dict[str, Any] = dict(
        prompt=_style_prompt(spec),
        image=control,
        num_inference_steps=30,
        generator=generator,
    )
    if getattr(pipe, "_has_ip_adapter", False):
        kwargs["ip_adapter_image"] = palette_swatch_image(spec.palette)
    out = None
    try:
        pipe.to(device)
        out = pipe(**kwargs).images[0].resize(room.size)
    except Exception as e:
        logger.warning(f"[restyle] Diffusion inference failed ({e}); falling back to palette-projection baseline.")
        if getattr(pipe, "_has_ip_adapter", False):
            # IP-Adapter may be incompatible with this diffusers version; retry without it.
            kwargs.pop("ip_adapter_image", None)
            pipe._has_ip_adapter = False
            try:
                out = pipe(**kwargs).images[0].resize(room.size)
            except Exception as e2:
                logger.warning(f"[restyle] Diffusion retry also failed ({e2}); palette-projection only.")
    finally:
        pipe.to("cpu")
        rp.cleanup_gpu(False)

    if out is None:
        out = apply_palette_projection(room, spec.palette, strength=plan.strength)

    # Snap colors onto the target palette deterministically (the palette knob).
    out = apply_palette_projection(out, spec.palette, strength=min(0.6, plan.strength))

    # M4: paste locked slot regions verbatim from the parent variant.
    if parent_image is not None:
        out = _composite_locked_slots(
            out, parent_image, structure.get("boxes", []),
            plan.locked_fields.get("slots", {}) if plan.locked_fields else {},
        )

    out.save(out_path)
    return out_path


def generate_variants(
    room_photo_path: str,
    scene_folder: str,
    scene_id: str,
    spec: DesignSpec,
    n: int,
    start_index: int = 0,
    locked: Optional[Dict[str, Any]] = None,
    parent_image_path: Optional[str] = None,
) -> List[str]:
    """Produce N 2D variant PNGs under ``scene_folder/variants/``. Reuses one
    structure-conditioning build for all variants.

    ``parent_image_path``: when provided (regeneration case), pixel regions for
    locked furniture slots are composited verbatim from this image (M4)."""
    variants_dir = Path(scene_folder) / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    parent_image: Optional[Image.Image] = None
    if parent_image_path:
        try:
            parent_image = Image.open(parent_image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[restyle] Could not load parent image for slot compositing ({e}).")

    structure = build_structure_conditioning(room_photo_path, scene_folder, scene_id)
    plans = build_variant_plans(spec, n, start_index=start_index, locked=locked)

    paths: List[str] = []
    for plan in plans:
        out_path = str(variants_dir / f"variant_{plan.variant_index}.png")
        render_variant(structure, spec, plan, out_path, parent_image=parent_image)
        paths.append(out_path)
    return paths
