"""Mood board analysis — the disentangled {palette, furniture} representation.

This module turns a set of mood-board images into a ``DesignSpec`` whose two
channels are deliberately independent:

* ``Palette``       — the APPEARANCE / style channel (color), via CIELAB clustering.
* ``FurnitureSpec`` — the IDENTITY / content channel, via the existing D-FINE
  detector + CLIP image embeddings.

The palette channel is pure NumPy + Pillow (CPU, fully testable). The furniture
channel needs the GPU model stack, so its heavy imports are deferred to call
time and the module stays importable without torch/transformers installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("ThesisPipeline")

# Default palette size; sensible for interior mood boards.
DEFAULT_K = 6
# Longest edge images are downsampled to before pixel pooling (speed).
_MAX_EDGE = 256


# ──────────────────────────────────────────────────────────────────────────────
# Disentangled representation (JSON-serializable dataclasses)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Swatch:
    lab: Tuple[float, float, float]   # CIELAB centroid
    rgb: Tuple[int, int, int]         # display sRGB (0-255)
    proportion: float                 # 0..1 cluster mass

    @classmethod
    def from_dict(cls, d: dict) -> "Swatch":
        return cls(
            lab=tuple(float(x) for x in d["lab"]),
            rgb=tuple(int(x) for x in d["rgb"]),
            proportion=float(d["proportion"]),
        )


@dataclass
class Palette:
    swatches: List[Swatch]            # ordered by proportion desc
    k: int
    space: str = "CIELAB"

    @classmethod
    def from_dict(cls, d: dict) -> "Palette":
        return cls(
            swatches=[Swatch.from_dict(s) for s in d.get("swatches", [])],
            k=int(d.get("k", len(d.get("swatches", [])))),
            space=d.get("space", "CIELAB"),
        )


@dataclass
class FurnitureStyleItem:
    obj365_label: int
    category_name: str
    clip_embedding: List[float]       # L2-normalized
    source_crop_path: str
    score: float

    @classmethod
    def from_dict(cls, d: dict) -> "FurnitureStyleItem":
        return cls(
            obj365_label=int(d["obj365_label"]),
            category_name=d["category_name"],
            clip_embedding=[float(x) for x in d.get("clip_embedding", [])],
            source_crop_path=d.get("source_crop_path", ""),
            score=float(d.get("score", 0.0)),
        )


@dataclass
class FurnitureSpec:
    items: List[FurnitureStyleItem] = field(default_factory=list)
    style_centroid: List[float] = field(default_factory=list)   # mean CLIP embedding

    @classmethod
    def from_dict(cls, d: dict) -> "FurnitureSpec":
        return cls(
            items=[FurnitureStyleItem.from_dict(i) for i in d.get("items", [])],
            style_centroid=[float(x) for x in d.get("style_centroid", [])],
        )


@dataclass
class DesignSpec:
    palette: Palette
    furniture: FurnitureSpec
    seed: int = 12345
    spec_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DesignSpec":
        return cls(
            palette=Palette.from_dict(d.get("palette", {})),
            furniture=FurnitureSpec.from_dict(d.get("furniture", {})),
            seed=int(d.get("seed", 12345)),
            spec_version=int(d.get("spec_version", 1)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# sRGB <-> CIELAB (D65) — vectorized NumPy, no skimage dependency
# ──────────────────────────────────────────────────────────────────────────────

# D65 reference white
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0

_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3 * _DELTA ** 2) + 4.0 / 29.0)


def _f_inv(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (...,3) in [0,1] sRGB -> (...,3) CIELAB."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    xyz = lin @ _RGB2XYZ.T
    fx, fy, fz = [_f(xyz[..., i] / _WHITE[i]) for i in range(3)]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """lab: (...,3) CIELAB -> (...,3) in [0,1] sRGB (clipped)."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xyz = np.stack([_f_inv(fx) * _WHITE[0], _f_inv(fy) * _WHITE[1], _f_inv(fz) * _WHITE[2]], axis=-1)
    rgb = _linear_to_srgb(xyz @ _XYZ2RGB.T)
    return np.clip(rgb, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Palette extraction (APPEARANCE channel)
# ──────────────────────────────────────────────────────────────────────────────

def _load_pixels(image_paths: List[str], max_edge: int = _MAX_EDGE) -> np.ndarray:
    """Pool downsampled sRGB[0,1] pixels across all mood-board images -> (N,3)."""
    chunks = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            logger.warning(f"[moodboard] skipping unreadable image {p}: {e}")
            continue
        w, h = img.size
        scale = max_edge / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float64).reshape(-1, 3) / 255.0
        chunks.append(arr)
    if not chunks:
        raise ValueError("No readable mood-board images for palette extraction.")
    return np.concatenate(chunks, axis=0)


def _kmeans_labels_centers(pixels_lab: np.ndarray, k: int, seed: int):
    """Cluster LAB pixels. Prefer sklearn KMeans; fall back to NumPy median-cut."""
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(pixels_lab)
        return labels, km.cluster_centers_
    except Exception as e:
        logger.warning(f"[moodboard] sklearn unavailable ({e}); using median-cut fallback.")
        return _median_cut(pixels_lab, k)


def _median_cut(pixels_lab: np.ndarray, k: int):
    """Simple median-cut in LAB. Returns (labels, centers) approximating k clusters."""
    # Each box: indices into pixels_lab. Repeatedly split the box with largest extent.
    boxes = [np.arange(len(pixels_lab))]
    while len(boxes) < k:
        # pick the box with the largest single-channel spread
        spreads = [np.ptp(pixels_lab[b], axis=0).max() if len(b) > 1 else 0.0 for b in boxes]
        i = int(np.argmax(spreads))
        if spreads[i] <= 0.0:
            break
        b = boxes.pop(i)
        sub = pixels_lab[b]
        ch = int(np.argmax(np.ptp(sub, axis=0)))
        order = b[np.argsort(sub[:, ch])]
        mid = len(order) // 2
        boxes.append(order[:mid])
        boxes.append(order[mid:])
    centers = np.stack([pixels_lab[b].mean(axis=0) for b in boxes], axis=0)
    labels = np.empty(len(pixels_lab), dtype=int)
    for idx, b in enumerate(boxes):
        labels[b] = idx
    return labels, centers


def extract_palette(
    image_paths: List[str], k: int = DEFAULT_K, seed: int = 12345
) -> Palette:
    """Extract an ordered color palette from mood-board images via CIELAB clustering.

    Pixels are pooled across all boards, converted to CIELAB, and clustered. Each
    cluster becomes a Swatch whose ``proportion`` is its share of the pixel mass.
    Swatches are returned ordered by proportion (descending).
    """
    pixels_rgb = _load_pixels(image_paths)
    pixels_lab = rgb_to_lab(pixels_rgb)

    k_eff = max(1, min(k, len(np.unique(pixels_rgb, axis=0))))
    labels, centers_lab = _kmeans_labels_centers(pixels_lab, k_eff, seed)

    total = len(labels)
    swatches: List[Swatch] = []
    for ci in range(len(centers_lab)):
        mass = int(np.count_nonzero(labels == ci))
        if mass == 0:
            continue
        lab = centers_lab[ci].astype(np.float64)
        rgb01 = lab_to_rgb(lab)
        rgb = tuple(int(round(v * 255)) for v in rgb01)
        swatches.append(
            Swatch(
                lab=(float(lab[0]), float(lab[1]), float(lab[2])),
                rgb=rgb,
                proportion=mass / total,
            )
        )

    swatches.sort(key=lambda s: s.proportion, reverse=True)
    return Palette(swatches=swatches, k=len(swatches), space="CIELAB")


# ──────────────────────────────────────────────────────────────────────────────
# Furniture style extraction (IDENTITY channel) — GPU; heavy imports deferred
# ──────────────────────────────────────────────────────────────────────────────

_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def get_clip_model():
    """Lazy-load CLIP into the shared model cache, mirroring the on/off-device
    discipline used by the rest of the pipeline. Returns (model, processor)."""
    from app import reconstructor_pipeline as rp  # deferred (pulls torch)

    cache = rp._models_cache
    if cache.get("clip") is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        logger.info("Initializing CLIP (ViT-B/32) for furniture style embeddings...")
        model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).eval()
        processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        cache["clip"] = (model, processor)
    return cache["clip"]


def _embed_crops(crop_paths: List[str]) -> np.ndarray:
    """CLIP image embeddings for a list of crops -> (N, D), L2-normalized."""
    import torch

    from app import reconstructor_pipeline as rp

    model, processor = get_clip_model()
    device = rp.PIPELINE_DEVICE
    embeds = []
    try:
        model.to(device)
        for cp in crop_paths:
            try:
                img = Image.open(cp).convert("RGB")
            except Exception:
                continue
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            embeds.append(feat[0].cpu().float().numpy())
    finally:
        model.to("cpu")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return np.stack(embeds, axis=0) if embeds else np.zeros((0, 512), dtype=np.float32)


def extract_furniture_style(
    image_paths: List[str], work_dir: str, scene_id: str
) -> FurnitureSpec:
    """Detect furniture in mood-board images (reusing D-FINE) and embed each crop
    with CLIP to form the IDENTITY channel."""
    from app.reconstructor_pipeline import detect_objects, INDOOR_FURNITURE_CLASSES, OBJ365_NAMES

    items: List[FurnitureStyleItem] = []
    for board_idx, img_path in enumerate(image_paths):
        try:
            detections = detect_objects(img_path, work_dir, f"{scene_id}_board{board_idx}")
        except Exception as e:
            logger.warning(f"[moodboard] detection failed on {img_path}: {e}")
            continue
        for crop_path, _box, score, label in detections:
            if INDOOR_FURNITURE_CLASSES and label not in INDOOR_FURNITURE_CLASSES:
                continue
            items.append(
                FurnitureStyleItem(
                    obj365_label=int(label),
                    category_name=OBJ365_NAMES.get(int(label), str(label)),
                    clip_embedding=[],  # filled below in one batch
                    source_crop_path=crop_path,
                    score=float(score),
                )
            )

    if items:
        embeds = _embed_crops([it.source_crop_path for it in items])
        # _embed_crops may drop unreadable crops; align by re-embedding 1:1 length.
        if len(embeds) == len(items):
            for it, emb in zip(items, embeds):
                it.clip_embedding = [float(x) for x in emb]
            centroid = embeds.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            style_centroid = [float(x) for x in centroid]
        else:
            logger.warning("[moodboard] embedding count mismatch; storing items without embeddings.")
            style_centroid = []
    else:
        style_centroid = []

    return FurnitureSpec(items=items, style_centroid=style_centroid)


def analyze_moodboard(
    image_paths: List[str],
    work_dir: str,
    scene_id: str,
    k: int = DEFAULT_K,
    seed: int = 12345,
) -> DesignSpec:
    """Build the disentangled DesignSpec from mood-board images.

    The palette channel always runs (CPU). The furniture channel is best-effort:
    if the GPU model stack is unavailable, the spec still carries a valid palette.
    """
    palette = extract_palette(image_paths, k=k, seed=seed)
    try:
        furniture = extract_furniture_style(image_paths, work_dir, scene_id)
    except Exception as e:
        logger.warning(f"[moodboard] furniture channel unavailable ({e}); palette-only spec.")
        furniture = FurnitureSpec()
    return DesignSpec(palette=palette, furniture=furniture, seed=seed)
