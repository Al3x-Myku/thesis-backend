"""Unit tests for app/restyle_2d.py — the CPU-verifiable parts of the 2D restyle
engine: LAB palette projection (the engineered baseline / histogram-match ablation),
the palette swatch image, deterministic variant planning, and the diffusion-free
``generate_variants`` fallback path.

The diffusion path (SD1.5 + ControlNet + IP-Adapter) is GPU-only and is exercised
on the model box.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.moodboard import Palette, Swatch, FurnitureSpec, DesignSpec, rgb_to_lab
from app.restyle_2d import (
    apply_palette_projection,
    palette_swatch_image,
    build_variant_plans,
    generate_variants,
    _palette_target_stats,
)


def _blue_palette() -> Palette:
    # Deep blue dominant + a dark neutral.
    return Palette(
        swatches=[
            Swatch(lab=tuple(rgb_to_lab(np.array([40, 70, 200]) / 255.0)), rgb=(40, 70, 200), proportion=0.7),
            Swatch(lab=tuple(rgb_to_lab(np.array([30, 30, 35]) / 255.0)), rgb=(30, 30, 35), proportion=0.3),
        ],
        k=2,
    )


def _gray_room(tmpdir, color=(128, 128, 128)) -> str:
    arr = np.full((64, 96, 3), color, dtype=np.uint8)
    # add a little structure so it's not perfectly uniform
    arr[:, :32, :] = (160, 150, 140)
    p = os.path.join(tmpdir, "input.png")
    Image.fromarray(arr).save(p)
    return p


def _mean_lab(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    return rgb_to_lab(rgb).reshape(-1, 3).mean(axis=0)


def test_palette_projection_moves_toward_palette(tmp_path):
    pal = _blue_palette()
    room = Image.open(_gray_room(tmp_path))
    t_mean, _ = _palette_target_stats(pal)

    before = np.linalg.norm(_mean_lab(room) - t_mean)
    out = apply_palette_projection(room, pal, strength=1.0)
    after = np.linalg.norm(_mean_lab(out) - t_mean)
    assert after < before, (before, after)
    # the bluening should drive the b* channel negative
    assert _mean_lab(out)[2] < _mean_lab(room)[2]


def test_palette_projection_strength_zero_is_identity(tmp_path):
    pal = _blue_palette()
    room = Image.open(_gray_room(tmp_path))
    out = apply_palette_projection(room, pal, strength=0.0)
    assert np.allclose(np.asarray(out), np.asarray(room.convert("RGB")), atol=1)


def test_palette_projection_monotonic_in_strength(tmp_path):
    pal = _blue_palette()
    room = Image.open(_gray_room(tmp_path))
    t_mean, _ = _palette_target_stats(pal)
    dists = [
        np.linalg.norm(_mean_lab(apply_palette_projection(room, pal, s)) - t_mean)
        for s in (0.0, 0.5, 1.0)
    ]
    assert dists[0] > dists[1] > dists[2]


def test_swatch_image(tmp_path):
    pal = _blue_palette()
    img = palette_swatch_image(pal, size=100)
    assert img.size == (100, 100)
    colors = {tuple(c) for c in np.asarray(img).reshape(-1, 3).tolist()}
    # both swatch colors should appear
    assert (40, 70, 200) in colors and (30, 30, 35) in colors


def test_variant_plans_deterministic_and_seeded():
    spec = DesignSpec(palette=_blue_palette(), furniture=FurnitureSpec(), seed=1000)
    plans = build_variant_plans(spec, 4, start_index=2)
    assert [p.variant_index for p in plans] == [2, 3, 4, 5]
    assert [p.seed for p in plans] == [1002, 1003, 1004, 1005]
    # distinct strengths across the baseline spread
    assert len({p.strength for p in plans}) > 1


def test_locked_palette_pins_strength():
    spec = DesignSpec(palette=_blue_palette(), furniture=FurnitureSpec(), seed=0)
    plans = build_variant_plans(spec, 3, locked={"palette": True, "slots": {}})
    assert len({p.strength for p in plans}) == 1   # palette locked → stable color


def test_generate_variants_cpu_fallback(tmp_path):
    # No torch/diffusers here → engineered baseline path produces real PNGs.
    folder = str(tmp_path)
    room = _gray_room(tmp_path)
    spec = DesignSpec(palette=_blue_palette(), furniture=FurnitureSpec(), seed=7)
    paths = generate_variants(room, folder, "t", spec, n=3)
    assert len(paths) == 3
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    for arr in imgs:
        assert arr.shape == (64, 96, 3)
    # different strengths → visibly different variants
    assert not np.array_equal(imgs[0], imgs[2])
