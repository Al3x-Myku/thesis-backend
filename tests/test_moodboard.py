"""Unit tests for app/moodboard.py — the disentangled {palette, furniture} representation.

Covers the CPU-only APPEARANCE channel (CIELAB conversion + palette clustering)
and DesignSpec JSON (de)serialization. The furniture/CLIP channel is GPU-only and
is exercised separately on the model box.
"""

import os
import sys
import json
import tempfile

import numpy as np
import pytest
from PIL import Image

# Allow import of app.moodboard without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.moodboard import (
    extract_palette,
    rgb_to_lab,
    lab_to_rgb,
    DesignSpec,
    Palette,
    FurnitureSpec,
    analyze_moodboard,
    _median_cut,
)


def _synthetic_board(tmpdir) -> str:
    """50% red, 30% green, 20% blue."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :, :] = [220, 30, 40]
    arr[50:80, :, :] = [40, 200, 60]
    arr[80:, :, :] = [50, 60, 210]
    path = os.path.join(tmpdir, "board.png")
    Image.fromarray(arr).save(path)
    return path


def test_lab_white_black():
    lab_white = rgb_to_lab(np.array([1.0, 1.0, 1.0]))
    assert lab_white[0] == pytest.approx(100.0, abs=1e-3)
    assert lab_white[1] == pytest.approx(0.0, abs=1e-3)
    assert lab_white[2] == pytest.approx(0.0, abs=1e-3)
    assert rgb_to_lab(np.array([0.0, 0.0, 0.0]))[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("c", [[0.2, 0.5, 0.8], [0.9, 0.1, 0.3], [0.5, 0.5, 0.5]])
def test_lab_roundtrip(c):
    back = lab_to_rgb(rgb_to_lab(np.array(c)))
    assert np.allclose(back, c, atol=1e-3)


def test_palette_proportions_and_order(tmp_path):
    pal = extract_palette([_synthetic_board(tmp_path)], k=3, seed=0)
    assert isinstance(pal, Palette)
    assert len(pal.swatches) == 3
    # ordered by proportion desc, dominant ≈ 0.5
    props = [s.proportion for s in pal.swatches]
    assert props[0] > props[1] > props[2]
    assert props[0] == pytest.approx(0.5, abs=0.05)
    # dominant swatch is reddish
    r, g, b = pal.swatches[0].rgb
    assert r > g and r > b


def test_median_cut_fallback(tmp_path):
    arr = np.asarray(Image.open(_synthetic_board(tmp_path)).convert("RGB"), dtype=np.float64)
    lab = rgb_to_lab(arr.reshape(-1, 3) / 255.0)
    labels, centers = _median_cut(lab, 3)
    assert centers.shape == (3, 3)
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_designspec_json_roundtrip(tmp_path):
    pal = extract_palette([_synthetic_board(tmp_path)], k=3, seed=0)
    spec = DesignSpec(palette=pal, furniture=FurnitureSpec())
    parsed = json.loads(json.dumps(spec.to_dict()))
    # shape consumed by design_service.session_summary
    assert parsed["palette"]["swatches"][0]["rgb"]
    spec2 = DesignSpec.from_dict(parsed)
    assert spec2.palette.swatches[0].rgb == pal.swatches[0].rgb


def test_analyze_palette_only_without_gpu(tmp_path):
    # furniture channel degrades gracefully when the model stack is unavailable
    spec = analyze_moodboard([_synthetic_board(tmp_path)], work_dir=str(tmp_path), scene_id="t", k=3, seed=0)
    assert len(spec.palette.swatches) == 3
    assert spec.furniture.items == []
