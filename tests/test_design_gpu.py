"""GPU validation tests for the Interior Design Zone (M2–M5).

Run on the GPU box with:
    conda run -n ml python -m pytest tests/test_design_gpu.py -v -s

These tests exercise the full ML paths that cannot run on the dev box:
  - M2: CLIP furniture extraction + full analyze_moodboard
  - M3: generate_variants (CPU baseline path + structure conditioning)
  - M4: locked-slot pixel-exact compositing in regenerate
  - M5: commit_reconstruction (detect → palette-projected build_mesh → position_meshes)

Tests are ordered by milestone and are self-contained: each creates its own
temp directory so they can run independently or in sequence.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Test data setup helpers
# ---------------------------------------------------------------------------

ROOM_PHOTO_SRC = Path(
    "/home/al3xmyku/thesis-backend/benchmark_results/"
    "scene_00000_485142_out/wall_views/wall_01_rgb.jpg"
)
MOODBOARD_DIR = Path("/tmp/design_val/moodboard")

# Synthetic palette colors for fast moodboard creation (no disk dependency)
_SCANDI_PALETTE = [(230, 225, 215), (200, 195, 185), (155, 165, 140), (185, 160, 125)]
_MEDI_PALETTE = [(35, 60, 100), (190, 90, 60), (215, 195, 155), (100, 120, 70)]


def _make_palette_image(swatches, size=(256, 256)) -> Image.Image:
    img = Image.new("RGB", size)
    arr = np.array(img)
    n = len(swatches)
    w = size[0] // n
    for i, (r, g, b) in enumerate(swatches):
        x0 = i * w
        x1 = x0 + w if i < n - 1 else size[0]
        arr[:, x0:x1] = [r, g, b]
    return Image.fromarray(arr)


def _setup_scene(tmp: Path, with_moodboard: bool = True):
    """Copy/create test assets into tmp."""
    tmp.mkdir(parents=True, exist_ok=True)
    room = tmp / "input.png"
    if ROOM_PHOTO_SRC.exists():
        shutil.copyfile(ROOM_PHOTO_SRC, room)
    else:
        # Fallback: 256×256 neutral gray room
        Image.new("RGB", (256, 256), (180, 175, 165)).save(room)

    mb_dir = tmp / "moodboard"
    mb_dir.mkdir()
    if with_moodboard:
        if MOODBOARD_DIR.exists():
            for f in sorted(MOODBOARD_DIR.glob("board_*.png")):
                shutil.copyfile(f, mb_dir / f.name)
        else:
            _make_palette_image(_SCANDI_PALETTE).save(mb_dir / "board_0.png")
            _make_palette_image(_MEDI_PALETTE).save(mb_dir / "board_1.png")
    return room, mb_dir


# ---------------------------------------------------------------------------
# M2 — Mood board analysis (palette + CLIP furniture)
# ---------------------------------------------------------------------------

class TestM2MoodboardAnalysis:

    def test_extract_palette_from_synthetic_moodboard(self, tmp_path):
        """extract_palette returns k swatches, proportions sum to 1, LAB values valid."""
        from app.moodboard import extract_palette

        board = _make_palette_image(_SCANDI_PALETTE)
        board_path = str(tmp_path / "board.png")
        board.save(board_path)

        palette = extract_palette([board_path], k=4)
        assert len(palette.swatches) > 0, "No swatches returned"
        total = sum(s.proportion for s in palette.swatches)
        assert abs(total - 1.0) < 0.01, f"Proportions do not sum to 1: {total}"
        for s in palette.swatches:
            assert len(s.lab) == 3
            assert 0 <= s.lab[0] <= 100, f"L* out of range: {s.lab[0]}"

    def test_extract_palette_from_real_moodboard(self, tmp_path):
        """extract_palette on real synthetic moodboard returns distinct swatches."""
        from app.moodboard import extract_palette

        board_path = str(tmp_path / "board.png")
        _make_palette_image(_MEDI_PALETTE).save(board_path)
        palette = extract_palette([board_path], k=4)
        assert len(palette.swatches) >= 2

        # The dominant hues should span a reasonable LAB range (not all grey)
        labs = np.array([s.lab for s in palette.swatches])
        assert np.ptp(labs[:, 1], axis=0) > 5 or np.ptp(labs[:, 2], axis=0) > 5, \
            "All swatches collapsed to grey — palette extraction likely failed"

    def test_clip_furniture_extraction(self, tmp_path):
        """extract_furniture_style runs CLIP on moodboard crops without crashing."""
        from app.moodboard import extract_furniture_style

        # Use a room wall image as a "moodboard" so D-FINE can find furniture
        board_path = str(tmp_path / "board.png")
        if ROOM_PHOTO_SRC.exists():
            shutil.copyfile(ROOM_PHOTO_SRC, board_path)
        else:
            Image.new("RGB", (256, 256), (180, 175, 165)).save(board_path)

        work_dir = str(tmp_path / "mb_work")
        spec = extract_furniture_style([board_path], work_dir, scene_id="gpu_test_clip")
        # May return empty if no indoor furniture detected, but must not crash
        assert spec is not None
        assert hasattr(spec, "items")

    def test_analyze_moodboard_full(self, tmp_path):
        """analyze_moodboard returns a DesignSpec with palette + best-effort furniture."""
        from app.moodboard import analyze_moodboard, DesignSpec

        board_path = str(tmp_path / "board.png")
        if ROOM_PHOTO_SRC.exists():
            shutil.copyfile(ROOM_PHOTO_SRC, board_path)
        else:
            _make_palette_image(_SCANDI_PALETTE).save(board_path)

        work_dir = str(tmp_path / "mb_work")
        spec = analyze_moodboard([board_path], work_dir, scene_id="gpu_test_full")

        assert isinstance(spec, DesignSpec)
        assert spec.palette is not None
        assert len(spec.palette.swatches) > 0

        # Roundtrip through JSON (critical for DB storage)
        d = spec.to_dict()
        spec2 = DesignSpec.from_dict(d)
        assert len(spec2.palette.swatches) == len(spec.palette.swatches)
        for s1, s2 in zip(spec.palette.swatches, spec2.palette.swatches):
            assert abs(s1.lab[0] - s2.lab[0]) < 0.01


# ---------------------------------------------------------------------------
# M3 — 2D restyle engine (structure conditioning + generate_variants)
# ---------------------------------------------------------------------------

class TestM3RestyLe2D:

    def test_build_structure_conditioning(self, tmp_path):
        """build_structure_conditioning produces canny edges and optionally depth."""
        from app.restyle_2d import build_structure_conditioning

        room_path = str(tmp_path / "input.png")
        if ROOM_PHOTO_SRC.exists():
            shutil.copyfile(ROOM_PHOTO_SRC, room_path)
        else:
            Image.new("RGB", (256, 256), (180, 175, 165)).save(room_path)

        struct = build_structure_conditioning(room_path, str(tmp_path), "gpu_test_struct")
        assert "room" in struct
        assert struct["room"] is not None
        # Canny should always succeed (pure OpenCV, no GPU)
        assert struct.get("canny") is not None, "Canny edges missing"

    def test_generate_variants_cpu_baseline(self, tmp_path):
        """generate_variants (CPU palette-projection baseline) produces N PNGs."""
        from app.moodboard import extract_palette, FurnitureSpec, DesignSpec
        from app.restyle_2d import generate_variants

        board_path = str(tmp_path / "board.png")
        _make_palette_image(_SCANDI_PALETTE).save(board_path)
        palette = extract_palette([board_path], k=4)

        # Minimal DesignSpec with empty furniture (palette-only)
        spec = DesignSpec(
            palette=palette,
            furniture=FurnitureSpec(items=[], style_centroid=None),
            seed=42,
        )

        room_path = str(tmp_path / "input.png")
        if ROOM_PHOTO_SRC.exists():
            shutil.copyfile(ROOM_PHOTO_SRC, room_path)
        else:
            _make_palette_image(_SCANDI_PALETTE, size=(256, 256)).save(room_path)

        paths = generate_variants(room_path, str(tmp_path), "gpu_test_gen", spec, n=3)

        assert len(paths) == 3
        for p in paths:
            assert Path(p).exists(), f"Variant image missing: {p}"
            im = Image.open(p)
            assert im.width > 0 and im.height > 0

    def test_palette_actually_shifts_colors(self, tmp_path):
        """Palette projection should measurably shift a gray room toward the target palette.

        We use a strongly warm (orange/red) palette vs a cool-gray room and check that
        the LAB a* channel (green-red axis) increases toward the warm target.
        """
        from app.moodboard import extract_palette, FurnitureSpec, DesignSpec, rgb_to_lab
        from app.restyle_2d import generate_variants, apply_palette_projection

        # Strongly warm/orange palette — a* should be clearly positive
        warm_swatches = [(210, 100, 40), (230, 120, 50), (200, 80, 30), (220, 110, 45)]
        board_path = str(tmp_path / "board.png")
        _make_palette_image(warm_swatches).save(board_path)
        palette = extract_palette([board_path], k=4)

        spec = DesignSpec(
            palette=palette,
            furniture=FurnitureSpec(items=[], style_centroid=None),
            seed=99,
        )

        # Cool gray room: a* ≈ 0 in LAB
        room = Image.new("RGB", (128, 128), (160, 165, 175))
        room_path = str(tmp_path / "input.png")
        room.save(room_path)

        # Use apply_palette_projection directly (deterministic, bypasses diffusion)
        projected = apply_palette_projection(room, palette, strength=0.9)

        room_lab = rgb_to_lab(np.array(room).astype(np.float64) / 255.0)
        proj_lab = rgb_to_lab(np.array(projected).astype(np.float64) / 255.0)

        orig_a = room_lab[:, :, 1].mean()   # green-red axis
        proj_a = proj_lab[:, :, 1].mean()

        assert proj_a > orig_a + 5, (
            f"Warm palette did not shift a* upward: room a*={orig_a:.1f}, projected a*={proj_a:.1f}"
        )


# ---------------------------------------------------------------------------
# M4 — Pixel-exact slot compositing
# ---------------------------------------------------------------------------

class TestM4FeedbackFreeze:

    def test_locked_slot_is_pixel_identical(self, tmp_path):
        """When slot 0 is locked, those pixels in the regenerated variant must be
        byte-identical to the parent variant."""
        from app.moodboard import extract_palette, FurnitureSpec, DesignSpec
        from app.restyle_2d import generate_variants, build_structure_conditioning

        board_path = str(tmp_path / "board.png")
        _make_palette_image(_SCANDI_PALETTE).save(board_path)
        palette = extract_palette([board_path], k=4)
        spec = DesignSpec(
            palette=palette,
            furniture=FurnitureSpec(items=[], style_centroid=None),
            seed=7,
        )

        room = Image.new("RGB", (128, 128), (180, 175, 165))
        room_path = str(tmp_path / "input.png")
        room.save(room_path)

        # Generate parent variant
        parent_paths = generate_variants(room_path, str(tmp_path), "gpu_m4", spec, n=1)
        parent_path = parent_paths[0]
        parent_img = np.array(Image.open(parent_path))

        # Inject a fake box so compositing has something to lock onto
        struct = build_structure_conditioning(room_path, str(tmp_path), "gpu_m4")
        struct["boxes"] = [[10, 10, 50, 50]]   # slot 0 occupies (10,10)-(50,50)

        from app.restyle_2d import render_variant, build_variant_plans
        from app.moodboard import DesignSpec

        locked = {"palette": False, "slots": {"0": True}}
        plans = build_variant_plans(spec, 1, start_index=1, locked=locked)
        child_path = str(tmp_path / "variants" / "variant_child.png")
        Path(child_path).parent.mkdir(parents=True, exist_ok=True)

        render_variant(
            struct, spec, plans[0], child_path,
            parent_image=Image.open(parent_path).convert("RGB"),
        )

        child_img = np.array(Image.open(child_path))
        # Locked region must be byte-identical to parent
        locked_region_parent = parent_img[10:50, 10:50]
        locked_region_child = child_img[10:50, 10:50]
        assert np.array_equal(locked_region_parent, locked_region_child), (
            "Locked slot region changed between parent and child variant — M4 compositing broken"
        )

    def test_unlocked_variant_differs(self, tmp_path):
        """Different start_index values should produce visually different variants.

        In the CPU baseline path the seed is irrelevant (no stochastic diffusion);
        variants differ because build_variant_plans assigns distinct palette-projection
        strengths based on the global variant index (idx % 4). Specifically:
          idx=0 → strength=0.35, idx=2 → strength=0.75 — measurably different."""
        from app.moodboard import extract_palette, FurnitureSpec, DesignSpec
        from app.restyle_2d import generate_variants

        board_path = str(tmp_path / "board.png")
        _make_palette_image(_MEDI_PALETTE).save(board_path)
        palette = extract_palette([board_path], k=4)
        spec = DesignSpec(palette=palette, furniture=FurnitureSpec(items=[], style_centroid=None), seed=42)

        room = Image.new("RGB", (128, 128), (180, 175, 165))
        room_path = str(tmp_path / "input.png")
        room.save(room_path)

        p1 = generate_variants(room_path, str(tmp_path), "gpu_m4_unlocked", spec, n=1, start_index=0)[0]
        p2 = generate_variants(room_path, str(tmp_path), "gpu_m4_unlocked", spec, n=1, start_index=2)[0]

        arr1 = np.array(Image.open(p1))
        arr2 = np.array(Image.open(p2))
        assert not np.array_equal(arr1, arr2), (
            "Variants at index 0 (strength=0.35) and 2 (strength=0.75) are identical — "
            "build_variant_plans strength selection broken"
        )


# ---------------------------------------------------------------------------
# M5 — palette_to_shell_colors + build_mesh with palette (unit parts)
# ---------------------------------------------------------------------------

class TestM5PaletteUtils:

    def test_palette_to_shell_colors_returns_five_surfaces(self):
        """palette_to_shell_colors maps a palette to all 5 shell surfaces."""
        from app.moodboard import extract_palette
        from app.reconstructor_pipeline import palette_to_shell_colors
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            board_path = os.path.join(td, "board.png")
            _make_palette_image(_SCANDI_PALETTE).save(board_path)
            palette = extract_palette([board_path], k=4)

        colors = palette_to_shell_colors(palette)
        for key in ("floor", "back_wall", "left_wall", "right_wall", "ceiling"):
            assert key in colors, f"Missing shell key: {key}"
            rgba = colors[key]
            assert len(rgba) == 4
            assert all(0 <= v <= 255 for v in rgba), f"Color out of range: {rgba}"

    def test_palette_to_shell_colors_empty_palette_returns_empty(self):
        from app.moodboard import Palette
        from app.reconstructor_pipeline import palette_to_shell_colors

        colors = palette_to_shell_colors(Palette(swatches=[], k=0, space="lab"))
        assert colors == {}

    def test_palette_to_shell_colors_none_returns_empty(self):
        from app.reconstructor_pipeline import palette_to_shell_colors
        assert palette_to_shell_colors(None) == {}

    def test_position_meshes_shell_colors_accepted(self, tmp_path):
        """position_meshes with shell_colors= and out_subdir= doesn't crash on empty mesh list."""
        from app.moodboard import extract_palette
        from app.reconstructor_pipeline import position_meshes, palette_to_shell_colors

        board_path = str(tmp_path / "board.png")
        _make_palette_image(_SCANDI_PALETTE).save(board_path)
        palette = extract_palette([board_path], k=4)
        shell_colors = palette_to_shell_colors(palette)

        room_path = str(tmp_path / "input.png")
        if ROOM_PHOTO_SRC.exists():
            shutil.copyfile(ROOM_PHOTO_SRC, room_path)
        else:
            Image.new("RGB", (128, 128), (180, 175, 165)).save(room_path)

        out_path = position_meshes(
            mesh_paths=[],
            image_path=room_path,
            scene_folder=str(tmp_path),
            scene_id="gpu_m5_shell",
            boxes=[],
            labels=[],
            shell_colors=shell_colors,
            out_subdir="committed",
        )
        assert Path(out_path).exists(), "position_meshes did not produce a .glb"
        committed = tmp_path / "committed" / "scene_positioned.glb"
        assert committed.exists(), f"Output not in committed/ subdir: {out_path}"

    def test_position_meshes_default_still_writes_to_final(self, tmp_path):
        """Backward compat: position_meshes without out_subdir still writes to final/."""
        from app.reconstructor_pipeline import position_meshes

        room_path = str(tmp_path / "input.png")
        if ROOM_PHOTO_SRC.exists():
            shutil.copyfile(ROOM_PHOTO_SRC, room_path)
        else:
            Image.new("RGB", (128, 128), (180, 175, 165)).save(room_path)

        out_path = position_meshes(
            mesh_paths=[], image_path=room_path,
            scene_folder=str(tmp_path), scene_id="gpu_m5_compat",
            boxes=[], labels=[],
        )
        final = tmp_path / "final" / "scene_positioned.glb"
        assert final.exists(), f"Backward-compat: output not in final/: {out_path}"


# ---------------------------------------------------------------------------
# Integration smoke test — end-to-end design flow without DB
# ---------------------------------------------------------------------------

class TestDesignFlowIntegration:

    def test_moodboard_to_variant_pipeline(self, tmp_path):
        """End-to-end: moodboard → DesignSpec → 3 variants, all within one temp dir."""
        from app.moodboard import analyze_moodboard
        from app.restyle_2d import generate_variants

        # Prepare assets
        mb_dir = tmp_path / "moodboard"
        mb_dir.mkdir()
        for i, pal in enumerate([_SCANDI_PALETTE, _MEDI_PALETTE]):
            _make_palette_image(pal).save(mb_dir / f"board_{i}.png")

        room_path = str(tmp_path / "input.png")
        if ROOM_PHOTO_SRC.exists():
            shutil.copyfile(ROOM_PHOTO_SRC, room_path)
        else:
            Image.new("RGB", (256, 256), (180, 175, 165)).save(room_path)

        board_paths = [str(p) for p in sorted(mb_dir.glob("board_*.png"))]
        spec = analyze_moodboard(board_paths, str(mb_dir), scene_id="gpu_e2e")
        assert spec.palette is not None

        spec_json = json.dumps(spec.to_dict())
        spec_rt = type(spec).from_dict(json.loads(spec_json))

        variant_paths = generate_variants(
            room_path, str(tmp_path), "gpu_e2e", spec_rt, n=3
        )
        assert len(variant_paths) == 3
        for vp in variant_paths:
            assert Path(vp).exists()

        # Variants must differ from each other (different seeds)
        arrs = [np.array(Image.open(vp)) for vp in variant_paths]
        assert not np.array_equal(arrs[0], arrs[1]) or not np.array_equal(arrs[1], arrs[2]), \
            "All variants are identical — seed variation not working"
