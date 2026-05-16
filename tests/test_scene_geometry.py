"""
Unit tests for scene_geometry.py — Geometric Grounding Algorithms.

Validates the 6 core geometric functions with synthetic data:
1. VP intrinsics fallback
2. RANSAC floor plane on flat ground
3. Bottom-10% mask depth sampling
4. Mesh anchoring to floor
5. Metric scaling (5.4m → 2.7m)
6. Overlap resolution (two intersecting boxes)
"""

import sys
import os
import math
import numpy as np
import pytest
import trimesh

# Allow import of app.scene_geometry even without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.scene_geometry import (
    estimate_intrinsics_vp,
    extract_floor_plane_ransac,
    sample_bottom_mask_depth,
    anchor_mesh_to_floor,
    scale_scene_metric,
    resolve_overlaps,
)


# ─── 1. VP Intrinsics Fallback ────────────────────────────────────────

class TestVPIntrinsics:
    def test_fallback_on_blank_image(self):
        """A blank (uniform) image has no lines → should return fallback intrinsics."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        f, cx, cy = estimate_intrinsics_vp(blank, 640, 480)

        # Fallback: f = max(W,H) * 1.2
        assert abs(f - 640 * 1.2) < 1.0, f"Expected f≈768, got {f}"
        assert abs(cx - 320) < 1.0, f"Expected cx≈320, got {cx}"
        assert abs(cy - 240) < 1.0, f"Expected cy≈240, got {cy}"

    def test_fallback_on_noise_image(self):
        """Random noise has no coherent lines → should return fallback."""
        rng = np.random.RandomState(42)
        noise = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        f, cx, cy = estimate_intrinsics_vp(noise, 640, 480)

        # Should still get reasonable values
        assert f > 0, "Focal length must be positive"
        assert 0 < cx < 640, "cx must be within image"
        assert 0 < cy < 480, "cy must be within image"


# ─── 2. RANSAC Floor Plane ─────────────────────────────────────────

class TestRANSACFloor:
    def test_flat_floor_at_known_y(self):
        """Create a depth map where the bottom 40% projects to Y ≈ -1.5."""
        W, H = 640, 480
        f = 500.0
        cx, cy = W / 2, H / 2

        # Create a depth map: constant depth 3m everywhere
        depth = np.full((H, W), 3.0, dtype=np.float32)

        floor_y, normal = extract_floor_plane_ransac(
            depth, f, cx, cy, W, H, scene_id="test"
        )

        # Expected floor Y: -(v - cy) * Z / f for v in bottom rows
        # For v = H-1 = 479, Y = -(479 - 240) * 3.0 / 500 = -1.434
        assert floor_y < 0, f"Floor should be negative Y (got {floor_y})"
        assert abs(normal[1]) > 0.9, f"Floor normal should be ~vertical (got {normal})"

    def test_insufficient_points(self):
        """Empty depth map should use fallback."""
        depth = np.zeros((480, 640), dtype=np.float32)
        floor_y, normal = extract_floor_plane_ransac(
            depth, 500.0, 320, 240, 640, 480, scene_id="test_empty"
        )

        # Should still return some reasonable default
        assert isinstance(floor_y, float)
        assert len(normal) == 3


# ─── 3. Bottom-10% Mask Depth ──────────────────────────────────────

class TestBottomMaskDepth:
    def test_bottom_depth_vs_full(self):
        """Bottom 10% should return the depth at the bottom of the mask,
        which differs from the full-mask average."""
        H, W = 100, 50

        # Depth increases linearly from 2m (top) to 5m (bottom)
        depth = np.linspace(2.0, 5.0, H).reshape(-1, 1).repeat(W, axis=1).astype(np.float32)

        # Mask covers the full image
        mask = np.ones((H, W), dtype=bool)

        Z_bottom = sample_bottom_mask_depth(depth, mask, bottom_fraction=0.10)
        Z_full = float(np.median(depth[mask]))

        # Bottom 10% should be close to 5m, while full median is ~3.5m
        assert Z_bottom > Z_full, f"Bottom depth {Z_bottom} should exceed full median {Z_full}"
        assert Z_bottom > 4.5, f"Bottom depth should be near 5m, got {Z_bottom}"

    def test_empty_mask_returns_zero(self):
        depth = np.full((100, 50), 3.0, dtype=np.float32)
        mask = np.zeros((100, 50), dtype=bool)

        result = sample_bottom_mask_depth(depth, mask)
        assert result == 0.0


# ─── 4. Mesh Anchoring ─────────────────────────────────────────────

class TestAnchorMesh:
    def test_anchor_box_to_y_zero(self):
        """A box mesh centred at (0, 2, 0) should drop its bottom to Y=0."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        # Centre is at origin, bottom at Y=-0.5
        mesh.apply_translation([0, 2, 0])
        # Now bottom is at Y=1.5

        anchor_mesh_to_floor(mesh, floor_y=0.0)

        bottom_y = float(mesh.bounds[0][1])
        assert abs(bottom_y - 0.0) < 0.01, f"Bottom Y should be 0, got {bottom_y}"

    def test_anchor_to_negative_floor(self):
        """Anchor to a floor at Y=-1.5 (as RANSAC might return)."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        mesh.apply_translation([0, 5, 0])

        anchor_mesh_to_floor(mesh, floor_y=-1.5)

        bottom_y = float(mesh.bounds[0][1])
        assert abs(bottom_y - (-1.5)) < 0.01, f"Bottom Y should be -1.5, got {bottom_y}"


# ─── 5. Metric Scaling ─────────────────────────────────────────────

class TestMetricScaling:
    def test_double_height_halves(self):
        """A scene with 5.4m floor-to-ceiling should scale by 0.5 to reach 2.7m."""
        scene = trimesh.Scene()
        box = trimesh.creation.box(extents=[1, 1, 1])
        scene.add_geometry(box)

        scene, s = scale_scene_metric(scene, floor_y=0.0, ceiling_y=5.4,
                                       target_height=2.7, scene_id="test")

        assert abs(s - 0.5) < 0.01, f"Scale should be 0.5, got {s}"

    def test_skip_tiny_gap(self):
        """Floor-ceiling gap < 0.1m should skip scaling."""
        scene = trimesh.Scene()
        box = trimesh.creation.box(extents=[1, 1, 1])
        scene.add_geometry(box)

        scene, s = scale_scene_metric(scene, floor_y=0.0, ceiling_y=0.05,
                                       target_height=2.7, scene_id="test")

        assert s == 1.0, "Should skip scaling for tiny gap"


# ─── 6. Overlap Resolution ─────────────────────────────────────────

class TestOverlapResolution:
    def test_two_overlapping_boxes(self):
        """Two boxes at (0,0,0) and (0.5,0,0) with extent (1,1,1) overlap → should be pushed apart."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        extents = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])

        adjusted = resolve_overlaps(positions, extents, scene_id="test")

        # After resolution, the boxes should no longer overlap
        half_0 = extents[0] / 2
        half_1 = extents[1] / 2
        overlap = np.minimum(
            adjusted[0] + half_0, adjusted[1] + half_1
        ) - np.maximum(
            adjusted[0] - half_0, adjusted[1] - half_1
        )

        has_overlap = np.all(overlap > 0)
        assert not has_overlap, f"Boxes should not overlap after resolution (overlap={overlap})"

    def test_non_overlapping_unchanged(self):
        """Two distant boxes should not be moved."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        extents = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])

        adjusted = resolve_overlaps(positions, extents, scene_id="test")

        np.testing.assert_array_almost_equal(adjusted, positions,
                                              err_msg="Non-overlapping boxes should stay in place")

    def test_single_object(self):
        """Single object should be returned unchanged."""
        positions = np.array([[1.0, 2.0, 3.0]])
        extents = np.array([[1.0, 1.0, 1.0]])

        adjusted = resolve_overlaps(positions, extents)
        np.testing.assert_array_equal(adjusted, positions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
