
import math
import logging
from typing import Tuple, Optional, List, Dict

import numpy as np
import trimesh

logger = logging.getLogger("ThesisPipeline")

def _fit_vanishing_point(lines: np.ndarray, angles: np.ndarray,
                         angle_centre: float, angle_tol: float = 15.0) -> Optional[np.ndarray]:

    mask = np.abs(angles - angle_centre) < angle_tol
    mask |= np.abs(angles - angle_centre + 180) < angle_tol
    mask |= np.abs(angles - angle_centre - 180) < angle_tol

    subset = lines[mask]
    if len(subset) < 2:
        return None

    homo_lines = []
    for seg in subset:
        p1 = np.array([seg[0], seg[1], 1.0])
        p2 = np.array([seg[2], seg[3], 1.0])
        l = np.cross(p1, p2)
        norm = np.linalg.norm(l[:2])
        if norm > 1e-8:
            homo_lines.append(l / norm)

    if len(homo_lines) < 2:
        return None

    homo_lines = np.array(homo_lines)

    A = homo_lines
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        vp = Vt[-1]
        if abs(vp[2]) < 1e-12:
            return None
        vp = vp / vp[2]
        return vp[:2]
    except np.linalg.LinAlgError:
        return None

def estimate_intrinsics_vp(image_rgb: np.ndarray, W: int, H: int) -> Tuple[float, float, float]:

    f_fallback = max(W, H) * 1.2
    cx_fallback = W / 2.0
    cy_fallback = H / 2.0

    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available — using fallback intrinsics")
        return f_fallback, cx_fallback, cy_fallback

    if len(image_rgb.shape) == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segments, widths, precs, nfas = lsd.detect(gray)

    if segments is None or len(segments) < 6:
        logger.info(f"LSD found {0 if segments is None else len(segments)} segments — using fallback intrinsics")
        return f_fallback, cx_fallback, cy_fallback

    segments = segments.reshape(-1, 4)

    diag = math.sqrt(W * W + H * H)
    min_len = diag * 0.03
    lengths = np.sqrt((segments[:, 2] - segments[:, 0]) ** 2 +
                      (segments[:, 3] - segments[:, 1]) ** 2)
    long_mask = lengths > min_len
    segments = segments[long_mask]
    lengths = lengths[long_mask]

    if len(segments) < 6:
        logger.info(f"Only {len(segments)} long segments — using fallback intrinsics")
        return f_fallback, cx_fallback, cy_fallback

    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))

    hist, bin_edges = np.histogram(angles, bins=180, range=(-90, 90))

    peak_angles = []
    smooth_hist = np.convolve(hist, np.ones(5) / 5, mode='same')
    for _ in range(3):
        if smooth_hist.max() < 2:
            break
        peak_idx = np.argmax(smooth_hist)
        peak_angle = bin_edges[peak_idx] + 0.5
        peak_angles.append(peak_angle)
        suppress_lo = max(0, peak_idx - 15)
        suppress_hi = min(len(smooth_hist), peak_idx + 16)
        smooth_hist[suppress_lo:suppress_hi] = 0

    if len(peak_angles) < 2:
        logger.info(f"Only {len(peak_angles)} dominant directions — using fallback intrinsics")
        return f_fallback, cx_fallback, cy_fallback

    vps = []
    for pa in peak_angles:
        vp = _fit_vanishing_point(segments, angles, pa, angle_tol=12.0)
        if vp is not None:
            vps.append(vp)

    if len(vps) < 2:
        logger.info(f"Could not estimate 2+ vanishing points — using fallback intrinsics")
        return f_fallback, cx_fallback, cy_fallback

    logger.info(f"Detected {len(vps)} vanishing points: {[f'({v[0]:.0f},{v[1]:.0f})' for v in vps]}")

    if len(vps) >= 3:
        cx_est = float(np.mean([v[0] for v in vps]))
        cy_est = float(np.mean([v[1] for v in vps]))
    else:
        cx_est = cx_fallback
        cy_est = cy_fallback

    cx_est = np.clip(cx_est, W * 0.3, W * 0.7)
    cy_est = np.clip(cy_est, H * 0.3, H * 0.7)

    best_f2 = None
    for i in range(len(vps)):
        for j in range(i + 1, len(vps)):
            v1c = np.array([vps[i][0] - cx_est, vps[i][1] - cy_est])
            v2c = np.array([vps[j][0] - cx_est, vps[j][1] - cy_est])
            f2_candidate = -(v1c[0] * v2c[0] + v1c[1] * v2c[1])

            if f2_candidate > 0:
                if best_f2 is None or abs(f2_candidate - (max(W, H) * 1.0) ** 2) < abs(best_f2 - (max(W, H) * 1.0) ** 2):
                    best_f2 = f2_candidate

    if best_f2 is not None and best_f2 > 0:
        f_est = math.sqrt(best_f2)
        if diag * 0.3 < f_est < diag * 3.0:
            logger.info(f"VP intrinsics: f={f_est:.1f}px, cx={cx_est:.1f}, cy={cy_est:.1f}")
            return float(f_est), float(cx_est), float(cy_est)

    logger.info(f"VP focal length estimation failed (f²={best_f2}) — using fallback")
    return f_fallback, cx_fallback, cy_fallback

def _backproject_depth_region(
    depth_map: np.ndarray,
    f: float, cx: float, cy: float,
    W: int, H: int,
    row_start_frac: float = 0.0,
    row_end_frac: float = 1.0,
    max_points: int = 50000,
) -> np.ndarray:

    row_start = int(H * row_start_frac)
    row_end = int(H * row_end_frac)

    region = depth_map[row_start:row_end, :]
    valid_mask = (region > 0.05) & (region < 40.0)

    v_local, u_local = np.where(valid_mask)
    Z = region[valid_mask]

    if len(Z) == 0:
        return np.zeros((0, 3))

    if len(Z) > max_points:
        indices = np.random.choice(len(Z), max_points, replace=False)
        v_local = v_local[indices]
        u_local = u_local[indices]
        Z = Z[indices]

    v_global = v_local + row_start
    X = (u_local.astype(np.float64) - cx) * Z / f
    Y = -(v_global.astype(np.float64) - cy) * Z / f

    return np.column_stack([X, Y, Z])

def extract_floor_plane_ransac(
    depth_map: np.ndarray,
    f: float, cx: float, cy: float,
    W: int, H: int,
    scene_id: str = "",
) -> Tuple[float, np.ndarray]:

    pts = _backproject_depth_region(depth_map, f, cx, cy, W, H,
                                    row_start_frac=0.60, row_end_frac=1.0,
                                    max_points=50000)

    if len(pts) < 100:
        logger.warning(f"[{scene_id}] Too few points for RANSAC floor ({len(pts)}) — using percentile fallback")
        return _floor_percentile_fallback(depth_map, f, cx, cy, W, H, scene_id)

    try:
        from sklearn.linear_model import RANSACRegressor

        XZ = pts[:, [0, 2]]
        Y = pts[:, 1]

        ransac = RANSACRegressor(
            min_samples=3,
            residual_threshold=0.05,
            max_trials=500,
            random_state=42,
        )
        ransac.fit(XZ, Y)

        inlier_mask = ransac.inlier_mask_
        inlier_ratio = inlier_mask.sum() / len(inlier_mask)

        if inlier_ratio < 0.15:
            logger.warning(f"[{scene_id}] RANSAC inlier ratio too low ({inlier_ratio:.1%}) — using fallback")
            return _floor_percentile_fallback(depth_map, f, cx, cy, W, H, scene_id)

        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_

        normal = np.array([-a, 1.0, -b])
        normal = normal / np.linalg.norm(normal)

        if normal[1] < 0:
            normal = -normal

        median_z = float(np.median(pts[inlier_mask, 2]))
        floor_y = float(c + b * median_z)

        logger.info(
            f"[{scene_id}] RANSAC floor plane: Y={floor_y:.3f}m, "
            f"normal=({normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}), "
            f"inlier_ratio={inlier_ratio:.1%} ({inlier_mask.sum()}/{len(inlier_mask)} pts)"
        )
        return floor_y, normal

    except ImportError:
        logger.warning(f"[{scene_id}] scikit-learn not available for RANSAC — using fallback")
        return _floor_percentile_fallback(depth_map, f, cx, cy, W, H, scene_id)
    except Exception as e:
        logger.warning(f"[{scene_id}] RANSAC failed ({e}) — using fallback")
        return _floor_percentile_fallback(depth_map, f, cx, cy, W, H, scene_id)

def _floor_percentile_fallback(
    depth_map: np.ndarray,
    f: float, cx: float, cy: float,
    W: int, H: int,
    scene_id: str = "",
) -> Tuple[float, np.ndarray]:

    pts = _backproject_depth_region(depth_map, f, cx, cy, W, H,
                                    row_start_frac=0.8, row_end_frac=1.0,
                                    max_points=20000)
    if len(pts) < 10:
        floor_y = -1.35
        logger.warning(f"[{scene_id}] No depth points for floor estimation — defaulting to Y={floor_y}")
        return floor_y, np.array([0.0, 1.0, 0.0])

    floor_y = float(np.percentile(pts[:, 1], 10))
    logger.info(f"[{scene_id}] Floor Y (percentile fallback): {floor_y:.3f}m")
    return floor_y, np.array([0.0, 1.0, 0.0])

def sample_bottom_mask_depth(
    depth_map: np.ndarray,
    mask: np.ndarray,
    bottom_fraction: float = 0.10,
) -> float:

    rows_with_mask = np.any(mask, axis=1)
    if not rows_with_mask.any():
        return 0.0

    row_indices = np.where(rows_with_mask)[0]
    row_min, row_max = row_indices[0], row_indices[-1]
    mask_height = row_max - row_min + 1

    if mask_height < 2:
        bottom_start = row_min
    else:
        bottom_start = row_max - max(1, int(mask_height * bottom_fraction))

    bottom_mask = mask.copy()
    bottom_mask[:bottom_start, :] = False

    depths = depth_map[bottom_mask]
    valid = depths[(depths > 0.05) & (depths < 40.0)]

    if valid.size == 0:
        depths = depth_map[mask]
        valid = depths[(depths > 0.05) & (depths < 40.0)]
        if valid.size == 0:
            return 0.0

    return float(np.median(valid))

def anchor_mesh_to_floor(mesh, floor_y: float = 0.0):

    try:
        if hasattr(mesh, 'bounds') and mesh.bounds is not None:
            current_min_y = float(mesh.bounds[0][1])
        else:
            all_y = []
            if hasattr(mesh, 'geometry'):
                for g in mesh.geometry.values():
                    if hasattr(g, 'vertices') and len(g.vertices) > 0:
                        all_y.append(g.vertices[:, 1].min())
            if all_y:
                current_min_y = min(all_y)
            else:
                return mesh

        shift = floor_y - current_min_y
        mesh.apply_translation([0.0, shift, 0.0])
    except Exception as e:
        logger.warning(f"anchor_mesh_to_floor failed: {e}")

    return mesh

def manhattan_align(
    scene: trimesh.Scene,
    depth_map: np.ndarray,
    f: float, cx: float, cy: float,
    W: int, H: int,
    floor_normal: np.ndarray,
    scene_id: str = "",
) -> trimesh.Scene:

    up = np.array([0.0, 1.0, 0.0])
    floor_n = floor_normal / (np.linalg.norm(floor_normal) + 1e-12)

    dot = np.clip(np.dot(floor_n, up), -1.0, 1.0)
    if abs(dot) < 0.999:
        axis = np.cross(floor_n, up)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
            angle = math.acos(dot)
            R_tilt = trimesh.transformations.rotation_matrix(angle, axis)
            scene.apply_transform(R_tilt)
            logger.info(f"[{scene_id}] Manhattan: tilt correction {math.degrees(angle):.1f}°")

    left_pts = _backproject_depth_region(depth_map, f, cx, cy, W, H,
                                         row_start_frac=0.2, row_end_frac=0.8,
                                         max_points=10000)
    right_pts = _backproject_depth_region(depth_map, f, cx, cy, W, H,
                                          row_start_frac=0.2, row_end_frac=0.8,
                                          max_points=10000)

    if len(left_pts) > 0:
        pass

    all_pts = _backproject_depth_region(depth_map, f, cx, cy, W, H,
                                        row_start_frac=0.1, row_end_frac=0.9,
                                        max_points=30000)

    if len(all_pts) < 100:
        logger.info(f"[{scene_id}] Manhattan: insufficient points for yaw alignment")
        return scene

    try:
        dz_du = np.gradient(depth_map, axis=1)
        dz_dv = np.gradient(depth_map, axis=0)

        valid = (depth_map > 0.1) & (depth_map < 30.0)
        band = np.zeros_like(valid)
        band[int(H * 0.2):int(H * 0.8), :] = True
        valid &= band

        dz_du_valid = dz_du[valid]
        dz_dv_valid = dz_dv[valid]
        Z_valid = depth_map[valid]

        grad_mag = np.sqrt(dz_du ** 2 + dz_dv ** 2)
        wall_pixels = valid & (grad_mag > 0.02) & (np.abs(dz_dv) < np.abs(dz_du) * 2)

        if wall_pixels.sum() < 50:
            logger.info(f"[{scene_id}] Manhattan: insufficient wall pixels for yaw alignment")
            return scene

        wall_angles = np.arctan2(dz_dv[wall_pixels], dz_du[wall_pixels])

        wall_angles_mod = np.mod(np.degrees(wall_angles), 90.0)
        hist, bin_edges = np.histogram(wall_angles_mod, bins=90, range=(0, 90))
        smooth = np.convolve(hist, np.ones(5) / 5, mode='same')
        peak_bin = np.argmax(smooth)
        dominant_angle = bin_edges[peak_bin] + 0.5

        if dominant_angle > 45:
            yaw_correction = -(90 - dominant_angle)
        else:
            yaw_correction = -dominant_angle

        if 1.0 < abs(yaw_correction) < 20.0:
            R_yaw = trimesh.transformations.rotation_matrix(
                math.radians(yaw_correction), [0, 1, 0]
            )
            scene.apply_transform(R_yaw)
            logger.info(f"[{scene_id}] Manhattan: yaw correction {yaw_correction:.1f}° "
                       f"(dominant wall angle: {dominant_angle:.1f}°)")
        else:
            logger.info(f"[{scene_id}] Manhattan: yaw correction {yaw_correction:.1f}° "
                       f"(below threshold or too large — skipped)")

    except Exception as e:
        logger.warning(f"[{scene_id}] Manhattan yaw alignment failed: {e}")

    return scene

def estimate_ceiling_y(
    depth_map: np.ndarray,
    f: float, cx: float, cy: float,
    W: int, H: int,
    scene_id: str = "",
) -> float:

    pts = _backproject_depth_region(depth_map, f, cx, cy, W, H,
                                    row_start_frac=0.0, row_end_frac=0.15,
                                    max_points=20000)
    if len(pts) < 10:
        logger.warning(f"[{scene_id}] Cannot estimate ceiling — defaulting to floor_y + 2.7m")
        return None

    ceiling_y = float(np.percentile(pts[:, 1], 90))
    logger.info(f"[{scene_id}] Estimated ceiling Y: {ceiling_y:.3f}m")
    return ceiling_y

def scale_scene_metric(
    scene: trimesh.Scene,
    floor_y: float,
    ceiling_y: float,
    target_height: float = 2.7,
    scene_id: str = "",
) -> Tuple[trimesh.Scene, float]:

    h_current = ceiling_y - floor_y

    if h_current < 0.1:
        logger.warning(f"[{scene_id}] Metric scaling: floor-ceiling gap too small ({h_current:.3f}m) — skipping")
        return scene, 1.0

    scale = target_height / h_current

    if scale < 0.1 or scale > 10.0:
        logger.warning(f"[{scene_id}] Metric scaling factor {scale:.2f} out of range — clamping")
        scale = np.clip(scale, 0.3, 5.0)

    scene.apply_scale(scale)

    logger.info(
        f"[{scene_id}] Metric scaling: {h_current:.2f}m → {target_height:.1f}m "
        f"(scale factor: {scale:.3f})"
    )
    return scene, float(scale)

def resolve_overlaps(
    positions: np.ndarray,
    extents: np.ndarray,
    max_iterations: int = 20,
    scene_id: str = "",
) -> np.ndarray:

    n = len(positions)
    if n < 2:
        return positions.copy()

    pos = positions.copy().astype(np.float64)
    ext = extents.copy().astype(np.float64)
    volumes = ext[:, 0] * ext[:, 1] * ext[:, 2]

    total_pushes = 0

    for iteration in range(max_iterations):
        any_overlap = False

        for i in range(n):
            for j in range(i + 1, n):
                half_i = ext[i] / 2.0
                half_j = ext[j] / 2.0

                overlap = np.minimum(
                    pos[i] + half_i, pos[j] + half_j
                ) - np.maximum(
                    pos[i] - half_i, pos[j] - half_j
                )

                if np.all(overlap > 0):
                    any_overlap = True
                    direction = pos[j] - pos[i]
                    dist = np.linalg.norm(direction)

                    if dist < 1e-6:
                        direction = np.array([1.0, 0.0, 0.0])
                        dist = 1.0

                    direction = direction / dist

                    min_push = np.min(overlap) + 0.02

                    if volumes[i] <= volumes[j]:
                        pos[i] -= direction * min_push
                    else:
                        pos[j] += direction * min_push

                    total_pushes += 1

        if not any_overlap:
            break

    if total_pushes > 0:
        logger.info(f"[{scene_id}] Overlap resolution: {total_pushes} pushes in {iteration + 1} iterations")

    return pos
