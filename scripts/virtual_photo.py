
import cv2
import json
import logging
import math
import numpy as np
import os
import glob
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger("WallDecomposition")

DEFAULT_OUT_W = 1024
DEFAULT_OUT_H = 1024
DEFAULT_PITCH = 0.0
MIN_WALL_ANGLE_DEG = 10.0
FOV_MARGIN_DEG = 20.0
FOV_CLAMP = (70.0, 110.0)

@dataclass
class LayoutCorner:

    x_pixel: float
    y_ceiling_pixel: float
    y_floor_pixel: float
    azimuth_rad: float = 0.0

@dataclass
class WallInfo:

    wall_idx: int
    yaw_rad: float
    yaw_deg: float
    angular_width_rad: float
    angular_width_deg: float
    fov_deg: float

    normal: List[float] = field(default_factory=lambda: [0, 0, 0])
    center_3d: List[float] = field(default_factory=lambda: [0, 0, 0])
    left_corner_3d: List[float] = field(default_factory=lambda: [0, 0, 0])
    right_corner_3d: List[float] = field(default_factory=lambda: [0, 0, 0])
    width_m: float = 0.0
    height_m: float = 0.0
    floor_y: float = 0.0
    ceiling_y: float = 0.0

    pano_w: int = 0
    pano_h: int = 0
    left_corner_azimuth_rad: float = 0.0
    right_corner_azimuth_rad: float = 0.0

    rgb_path: str = ""
    depth_path: str = ""
    semantic_path: str = ""
    meta_path: str = ""

    overlaps_left: bool = False
    overlaps_right: bool = False

def build_perspective_map(
    yaw_rad: float,
    pitch_rad: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
    pano_w: int,
    pano_h: int,
) -> Tuple[np.ndarray, np.ndarray]:

    fov_rad = np.radians(fov_deg)
    f = (out_w / 2.0) / np.tan(fov_rad / 2.0)
    cx, cy = out_w / 2.0, out_h / 2.0

    x, y = np.meshgrid(np.arange(out_w), np.arange(out_h))
    rays = np.stack([(x - cx) / f, (y - cy) / f, np.ones_like(x)], axis=-1)

    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)

    Rx = np.array([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]])
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    R = Ry @ Rx

    rays_world = rays @ R.T
    rays_world /= np.linalg.norm(rays_world, axis=-1, keepdims=True)

    lat = np.arcsin(np.clip(rays_world[..., 1], -1, 1))
    lon = np.arctan2(rays_world[..., 0], rays_world[..., 2])
    map_x = ((lon / (2 * np.pi)) + 0.5) * pano_w
    map_y = ((lat / np.pi) + 0.5) * pano_h

    return map_x.astype(np.float32), map_y.astype(np.float32)

def read_layout_corners(layout_path: str, pano_w: int) -> List[LayoutCorner]:

    corners = []
    with open(layout_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                x = float(parts[0])
                y_ceil = float(parts[1]) if len(parts) >= 2 else 0.0
                y_floor = float(parts[2]) if len(parts) >= 3 else 512.0
                azimuth = (x / pano_w - 0.5) * 2 * np.pi
                corners.append(LayoutCorner(x, y_ceil, y_floor, azimuth))
            except ValueError:
                continue
    return corners

def _angular_midpoint(theta1: float, theta2: float) -> Tuple[float, float]:

    d = theta2 - theta1
    if d > np.pi:
        d -= 2 * np.pi
    elif d < -np.pi:
        d += 2 * np.pi
    center = theta1 + d / 2.0
    center = (center + np.pi) % (2 * np.pi) - np.pi
    return center, abs(d)

def _pixel_y_to_elevation(y_pixel: float, pano_h: int) -> float:

    return (0.5 - y_pixel / pano_h) * np.pi

def _estimate_wall_height_m(
    y_ceil_px: float,
    y_floor_px: float,
    pano_h: int,
    wall_distance: float,
) -> Tuple[float, float, float]:

    elev_ceil = _pixel_y_to_elevation(y_ceil_px, pano_h)
    elev_floor = _pixel_y_to_elevation(y_floor_px, pano_h)

    max_elev = np.radians(80.0)
    elev_ceil = np.clip(elev_ceil, -max_elev, max_elev)
    elev_floor = np.clip(elev_floor, -max_elev, max_elev)

    ceil_y = wall_distance * np.tan(elev_ceil)
    floor_y = wall_distance * np.tan(elev_floor)

    ceil_y = np.clip(ceil_y, -10.0, 10.0)
    floor_y = np.clip(floor_y, -10.0, 10.0)

    height = ceil_y - floor_y
    return max(height, 0.1), float(floor_y), float(ceil_y)

def compute_walls(
    corners: List[LayoutCorner],
    pano_w: int,
    pano_h: int,
    depth_map: Optional[np.ndarray] = None,
    min_angle_deg: float = MIN_WALL_ANGLE_DEG,
) -> List[WallInfo]:

    n = len(corners)
    if n < 3:
        return []

    walls: List[WallInfo] = []
    wall_idx = 0

    for i in range(n):
        c_left = corners[i]
        c_right = corners[(i + 1) % n]

        center_yaw, width_rad = _angular_midpoint(c_left.azimuth_rad, c_right.azimuth_rad)
        width_deg = np.degrees(width_rad)

        if width_deg < min_angle_deg:
            continue

        adapted_fov = width_deg + FOV_MARGIN_DEG
        final_fov = float(np.clip(adapted_fov, *FOV_CLAMP))

        wall_dist = 3.0
        if depth_map is not None:
            wall_dist = _sample_wall_depth(
                depth_map, c_left.x_pixel, c_right.x_pixel, pano_w, pano_h
            )

        left_3d = [
            wall_dist * np.sin(c_left.azimuth_rad),
            0.0,
            wall_dist * np.cos(c_left.azimuth_rad),
        ]
        right_3d = [
            wall_dist * np.sin(c_right.azimuth_rad),
            0.0,
            wall_dist * np.cos(c_right.azimuth_rad),
        ]

        center_3d = [
            wall_dist * np.sin(center_yaw),
            0.0,
            wall_dist * np.cos(center_yaw),
        ]

        nx = -np.sin(center_yaw)
        nz = -np.cos(center_yaw)
        normal = [float(nx), 0.0, float(nz)]

        phys_width = np.sqrt(
            (left_3d[0] - right_3d[0]) ** 2 + (left_3d[2] - right_3d[2]) ** 2
        )

        avg_ceil_px = (c_left.y_ceiling_pixel + c_right.y_ceiling_pixel) / 2.0
        avg_floor_px = (c_left.y_floor_pixel + c_right.y_floor_pixel) / 2.0
        phys_height, floor_y, ceil_y = _estimate_wall_height_m(
            avg_ceil_px, avg_floor_px, pano_h, wall_dist
        )

        center_3d[1] = (floor_y + ceil_y) / 2.0
        left_3d[1] = floor_y
        right_3d[1] = floor_y

        wall = WallInfo(
            wall_idx=wall_idx,
            yaw_rad=float(center_yaw),
            yaw_deg=float(np.degrees(center_yaw)),
            angular_width_rad=float(width_rad),
            angular_width_deg=float(width_deg),
            fov_deg=final_fov,
            normal=[round(v, 6) for v in normal],
            center_3d=[round(v, 4) for v in center_3d],
            left_corner_3d=[round(v, 4) for v in left_3d],
            right_corner_3d=[round(v, 4) for v in right_3d],
            width_m=round(float(phys_width), 4),
            height_m=round(float(phys_height), 4),
            floor_y=round(float(floor_y), 4),
            ceiling_y=round(float(ceil_y), 4),
            pano_w=pano_w,
            pano_h=pano_h,
            left_corner_azimuth_rad=float(c_left.azimuth_rad),
            right_corner_azimuth_rad=float(c_right.azimuth_rad),
        )
        walls.append(wall)
        wall_idx += 1

    _detect_overlaps(walls)

    return walls

def _sample_wall_depth(
    depth_map: np.ndarray,
    x_left: float,
    x_right: float,
    pano_w: int,
    pano_h: int,
) -> float:

    row_start = int(pano_h * 0.2)
    row_end = int(pano_h * 0.8)

    x_l = int(x_left) % pano_w
    x_r = int(x_right) % pano_w

    if x_l <= x_r:
        strip = depth_map[row_start:row_end, x_l:x_r]
    else:
        strip = np.concatenate([
            depth_map[row_start:row_end, x_l:],
            depth_map[row_start:row_end, :x_r],
        ], axis=1)

    if strip.size == 0:
        return 3.0

    valid = strip[strip > 0].astype(np.float64)
    if valid.size == 0:
        return 3.0

    depth_m = np.median(valid) / 1000.0 if np.median(valid) > 100 else np.median(valid)
    return max(float(depth_m), 0.5)

def _detect_overlaps(walls: List[WallInfo]) -> None:

    n = len(walls)
    if n < 2:
        return

    for i in range(n):
        curr = walls[i]
        next_wall = walls[(i + 1) % n]

        half_fov_curr = np.radians(curr.fov_deg / 2.0)
        half_fov_next = np.radians(next_wall.fov_deg / 2.0)

        right_edge = curr.yaw_rad + half_fov_curr
        left_edge = next_wall.yaw_rad - half_fov_next

        gap = left_edge - right_edge
        if gap > np.pi:
            gap -= 2 * np.pi
        elif gap < -np.pi:
            gap += 2 * np.pi

        if gap < 0:
            curr.overlaps_right = True
            next_wall.overlaps_left = True

def project_wall_views(
    walls: List[WallInfo],
    pano_rgb: np.ndarray,
    pano_depth: Optional[np.ndarray],
    pano_semantic: Optional[np.ndarray],
    output_dir: str,
    out_w: int = DEFAULT_OUT_W,
    out_h: int = DEFAULT_OUT_H,
    pitch: float = DEFAULT_PITCH,
) -> List[WallInfo]:

    os.makedirs(output_dir, exist_ok=True)
    pano_h, pano_w = pano_rgb.shape[:2]

    for wall in walls:
        map_x, map_y = build_perspective_map(
            wall.yaw_rad, pitch, wall.fov_deg, out_w, out_h, pano_w, pano_h
        )

        view_rgb = cv2.remap(
            pano_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
        )

        base = os.path.join(output_dir, f"wall_{wall.wall_idx:02d}")
        wall.rgb_path = f"{base}_rgb.jpg"
        cv2.imwrite(wall.rgb_path, view_rgb)

        if pano_depth is not None:
            view_depth = cv2.remap(
                pano_depth, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP
            )
            wall.depth_path = f"{base}_depth.png"
            cv2.imwrite(wall.depth_path, view_depth)

        if pano_semantic is not None:
            view_sem = cv2.remap(
                pano_semantic, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP
            )
            wall.semantic_path = f"{base}_semantic.png"
            cv2.imwrite(wall.semantic_path, view_sem)

        wall.meta_path = f"{base}_meta.json"
        with open(wall.meta_path, "w") as mf:
            json.dump(asdict(wall), mf, indent=2)

    return walls

def decompose_panorama(
    pano_folder: str,
    output_dir: str,
    out_w: int = DEFAULT_OUT_W,
    out_h: int = DEFAULT_OUT_H,
    min_wall_angle_deg: float = MIN_WALL_ANGLE_DEG,
) -> List[WallInfo]:

    img_path = os.path.join(pano_folder, "full", "rgb_rawlight.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(pano_folder, "full", "rgb_coldlight.png")
    depth_path = os.path.join(pano_folder, "full", "depth.png")
    semantic_path = os.path.join(pano_folder, "full", "semantic.png")
    layout_path = os.path.join(pano_folder, "layout.txt")

    if not os.path.exists(img_path):
        logger.warning(f"No panorama RGB found in {pano_folder}")
        return []
    if not os.path.exists(layout_path):
        logger.warning(f"No layout.txt found in {pano_folder}")
        return []

    pano_rgb = cv2.imread(img_path)
    if pano_rgb is None:
        logger.warning(f"Failed to read panorama: {img_path}")
        return []

    pano_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) if os.path.exists(depth_path) else None
    pano_semantic = cv2.imread(semantic_path) if os.path.exists(semantic_path) else None

    pano_h, pano_w = pano_rgb.shape[:2]

    corners = read_layout_corners(layout_path, pano_w)
    if len(corners) < 3:
        logger.warning(f"Too few corners ({len(corners)}) in {layout_path}")
        return []

    walls = compute_walls(
        corners, pano_w, pano_h,
        depth_map=pano_depth,
        min_angle_deg=min_wall_angle_deg,
    )

    if not walls:
        logger.warning(f"No valid walls after filtering in {pano_folder}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "original_panorama.jpg"), pano_rgb)

    walls = project_wall_views(
        walls, pano_rgb, pano_depth, pano_semantic,
        output_dir, out_w, out_h,
    )

    summary = {
        "pano_folder": pano_folder,
        "pano_w": pano_w,
        "pano_h": pano_h,
        "n_corners": len(corners),
        "n_walls": len(walls),
        "corners": [
            {
                "x_pixel": c.x_pixel,
                "y_ceiling_pixel": c.y_ceiling_pixel,
                "y_floor_pixel": c.y_floor_pixel,
                "azimuth_rad": round(c.azimuth_rad, 6),
            }
            for c in corners
        ],
        "walls": [asdict(w) for w in walls],
    }
    with open(os.path.join(output_dir, "room_decomposition.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Decomposed {pano_folder} → {len(walls)} walls "
        f"(corners: {len(corners)}, pano: {pano_w}x{pano_h})"
    )
    return walls

def main():
    parser = argparse.ArgumentParser(
        description="Decompose Structured3D panoramas into per-wall perspective views"
    )
    parser.add_argument("--dataset", default="Structured3D", help="Dataset root dir")
    parser.add_argument("--output", default="output", help="Output root dir")
    parser.add_argument("--n_scenes", type=int, default=5, help="Max scenes to process")
    parser.add_argument("--out_w", type=int, default=DEFAULT_OUT_W)
    parser.add_argument("--out_h", type=int, default=DEFAULT_OUT_H)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    all_scenes = sorted(glob.glob(os.path.join(args.dataset, "scene_*")))
    test_scenes = all_scenes[: args.n_scenes]

    if not test_scenes:
        print(f"ERROR: No 'scene_' folders found in {args.dataset}")
        return

    panorama_folders = []
    for scene in test_scenes:
        pattern = os.path.join(scene, "2D_rendering", "*", "panorama")
        panorama_folders.extend(glob.glob(pattern))

    print(f"Found {len(panorama_folders)} panoramas in {len(test_scenes)} scenes\n")

    total_views = 0
    for pano_folder in panorama_folders:
        parts = os.path.normpath(pano_folder).split(os.sep)
        scene_id = parts[-4]
        room_id = parts[-2]

        room_output_dir = os.path.join(args.output, scene_id, room_id)
        walls = decompose_panorama(
            pano_folder, room_output_dir, args.out_w, args.out_h
        )

        if walls:
            n_overlap = sum(1 for w in walls if w.overlaps_left or w.overlaps_right)
            print(
                f"  {scene_id} / room {room_id} → {len(walls)} walls "
                f"(overlap flags: {n_overlap})"
            )
            total_views += len(walls)
        else:
            print(f"  SKIP {scene_id} / room {room_id}")

    print(f"\nTotal wall views generated: {total_views}")

if __name__ == "__main__":
    main()
