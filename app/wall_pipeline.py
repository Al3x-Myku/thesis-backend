
import json
import logging
import math
import os
import traceback
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

logger = logging.getLogger("ThesisPipeline")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from virtual_photo import decompose_panorama, WallInfo

_pipeline_imported = False

def _ensure_pipeline():

    global _pipeline_imported
    if _pipeline_imported:
        return
    _pipeline_imported = True
    from . import reconstructor_pipeline  # noqa: F401

def _deduplicate_cross_wall(
    wall_detections: Dict[int, List[Dict]],
    walls: List[WallInfo],
    iou_threshold: float = 0.3,
) -> Dict[int, List[Dict]]:

    n_walls = len(walls)
    if n_walls < 2:
        return wall_detections

    total_removed = 0

    for i in range(n_walls):
        j = (i + 1) % n_walls
        wall_i = walls[i]
        wall_j = walls[j]

        if not (wall_i.overlaps_right and wall_j.overlaps_left):
            continue

        dets_i = wall_detections.get(wall_i.wall_idx, [])
        dets_j = wall_detections.get(wall_j.wall_idx, [])

        if not dets_i or not dets_j:
            continue

        remove_from_i = set()
        remove_from_j = set()

        for idx_i, det_i in enumerate(dets_i):
            box_i = det_i["box"]
            xc_i = (box_i[0] + box_i[2]) / 2.0
            img_w = det_i.get("img_w", 1024)

            if xc_i < img_w * 0.7:
                continue

            for idx_j, det_j in enumerate(dets_j):
                box_j = det_j["box"]
                xc_j = (box_j[0] + box_j[2]) / 2.0

                if xc_j > img_w * 0.3:
                    continue

                yc_i = (box_i[1] + box_i[3]) / 2.0
                yc_j = (box_j[1] + box_j[3]) / 2.0
                h_i = box_i[3] - box_i[1]
                h_j = box_j[3] - box_j[1]

                y_overlap = abs(yc_i - yc_j) < max(h_i, h_j) * 0.5
                size_similar = min(h_i, h_j) / max(h_i, h_j) > 0.5 if max(h_i, h_j) > 0 else True

                if y_overlap and size_similar:
                    centrality_i = abs(xc_i - img_w / 2.0) / img_w
                    centrality_j = abs(xc_j - img_w / 2.0) / img_w

                    if centrality_i > centrality_j:
                        remove_from_i.add(idx_i)
                    else:
                        remove_from_j.add(idx_j)

        if remove_from_i:
            wall_detections[wall_i.wall_idx] = [
                d for k, d in enumerate(dets_i) if k not in remove_from_i
            ]
            total_removed += len(remove_from_i)
        if remove_from_j:
            wall_detections[wall_j.wall_idx] = [
                d for k, d in enumerate(dets_j) if k not in remove_from_j
            ]
            total_removed += len(remove_from_j)

    if total_removed > 0:
        logger.info(f"Cross-wall deduplication: removed {total_removed} duplicate detections")

    return wall_detections

def _place_object_on_wall(
    mesh: trimesh.Scene,
    wall: WallInfo,
    box: List[int],
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray]:

    x0, y0, x1, y1 = box
    xc = (x0 + x1) / 2.0
    yc = (y0 + y1) / 2.0

    t_horizontal = xc / img_w

    left = np.array(wall.left_corner_3d)
    right = np.array(wall.right_corner_3d)
    pos_along_wall = left + t_horizontal * (right - left)

    normal = np.array(wall.normal)
    extents = mesh.extents if hasattr(mesh, 'extents') else np.array([0.3, 0.3, 0.3])
    depth_offset = extents[2] / 2.0

    position = np.array([
        pos_along_wall[0] + normal[0] * depth_offset,
        wall.floor_y,
        pos_along_wall[2] + normal[2] * depth_offset,
    ])

    rotation_y = math.atan2(normal[0], normal[2])

    return position, rotation_y

def _build_room_shell(walls: List[WallInfo]) -> trimesh.Scene:

    scene = trimesh.Scene()

    if not walls:
        return scene

    floor_y = min(w.floor_y for w in walls)
    ceil_y = max(w.ceiling_y for w in walls)
    room_height = ceil_y - floor_y

    floor_verts_2d = []
    for w in walls:
        floor_verts_2d.append([w.left_corner_3d[0], w.left_corner_3d[2]])
    floor_verts_2d.append(floor_verts_2d[0])

    for i, wall in enumerate(walls):
        left = wall.left_corner_3d
        right = wall.right_corner_3d

        v0 = [left[0], floor_y, left[2]]
        v1 = [right[0], floor_y, right[2]]
        v2 = [right[0], ceil_y, right[2]]
        v3 = [left[0], ceil_y, left[2]]

        vertices = np.array([v0, v1, v2, v3])
        faces = np.array([[0, 1, 2], [0, 2, 3]])

        wall_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        wall_mesh.visual.vertex_colors = [235, 235, 230, 255]

        scene.add_geometry(
            wall_mesh,
            geom_name=f"room_wall_{i:02d}",
            node_name=f"room_wall_{i:02d}",
        )

    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        floor_poly = ShapelyPolygon(floor_verts_2d[:-1])
        if floor_poly.is_valid:
            floor_3d = []
            for v in floor_verts_2d[:-1]:
                floor_3d.append([v[0], floor_y, v[1]])
            floor_3d = np.array(floor_3d)

            floor_faces = []
            for k in range(1, len(floor_3d) - 1):
                floor_faces.append([0, k, k + 1])
            floor_faces = np.array(floor_faces)

            floor_mesh = trimesh.Trimesh(vertices=floor_3d, faces=floor_faces)
            floor_mesh.visual.vertex_colors = [230, 228, 220, 255]
            scene.add_geometry(floor_mesh, geom_name="room_floor", node_name="room_floor")

            ceil_3d = floor_3d.copy()
            ceil_3d[:, 1] = ceil_y
            ceil_mesh = trimesh.Trimesh(vertices=ceil_3d, faces=floor_faces)
            ceil_mesh.visual.vertex_colors = [245, 245, 240, 255]
            scene.add_geometry(ceil_mesh, geom_name="room_ceiling", node_name="room_ceiling")
    except ImportError:
        all_x = [w.left_corner_3d[0] for w in walls] + [w.right_corner_3d[0] for w in walls]
        all_z = [w.left_corner_3d[2] for w in walls] + [w.right_corner_3d[2] for w in walls]
        rx_min, rx_max = min(all_x), max(all_x)
        rz_min, rz_max = min(all_z), max(all_z)
        rw = rx_max - rx_min
        rd = rz_max - rz_min
        cx = (rx_min + rx_max) / 2.0
        cz = (rz_min + rz_max) / 2.0

        floor_mesh = trimesh.creation.box(extents=[rw, 0.02, rd])
        floor_mesh.apply_translation([cx, floor_y - 0.01, cz])
        floor_mesh.visual.vertex_colors = [230, 228, 220, 255]
        scene.add_geometry(floor_mesh, geom_name="room_floor", node_name="room_floor")

        ceil_mesh = trimesh.creation.box(extents=[rw, 0.02, rd])
        ceil_mesh.apply_translation([cx, ceil_y + 0.01, cz])
        ceil_mesh.visual.vertex_colors = [245, 245, 240, 255]
        scene.add_geometry(ceil_mesh, geom_name="room_ceiling", node_name="room_ceiling")

    return scene

def full_reconstruction_panoramic(
    pano_folder: str,
    scene_folder: str,
    out_w: int = 1024,
    out_h: int = 1024,
) -> str:

    from .reconstructor_pipeline import (
        detect_objects, build_mesh, cleanup_gpu,
        filter_detections, ThesisProfiler,
        get_depth_model, PIPELINE_DEVICE,
    )
    from .scene_geometry import anchor_mesh_to_floor

    scene_id = Path(scene_folder).name
    logger.info(f"[{scene_id}] === WALL-FIRST PANORAMIC RECONSTRUCTION ===")

    wall_output_dir = os.path.join(scene_folder, "wall_views")
    with ThesisProfiler("Wall_Decomposition", scene_id):
        walls = decompose_panorama(pano_folder, wall_output_dir, out_w, out_h)

    if not walls:
        raise RuntimeError(f"Wall decomposition failed — no valid walls in {pano_folder}")

    logger.info(f"[{scene_id}] Decomposed into {len(walls)} wall views")

    wall_meshes: Dict[int, List[Dict]] = {}
    wall_detections: Dict[int, List[Dict]] = {}
    total_objects = 0

    for wall in walls:
        if not wall.rgb_path or not os.path.exists(wall.rgb_path):
            continue

        wall_scene_dir = os.path.join(scene_folder, f"wall_{wall.wall_idx:02d}")
        os.makedirs(wall_scene_dir, exist_ok=True)

        logger.info(
            f"[{scene_id}] Processing wall {wall.wall_idx}: "
            f"yaw={wall.yaw_deg:.1f}°, width={wall.angular_width_deg:.1f}°, "
            f"FOV={wall.fov_deg:.1f}°"
        )

        try:
            crops_and_boxes = detect_objects(wall.rgb_path, wall_scene_dir, f"{scene_id}_w{wall.wall_idx}")

            if not crops_and_boxes:
                logger.info(f"[{scene_id}] Wall {wall.wall_idx}: no detections")
                continue

            crops_and_boxes = filter_detections(crops_and_boxes, f"{scene_id}_w{wall.wall_idx}")

            dets = []
            for crop_path, box, score, label in crops_and_boxes:
                dets.append({
                    "crop_path": crop_path,
                    "box": box,
                    "score": score,
                    "label": label,
                    "img_w": out_w,
                    "img_h": out_h,
                })
            wall_detections[wall.wall_idx] = dets

        except Exception as e:
            logger.error(f"[{scene_id}] Wall {wall.wall_idx} detection failed: {e}")
            traceback.print_exc()

    wall_detections = _deduplicate_cross_wall(wall_detections, walls)

    for wall in walls:
        dets = wall_detections.get(wall.wall_idx, [])
        if not dets:
            continue

        wall_scene_dir = os.path.join(scene_folder, f"wall_{wall.wall_idx:02d}")
        meshes_for_wall = []

        for det_idx, det in enumerate(dets):
            obj_id = f"{scene_id}_w{wall.wall_idx}_o{det_idx}"
            try:
                mesh_path = build_mesh(
                    det["crop_path"], wall_scene_dir, obj_id, obj_idx=det_idx
                )
                meshes_for_wall.append({
                    "mesh_path": mesh_path,
                    "box": det["box"],
                    "score": det["score"],
                    "label": det["label"],
                })
                total_objects += 1
                cleanup_gpu(aggressive=False)
            except Exception as e:
                logger.error(f"[{obj_id}] Mesh generation failed: {e}")
                traceback.print_exc()

        wall_meshes[wall.wall_idx] = meshes_for_wall

    logger.info(f"[{scene_id}] Total objects across all walls: {total_objects}")

    with ThesisProfiler("Wall_Aware_Assembly", scene_id):
        scene = _build_room_shell(walls)

        floor_y = min(w.floor_y for w in walls)

        gt_depth = None
        pano_depth_path = os.path.join(pano_folder, "full", "depth.png")
        if os.path.exists(pano_depth_path):
            try:
                from PIL import Image
                depth_img = np.array(Image.open(pano_depth_path))
                if depth_img.ndim == 3:
                    depth_img = depth_img[:, :, 0]
                gt_depth = depth_img.astype(np.float32)
                if gt_depth.max() > 1000:
                    gt_depth = gt_depth / 1000.0
                elif gt_depth.max() > 100:
                    gt_depth = gt_depth / 100.0
                logger.info(f"[{scene_id}] Loaded GT depth map: {gt_depth.shape}, range [{gt_depth.min():.2f}, {gt_depth.max():.2f}]m")
            except Exception as e:
                logger.warning(f"[{scene_id}] Could not load GT depth: {e}")

        placed_count = 0
        for wall in walls:
            meshes = wall_meshes.get(wall.wall_idx, [])
            for m_info in meshes:
                try:
                    mesh = trimesh.load(m_info["mesh_path"], force="scene")
                    position, rot_y = _place_object_on_wall(
                        mesh, wall, m_info["box"], out_w, out_h
                    )

                    mesh_center = (mesh.bounds[0] + mesh.bounds[1]) / 2 if mesh.bounds is not None else np.zeros(3)
                    mesh.apply_translation(-mesh_center)

                    fov_rad = np.radians(wall.fov_deg)
                    f_px = (out_w / 2.0) / np.tan(fov_rad / 2.0)
                    box = m_info["box"]
                    obj_w_px = box[2] - box[0]
                    obj_h_px = box[3] - box[1]

                    wall_dist = np.sqrt(
                        wall.center_3d[0] ** 2 + wall.center_3d[2] ** 2
                    )
                    obj_dist = wall_dist

                    if gt_depth is not None:
                        xc_view = (box[0] + box[2]) / 2.0
                        yc_view = (box[1] + box[3]) / 2.0
                        t_h = xc_view / out_w
                        obj_az = wall.left_corner_azimuth_rad + t_h * (wall.right_corner_azimuth_rad - wall.left_corner_azimuth_rad) \
                            if hasattr(wall, 'left_corner_azimuth_rad') else None
                        if obj_az is not None:
                            pH = gt_depth.shape[0]
                            pW = gt_depth.shape[1]
                            px = int(((obj_az / (2 * math.pi)) + 0.5) % 1.0 * pW)
                            py = int(pH * 0.5)
                            r = 5
                            region = gt_depth[max(0,py-r):py+r, max(0,px-r):px+r]
                            valid = region[region > 0.1]
                            if len(valid) > 0:
                                obj_dist = float(np.median(valid))

                    target_w = (obj_w_px * obj_dist) / f_px
                    target_h = (obj_h_px * obj_dist) / f_px

                    raw_ext = mesh.extents
                    if raw_ext is not None and min(raw_ext[:2]) > 1e-6:
                        scale = min(target_w / raw_ext[0], target_h / raw_ext[1])
                        mesh.apply_scale(scale)

                    R = trimesh.transformations.rotation_matrix(rot_y, [0, 1, 0])
                    mesh.apply_transform(R)

                    anchor_mesh_to_floor(mesh, floor_y=floor_y)

                    current_center = (mesh.bounds[0] + mesh.bounds[1]) / 2 if mesh.bounds is not None else np.zeros(3)

                    if obj_dist < wall_dist:
                        normal = np.array(wall.normal)
                        depth_pull = wall_dist - obj_dist
                        position[0] += normal[0] * depth_pull
                        position[2] += normal[2] * depth_pull

                    mesh.apply_translation([
                        position[0] - current_center[0],
                        0,
                        position[2] - current_center[2],
                    ])

                    name = f"w{wall.wall_idx}_{Path(m_info['mesh_path']).stem}"
                    scene.add_geometry(mesh, node_name=name)
                    placed_count += 1

                    logger.info(
                        f"[{scene_id}] Placed {name} at "
                        f"({position[0]:.2f}, {floor_y:.2f}, {position[2]:.2f})m "
                        f"on wall {wall.wall_idx} (dist={obj_dist:.2f}m)"
                    )

                except Exception as e:
                    logger.error(f"[{scene_id}] Failed to place object: {e}")
                    traceback.print_exc()

    logger.info(f"[{scene_id}] Placed {placed_count} objects in wall-aware scene")

    out_dir = Path(scene_folder) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_positioned.glb"
    scene.export(str(out_path))

    cleanup_gpu(aggressive=True)
    logger.info(f"[{scene_id}] Wall-first pipeline complete! Output: {out_path}")

    return str(out_path)
