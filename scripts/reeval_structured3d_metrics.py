"""
Structured3D Benchmark RE-EVALUATION — CPU-Only, No GPU Required

Re-processes existing GLB scene outputs against Structured3D ground-truth
to generate per-object CSV data without re-running the GPU pipeline.

Outputs:
  1. structured3d_metrics.csv          — scene-level aggregated metrics
  2. structured3d_per_object_metrics.csv — one row per GT visible object

Usage:
    python3 scripts/reeval_structured3d_metrics.py
"""
import os
import json
import logging
import numpy as np
import trimesh
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── Camera Pose Utilities ────────────────────────────────────────────

def parse_camera_pose(pose_path: str):
    with open(pose_path) as f:
        vals = list(map(float, f.read().strip().split()))
    pos = np.array(vals[0:3])
    r0 = np.array(vals[3:6])
    r1 = np.array(vals[6:9])
    fov_x = vals[9]
    fov_y = vals[10] if len(vals) > 10 else fov_x
    r2 = np.cross(r0, r1)
    R_c2w = np.array([r0, r1, r2])
    return pos, R_c2w, fov_x, fov_y


def world_to_camera(points_world_mm, cam_pos_mm, R_c2w):
    p_local = points_world_mm - cam_pos_mm
    R_w2c = R_c2w.T
    p_cam = (R_w2c @ p_local.T).T
    p_cam = p_cam / 1000.0
    return p_cam


def filter_visible_objects(gt_objects, cam_pos, R_c2w, fov_x, fov_y,
                           img_w=1280, img_h=720, min_size_m=0.05):
    f_x = (img_w / 2) / np.tan(fov_x / 2)
    f_y = (img_h / 2) / np.tan(fov_y / 2)
    visible = []
    for obj in gt_objects:
        centroid_mm = np.array(obj['centroid'])
        coeffs_mm = np.array(obj['coeffs'])
        full_extents_m = coeffs_mm * 2 / 1000.0
        max_dim = max(full_extents_m)
        if max_dim > 3.0:
            continue
        if max_dim < min_size_m:
            continue
        p_cam = world_to_camera(centroid_mm.reshape(1, 3), cam_pos, R_c2w)[0]
        if p_cam[2] <= 0.1:
            continue
        u = f_x * p_cam[0] / p_cam[2] + img_w / 2
        v = f_y * p_cam[1] / p_cam[2] + img_h / 2
        margin = 50
        if -margin <= u <= img_w + margin and -margin <= v <= img_h + margin:
            visible.append({
                'id': obj['ID'],
                'centroid_world_mm': centroid_mm,
                'centroid_cam_m': p_cam,
                'extents_m': full_extents_m,
                'max_dim': max_dim,
            })
    return visible


# ─── Metric Computation ──────────────────────────────────────────────

def extract_pred_object_centroids(scene_path: str):
    scene = trimesh.load(scene_path, force="scene")
    T_w2c = np.eye(4)
    if 'photo_perspective' in scene.graph.nodes:
        try:
            T_c2w, _ = scene.graph['photo_perspective']
            T_w2c = np.linalg.inv(T_c2w)
        except Exception:
            pass
    room_names = {"room_floor", "room_back_wall", "room_left_wall", "room_right_wall", "room_ceiling"}
    objects = []
    for node_name in scene.graph.nodes_geometry:
        if any(rn in node_name for rn in room_names):
            continue
        try:
            T_node2w, geom_name = scene.graph[node_name]
            geom = scene.geometry[geom_name]
        except Exception:
            continue
        T_node2cam = T_w2c @ T_node2w
        if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
            verts = np.array(geom.vertices)
            ones = np.ones((verts.shape[0], 1))
            verts_h = np.hstack([verts, ones])
            cam_verts = (T_node2cam @ verts_h.T).T[:, :3]
            b_min = cam_verts.min(axis=0)
            b_max = cam_verts.max(axis=0)
        else:
            local_bounds = geom.bounds if hasattr(geom, 'bounds') else np.array([[0, 0, 0], [0, 0, 0]])
            v_min = (T_node2cam @ np.append(local_bounds[0], 1.0))[:3]
            v_max = (T_node2cam @ np.append(local_bounds[1], 1.0))[:3]
            b_min = np.minimum(v_min, v_max)
            b_max = np.maximum(v_min, v_max)
        extents = b_max - b_min
        centroid = (b_min + b_max) / 2.0
        centroid[0] = -centroid[0]
        centroid[2] = -centroid[2]
        objects.append({
            'node_name': node_name,
            'centroid': centroid,
            'extents': extents,
            'max_dim': max(extents),
        })
    return objects


def compute_per_object_metrics(gt_visible, pred_objects):
    n_gt = len(gt_visible)
    n_pred = len(pred_objects)
    match_details = []
    for i, gt_obj in enumerate(gt_visible):
        match_details.append({
            'gt_idx': i,
            'gt_id': gt_obj['id'],
            'gt_centroid_cam_m': gt_obj['centroid_cam_m'],
            'gt_max_dim_m': gt_obj['max_dim'],
            'gt_extents_m': gt_obj['extents_m'],
            'matched': False,
            'pred_node_name': '',
            'placement_err_m': 0.0,
            'size_err_pct': 0.0,
        })
    if n_gt == 0 or n_pred == 0:
        return [], 0.0, 0.0, [], match_details

    gt_centroids = np.array([o['centroid_cam_m'] for o in gt_visible])
    pred_centroids = np.array([o['centroid'] for o in pred_objects])

    if len(gt_centroids) > 0 and len(pred_centroids) > 0:
        gt_floor = np.min(gt_centroids[:, 1])
        pred_floor = np.min(pred_centroids[:, 1])
        gt_centroids[:, 1] += (pred_floor - gt_floor)

    cost_matrix = cdist(gt_centroids, pred_centroids, metric='euclidean')
    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

    match_threshold = 3.0
    valid_mask = cost_matrix[gt_idx, pred_idx] < match_threshold
    gt_idx = gt_idx[valid_mask]
    pred_idx = pred_idx[valid_mask]

    placement_errors = cost_matrix[gt_idx, pred_idx].tolist()

    size_errors = []
    for gi, pi in zip(gt_idx, pred_idx):
        gt_size = float(np.prod(gt_visible[gi]['extents_m']) ** (1.0 / 3.0))
        pred_size = float(np.prod(pred_objects[pi]['extents']) ** (1.0 / 3.0))
        if gt_size > 0.01:
            se = abs(gt_size - pred_size) / gt_size
        else:
            se = 0.0
        size_errors.append(se)
        match_details[gi]['matched'] = True
        match_details[gi]['pred_node_name'] = pred_objects[pi].get('node_name', f'pred_{pi}')
        match_details[gi]['placement_err_m'] = float(cost_matrix[gi, pi])
        match_details[gi]['size_err_pct'] = se * 100.0

    recall = len(gt_idx) / len(gt_visible) if len(gt_visible) > 0 else 0.0

    rel_dist_error = 0.0
    if len(gt_idx) >= 2:
        matched_gt = gt_centroids[gt_idx]
        matched_pred = pred_centroids[pred_idx]
        gt_pdist = cdist(matched_gt, matched_gt)
        pred_pdist = cdist(matched_pred, matched_pred)
        triu_idx = np.triu_indices(len(gt_idx), k=1)
        gt_dists = gt_pdist[triu_idx]
        pred_dists = pred_pdist[triu_idx]
        nonzero = gt_dists > 0.1
        if nonzero.sum() > 0:
            rel_errors = np.abs(pred_dists[nonzero] - gt_dists[nonzero]) / gt_dists[nonzero]
            rel_dist_error = float(np.mean(rel_errors))

    return placement_errors, rel_dist_error, recall, size_errors, match_details


# ─── Main Re-evaluation ──────────────────────────────────────────────

def find_gt_for_scene(scene_id: str, data_dir: Path):
    """Given a scene_id like 'scene_01058_400641_1', locate the GT files."""
    parts = scene_id.split('_')
    # scene_XXXXX_ROOM_VIEW
    if len(parts) < 4:
        return None, None
    scene_name = f"{parts[0]}_{parts[1]}"
    room_id = parts[2]
    view_id = parts[3]

    scene_dir = data_dir / "Structured3D" / scene_name
    if not scene_dir.exists():
        return None, None

    bbox_path = scene_dir / "bbox_3d.json"
    cam_pose_path = scene_dir / "2D_rendering" / room_id / "perspective" / "full" / view_id / "camera_pose.txt"

    if not bbox_path.exists() or not cam_pose_path.exists():
        return None, None

    return str(cam_pose_path), str(bbox_path)


def main():
    DATA_DIR = Path(os.getcwd()) / "data" / "Structured3D"
    OUTPUT_DIR = Path(os.getcwd()) / "benchmark_results"
    SCENES_DIR = OUTPUT_DIR / "scenes"

    scene_csv = OUTPUT_DIR / "structured3d_metrics.csv"
    obj_csv = OUTPUT_DIR / "structured3d_per_object_metrics.csv"

    # Write headers
    with open(scene_csv, 'w') as f:
        f.write("scene_id,n_gt_visible,n_pred_objects,n_matched,recall,"
                "mean_placement_err,median_placement_err,"
                "mean_size_err_pct,mean_rel_dist_err_m\n")

    with open(obj_csv, 'w') as f:
        f.write("scene_id,gt_object_id,"
                "gt_centroid_cam_x,gt_centroid_cam_y,gt_centroid_cam_z,"
                "gt_max_dim_m,matched,pred_node_name,"
                "placement_err_m,size_err_pct\n")

    # Find all existing scene GLBs
    scene_dirs = sorted([
        d for d in SCENES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("scene_") and len(d.name.split('_')) >= 4
    ])

    logger.info(f"Found {len(scene_dirs)} scene directories to re-evaluate")

    total_gt_objects = 0
    total_matched = 0
    processed = 0

    for scene_result_dir in scene_dirs:
        scene_id = scene_result_dir.name
        glb_path = scene_result_dir / "final" / "scene_positioned.glb"
        if not glb_path.exists():
            logger.warning(f"[{scene_id}] No GLB found, skipping")
            continue

        # Find GT
        cam_pose_path, bbox_path = find_gt_for_scene(scene_id, DATA_DIR)
        if cam_pose_path is None:
            logger.warning(f"[{scene_id}] No GT found, skipping")
            continue

        try:
            cam_pos, R_c2w, fov_x, fov_y = parse_camera_pose(cam_pose_path)
            with open(bbox_path, 'r') as f:
                gt_objects = json.load(f)

            gt_visible = filter_visible_objects(gt_objects, cam_pos, R_c2w, fov_x, fov_y)
            pred_objects = extract_pred_object_centroids(str(glb_path))

            placement_errors, rel_dist_error, recall, size_errors, match_details = \
                compute_per_object_metrics(gt_visible, pred_objects)

            n_gt = len(gt_visible)
            n_pred = len(pred_objects)
            n_matched = len(placement_errors)
            mean_place = float(np.mean(placement_errors)) if placement_errors else 0.0
            median_place = float(np.median(placement_errors)) if placement_errors else 0.0
            mean_size = float(np.mean(size_errors)) if size_errors else 0.0

            total_gt_objects += n_gt
            total_matched += n_matched

            # Write scene-level row
            with open(scene_csv, 'a') as f:
                f.write(f"{scene_id},{n_gt},{n_pred},{n_matched},"
                        f"{recall:.4f},{mean_place:.4f},"
                        f"{median_place:.4f},{mean_size * 100.0:.4f},"
                        f"{rel_dist_error:.4f}\n")

            # Write per-object rows
            with open(obj_csv, 'a') as f:
                for detail in match_details:
                    c = detail['gt_centroid_cam_m']
                    matched_int = 1 if detail['matched'] else 0
                    pred_name = str(detail['pred_node_name']).replace(',', '_')
                    f.write(
                        f"{scene_id},{detail['gt_id']},"
                        f"{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},"
                        f"{detail['gt_max_dim_m']:.4f},{matched_int},{pred_name},"
                        f"{detail['placement_err_m']:.4f},{detail['size_err_pct']:.4f}\n"
                    )

            processed += 1
            logger.info(
                f"[{scene_id}] GT={n_gt} Pred={n_pred} Matched={n_matched} "
                f"Recall={recall:.1%} PlaceErr={mean_place:.2f}m"
            )

        except Exception as e:
            logger.error(f"[{scene_id}] Failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n{'='*60}")
    logger.info(f"RE-EVALUATION COMPLETE")
    logger.info(f"Scenes processed: {processed}")
    logger.info(f"Total GT visible objects: {total_gt_objects}")
    logger.info(f"Total matched: {total_matched}")
    logger.info(f"Overall recall: {total_matched/total_gt_objects:.1%}" if total_gt_objects > 0 else "N/A")
    logger.info(f"Scene-level CSV: {scene_csv}")
    logger.info(f"Per-object CSV:  {obj_csv}")


if __name__ == "__main__":
    main()
