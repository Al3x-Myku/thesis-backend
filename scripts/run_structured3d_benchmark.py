
import os
import gc
import json
import logging
import traceback
import torch
import numpy as np
import trimesh
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.reconstructor_pipeline import full_reconstruction, full_reconstruction_panoramic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def parse_camera_xyz(camera_xyz_path: str):

    with open(camera_xyz_path) as f:
        vals = list(map(float, f.read().strip().split()))
    return np.array(vals[:3])

def filter_visible_objects_panoramic(gt_objects, cam_pos_mm, min_size_m=0.15, max_dist_m=6.0):

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

        delta_world = (centroid_mm - cam_pos_mm) / 1000.0

        p_room = np.array([delta_world[0], delta_world[2], delta_world[1]])

        dist_horiz = np.sqrt(p_room[0]**2 + p_room[2]**2)
        if dist_horiz > max_dist_m:
            continue

        visible.append({
            'id': obj['ID'],
            'centroid_world_mm': centroid_mm,
            'centroid_cam_m': p_room,
            'extents_m': full_extents_m,
            'max_dim': max_dim,
        })

    return visible

def extract_pred_object_centroids_panoramic(scene_path: str):

    scene = trimesh.load(scene_path, force="scene")

    room_names = {"room_floor", "room_ceiling", "room_wall_"}
    objects = []

    for node_name in scene.graph.nodes_geometry:
        if any(rn in node_name for rn in room_names):
            continue

        try:
            T_node2w, geom_name = scene.graph[node_name]
            geom = scene.geometry[geom_name]
        except Exception:
            continue

        if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
            verts = np.array(geom.vertices)
            ones = np.ones((verts.shape[0], 1))
            verts_h = np.hstack([verts, ones])
            world_verts = (T_node2w @ verts_h.T).T[:, :3]
            b_min = world_verts.min(axis=0)
            b_max = world_verts.max(axis=0)
        else:
            local_bounds = geom.bounds if hasattr(geom, 'bounds') else np.array([[0,0,0],[0,0,0]])
            v_min = (T_node2w @ np.append(local_bounds[0], 1.0))[:3]
            v_max = (T_node2w @ np.append(local_bounds[1], 1.0))[:3]
            b_min = np.minimum(v_min, v_max)
            b_max = np.maximum(v_min, v_max)

        extents = b_max - b_min
        centroid = (b_min + b_max) / 2.0

        objects.append({
            'node_name': node_name,
            'centroid': centroid,
            'extents': extents,
            'max_dim': max(extents),
        })

    return objects

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
        gt_median_y = np.median(gt_centroids[:, 1])
        pred_median_y = np.median(pred_centroids[:, 1])
        gt_centroids[:, 1] += (pred_median_y - gt_median_y)

    cost_matrix = cdist(gt_centroids, pred_centroids, metric='euclidean')

    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

    match_threshold = 2.0
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

class Structured3DBenchmark:
    def __init__(self, data_dir: str, output_dir: str, mode: str = "perspective"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode

        suffix = f"_{mode}" if mode != "perspective" else ""
        self.metrics_log = self.output_dir / f"structured3d_metrics{suffix}.csv"
        self.per_object_log = self.output_dir / f"structured3d_per_object_metrics{suffix}.csv"
        self.graphs_dir = self.output_dir / "thesis_graphs"
        self.examples_dir = self.output_dir / "structured3d_examples"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)

        if self.metrics_log.exists():
            self.metrics_log.unlink()
        if self.per_object_log.exists():
            self.per_object_log.unlink()

        with open(self.metrics_log, 'w') as f:
            f.write("scene_id,mode,n_gt_visible,n_pred_objects,n_matched,recall,"
                    "mean_placement_err,median_placement_err,"
                    "mean_size_err_pct,mean_rel_dist_err_m\n")

        with open(self.per_object_log, 'w') as f:
            f.write("scene_id,mode,gt_object_id,"
                    "gt_centroid_cam_x,gt_centroid_cam_y,gt_centroid_cam_z,"
                    "gt_max_dim_m,matched,pred_node_name,"
                    "placement_err_m,size_err_pct\n")

    def get_test_suites(self):

        if self.mode == "panoramic":
            return self._get_panorama_suites()

        images = []
        if self.data_dir.exists():
            images.extend(list(self.data_dir.rglob("rgb_rawlight.png")))
            if len(images) == 0:
                images.extend(list(self.data_dir.rglob("rgb.png")))
        images = [p for p in images if "panorama" not in str(p)]
        return images

    def _get_panorama_suites(self):

        pano_folders = []
        if self.data_dir.exists():
            for layout_path in self.data_dir.rglob("layout.txt"):
                pano_folder = layout_path.parent
                rgb_path = pano_folder / "full" / "rgb_rawlight.png"
                if not rgb_path.exists():
                    rgb_path = pano_folder / "full" / "rgb_coldlight.png"
                if rgb_path.exists():
                    pano_folders.append(pano_folder)
        logger.info(f"Found {len(pano_folders)} panorama folders for wall-first reconstruction")
        return pano_folders

    def _find_gt_files(self, img_path: Path):

        cam_pose_path = img_path.parent / "camera_pose.txt"

        scene_dir = None
        for p in img_path.parents:
            if p.name.startswith("scene_"):
                scene_dir = p
                break

        if scene_dir is None:
            return None, None, None

        bbox_path = scene_dir / "bbox_3d.json"

        return cam_pose_path, bbox_path, scene_dir

    def _find_gt_files_panoramic(self, pano_folder: Path):

        cam_xyz_path = pano_folder / "camera_xyz.txt"

        scene_dir = None
        for p in pano_folder.parents:
            if p.name.startswith("scene_"):
                scene_dir = p
                break

        if scene_dir is None:
            return None, None, None

        bbox_path = scene_dir / "bbox_3d.json"
        return cam_xyz_path, bbox_path, scene_dir

    def _extract_metrics(self, pred_scene_path: str, gt_visible: list, panoramic: bool = False):

        if panoramic:
            pred_objects = extract_pred_object_centroids_panoramic(pred_scene_path)
        else:
            pred_objects = extract_pred_object_centroids(pred_scene_path)

        placement_errors, rel_dist_error, recall, size_errors, match_details = \
            compute_per_object_metrics(gt_visible, pred_objects)

        n_matched = len(placement_errors)
        n_pred = len(pred_objects)
        mean_place = float(np.mean(placement_errors)) if placement_errors else 0.0
        median_place = float(np.median(placement_errors)) if placement_errors else 0.0
        mean_size = float(np.mean(size_errors)) if size_errors else 0.0

        return {
            'n_gt_visible': len(gt_visible),
            'n_pred_objects': n_pred,
            'n_matched': n_matched,
            'recall': recall,
            'mean_placement_err': mean_place,
            'median_placement_err': median_place,
            'mean_size_err_pct': mean_size * 100.0,
            'mean_rel_dist_err_m': rel_dist_error,
            'match_details': match_details,
        }

    def _plot_metrics(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        logger.info("Generating Academic Thesis Plots for Structured3D...")
        try:
            plt.style.use('ggplot')
            sns.set_context("paper", font_scale=1.5)
            sns.set_style("whitegrid")

            df = pd.read_csv(self.metrics_log)
            if not df.empty:
                df['base_scene'] = df['scene_id'].apply(lambda x: "_".join(str(x).split('_')[:2]))
                summary_df = df.groupby('base_scene').mean(numeric_only=True).reset_index()
                summary_df = summary_df.sort_values(by='recall', ascending=False)

                fig, axs = plt.subplots(1, 3, figsize=(20, 6))
                fig.suptitle('Global Summary by Scene', fontsize=18, weight='bold')

                sns.barplot(data=summary_df, x='base_scene', y='recall', ax=axs[0], color='#3498db', edgecolor='black')
                axs[0].set_title('Average Recall')
                axs[0].set_ylabel('Recall (0-1)')

                sns.barplot(data=summary_df, x='base_scene', y='mean_placement_err', ax=axs[1], color='#e74c3c', edgecolor='black')
                axs[1].set_title('Avg Placement Error (m)')
                axs[1].set_ylabel('Placement Error (m)')

                sns.barplot(data=summary_df, x='base_scene', y='mean_size_err_pct', ax=axs[2], color='#2ecc71', edgecolor='black')
                axs[2].set_title('Avg Size Error (%)')
                axs[2].set_ylabel('Size Error (%)')

                for ax in axs:
                    ax.set_xlabel('')
                    ax.tick_params(axis='x', rotation=90)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                out_path = self.graphs_dir / '1_global_scene_summary.png'
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved {out_path}")

            if self.per_object_log.exists():
                obj_df = pd.read_csv(self.per_object_log)
                if not obj_df.empty:
                    matched_only = obj_df[obj_df['matched'] == 1]

                    if not matched_only.empty:
                        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
                        fig.suptitle('Error Distributions for Successfully Reconstructed Objects', fontsize=18, weight='bold')

                        sns.histplot(matched_only['placement_err_m'], bins=20, kde=True, ax=axs[0], color='#9b59b6')
                        axs[0].set_title('Placement Error Distribution')
                        axs[0].set_xlabel('Placement Error (meters)')
                        axs[0].set_ylabel('Number of Objects')

                        size_err_filtered = matched_only[matched_only['size_err_pct'] < 200]['size_err_pct']
                        sns.histplot(size_err_filtered, bins=20, kde=True, ax=axs[1], color='#f1c40f')
                        axs[1].set_title('Size Error Distribution')
                        axs[1].set_xlabel('Size Error (%)')
                        axs[1].set_ylabel('Number of Objects')

                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        out_path = self.graphs_dir / '2_error_distributions.png'
                        fig.savefig(out_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Saved {out_path}")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.suptitle('Detection Probability by Object Size', fontsize=16, weight='bold')

                    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
                    labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0']
                    obj_df['size_bin'] = pd.cut(obj_df['gt_max_dim_m'], bins=bins, labels=labels, include_lowest=True)

                    recall_by_size = obj_df.groupby('size_bin', observed=False)['matched'].mean().reset_index()

                    sns.barplot(data=recall_by_size, x='size_bin', y='matched', hue='size_bin', ax=ax, palette='viridis', edgecolor='black', legend=False)
                    ax.set_title('Does object scale affect reconstruction success?')
                    ax.set_xlabel('Ground Truth Object Max Dimension (meters)')
                    ax.set_ylabel('Recall Probability')
                    ax.set_ylim(0, 1.0)

                    counts = obj_df.groupby('size_bin', observed=False).size()
                    for i, p in enumerate(ax.patches):
                        if pd.notna(p.get_height()) and p.get_height() > 0:
                            ax.annotate(f'n={counts.iloc[i]}',
                                        (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha='center', va='center', xytext=(0, 10),
                                        textcoords='offset points', fontsize=12)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    out_path = self.graphs_dir / '3_detection_rate_by_size.png'
                    fig.savefig(out_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Saved {out_path}")

        except Exception as e:
            logger.error(f"Plotting failed: {e}")
            traceback.print_exc()

    def benchmark_pipeline(self, max_samples=20):
        images = self.get_test_suites()

        if not images:
            logger.error(f"FATAL: No Structured3D images found at {self.data_dir} (mode={self.mode}).")
            return

        logger.info(f"--- THESIS EVALUATION (Structured3D, mode={self.mode}) STARTING [{len(images)} scenes] ---")

        for idx, img_path in enumerate(images[:max_samples]):
            is_panoramic = (self.mode == "panoramic")

            scene_base = "unknown"
            room_id = "000"

            if is_panoramic:
                view_id = "pano"
                for p in img_path.parents:
                    if p.name.startswith("scene_"):
                        scene_base = p.name
                        try:
                            rel = img_path.relative_to(p)
                            if len(rel.parts) >= 2:
                                room_id = rel.parts[1]
                        except:
                            pass
                        break
            else:
                view_id = img_path.parent.name
                for p in img_path.parents:
                    if p.name.startswith("scene_"):
                        scene_base = p.name
                        try:
                            rel = img_path.relative_to(p)
                            if len(rel.parts) >= 2:
                                room_id = rel.parts[1]
                        except:
                            pass
                        break

            scene_id = f"{scene_base}_{room_id}_{view_id}"
            if not scene_id:
                logger.warning(f"Could not extract scene_id from {img_path}")
                continue

            scene_results_dir = self.output_dir / "scenes" / scene_id
            scene_results_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"==> Benchmarking {scene_id} ...")
            try:
                if is_panoramic:
                    cam_xyz_path, bbox_path, scene_dir = self._find_gt_files_panoramic(img_path)

                    if not cam_xyz_path or not cam_xyz_path.exists():
                        logger.warning(f"No camera_xyz.txt found for {scene_id}")
                        continue
                    if not bbox_path or not bbox_path.exists():
                        logger.warning(f"No bbox_3d.json found for {scene_id}")
                        continue

                    cam_pos_mm = parse_camera_xyz(str(cam_xyz_path))

                    with open(bbox_path, 'r') as f:
                        gt_objects = json.load(f)

                    gt_visible = filter_visible_objects_panoramic(gt_objects, cam_pos_mm)
                    logger.info(f"[{scene_id}] GT: {len(gt_objects)} total → {len(gt_visible)} in room (panoramic)")
                else:
                    cam_pose_path, bbox_path, scene_dir = self._find_gt_files(img_path)

                    if not cam_pose_path or not cam_pose_path.exists():
                        logger.warning(f"No camera_pose.txt found for {scene_id}")
                        continue
                    if not bbox_path or not bbox_path.exists():
                        logger.warning(f"No bbox_3d.json found for {scene_id}")
                        continue

                    cam_pos, R_c2w, fov_x, fov_y = parse_camera_pose(str(cam_pose_path))

                    with open(bbox_path, 'r') as f:
                        gt_objects = json.load(f)

                    gt_visible = filter_visible_objects(gt_objects, cam_pos, R_c2w, fov_x, fov_y)
                    logger.info(f"[{scene_id}] GT: {len(gt_objects)} total → {len(gt_visible)} visible from camera")

                for i, obj in enumerate(gt_visible[:10]):
                    c = obj['centroid_cam_m']
                    logger.info(f"  GT obj {obj['id']:>3}: cam=({c[0]:+.2f},{c[1]:+.2f},{c[2]:.2f})m  size={obj['max_dim']:.2f}m")

                try:
                    if is_panoramic:
                        pred_scene_path = full_reconstruction_panoramic(str(img_path), str(scene_results_dir))
                    else:
                        pred_scene_path = full_reconstruction(str(img_path), str(scene_results_dir))
                except Exception as e:
                    logger.error(f"CRITICAL HARDWARE/BACKEND CRASH: {e}")
                    traceback.print_exc()
                    pred_scene_path = str(scene_results_dir / "fallback_mesh.glb")
                    m = trimesh.creation.box()
                    m.export(pred_scene_path)

                metrics = self._extract_metrics(pred_scene_path, gt_visible, panoramic=is_panoramic)

                if is_panoramic:
                    pred_objects = extract_pred_object_centroids_panoramic(pred_scene_path)
                else:
                    pred_objects = extract_pred_object_centroids(pred_scene_path)

                if len(gt_visible) >= 2 and len(pred_objects) >= 2:
                    gt_centroids = np.array([o['centroid_cam_m'] for o in gt_visible])
                    pred_centroids = np.array([o['centroid'] for o in pred_objects])
                    cost_matrix = cdist(gt_centroids, pred_centroids, metric='euclidean')
                    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

                    valid = cost_matrix[gt_idx, pred_idx] < 3.0
                    gt_idx, pred_idx = gt_idx[valid], pred_idx[valid]

                    if len(gt_idx) >= 2:
                        logger.info(f"--- Origin-Independent Relative Distances ({scene_id}) ---")
                        for i in range(len(gt_idx)):
                            for j in range(i + 1, len(gt_idx)):
                                gi, gj = gt_idx[i], gt_idx[j]
                                pi, pj = pred_idx[i], pred_idx[j]

                                gt_dist = np.linalg.norm(gt_centroids[gi] - gt_centroids[gj])
                                pred_dist = np.linalg.norm(pred_centroids[pi] - pred_centroids[pj])

                                name_i = gt_visible[gi]['id']
                                name_j = gt_visible[gj]['id']
                                error = abs(pred_dist - gt_dist)

                                log_msg = f"  Dist {name_i}<->{name_j}: Pred={pred_dist:.2f}m (GT={gt_dist:.2f}m, Err={error:.2f}m)"
                                if error > 1.0:
                                    logger.warning(f"{log_msg} <<< LARGE ERROR (>1m) >>>")
                                else:
                                    logger.info(log_msg)

                logger.info(
                    f"[{scene_id}] Results: "
                    f"recall={metrics['recall']:.2%}, "
                    f"placement_err={metrics['mean_placement_err']:.2f}m, "
                    f"size_err={metrics['mean_size_err_pct']:.2f}%, "
                    f"rel_dist_err={metrics['mean_rel_dist_err_m']:.2f}m"
                )

                with open(self.metrics_log, 'a') as f:
                    m = metrics
                    f.write(f"{scene_id},{self.mode},{m['n_gt_visible']},"
                            f"{m['n_pred_objects']},{m['n_matched']},"
                            f"{m['recall']:.4f},{m['mean_placement_err']:.4f},"
                            f"{m['median_placement_err']:.4f},{m['mean_size_err_pct']:.4f},"
                            f"{m['mean_rel_dist_err_m']:.4f}\n")

                with open(self.per_object_log, 'a') as f:
                    for detail in metrics.get('match_details', []):
                        c = detail['gt_centroid_cam_m']
                        matched_int = 1 if detail['matched'] else 0
                        pred_name = detail['pred_node_name'].replace(',', '_')
                        f.write(
                            f"{scene_id},{self.mode},{detail['gt_id']},"
                            f"{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},"
                            f"{detail['gt_max_dim_m']:.4f},{matched_int},{pred_name},"
                            f"{detail['placement_err_m']:.4f},{detail['size_err_pct']:.4f}\n"
                        )

                if idx < 5:
                    import shutil
                    example_path = self.examples_dir / f"{scene_id}_example_prediction.glb"
                    shutil.copy(pred_scene_path, example_path)
                    logger.info(f"[Exported Example Model for {scene_id}]")

            except Exception as e:
                logger.error(f"Failed benchmark suite for {scene_id}: {e}")
                traceback.print_exc()

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        self._plot_metrics()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Structured3D Benchmark")
    parser.add_argument("--mode", choices=["perspective", "panoramic"], default="perspective",
                        help="Reconstruction mode: 'perspective' (single image) or 'panoramic' (wall-first)")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    DATA_DIR = args.data_dir or os.path.join(os.getcwd(), "data/Structured3D")
    OUTPUT_DIR = args.output_dir or os.path.join(os.getcwd(), "benchmark_results")

    suite = Structured3DBenchmark(DATA_DIR, OUTPUT_DIR, mode=args.mode)
    suite.benchmark_pipeline(max_samples=args.max_samples)

if __name__ == "__main__":
    main()
