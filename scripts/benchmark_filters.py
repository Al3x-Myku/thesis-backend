"""
Shared filtering logic for all benchmark visualization scripts.

Scientific justification for each filter:
- min_object_size (0.15m): Objects below 15cm are below the spatial resolution
  of monocular depth estimation networks (DepthAnythingV2, MiDaS, etc.) and 
  cannot be reliably reconstructed from a single image.
- max_depth (8.0m): Objects beyond 8m from the camera are in the far-field where
  depth estimation error grows quadratically with distance. Industry benchmarks
  (NYUv2, ScanNet) typically evaluate within 10m.
- exclude_zero_gt: Viewpoints with 0 evaluable GT objects contribute no signal
  and artificially deflate recall averages.
"""

import csv
from pathlib import Path

# ── Thresholds ──────────────────────────────────────────────────────────
MIN_OBJECT_SIZE_M = 0.15   # exclude objects smaller than 15 cm
MAX_DEPTH_M = 8.0          # exclude objects farther than 8 m from camera
SIZE_ERR_CAP_PCT = 200.0   # cap size error at 200% for robust statistics


def load_objects(csv_path):
    """Load per-object metrics CSV as list of dicts."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def load_scenes(csv_path):
    """Load scene-level metrics CSV as list of dicts."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def filter_objects(objects,
                   min_size=MIN_OBJECT_SIZE_M,
                   max_depth=MAX_DEPTH_M):
    """Apply scientific filters to per-object rows.
    
    Returns filtered list (keeps both matched=0 and matched=1).
    """
    filtered = []
    for o in objects:
        size = float(o["gt_max_dim_m"])
        depth = float(o["gt_centroid_cam_z"])

        if size < min_size:
            continue
        if depth > max_depth:
            continue
        filtered.append(o)
    return filtered


def get_matched(objects):
    """Return only matched objects from a list."""
    return [o for o in objects if o["matched"] == "1"]


def recompute_scene_metrics(filtered_objects):
    """Recompute scene-level metrics from filtered per-object data.
    
    Returns a list of dicts, one per scene_id, with:
      scene_id, n_gt, n_matched, recall, median_place_err, median_size_err
    """
    from collections import defaultdict
    import numpy as np

    by_scene = defaultdict(list)
    for o in filtered_objects:
        by_scene[o["scene_id"]].append(o)

    results = []
    for scene_id in sorted(by_scene.keys()):
        rows = by_scene[scene_id]
        n_gt = len(rows)
        matched = [r for r in rows if r["matched"] == "1"]
        n_matched = len(matched)
        recall = n_matched / n_gt if n_gt > 0 else 0.0

        if matched:
            pe = [float(r["placement_err_m"]) for r in matched]
            se = [min(float(r["size_err_pct"]), SIZE_ERR_CAP_PCT) for r in matched]
            med_pe = float(np.median(pe))
            med_se = float(np.median(se))
        else:
            med_pe = 0.0
            med_se = 0.0

        results.append({
            "scene_id": scene_id,
            "n_gt": n_gt,
            "n_matched": n_matched,
            "recall": recall,
            "median_place_err": med_pe,
            "median_size_err": med_se,
        })
    return results


def get_results_dir():
    """Return the benchmark_results directory path."""
    return Path(__file__).resolve().parent.parent / "benchmark_results"


def get_graphs_dir():
    """Return the thesis_graphs directory path, creating if needed."""
    d = get_results_dir() / "thesis_graphs"
    d.mkdir(parents=True, exist_ok=True)
    return d
