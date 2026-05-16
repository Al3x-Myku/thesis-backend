#!/usr/bin/env python3
"""Generate per-scene recall bar chart for thesis (filtered)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from benchmark_filters import (
    load_objects, filter_objects, recompute_scene_metrics,
    get_results_dir, get_graphs_dir, MIN_OBJECT_SIZE_M, MAX_DEPTH_M,
)


def main():
    results_dir = get_results_dir()
    graphs_dir = get_graphs_dir()

    raw = load_objects(results_dir / "structured3d_per_object_metrics.csv")
    filtered = filter_objects(raw)
    scene_metrics = recompute_scene_metrics(filtered)

    # Exclude viewpoints with 0 evaluable GT objects
    scene_metrics = [s for s in scene_metrics if s["n_gt"] > 0]

    # Group by base scene (scene_XXXXX_YYYYY)
    grouped = defaultdict(list)
    for s in scene_metrics:
        base = "_".join(s["scene_id"].split("_")[:3])
        grouped[base].append(s["recall"])

    base_scenes = sorted(grouped.keys())
    mean_recalls = [np.mean(grouped[b]) for b in base_scenes]
    labels = [b.replace("scene_", "") for b in base_scenes]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = plt.cm.RdYlGn(np.array(mean_recalls))
    bars = ax.bar(range(len(labels)), mean_recalls, color=colors,
                  edgecolor="#333", linewidth=0.7, width=0.7)

    overall_mean = np.mean(mean_recalls)
    ax.axhline(overall_mean, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Mean = {overall_mean:.2f}")

    for bar, val in zip(bars, mean_recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Recall", fontsize=12)
    ax.set_xlabel("Scene ID", fontsize=12)
    ax.set_title("Object Reconstruction Recall per Scene", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for i, b in enumerate(base_scenes):
        n = len(grouped[b])
        ax.text(i, -0.06, f"n={n}", ha="center", va="top", fontsize=7, color="#666",
                transform=ax.get_xaxis_transform())

    # Filter note
    fig.text(0.5, -0.02,
             f"Evaluated on objects ≥ {MIN_OBJECT_SIZE_M}m, depth ≤ {MAX_DEPTH_M}m | "
             f"Zero-GT viewpoints excluded",
             ha="center", fontsize=8, style="italic", color="#888")

    plt.tight_layout()
    out = graphs_dir / "recall_overview.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
