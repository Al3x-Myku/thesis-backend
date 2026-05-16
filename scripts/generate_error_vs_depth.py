#!/usr/bin/env python3
"""Generate placement error vs. depth as a binned box plot for thesis (filtered)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_filters import (
    load_objects, filter_objects, get_matched,
    get_results_dir, get_graphs_dir, MIN_OBJECT_SIZE_M, MAX_DEPTH_M,
)


def main():
    results_dir = get_results_dir()
    graphs_dir = get_graphs_dir()

    raw = load_objects(results_dir / "structured3d_per_object_metrics.csv")
    filtered = filter_objects(raw)
    matched = get_matched(filtered)

    if not matched:
        print("No matched objects after filtering.")
        return

    depths = np.array([float(o["gt_centroid_cam_z"]) for o in matched])
    place_errs = np.array([float(o["placement_err_m"]) for o in matched])

    # --- Depth bins ---
    bin_edges = [0, 1, 2, 3, 4, 5, 6, MAX_DEPTH_M]
    bin_labels = ["0–1 m", "1–2 m", "2–3 m", "3–4 m", "4–5 m", "5–6 m", f"6–{MAX_DEPTH_M:.0f} m"]

    binned_data = []
    bin_counts = []
    bin_medians = []
    used_labels = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (depths >= lo) & (depths < hi)
        errs = place_errs[mask]
        if len(errs) >= 3:  # need at least 3 points for a meaningful box
            binned_data.append(errs)
            bin_counts.append(len(errs))
            bin_medians.append(np.median(errs))
            used_labels.append(bin_labels[i])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5.5))

    palette = ["#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6", "#1abc9c"]
    bp = ax.boxplot(binned_data, tick_labels=used_labels, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                   markerfacecolor="#999"),
                    medianprops=dict(color="#c0392b", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))

    for patch, color in zip(bp["boxes"], palette[:len(binned_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Overlay median trend line
    x_positions = range(1, len(binned_data) + 1)
    ax.plot(x_positions, bin_medians, color="#c0392b", marker="D", markersize=6,
            linewidth=2, linestyle="--", label="Median trend", zorder=5)

    # Annotate counts and medians
    for i, (n, med) in enumerate(zip(bin_counts, bin_medians)):
        ax.text(i + 1, -0.25, f"n={n}\nmed={med:.2f}m",
                ha="center", va="top", fontsize=8, color="#555")

    # Overall correlation
    corr = np.corrcoef(depths, place_errs)[0, 1]
    ax.text(0.03, 0.97, f"Overall r = {corr:.3f}",
            transform=ax.transAxes, fontsize=10, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="#ccc", alpha=0.9,
                      boxstyle="round,pad=0.3"))

    ax.set_xlabel("Object Depth — Z distance from camera", fontsize=12)
    ax.set_ylabel("Placement Error (meters)", fontsize=12)
    ax.set_title("Placement Error by Object Depth", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.04,
             f"Objects ≥ {MIN_OBJECT_SIZE_M}m, depth ≤ {MAX_DEPTH_M}m | "
             f"N = {len(matched)} matched objects | Bins with < 3 objects excluded",
             ha="center", fontsize=8, style="italic", color="#888")

    plt.tight_layout()
    out = graphs_dir / "error_vs_depth.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
