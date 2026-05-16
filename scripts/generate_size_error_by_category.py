#!/usr/bin/env python3
"""Generate box plot of size error by object size category for thesis (filtered)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_filters import (
    load_objects, filter_objects, get_matched,
    get_results_dir, get_graphs_dir,
    MIN_OBJECT_SIZE_M, MAX_DEPTH_M, SIZE_ERR_CAP_PCT,
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

    categories = [
        (f"{MIN_OBJECT_SIZE_M}–0.5 m", MIN_OBJECT_SIZE_M, 0.5),
        ("0.5–1.0 m", 0.5, 1.0),
        ("1.0–2.0 m", 1.0, 2.0),
        ("2.0+ m", 2.0, 100.0),
    ]

    data = []
    labels = []
    counts = []
    medians_list = []

    for label, lo, hi in categories:
        errs = [min(float(o["size_err_pct"]), SIZE_ERR_CAP_PCT)
                for o in matched if lo <= float(o["gt_max_dim_m"]) < hi]
        if errs:
            data.append(errs)
            labels.append(label)
            counts.append(len(errs))
            medians_list.append(np.median(errs))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = ["#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                   markerfacecolor="#999"),
                    medianprops=dict(color="#c0392b", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))

    for patch, color in zip(bp["boxes"], palette[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Annotations
    for i, (med, n) in enumerate(zip(medians_list, counts)):
        ax.text(i + 1, -15, f"n={n}\nmed={med:.0f}%",
                ha="center", va="top", fontsize=8, color="#555")

    ax.set_ylabel("Size Error (%)", fontsize=12)
    ax.set_xlabel("Ground Truth Object Size (max dimension)", fontsize=12)
    ax.set_title("Size Error Distribution by Object Scale", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.02,
             f"Objects ≥ {MIN_OBJECT_SIZE_M}m, depth ≤ {MAX_DEPTH_M}m | "
             f"Size errors capped at {SIZE_ERR_CAP_PCT:.0f}%",
             ha="center", fontsize=8, style="italic", color="#888")

    plt.tight_layout()
    out = graphs_dir / "size_error_by_category.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
