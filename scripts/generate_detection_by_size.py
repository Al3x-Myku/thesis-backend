#!/usr/bin/env python3
"""Generate reconstruction rate by object size category for thesis (filtered)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_filters import (
    load_objects, filter_objects,
    get_results_dir, get_graphs_dir, MIN_OBJECT_SIZE_M, MAX_DEPTH_M,
)


def main():
    results_dir = get_results_dir()
    graphs_dir = get_graphs_dir()

    raw = load_objects(results_dir / "structured3d_per_object_metrics.csv")
    filtered = filter_objects(raw)

    # Bin by size — starting from MIN_OBJECT_SIZE_M
    categories = [
        (f"{MIN_OBJECT_SIZE_M}–0.5 m", MIN_OBJECT_SIZE_M, 0.5),
        ("0.5–1.0 m", 0.5, 1.0),
        ("1.0–1.5 m", 1.0, 1.5),
        ("1.5–2.0 m", 1.5, 2.0),
        ("2.0–3.0 m", 2.0, 3.0),
    ]

    data = []
    for label, lo, hi in categories:
        subset = [o for o in filtered if lo <= float(o["gt_max_dim_m"]) < hi]
        if not subset:
            data.append((label, 0, 0, 0.0))
            continue
        n = len(subset)
        m = sum(1 for o in subset if o["matched"] == "1")
        data.append((label, n, m, m / n))

    labels = [d[0] for d in data]
    totals = [d[1] for d in data]
    matches = [d[2] for d in data]
    recalls = [d[3] for d in data]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5.5))

    palette = ["#e67e22", "#f1c40f", "#2ecc71", "#27ae60", "#1abc9c"]
    bars = ax.bar(range(len(labels)), recalls, color=palette[:len(labels)],
                  edgecolor="#333", linewidth=0.7, width=0.6)

    for i, (bar, n, m, r) in enumerate(zip(bars, totals, matches, recalls)):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{r:.0%}\n({m}/{n})",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Reconstruction Recall", fontsize=12)
    ax.set_xlabel("Ground Truth Object Size (max dimension)", fontsize=12)
    ax.set_title("Reconstruction Rate by Object Size Category", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.02,
             f"Depth ≤ {MAX_DEPTH_M}m | Objects < {MIN_OBJECT_SIZE_M}m excluded "
             f"(below monocular depth resolution)",
             ha="center", fontsize=8, style="italic", color="#888")

    plt.tight_layout()
    out = graphs_dir / "detection_by_size.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
