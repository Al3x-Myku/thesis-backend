#!/usr/bin/env python3
"""Generate placement & size error distribution histograms for thesis (filtered)."""

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

    placement_errs = np.array([float(o["placement_err_m"]) for o in matched])
    size_errs_raw = np.array([float(o["size_err_pct"]) for o in matched])
    # Cap size errors for robust statistics
    size_errs = np.clip(size_errs_raw, 0, SIZE_ERR_CAP_PCT)

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Error Distributions for Matched Objects", fontsize=14, fontweight="bold", y=1.01)

    # Placement error
    ax1.hist(placement_errs, bins=20, color="#8e44ad", edgecolor="white", alpha=0.85)
    ax1.axvline(np.median(placement_errs), color="#e74c3c", linestyle="--", linewidth=1.5,
                label=f"Median = {np.median(placement_errs):.2f} m")
    ax1.axvline(np.mean(placement_errs), color="#2980b9", linestyle="-.", linewidth=1.5,
                label=f"Mean = {np.mean(placement_errs):.2f} m")
    ax1.set_xlabel("Placement Error (meters)", fontsize=11)
    ax1.set_ylabel("Number of Objects", fontsize=11)
    ax1.set_title("Placement Error", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    # Size error (already capped)
    ax2.hist(size_errs, bins=20, color="#f39c12", edgecolor="white", alpha=0.85)
    ax2.axvline(np.median(size_errs), color="#e74c3c", linestyle="--", linewidth=1.5,
                label=f"Median = {np.median(size_errs):.1f}%")
    ax2.axvline(np.mean(size_errs), color="#2980b9", linestyle="-.", linewidth=1.5,
                label=f"Mean = {np.mean(size_errs):.1f}%")
    ax2.set_xlabel("Size Error (%)", fontsize=11)
    ax2.set_ylabel("Number of Objects", fontsize=11)
    ax2.set_title("Size Error", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    n_capped = int(np.sum(size_errs_raw > SIZE_ERR_CAP_PCT))
    if n_capped > 0:
        ax2.text(0.97, 0.95, f"{n_capped} values\ncapped at {SIZE_ERR_CAP_PCT:.0f}%",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(facecolor="#fff3cd", edgecolor="#ffc107", alpha=0.9,
                           boxstyle="round,pad=0.3"))

    stats_text = (
        f"N = {len(matched)} matched objects (≥{MIN_OBJECT_SIZE_M}m, ≤{MAX_DEPTH_M}m)\n"
        f"Placement: median={np.median(placement_errs):.2f}m, "
        f"mean={np.mean(placement_errs):.2f}m\n"
        f"Size: median={np.median(size_errs):.1f}%, "
        f"mean={np.mean(size_errs):.1f}%"
    )
    fig.text(0.5, -0.04, stats_text, ha="center", fontsize=9, style="italic", color="#555")

    plt.tight_layout()
    out = graphs_dir / "error_distributions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
