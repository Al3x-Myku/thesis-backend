#!/usr/bin/env python3
"""Generate a summary statistics table as PNG for thesis inclusion (filtered)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_filters import (
    load_objects, filter_objects, get_matched, recompute_scene_metrics,
    get_results_dir, get_graphs_dir,
    MIN_OBJECT_SIZE_M, MAX_DEPTH_M, SIZE_ERR_CAP_PCT,
)


def main():
    results_dir = get_results_dir()
    graphs_dir = get_graphs_dir()

    raw = load_objects(results_dir / "structured3d_per_object_metrics.csv")
    filtered = filter_objects(raw)
    scene_metrics = recompute_scene_metrics(filtered)

    # Group by base scene
    grouped = defaultdict(list)
    for s in scene_metrics:
        base = "_".join(s["scene_id"].split("_")[:3])
        grouped[base].append(s)

    table_rows = []
    for base in sorted(grouped.keys()):
        views = grouped[base]
        # Exclude zero-GT viewpoints
        active_views = [v for v in views if v["n_gt"] > 0]
        n_views = len(active_views)
        if n_views == 0:
            continue

        total_gt = sum(v["n_gt"] for v in active_views)
        total_matched = sum(v["n_matched"] for v in active_views)
        mean_recall = np.mean([v["recall"] for v in active_views])

        # Collect matched objects for this base scene
        matched_objs = [o for o in get_matched(filtered)
                        if "_".join(o["scene_id"].split("_")[:3]) == base]
        if matched_objs:
            pe = [float(o["placement_err_m"]) for o in matched_objs]
            se = [min(float(o["size_err_pct"]), SIZE_ERR_CAP_PCT) for o in matched_objs]
            med_pe = np.median(pe)
            med_se = np.median(se)
        else:
            med_pe = 0.0
            med_se = 0.0

        short_name = base.replace("scene_", "")
        table_rows.append([
            short_name,
            str(n_views),
            str(total_gt),
            str(total_matched),
            f"{mean_recall:.2f}",
            f"{med_pe:.2f}",
            f"{med_se:.1f}",
        ])

    # ── Overall row ──────────────────────────────────────────────────
    active_all = [s for s in scene_metrics if s["n_gt"] > 0]
    all_gt = sum(s["n_gt"] for s in active_all)
    all_matched_count = sum(s["n_matched"] for s in active_all)
    all_recalls = [s["recall"] for s in active_all]
    all_matched_objs = get_matched(filtered)
    all_pe = [float(o["placement_err_m"]) for o in all_matched_objs] if all_matched_objs else [0]
    all_se = [min(float(o["size_err_pct"]), SIZE_ERR_CAP_PCT) for o in all_matched_objs] if all_matched_objs else [0]

    table_rows.append([
        "OVERALL",
        str(len(active_all)),
        str(all_gt),
        str(all_matched_count),
        f"{np.mean(all_recalls):.2f}",
        f"{np.median(all_pe):.2f}",
        f"{np.median(all_se):.1f}",
    ])

    col_labels = [
        "Scene", "Views", "GT\nObjects", "Matched",
        "Recall", "Med. Place\nErr (m)", "Med. Size\nErr (%)",
    ]

    # ── Render table ─────────────────────────────────────────────────
    n_rows = len(table_rows)
    fig_height = 0.5 + 0.35 * n_rows
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    table = ax.table(cellText=table_rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Header style
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Data row styles
    for i in range(1, n_rows + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i == n_rows:  # Overall
                cell.set_facecolor("#ecf0f1")
                cell.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#f8f9fa")

            # Color-code recall
            if j == 4:
                val = float(table_rows[i - 1][j])
                if val >= 0.6:
                    cell.set_text_props(color="#27ae60", fontweight="bold")
                elif val >= 0.3:
                    cell.set_text_props(color="#f39c12")
                else:
                    cell.set_text_props(color="#e74c3c")

    fig.suptitle("Structured3D Benchmark Results Summary", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01,
             f"Evaluated on objects ≥ {MIN_OBJECT_SIZE_M}m, depth ≤ {MAX_DEPTH_M}m | "
             f"Size errors capped at {SIZE_ERR_CAP_PCT:.0f}%",
             ha="center", fontsize=8, style="italic", color="#888")

    out = graphs_dir / "summary_table.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
