import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_thesis_plots(results_dir: Path):
    graphs_dir = results_dir / "thesis_graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    scene_csv = results_dir / "structured3d_metrics.csv"
    obj_csv = results_dir / "structured3d_per_object_metrics.csv"

    if not scene_csv.exists() or not obj_csv.exists():
        logger.error("CSV files not found. Please run the benchmark or re-evaluation script first.")
        return

    # Set academic plotting style
    plt.style.use('ggplot')
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")

    # 1. Global Scene-Level Summary (Recall & Errors)
    scene_df = pd.read_csv(scene_csv)
    if not scene_df.empty:
        # Group by base scene
        scene_df['base_scene'] = scene_df['scene_id'].apply(lambda x: "_".join(str(x).split('_')[:2]))
        
        summary_df = scene_df.groupby('base_scene').mean(numeric_only=True).reset_index()
        
        # Sort by recall for better visualization
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
        out_path = graphs_dir / '1_global_scene_summary.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {out_path}")

    # Load per-object data
    obj_df = pd.read_csv(obj_csv)
    if obj_df.empty:
        logger.warning("Per-object CSV is empty.")
        return

    matched_only = obj_df[obj_df['matched'] == 1]
    
    # 2. Histogram of Placement and Size Errors (Matched Objects)
    if not matched_only.empty:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Error Distributions for Successfully Reconstructed Objects', fontsize=18, weight='bold')

        sns.histplot(matched_only['placement_err_m'], bins=20, kde=True, ax=axs[0], color='#9b59b6')
        axs[0].set_title('Placement Error Distribution')
        axs[0].set_xlabel('Placement Error (meters)')
        axs[0].set_ylabel('Number of Objects')

        # Filter out extreme outliers for size error plot to make it readable
        size_err_filtered = matched_only[matched_only['size_err_pct'] < 200]['size_err_pct']
        sns.histplot(size_err_filtered, bins=20, kde=True, ax=axs[1], color='#f1c40f')
        axs[1].set_title('Size Error Distribution')
        axs[1].set_xlabel('Size Error (%)')
        axs[1].set_ylabel('Number of Objects')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = graphs_dir / '2_error_distributions.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {out_path}")

    # 3. Detection Rate vs. Object Size
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Detection Probability by Object Size', fontsize=16, weight='bold')

    # Create size bins (m)
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0']
    obj_df['size_bin'] = pd.cut(obj_df['gt_max_dim_m'], bins=bins, labels=labels, include_lowest=True)

    # Calculate recall per bin
    recall_by_size = obj_df.groupby('size_bin')['matched'].mean().reset_index()

    sns.barplot(data=recall_by_size, x='size_bin', y='matched', ax=ax, palette='viridis', edgecolor='black')
    ax.set_title('Does object scale affect reconstruction success?')
    ax.set_xlabel('Ground Truth Object Max Dimension (meters)')
    ax.set_ylabel('Recall Probability')
    ax.set_ylim(0, 1.0)

    # Annotate with counts
    counts = obj_df.groupby('size_bin').size()
    for i, p in enumerate(ax.patches):
        ax.annotate(f'n={counts.iloc[i]}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = graphs_dir / '3_detection_rate_by_size.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {out_path}")
    
    # 4. Placement Error vs Size (Scatter)
    if not matched_only.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Placement Error vs. Object Scale', fontsize=16, weight='bold')

        sns.regplot(data=matched_only, x='gt_max_dim_m', y='placement_err_m', 
                    ax=ax, scatter_kws={'alpha':0.5, 's': 50}, line_kws={'color':'red'})
        
        ax.set_xlabel('Ground Truth Max Dimension (meters)')
        ax.set_ylabel('Placement Error (meters)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = graphs_dir / '4_placement_error_vs_size.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {out_path}")

    logger.info("All graphs generated successfully!")

if __name__ == "__main__":
    work_dir = Path(os.getcwd())
    results_dir = work_dir / "benchmark_results"
    generate_thesis_plots(results_dir)
