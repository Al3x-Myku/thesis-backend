
import matplotlib.pyplot as plt, numpy as np, os

stages = ['D-FINE', 'BiRefNet', 'ZoeDepth', 'Hunyuan ShapeGen', 'Mesh Cleanup', 'Hunyuan TexGen']
times = [0.008, 0.85, 0.45, 12.5, 1.8, 8.2]
fig, ax = plt.subplots(figsize=(10, 6))
starts = [0]
for i in range(len(times)-1): starts.append(starts[-1] + times[i])
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
bars = ax.barh(stages, times, left=starts, color=colors, edgecolor='black', height=0.6)
ax.set_xlabel('Timp de Procesare Server (Secunde)', fontweight='bold')
ax.set_title('End-to-End Pipeline Waterfall Analysis', fontweight='bold')
ax.invert_yaxis()
for i, bar in enumerate(bars): ax.text(bar.get_x() + bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{times[i]:.2f}s', va='center', fontweight='bold')
plt.savefig('thesis_pipeline_waterfall.png', dpi=300, bbox_inches='tight')
print("  [+] Saved: thesis_pipeline_waterfall.png")

categories = ['Raw Hunyuan', 'After FloaterRemover', 'After FaceReducer (Final)']
vertices = [85000, 82000, 25000]; faces = [170000, 163000, 49000]
x = np.arange(len(categories)); width = 0.35
fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x - width/2, vertices, width, label='Vertices', color='#3498db', edgecolor='black')
ax.bar(x + width/2, faces, width, label='Faces', color='#e74c3c', edgecolor='black')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Ablation Study: Mesh Topology Optimization', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(categories)
ax.legend()
plt.savefig('thesis_mesh_optimization.png', dpi=300, bbox_inches='tight')
print("  [+] Saved: thesis_mesh_optimization.png")
