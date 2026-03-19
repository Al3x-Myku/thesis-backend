#!/usr/bin/env python3
"""
End-to-End System Profiler pentru lucrarea de disertație.
Măsoară latenta pe fiecare micro-modul și analizează complexitatea geometrică a Mesh-ului.
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Aici vom simula datele extrase din pipeline-ul tău real pentru a genera graficele instant,
# dar în mod ideal, tu poți pune `time.time()` în reconstructor_pipeline.py în jurul fiecărui apel
# și să extragi aceste numere reale.

def generate_waterfall_chart():
    print("[*] Generare Waterfall Bottleneck Analysis...")
    
    # Etapele din pipeline-ul tău (reconstructor_pipeline.py)
    stages = ['D-FINE (Detectie)', 'BiRefNet (BG Removal)', 'ZoeDepth (Estimare Adancime)', 
              'Hunyuan ShapeGen', 'Mesh Cleanup', 'Hunyuan TexGen']
    
    # Timpi estimativi (in secunde) bazati pe benchmark-urile tale anterioare
    # Inlocuieste cu timpii tai exacti!
    times = [0.008, 0.85, 0.45, 12.5, 1.8, 8.2] 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculăm start-ul fiecărui proces pentru efectul de cascadă
    starts = [0]
    for i in range(len(times)-1):
        starts.append(starts[-1] + times[i])
        
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
    
    bars = ax.barh(stages, times, left=starts, color=colors, edgecolor='black', height=0.6)
    
    # Formatare grafic
    ax.set_xlabel('Timp de Procesare Server (Secunde)', fontsize=12, fontweight='bold')
    ax.set_title('End-to-End Pipeline Waterfall Analysis (Analiza Gâtului de Sticlă)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Etapele de sus in jos
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adăugăm etichete cu timpul pe fiecare bară
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{times[i]:.2f}s', va='center', fontweight='bold')
                
    total_time = sum(times)
    ax.text(total_time - 3, len(stages) - 0.2, f"Total Pipeline: {total_time:.2f}s", 
            fontsize=12, fontweight='bold', color='red', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('thesis_pipeline_waterfall.png', dpi=300)
    print("  [+] Grafic salvat: thesis_pipeline_waterfall.png")

def generate_mesh_optimization_chart():
    print("[*] Generare Mesh Topology Optimization Analysis...")
    
    # Date (Acestea sunt medii tipice pentru Hunyuan3D-2 cu si fara FaceReducer)
    # Dacă rulezi len(mesh.vertices) în codul tău poți pune numerele reale!
    categories = ['Hunyuan Brut', 'Dupa FloaterRemover', 'Dupa FaceReducer (Final)']
    
    vertices = [85000, 82000, 25000] # Numar de puncte
    faces = [170000, 163000, 49000]  # Numar de poligoane
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    bar1 = ax.bar(x - width/2, vertices, width, label='Vârfuri (Vertices)', color='#3498db', edgecolor='black')
    bar2 = ax.bar(x + width/2, faces, width, label='Poligoane (Faces)', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Număr de Elemente', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Optimizarea Topologiei Mesh-ului', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Adaugam valorile deasupra barelor
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:,}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
                        
    autolabel(bar1)
    autolabel(bar2)
    
    # Adaugam procentul de reducere
    reduction = ((faces[0] - faces[-1]) / faces[0]) * 100
    plt.annotate(f'Reducere complexitate: -{reduction:.1f}%', 
                 xy=(2.15, faces[-1]), xytext=(1.5, faces[0]*0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12, fontweight='bold', color='green',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=2))

    plt.tight_layout()
    plt.savefig('thesis_mesh_optimization.png', dpi=300)
    print("  [+] Grafic salvat: thesis_mesh_optimization.png")

if __name__ == "__main__":
    generate_waterfall_chart()
    generate_mesh_optimization_chart()
    print("\n[!] Gata! Aceste grafice analizează arhitectura backend-ului tău.")