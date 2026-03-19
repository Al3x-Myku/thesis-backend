#!/usr/bin/env python3
"""
Thesis Benchmark Part 2: Hunyuan3D-2 End-to-End Generation Pipeline.
Mimics the exact logic from app/reconstructor_pipeline.py to measure real-world performance.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- FIX: INJECT HUNYUAN3D-2 INTO PYTHON PATH ---
hunyuan_path = os.path.join(os.getcwd(), "Hunyuan3D-2")
if hunyuan_path not in sys.path:
    sys.path.insert(0, hunyuan_path)

# Silence warnings from ONNX/TensorRT
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import pynvml
except ImportError:
    print("[-] pynvml/nvidia-ml-py not found. Please run: pip install nvidia-ml-py")
    sys.exit(1)

# Now we can safely import the local hy3dgen package
try:
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    from hy3dgen.shapegen import (
        Hunyuan3DDiTFlowMatchingPipeline,
        FaceReducer,
        FloaterRemover,
        DegenerateFaceRemover,
    )
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
except ImportError as e:
    print(f"[-] Failed to import Hunyuan dependencies: {e}")
    sys.exit(1)

def get_vram_gb():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024**3)

def run_hunyuan_benchmark():
    pynvml.nvmlInit()
    device = torch.device("cuda:0")
    
    print("\n==============================================================")
    print("  PHASE 1: Loading Thesis Backend Models (BiRefNet + Hunyuan3D)")
    print("==============================================================")
    
    print("[INFO] Loading BiRefNet...")
    birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(device)
    birefnet.eval()
    
    print("[INFO] Loading Hunyuan3D-2 Mini ShapeGen (FP16)...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", variant="fp16"
    )
    
    print("[INFO] Loading Hunyuan3D-2 TexGen...")
    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    
    print(f"[INFO] Baseline VRAM with models loaded: {get_vram_gb():.2f} GB")

    test_image_path = os.path.join(hunyuan_path, "assets", "demo.png")
    if not os.path.exists(test_image_path):
        print(f"[-] Cannot find {test_image_path}.")
        sys.exit(1)
        
    raw_image = Image.open(test_image_path).convert("RGB")
    
    o_seg = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Benchmarking different quality steps
    step_counts = [10, 20, 30, 40, 50] 
    shape_times = []
    paint_times = []
    peak_vrams = []

    print("\n==============================================================")
    print("  PHASE 2: Benchmarking Pipeline Across Diffusion Steps")
    print("==============================================================")

    for steps in step_counts:
        print(f"\n[->] Benchmarking Generation at {steps} Diffusion Steps...")
        torch.cuda.reset_peak_memory_stats()
        
        # --- A. BiRefNet BG Removal ---
        inp = o_seg(raw_image).unsqueeze(0).to(device)
        model_dtype = next(birefnet.parameters()).dtype
        inp = inp.to(dtype=model_dtype)
        with torch.no_grad():
            mask = birefnet(inp)[-1].sigmoid()[0, 0].cpu().float().numpy()
        alpha = (mask * 255).astype(np.uint8)
        image_no_bg = raw_image.convert("RGBA")
        image_no_bg.putalpha(Image.fromarray(alpha).resize(raw_image.size))

        # --- B. Shape Generation ---
        t0 = time.time()
        mesh = shape_pipeline(
            image=image_no_bg,
            num_inference_steps=steps, 
            octree_resolution=380,     
            num_chunks=20000,          
            generator=torch.manual_seed(12345),
            output_type="trimesh",
        )[0]
        
        # --- C. Mesh Cleanup ---
        for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
            mesh = cleaner(mesh)
        shape_time = time.time() - t0
        
        # --- D. Texture Generation ---
        t1 = time.time()
        painted = paint_pipeline(mesh, image=image_no_bg)
        paint_time = time.time() - t1
        
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
        shape_times.append(shape_time)
        paint_times.append(paint_time)
        peak_vrams.append(vram_peak)
        
        print(f"     [+] Shape + Cleanup Time : {shape_time:.2f}s")
        print(f"     [+] Texture Paint Time   : {paint_time:.2f}s")
        print(f"     [+] Total Pipeline Time  : {(shape_time + paint_time):.2f}s")
        print(f"     [+] Peak VRAM Usage      : {vram_peak:.2f} GB")

    return step_counts, shape_times, paint_times, peak_vrams

def plot_thesis_graph(steps, shape_times, paint_times, vrams):
    print("\n==============================================================")
    print("  PHASE 3: Rendering 3D Pipeline Thesis Graph")
    print("==============================================================")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(steps, shape_times, width=4, label='Shape Gen + Cleanup', color='#66b3ff', edgecolor='black')
    ax1.bar(steps, paint_times, width=4, bottom=shape_times, label='Texture Painting', color='#ff9999', edgecolor='black')
    
    ax1.set_xlabel('Diffusion Inference Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Generation Time (Seconds)', fontsize=12, fontweight='bold')
    ax1.set_xticks(steps)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color = 'indigo'
    ax2.set_ylabel('Peak VRAM Usage (GB)', color=color, fontsize=12, fontweight='bold')  
    ax2.plot(steps, vrams, marker='D', color=color, linewidth=2.5, markersize=8, linestyle='dashed', label="Peak VRAM")
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax2.axhline(y=16.0, color='red', linestyle=':', linewidth=2, label='RTX 5080 VRAM Limit (16GB)')
    ax2.set_ylim(min(vrams)-2, 17)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title('Hunyuan3D-2 Pipeline Performance (RTX 5080)', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()  
    
    output_path = os.path.join(os.getcwd(), 'thesis_hunyuan_benchmark.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[+] Graph successfully saved to: {output_path}")

if __name__ == "__main__":
    steps, shape_t, paint_t, vrams = run_hunyuan_benchmark()
    plot_thesis_graph(steps, shape_t, paint_t, vrams)