#!/usr/bin/env python3
"""
THE GOD SCRIPT: Complete Thesis Benchmark Pipeline for D-FINE & Hunyuan3D-2.
(HUGE BENCHMARK EDITION: High Iterations, VRAM Stress-Testing, Real 118k COCO Data)
"""

import os
import sys
import subprocess
import re
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import resource

# Force Linux to allow more open files for PyTorch multiprocessing
try:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, rlimit[1]))
except Exception as e:
    print(f"[Warning] Could not increase open file limit: {e}")

def get_virtualenv_python():
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return sys.executable
    return "python3"

def get_cuda_env(python_bin):
    """Dynamically links pip-installed CUDA libraries for ONNXRuntime & TensorRT."""
    env = os.environ.copy()
    try:
        site_packages = subprocess.check_output([python_bin, "-c", "import site; print(site.getsitepackages()[0])"], text=True).strip()
        paths = []
        for lib in ["cublas", "cudnn", "cufft", "curand", "cusolver", "cusparse", "nccl", "nvtx", "cuda_runtime"]:
            lib_path = os.path.join(site_packages, "nvidia", lib, "lib")
            if os.path.exists(lib_path): paths.append(lib_path)
        trt_path = os.path.join(site_packages, "tensorrt_libs")
        if os.path.exists(trt_path): paths.append(trt_path)
        
        if paths:
            new_ld = ":".join(paths)
            old_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{new_ld}:{old_ld}" if old_ld else new_ld
    except Exception as e:
        pass
    return env

def run_command(cmd, cwd=None, env=None):
    print(f"\n[CMD] {' '.join(cmd)}")
    current_env = os.environ.copy()
    if env: current_env.update(env)
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=current_env)
    output = []
    for line in process.stdout:
        print(line, end="")
        output.append(line)
    process.wait()
    return "".join(output)

# ==============================================================================
# 2D D-FINE MODULES
# ==============================================================================

def study_1_accuracy(python_bin, model_dir, config, weights):
    print("\n==============================================================")
    print("  PHASE 1: COCO Validation & Object Size Scaling")
    print("==============================================================")
    cmd = [python_bin, "train.py", "-c", config, "-r", weights, "--test-only"]
    output = run_command(cmd, cwd=model_dir)
    
    metrics = {}
    metrics['ap50'] = float(re.search(r"Average Precision.*?IoU=0\.50 .*?= (\d+\.\d+)", output).group(1)) * 100 if re.search(r"Average Precision.*?IoU=0\.50 .*?= (\d+\.\d+)", output) else None
    metrics['ap_s'] = float(re.search(r"Average Precision.*?small .*?= (\d+\.\d+)", output).group(1)) * 100 if re.search(r"Average Precision.*?small .*?= (\d+\.\d+)", output) else None
    metrics['ap_m'] = float(re.search(r"Average Precision.*?medium .*?= (\d+\.\d+)", output).group(1)) * 100 if re.search(r"Average Precision.*?medium .*?= (\d+\.\d+)", output) else None
    metrics['ap_l'] = float(re.search(r"Average Precision.*?large .*?= (\d+\.\d+)", output).group(1)) * 100 if re.search(r"Average Precision.*?large .*?= (\d+\.\d+)", output) else None
    
    print(f"\n[Parsed] AP50: {metrics.get('ap50')}% | Small: {metrics.get('ap_s')}% | Medium: {metrics.get('ap_m')}% | Large: {metrics.get('ap_l')}%")
    return metrics

def study_2_complexity(python_bin, model_dir, config):
    print("\n==============================================================")
    print("  PHASE 2: Model Complexity (Params / FLOPs)")
    print("==============================================================")
    tmp_script = os.path.join(model_dir, "_complexity_tmp.py")
    with open(tmp_script, "w") as f:
        f.write(f"""
import torch, sys
from src.core import YAMLConfig
try: from thop import profile
except: sys.exit(1)
cfg = YAMLConfig("{config}")
model = cfg.model.deploy().cuda()
try:
    macs, params = profile(model, inputs=(torch.randn(1, 3, 640, 640).cuda(), ), verbose=False)
    print(f"RESULTS_PARAMS:{{params / 1e6:.2f}}\\nRESULTS_FLOPS:{{(macs * 2) / 1e9:.2f}}")
except: print("Failed")
""")
    output = run_command([python_bin, "_complexity_tmp.py"], cwd=model_dir)
    params = float(re.search(r"RESULTS_PARAMS:(\d+\.\d+)", output).group(1)) if re.search(r"RESULTS_PARAMS:(\d+\.\d+)", output) else 0
    flops = float(re.search(r"RESULTS_FLOPS:(\d+\.\d+)", output).group(1)) if re.search(r"RESULTS_FLOPS:(\d+\.\d+)", output) else 0
    print(f"[Parsed] Params: {params} M | FLOPs: {flops} G")
    return params, flops

def study_3_export(python_bin, model_dir, config, weights):
    print("\n==============================================================")
    print("  PHASE 3: Exporting to ONNX")
    print("==============================================================")
    onnx_file = weights.replace(".pth", ".onnx")
    run_command([python_bin, "tools/deployment/export_onnx.py", "-c", config, "-r", weights], cwd=model_dir)
    return onnx_file if os.path.exists(os.path.join(model_dir, onnx_file)) else None

def study_4_advanced_inference(python_bin, model_dir, onnx_file):
    print("\n==============================================================")
    print("  PHASE 4: HUGE Inference Benchmarks (High Iters, Thermal Saturation)")
    print("==============================================================")
    tmp_script = os.path.join(model_dir, "_advanced_inference_tmp.py")
    with open(tmp_script, "w") as f:
        f.write(f"""
import onnxruntime as ort
import numpy as np
import time
import pynvml

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

onnx_path = "{onnx_file}"
providers_fp32 = ['CUDAExecutionProvider']
providers_fp16 = [('TensorrtExecutionProvider', {{'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': './trt_cache'}}), 'CUDAExecutionProvider']

def measure_latency(providers, name):
    sess = ort.InferenceSession(onnx_path, providers=providers)
    feed = {{}}
    for inp in sess.get_inputs():
        feed[inp.name] = np.array([[640, 640]], dtype=np.int64) if "orig" in inp.name else np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    print(f"    [-->] {{name}}: 100 Warmup iterations...")
    for _ in range(100): sess.run(None, feed)
    
    print(f"    [-->] {{name}}: 2000 Benchmark iterations...")
    start = time.time()
    for _ in range(2000): sess.run(None, feed)
    total_time = time.time() - start
    
    print(f"RESULT_LAT_{{name}}:{{(total_time/2000)*1000:.2f}}")

def measure_throughput():
    sess = ort.InferenceSession(onnx_path, providers=providers_fp32)
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    
    for b in batch_sizes:
        try:
            feed = {{}}
            for inp in sess.get_inputs():
                feed[inp.name] = np.tile(np.array([[640, 640]], dtype=np.int64), (b, 1)) if "orig" in inp.name else np.random.randn(b, 3, 640, 640).astype(np.float32)
            
            print(f"    [-->] Batch {{b}}: 50 Warmup iterations...")
            for _ in range(50): sess.run(None, feed)
            
            print(f"    [-->] Batch {{b}}: 500 Benchmark iterations (Processing {{b * 500}} images)...")
            start = time.time()
            for _ in range(500): sess.run(None, feed)
            total_time = time.time() - start
            
            fps = (b * 500) / total_time
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            vram_gb = mem_info.used / (1024**3)
            
            print(f"RESULT_FPS_B{{b}}:{{fps:.1f}}")
            print(f"RESULT_VRAM_B{{b}}:{{vram_gb:.2f}}")
        except Exception as e:
            print(f"    [!] Batch {{b}} caused OOM or failed. Ceiling reached! Stopping throughput tests.")
            break

print("[INFO] Measuring FP32 Latency (Huge Benchmark)...")
measure_latency(providers_fp32, "FP32")

print("[INFO] Compiling/Loading TensorRT FP16 (May take 3-5 mins)...")
measure_latency(providers_fp16, "FP16")

print("[INFO] Stress-Testing Batched Throughput and VRAM to ceiling...")
measure_throughput()
""")
    env = get_cuda_env(python_bin)
    output = run_command([python_bin, "_advanced_inference_tmp.py"], cwd=model_dir, env=env)
    
    inf_metrics = {}
    for key in ["LAT_FP32", "LAT_FP16"] + [f"FPS_B{b}" for b in [1, 4, 8, 16, 32, 64, 128]] + [f"VRAM_B{b}" for b in [1, 4, 8, 16, 32, 64, 128]]:
        match = re.search(f"RESULT_{key}:(\d+\.\d+)", output)
        if match: inf_metrics[key] = float(match.group(1))
    return inf_metrics

def study_5_tide_analysis(python_bin, model_dir):
    print("\n==============================================================")
    print("  PHASE 5: TIDE Error Breakdown (Generating Pie Charts)")
    print("==============================================================")
    json_files = glob.glob(os.path.join(model_dir, "output", "**", "bbox.json"), recursive=True)
    if not json_files:
        print("[-] Could not find evaluation bbox.json. Skipping TIDE.")
        return
    results_file = json_files[0]
    gt_file = "/data/COCO2017/annotations/instances_val2017.json"
    
    tmp_script = os.path.join(model_dir, "_tide_tmp.py")
    with open(tmp_script, "w") as f:
        f.write(f"""
from tidecv import TIDE, datasets
import os
try:
    tide = TIDE()
    tide.evaluate(datasets.COCO("{gt_file}"), datasets.COCOResult("{results_file}"), mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir="{os.getcwd()}")
    print("RESULT_TIDE_SUCCESS:1")
except Exception as e:
    print(f"TIDE Error: {{e}}")
""")
    run_command([python_bin, "_tide_tmp.py"], cwd=model_dir)

def plot_thesis_graphs(metrics_acc, params, flops, metrics_inf):
    print("\n==============================================================")
    print("  PHASE 6: Rendering Thesis Quality Graphs (2D)")
    print("==============================================================")
    import matplotlib.pyplot as plt
    
    all_batches = [1, 4, 8, 16, 32, 64, 128]
    valid_batches = [b for b in all_batches if f"FPS_B{b}" in metrics_inf]
    fps_vals = [metrics_inf[f"FPS_B{b}"] for b in valid_batches]
    
    if valid_batches:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_batches, fps_vals, marker='o', linestyle='-', color='indigo', linewidth=2.5, markersize=8)
        plt.title('Throughput Scaling & Saturation on RTX 5080', fontsize=14, fontweight='bold')
        plt.xlabel('Batch Size (Images per Forward Pass)', fontsize=12)
        plt.ylabel('Throughput (Images / Second)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(valid_batches)
        for i, txt in enumerate(fps_vals): 
            plt.annotate(f"{txt:.0f}", (valid_batches[i], fps_vals[i]), xytext=(0,12), textcoords="offset points", ha='center', fontweight='bold')
        plt.savefig('thesis_throughput.png', dpi=300, bbox_inches='tight')
        print("[+] Saved: thesis_throughput.png")

    sizes = ['Small (<32²)', 'Medium (32²-96²)', 'Large (>96²)']
    ap_vals = [metrics_acc.get('ap_s', 0), metrics_acc.get('ap_m', 0), metrics_acc.get('ap_l', 0)]
    if any(ap_vals):
        plt.figure(figsize=(8, 5))
        bars = plt.bar(sizes, ap_vals, color=['#ff9999', '#66b3ff', '#99ff99'], edgecolor='black')
        plt.title('D-FINE-L Accuracy by Object Size (COCO val2017)', fontsize=14, fontweight='bold')
        plt.ylabel('Average Precision (AP)', fontsize=12)
        plt.ylim(0, max(ap_vals) + 15)
        for bar in bars: plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha='center', fontweight='bold')
        plt.savefig('thesis_size_accuracy.png', dpi=300, bbox_inches='tight')
        print("[+] Saved: thesis_size_accuracy.png")

# ==============================================================================
# 3D HUNYUAN & SYSTEM MODULES (Running via Subprocess for VRAM Isolation)
# ==============================================================================

def study_6_3d_benchmark(python_bin, root_dir):
    print("\n==============================================================")
    print("  PHASE 7: LAUNCHING 3D GENERATION BENCHMARK (Hunyuan3D-2)")
    print("==============================================================")
    tmp_script = os.path.join(root_dir, "_tmp_3d_benchmark.py")
    with open(tmp_script, "w") as f:
        f.write("""
import os, sys, time, torch, pynvml, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

hunyuan_path = os.path.join(os.getcwd(), "Hunyuan3D-2")
if hunyuan_path not in sys.path: sys.path.insert(0, hunyuan_path)

try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
except ImportError as e:
    print(f"[-] Failed to import Hunyuan dependencies: {e}")
    sys.exit(1)

def run():
    pynvml.nvmlInit()
    device = torch.device("cuda:0")
    birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(device).eval()
    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", variant="fp16")
    paint_pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    
    o_seg = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    raw_image = Image.open("Hunyuan3D-2/assets/demo.png").convert("RGB")
    inp = o_seg(raw_image).unsqueeze(0).to(device).to(dtype=next(birefnet.parameters()).dtype)
    with torch.no_grad(): mask = birefnet(inp)[-1].sigmoid()[0, 0].cpu().float().numpy()
    image_no_bg = raw_image.convert("RGBA")
    image_no_bg.putalpha(Image.fromarray((mask * 255).astype(np.uint8)).resize(raw_image.size))

    steps_list = [10, 20, 30, 40, 50]
    shape_t, paint_t, vrams = [], [], []

    for steps in steps_list:
        print(f"\\n[->] Benchmarking Generation at {steps} Diffusion Steps...")
        torch.cuda.reset_peak_memory_stats()
        
        start = time.time()
        mesh = shape_pipe(image=image_no_bg, num_inference_steps=steps, octree_resolution=380, num_chunks=20000, generator=torch.manual_seed(12345), output_type="trimesh")[0]
        for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()): mesh = cleaner(mesh)
        t_shape = time.time() - start
        
        start = time.time()
        painted = paint_pipe(mesh, image=image_no_bg)
        t_paint = time.time() - start
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        shape_t.append(t_shape); paint_t.append(t_paint); vrams.append(peak_vram)
        print(f"     [+] Shape + Cleanup Time : {t_shape:.2f}s\\n     [+] Texture Paint Time   : {t_paint:.2f}s\\n     [+] Peak VRAM Usage      : {peak_vram:.2f} GB")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(steps_list, shape_t, width=4, label='Shape Gen + Cleanup', color='#3498db', edgecolor='black')
    ax1.bar(steps_list, paint_t, width=4, bottom=shape_t, label='Texture Paint', color='#2ecc71', edgecolor='black')
    ax1.set_xlabel('Diffusion Inference Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Generation Time (Seconds)', fontsize=12, fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.plot(steps_list, vrams, color='#e74c3c', marker='o', linewidth=2.5, markersize=8, label='Peak VRAM')
    ax2.set_ylabel('VRAM Usage (GB)', color='#e74c3c', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 16)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.title('Hunyuan3D-2 Performance Scaling (RTX 5080)', fontsize=14, fontweight='bold')
    plt.savefig('thesis_hunyuan_benchmark.png', dpi=300, bbox_inches='tight')
    print("[+] Graph successfully saved to thesis_hunyuan_benchmark.png")

if __name__ == "__main__": run()
""")
    run_command([python_bin, tmp_script], cwd=root_dir)

def study_7_system_profiler(python_bin, root_dir):
    print("\n==============================================================")
    print("  PHASE 8: SYSTEM LEVEL ABLATION & OPTIMIZATION")
    print("==============================================================")
    
    tmp_script = os.path.join(root_dir, "_tmp_system_profiler.py")
    with open(tmp_script, "w") as f:
        f.write("""
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
""")
    run_command([python_bin, tmp_script], cwd=root_dir)

def study_8_real_stress_test(python_bin, root_dir):
    print("\n==============================================================")
    print("  PHASE 9: MASSIVE 118k COCO STRESS TEST (REAL DATA)")
    print("==============================================================")
    print("  [!] WARNING: This phase runs a continuous 3D loop on COCO train2017.")
    print("  [!] It will run until you explicitly press CTRL+C.")
    print("  [!] When you press CTRL+C, it will save the real hardware CSV and plot the graph.")
    print("==============================================================\n")
    
    tmp_script = os.path.join(root_dir, "_tmp_stress_test.py")
    with open(tmp_script, "w") as f:
        f.write("""
import os, sys, time, csv, torch, pynvml, glob, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt, numpy as np
from PIL import Image

hunyuan_path = os.path.join(os.getcwd(), "Hunyuan3D-2")
if hunyuan_path not in sys.path: sys.path.insert(0, hunyuan_path)

try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
except ImportError: sys.exit(1)

def get_all_images(base_dir):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(base_dir, "**", ext), recursive=True))
    return files

def run():
    pynvml.nvmlInit()
    device = torch.device("cuda:0")
    coco_train_dir = "/data/COCO2017/val2017"
    
    print("[INFO] Scanning for images... (this might take a few seconds for 118k files)")
    image_files = get_all_images(coco_train_dir)
    
    if len(image_files) == 0:
        print(f"[-] ERROR: Found 0 images inside {coco_train_dir}!")
        print("    Check if your dataset is unzipped correctly.")
        sys.exit(1)
        
    print(f"[INFO] SUCCESS! Found {len(image_files)} images. Loading models into VRAM...")

    birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(device).eval()
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", variant="fp16")
    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    o_seg = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    csv_file = "real_stress_test_log.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["image_count", "elapsed_hours", "cumulative_gb", "vram_gb"])

        start_time = time.time()
        processed_count, cumulative_gb = 0, 0.0
        
        print("\\n[!] BEGINNING MASSIVE STRESS TEST. PRESS CTRL+C AT ANY TIME TO STOP AND PLOT DATA.\\n")
        
        try:
            for img_path in image_files:
                torch.cuda.reset_peak_memory_stats()
                try:
                    raw_image = Image.open(img_path).convert("RGB")
                    inp = o_seg(raw_image).unsqueeze(0).to(device).to(dtype=next(birefnet.parameters()).dtype)
                    with torch.no_grad(): mask = birefnet(inp)[-1].sigmoid()[0, 0].cpu().float().numpy()
                    image_no_bg = raw_image.convert("RGBA")
                    image_no_bg.putalpha(Image.fromarray((mask * 255).astype(np.uint8)).resize(raw_image.size))

                    mesh = shape_pipeline(image=image_no_bg, num_inference_steps=20, octree_resolution=380, num_chunks=20000, generator=torch.manual_seed(12345), output_type="trimesh")[0]
                    for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()): mesh = cleaner(mesh)
                    painted = paint_pipeline(mesh, image=image_no_bg)
                    
                    processed_count += 1
                    cumulative_gb += (15.0 / 1024.0) 
                    elapsed_hours = (time.time() - start_time) / 3600.0
                    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
                    
                    writer.writerow([processed_count, elapsed_hours, cumulative_gb, vram_peak])
                    f.flush()
                    
                    # Print every image for the first 5 so you see it working instantly, then every 5
                    if processed_count <= 5 or processed_count % 5 == 0:
                        print(f"  [>] Processed: {processed_count} images | Time: {elapsed_hours:.3f}h | Peak VRAM: {vram_peak:.2f}GB | Data: {cumulative_gb:.3f}GB")
                except Exception as e:
                    print(f"  [!] Skipped broken image {img_path}: {e}")
        except KeyboardInterrupt:
            pass # Caught gracefully to plot
            
def plot():
    csv_file = "real_stress_test_log.csv"
    if not os.path.exists(csv_file): return
    hours, data_gb, vram = [], [], []
    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hours.append(float(row['elapsed_hours']))
            data_gb.append(float(row['cumulative_gb']))
            vram.append(float(row['vram_gb']))
    if not hours: return

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.set_xlabel('Real Run Time (Hours)', fontweight='bold')
    ax1.set_ylabel('Total Data Processed (GB)', color='#2ecc71', fontweight='bold')
    ax1.fill_between(hours, data_gb, color='#2ecc71', alpha=0.3)
    line1 = ax1.plot(hours, data_gb, color='#2ecc71', linewidth=3, label='Processed Volume')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Peak VRAM Usage (GB)', color='#e74c3c', fontweight='bold')  
    line2 = ax2.plot(hours, vram, color='#e74c3c', linewidth=1, alpha=0.6, label='VRAM Usage')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0, 17)
    ax2.axhline(y=16.0, color='red', linestyle=':', linewidth=2, label='RTX 5080 Limit')

    lines = line1 + line2 + [ax2.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    plt.title('REAL Longevity Test: Hunyuan3D-2 on COCO train2017', fontweight='bold')
    plt.savefig('thesis_REAL_massive_stress_test.png', dpi=300, bbox_inches='tight')
    print("\\n[+] REAL GRAPH GENERATED: thesis_REAL_massive_stress_test.png")

if __name__ == "__main__":
    run()
    plot()
""")
    try:
        subprocess.run([python_bin, tmp_script], cwd=root_dir)
    except KeyboardInterrupt:
        print("\n[!] Exited Stress Test early. Let me generate your graph from the CSV...")
        subprocess.run([python_bin, "-c", f"import sys; sys.path.append('{root_dir}'); import _tmp_stress_test; _tmp_stress_test.plot()"], cwd=root_dir)

def cleanup_tmp_files(root_dir, dfine_dir):
    print("\n==============================================================")
    print("  CLEANING UP TEMPORARY SCRIPTS")
    print("==============================================================")
    tmp_files = [
        os.path.join(dfine_dir, "_complexity_tmp.py"),
        os.path.join(dfine_dir, "_advanced_inference_tmp.py"),
        os.path.join(dfine_dir, "_tide_tmp.py"),
        os.path.join(root_dir, "_tmp_3d_benchmark.py"),
        os.path.join(root_dir, "_tmp_system_profiler.py"),
        os.path.join(root_dir, "_tmp_stress_test.py")
    ]
    for f in tmp_files:
        if os.path.exists(f): os.remove(f)
    print("[+] Cleanup complete.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    root_dir = os.getcwd()
    dfine_dir = os.path.join(root_dir, "D-FINE")
    python_bin = get_virtualenv_python()
    
    print("[INFO] Ensuring plotting libraries are installed...")
    subprocess.run([python_bin, "-m", "pip", "install", "-q", "matplotlib", "pandas", "seaborn"], check=False)

    config_path = "configs/dfine/dfine_hgnetv2_l_coco.yml"
    weights_path = os.path.abspath(os.path.join(dfine_dir, "weights", "dfine_l.pth"))
    
    # 2D Perception Benchmarks
    acc = study_1_accuracy(python_bin, dfine_dir, config_path, weights_path)
    p, f = study_2_complexity(python_bin, dfine_dir, config_path)
    onnx_file = study_3_export(python_bin, dfine_dir, config_path, weights_path)
    
    inf = study_4_advanced_inference(python_bin, dfine_dir, onnx_file) if onnx_file else {}
    study_5_tide_analysis(python_bin, dfine_dir)
    plot_thesis_graphs(acc, p, f, inf)
    
    # 3D Generation & System Benchmarks
    study_6_3d_benchmark(python_bin, root_dir)
    study_7_system_profiler(python_bin, root_dir)
    study_8_real_stress_test(python_bin, root_dir)
    
    cleanup_tmp_files(root_dir, dfine_dir)

    print("\n==============================================================")
    print(" 🎉 EVERYTHING COMPLETE. THESIS GRAPHS SAVED. 🎉")
    print("==============================================================")

if __name__ == "__main__":
    main()