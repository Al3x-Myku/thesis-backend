
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
        print(f"\n[->] Benchmarking Generation at {steps} Diffusion Steps...")
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
        print(f"     [+] Shape + Cleanup Time : {t_shape:.2f}s\n     [+] Texture Paint Time   : {t_paint:.2f}s\n     [+] Peak VRAM Usage      : {peak_vram:.2f} GB")

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
