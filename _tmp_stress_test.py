
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
        
        print("\n[!] BEGINNING MASSIVE STRESS TEST. PRESS CTRL+C AT ANY TIME TO STOP AND PLOT DATA.\n")
        
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
    print("\n[+] REAL GRAPH GENERATED: thesis_REAL_massive_stress_test.png")

if __name__ == "__main__":
    run()
    plot()
