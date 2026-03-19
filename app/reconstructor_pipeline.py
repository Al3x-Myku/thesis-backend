"""
Production-Level 3D Reconstruction Pipeline.
Master Thesis Backend Core Module.

This module orchestrates the end-to-end generation pipeline:
1. 2D Object Detection (D-FINE)
2. Background Matting (BiRefNet)
3. 3D Shape Generation (Hunyuan3D-2 Mini FP16)
4. Mesh Topology Cleanup (Trimesh)
5. Differentiable Texture Painting (Hunyuan3D-2 TexGen)
6. Spatial Scene Assembly (ZoeDepth)

Includes real-time hardware profiling for thesis metric extraction.
"""

import os
import gc
import csv
import time
import math
import logging
import traceback
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import numpy as np
from PIL import Image
import trimesh
from scipy.ndimage import median_filter

from transformers import (
    DFineForObjectDetection,
    AutoImageProcessor,
    AutoModelForImageSegmentation,
)
from torchvision import transforms
from torch.hub import load_state_dict_from_url

from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FaceReducer,
    FloaterRemover,
    DegenerateFaceRemover,
)
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from .dfine_wrapper import run_dfine_inference

# ==============================================================================
# CONFIGURATION & ENVIRONMENT
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ThesisPipeline")

DFINE_ROOT      = os.getenv("DFINE_ROOT", "./D-FINE")
DFINE_CONFIG    = os.getenv("DFINE_CONFIG", "configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml")
DFINE_CHECKPT   = os.getenv("DFINE_CHECKPT", "checkpoints/dfine_x_obj365.pth")
PIPELINE_DEVICE = os.getenv("PIPELINE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

HUNYUAN_SHAPEDIR           = os.getenv("HUNYUAN_SHAPEDIR", "tencent/Hunyuan3D-2mini")
HUNYUAN_SHAPEDIR_SUBFOLDER = os.getenv("HUNYUAN_SHAPEDIR_SUBFOLDER", "hunyuan3d-dit-v2-mini")
HUNYUAN_SHAPEDIR_VARIANT   = os.getenv("HUNYUAN_SHAPEDIR_VARIANT", "fp16")
HUNYUAN_PAINTDIR           = os.getenv("HUNYUAN_PAINTDIR", "tencent/Hunyuan3D-2")

# Global model cache for Celery worker persistence
_models_cache = {
    "birefnet": None,
    "shape_pipeline": None,
    "paint_pipeline": None,
    "depth_model": None
}

# ==============================================================================
# THESIS PROFILER (REAL DATA EXTRACTOR)
# ==============================================================================
class ThesisProfiler:
    """
    Context manager that silently records real hardware metrics during production.
    Ensures 100% authentic data for the Master's thesis methodology.
    """
    def __init__(self, stage_name: str, scene_id: str):
        self.stage_name = stage_name
        self.scene_id = scene_id
        self.start_time = 0.0
        self.csv_path = "production_metrics.csv"
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(["timestamp", "scene_id", "stage", "latency_sec", "peak_vram_gb"])

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        logger.info(f"[{self.scene_id}] Starting Stage: {self.stage_name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        
        status = "FAILED" if exc_type else "SUCCESS"
        logger.info(f"[{self.scene_id}] Stage {self.stage_name} [{status}] - Latency: {latency:.2f}s | Peak VRAM: {vram_gb:.2f}GB")
        
        if not exc_type:
            try:
                with open(self.csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.scene_id, self.stage_name, round(latency, 3), round(vram_gb, 3)])
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")

# ==============================================================================
# LAZY MODEL LOADERS (VRAM OPTIMIZATION)
# ==============================================================================
def get_birefnet():
    if _models_cache["birefnet"] is None:
        logger.info("Initializing BiRefNet...")
        _models_cache["birefnet"] = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        ).to(PIPELINE_DEVICE).eval()
    return _models_cache["birefnet"]

def get_hunyuan_shape():
    if _models_cache["shape_pipeline"] is None:
        logger.info("Initializing Hunyuan3D-2 Mini Shape Pipeline...")
        kwargs = {}
        if HUNYUAN_SHAPEDIR_SUBFOLDER: kwargs["subfolder"] = HUNYUAN_SHAPEDIR_SUBFOLDER
        if HUNYUAN_SHAPEDIR_VARIANT: kwargs["variant"] = HUNYUAN_SHAPEDIR_VARIANT
        _models_cache["shape_pipeline"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            HUNYUAN_SHAPEDIR, **kwargs
        )
    return _models_cache["shape_pipeline"]

def get_hunyuan_paint():
    if _models_cache["paint_pipeline"] is None:
        logger.info("Initializing Hunyuan3D-2 Texture Paint Pipeline...")
        _models_cache["paint_pipeline"] = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_PAINTDIR)
    return _models_cache["paint_pipeline"]

def get_zoe_depth():
    if _models_cache["depth_model"] is None:
        logger.info("Initializing ZoeDepth...")
        repo = "isl-org/ZoeDepth"
        model = torch.hub.load(repo, "ZoeD_NK", pretrained=False, trust_repo=True).to(PIPELINE_DEVICE).eval()
        url = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"
        sd  = load_state_dict_from_url(url, progress=True, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        for m in model.modules():
            if hasattr(m, "drop_path1") and not hasattr(m, "drop_path"):
                m.drop_path = m.drop_path1
        _models_cache["depth_model"] = model
    return _models_cache["depth_model"]

def cleanup_gpu(aggressive: bool = False):
    """
    Cleans up GPU memory to prevent Celery OOM crashes.
    If aggressive=True, unloads ALL massive generation models from VRAM.
    """
    if aggressive:
        logger.warning("Executing Aggressive VRAM Purge...")
        for key in _models_cache.keys():
            if _models_cache[key] is not None:
                del _models_cache[key]
                _models_cache[key] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def remove_bg_biref(image: Image.Image) -> Image.Image:
    """Removes background utilizing BiRefNet."""
    model = get_birefnet()
    o_seg = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    inp = o_seg(image).unsqueeze(0).to(PIPELINE_DEVICE)
    inp = inp.to(dtype=next(model.parameters()).dtype)
    
    with torch.no_grad():
        mask_logits = model(inp)[-1]
        mask = mask_logits.sigmoid()[0, 0].cpu().float().numpy()
        
    alpha = (mask * 255).astype(np.uint8)
    alpha_im = Image.fromarray(alpha).resize(image.size)
    out = image.convert("RGBA")
    out.putalpha(alpha_im)
    return out

def detect_objects(image_path: str, scene_folder: str, scene_id: str) -> List[str]:
    """Runs D-FINE to isolate bounding box crops."""
    crop_dir = Path(scene_folder) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    with ThesisProfiler("2D_Detection_DFINE", scene_id):
        run_dfine_inference(
            dfine_root=DFINE_ROOT,
            config_path=DFINE_CONFIG,
            checkpoint_path=DFINE_CHECKPT,
            input_image=image_path,
            device=PIPELINE_DEVICE,
            output_dir=str(crop_dir),
        )

    pattern = re.compile(r"crop(\d+)")
    crops = []
    for ext in ("png", "jpg", "jpeg"): 
        for p in crop_dir.rglob(f"input_crop*.{ext}"):
            m = pattern.search(p.name)
            if m: crops.append((int(m.group(1)), str(p)))
            
    if not crops:
        logger.warning(f"[{scene_id}] No objects detected by D-FINE.")
        return []
    return [path for _, path in sorted(crops, key=lambda x: x[0])]

def build_mesh(crop_path: str, scene_folder: str, scene_id: str) -> str:
    """Generates an optimized, textured 3D mesh from a 2D crop."""
    base = Path(crop_path).stem
    out_dir = Path(scene_folder) / "meshes" / base
    out_dir.mkdir(parents=True, exist_ok=True)
    
    image = Image.open(crop_path).convert("RGB")

    # 1. Background Removal
    with ThesisProfiler("BG_Removal_BiRefNet", scene_id):
        image_no_bg = remove_bg_biref(image)
        no_bg_path = Path(crop_path).with_name(f"{base}_no_bg.png")
        image_no_bg.save(no_bg_path)

    # 2. Shape Generation
    with ThesisProfiler("3D_ShapeGen_Hunyuan", scene_id):
        shape_pipe = get_hunyuan_shape()
        mesh = shape_pipe(
            image=image_no_bg,
            num_inference_steps=50,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type="trimesh",
        )[0]
        mesh.export(out_dir / f"{base}_raw.glb")

    # 3. Topology Optimization
    with ThesisProfiler("Mesh_Optimization", scene_id):
        for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
            mesh = cleaner(mesh)

    # 4. Texture Painting
    with ThesisProfiler("3D_TexGen_Hunyuan", scene_id):
        paint_pipe = get_hunyuan_paint()
        painted = paint_pipe(mesh, image=image_no_bg)
        
    out_path = out_dir / f"{base}_textured.glb"
    painted.export(out_path)
    return str(out_path)

# ==============================================================================
# SPATIAL ASSEMBLY & MATH
# ==============================================================================
def robust_depth(crop: np.ndarray) -> float:
    """Calculates median depth robustly, ignoring outliers."""
    if crop.size == 0: return 0.0
    p25, p75 = np.percentile(crop, [25, 75])
    trimmed = crop[(crop >= p25) & (crop <= p75)]
    return float(np.median(trimmed)) if trimmed.size else float(np.median(crop))

def position_meshes(
    mesh_paths: List[str],
    image_path: str,
    scene_folder: str,
    scene_id: str,
    valid_indices: List[int] = None,
    threshold: float = 0.6
) -> str:
    """Assembles individual meshes into a spatially accurate 3D scene using ZoeDepth."""
    with ThesisProfiler("Spatial_Assembly_ZoeDepth", scene_id):
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        cx, cy = W/2, H/2

        # Rerun basic detection to map bounding boxes to meshes
        processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_obj365")
        model_det = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_obj365")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model_det(**inputs)
            
        results = processor.post_process_object_detection(outputs, target_sizes=[(H, W)], threshold=threshold)[0]
        boxes = results["boxes"].cpu().numpy()
        if valid_indices is not None:
            boxes = boxes[valid_indices]

        zoe = get_zoe_depth()
        depth_map = zoe.cpu().infer_pil(image)
        depth_map = median_filter(depth_map, size=5)
        zoe.to(PIPELINE_DEVICE)

        order = np.argsort(boxes[:, 0])
        boxes_sorted = boxes[order]
        meshes_sorted = [mesh_paths[i] for i in order]

        fis = []
        for mp, bb in zip(meshes_sorted, boxes_sorted):
            x0, y0, x1, y1 = bb.astype(int)
            w_px = x1 - x0
            crop = depth_map[y0:y1, x0:x1]
            Z = robust_depth(crop)
            if w_px <= 0 or Z <= 0: continue
            
            mesh_ref = trimesh.load(mp, force="scene")
            w_mesh = float(mesh_ref.extents[0])
            if w_mesh <= 0: continue
            
            f_i = (w_px * Z) / w_mesh
            fis.append(f_i)

        if not fis:
            raise RuntimeError("No valid focal length estimates available")

        fis = np.array(fis)
        q1, q3 = np.percentile(fis, [25, 75])
        iqr = q3 - q1
        fis_filtered = fis[(fis >= q1 - 1.5*iqr) & (fis <= q3 + 1.5*iqr)]
        f = float(np.median(fis_filtered))

        scene = trimesh.Scene()
        for mp, bb, f_i in zip(meshes_sorted, boxes_sorted, fis):
            x0, y0, x1, y1 = bb.astype(int)
            xc, yc = (x0+x1)/2, (y0+y1)/2
            crop = depth_map[y0:y1, x0:x1]
            Z = robust_depth(crop)
            if Z <= 0: continue

            # Reprojection to World Coordinates
            X = (xc - cx) * Z / f
            Y = -(yc - cy) * Z / f

            mesh = trimesh.load(mp, force="scene")
            scale_i = (f_i / f)
            mesh.apply_scale(scale_i)
            mesh.apply_translation([X, Y, Z])
            scene.add_geometry(mesh, node_name=Path(mp).stem)

        out_dir = Path(scene_folder) / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scene_positioned.glb"
        scene.export(str(out_path))
        
    return str(out_path)

def full_reconstruction(image_path: str, scene_folder: str) -> str:
    """
    Main entry point for Celery tasks. 
    Executes the entire perception -> extraction -> generation pipeline.
    """
    scene_id = Path(scene_folder).name
    logger.info(f"[{scene_id}] Starting Full Pipeline Reconstruction...")
    
    try:
        crops = detect_objects(image_path, scene_folder, scene_id)
    except Exception as e:
        logger.error(f"[{scene_id}] Detection failed: {e}")
        traceback.print_exc()
        raise

    if not crops:
        raise RuntimeError("Pipeline aborted: No objects detected in the input image.")

    meshes = []
    valid_crop_indices = []
    
    for idx, crop in enumerate(crops):
        try:
            mesh_path = build_mesh(crop, scene_folder, scene_id)
            meshes.append(mesh_path)
            valid_crop_indices.append(idx)
            
            # Clean VRAM after each heavy generation iteration
            cleanup_gpu(aggressive=False)
            
        except Exception as e:
            logger.error(f"[{scene_id}] Mesh generation failed for crop {idx}: {e}")
            traceback.print_exc()

    if not meshes:
        raise RuntimeError("Pipeline aborted: No meshes could be built from the detected crops.")

    final_scene_path = position_meshes(
        mesh_paths=meshes,
        image_path=image_path,
        scene_folder=scene_folder,
        scene_id=scene_id,
        valid_indices=valid_crop_indices
    )
    
    # Aggressively purge massive generation models from VRAM after scene completion
    cleanup_gpu(aggressive=True)
    logger.info(f"[{scene_id}] Pipeline Complete! Final output saved to: {final_scene_path}")
    
    return final_scene_path