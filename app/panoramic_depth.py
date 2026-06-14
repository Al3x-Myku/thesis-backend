"""Panoramic depth estimation for equirectangular room images.

Primary:  MTPano (SIGGRAPH 2026, arXiv:2602.05330) — multi-task foundation model for
          360° panoramas; jointly predicts depth, surface normals, and semantics.
          Trained label-free via perspective-projection pseudo-labels; handles
          equirectangular distortion natively.
          HuggingFace: Evergreen0929/MTPano

Fallback: DepthPro on the equirectangular image directly (less accurate but present
          in _models_cache already).

The returned depth map is (H, W) float32 in metres, aligned to the input image.
`wall_pipeline.py` uses this instead of the GT depth file (which only exists for
Structured3D benchmark scenes).

2026 alternatives catalogued here for future comparison:
  - PanoVGGT (CVPR 2026, arXiv:2603.17571): takes multiple panoramas → calibrated
    3D + depth + poses in one forward pass; stronger for multi-view captures.
  - HY-World 2.0 (arXiv:2604.14268): full world generation from a panorama; overkill
    for depth estimation but useful if generating novel views is ever needed.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("ThesisPipeline")

_MTPANO_MODEL_ID = os.getenv("MTPANO_MODEL_ID", "Evergreen0929/MTPano")


def estimate_panoramic_depth(pano_path: str) -> Optional[np.ndarray]:
    """Return (H, W) float32 depth in metres for an equirectangular panorama.

    Tries MTPano first; if unavailable falls back to DepthPro on the raw panorama.
    Returns None if both paths fail so the caller can skip depth-based placement.
    """
    try:
        return _mtpano_depth(pano_path)
    except Exception as e:
        logger.warning(f"[panoramic_depth] MTPano unavailable ({e}); trying DepthPro fallback.")

    try:
        return _depthpro_fallback(pano_path)
    except Exception as e2:
        logger.warning(f"[panoramic_depth] DepthPro fallback also failed ({e2}); no panoramic depth.")
        return None


def _mtpano_depth(pano_path: str) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from app import reconstructor_pipeline as rp

    cache = rp._models_cache
    if cache.get("mtpano") is None:
        logger.info(f"Loading MTPano from {_MTPANO_MODEL_ID} ...")
        processor = AutoImageProcessor.from_pretrained(_MTPANO_MODEL_ID, trust_remote_code=True)
        model = AutoModel.from_pretrained(_MTPANO_MODEL_ID, trust_remote_code=True).eval()
        cache["mtpano"] = (model, processor)
        logger.info("MTPano loaded.")

    model, processor = cache["mtpano"]
    device = rp.PIPELINE_DEVICE

    img = Image.open(pano_path).convert("RGB")
    orig_w, orig_h = img.size

    try:
        model.to(device)
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # MTPano returns a dict/ModelOutput; depth is the first key or .predicted_depth
        depth = None
        for attr in ("predicted_depth", "depth", "depth_map"):
            if hasattr(outputs, attr):
                raw = getattr(outputs, attr)
                depth = raw.squeeze().cpu().float().numpy()
                break
        if depth is None:
            vals = list(outputs.values()) if hasattr(outputs, "values") else [outputs[0]]
            depth = vals[0].squeeze().cpu().float().numpy()

    finally:
        model.to("cpu")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Resize to match the panorama
    if depth.shape != (orig_h, orig_w):
        depth_img = Image.fromarray(depth).resize((orig_w, orig_h), Image.BILINEAR)
        depth = np.array(depth_img, dtype=np.float32)

    depth = np.clip(depth, 0.1, 50.0)
    logger.info(
        f"[panoramic_depth] MTPano: shape={depth.shape}, "
        f"range=[{depth.min():.2f}, {depth.max():.2f}]m"
    )
    return depth.astype(np.float32)


def _depthpro_fallback(pano_path: str) -> np.ndarray:
    """DepthPro run on the full equirectangular image as a coarse fallback.

    DepthPro expects perspective images; applying it to an equirectangular panorama
    produces distorted depth near the poles but is usable for the horizontal belt
    where most furniture sits.
    """
    import torch
    from app import reconstructor_pipeline as rp

    model, processor, model_type = rp.get_depth_model()
    device = rp.PIPELINE_DEVICE

    img = Image.open(pano_path).convert("RGB")
    orig_w, orig_h = img.size

    try:
        model.to(device)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        if model_type == "depth_pro":
            depth = outputs.predicted_depth.squeeze().cpu().float().numpy()
        else:
            depth = outputs.predicted_depth.squeeze().cpu().float().numpy()
            # DA-V2 outputs are relative; convert with a rough scale for indoor (≈3–5 m average)
            d_mean = depth.mean()
            if d_mean > 0:
                depth = depth * (3.5 / d_mean)

    finally:
        model.to("cpu")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if depth.shape != (orig_h, orig_w):
        depth_img = Image.fromarray(depth).resize((orig_w, orig_h), Image.BILINEAR)
        depth = np.array(depth_img, dtype=np.float32)

    depth = np.clip(depth, 0.1, 50.0)
    logger.info(
        f"[panoramic_depth] DepthPro fallback: shape={depth.shape}, "
        f"range=[{depth.min():.2f}, {depth.max():.2f}]m"
    )
    return depth.astype(np.float32)
