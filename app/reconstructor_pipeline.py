
import os
import gc
import csv
import json
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

from .scene_geometry import (
    estimate_intrinsics_vp,
    extract_floor_plane_ransac,
    sample_bottom_mask_depth,
    anchor_mesh_to_floor,
    manhattan_align,
    estimate_ceiling_y,
    scale_scene_metric,
    resolve_overlaps,
)

from transformers import (
    DFineForObjectDetection,
    AutoImageProcessor,
    AutoModelForImageSegmentation,
    AutoModelForDepthEstimation,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ThesisPipeline")

DFINE_ROOT      = os.getenv("DFINE_ROOT", "./D-FINE")
DFINE_CONFIG    = os.getenv("DFINE_CONFIG", "configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml")
DFINE_CHECKPT   = os.getenv("DFINE_CHECKPT", "weights/dfine_x_obj365.pth")
PIPELINE_DEVICE = os.getenv("PIPELINE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

HUNYUAN_SHAPEDIR           = os.getenv("HUNYUAN_SHAPEDIR", "tencent/Hunyuan3D-2")
HUNYUAN_SHAPEDIR_SUBFOLDER = os.getenv("HUNYUAN_SHAPEDIR_SUBFOLDER", "hunyuan3d-dit-v2-0")
HUNYUAN_SHAPEDIR_VARIANT   = os.getenv("HUNYUAN_SHAPEDIR_VARIANT", "")
HUNYUAN_PAINTDIR           = os.getenv("HUNYUAN_PAINTDIR", "tencent/Hunyuan3D-2")

_models_cache = {
    "birefnet": None,
    "shape_pipeline": None,
    "paint_pipeline": None,
    "depth_model": None
}

class ThesisProfiler:

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

def get_birefnet():
    if _models_cache["birefnet"] is None:
        logger.info("Initializing BiRefNet in FP32...")
        _models_cache["birefnet"] = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True, torch_dtype=torch.float32
        ).to(dtype=torch.float32).eval()
    return _models_cache["birefnet"]

def get_hunyuan_shape():
    if _models_cache["shape_pipeline"] is None:
        logger.info("Initializing Hunyuan3D-2 Mini Shape Pipeline in FP32...")
        kwargs = {"torch_dtype": torch.float32}
        if HUNYUAN_SHAPEDIR_SUBFOLDER: kwargs["subfolder"] = HUNYUAN_SHAPEDIR_SUBFOLDER
        kwargs["variant"] = HUNYUAN_SHAPEDIR_VARIANT if HUNYUAN_SHAPEDIR_VARIANT else None
        _models_cache["shape_pipeline"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            HUNYUAN_SHAPEDIR, **kwargs
        )
    return _models_cache["shape_pipeline"]

def get_hunyuan_paint():
    if _models_cache["paint_pipeline"] is None:
        logger.info("Initializing Hunyuan3D-2 Texture Paint Pipeline in FP32...")
        _models_cache["paint_pipeline"] = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_PAINTDIR)
    return _models_cache["paint_pipeline"]

def get_depth_model():

    if _models_cache["depth_model"] is None:
        try:
            from transformers import DepthProForDepthEstimation, DepthProImageProcessor
            model_id = "apple/DepthPro-hf"
            logger.info(f"Initializing Apple Depth Pro (metric depth + focal length)...")
            processor = DepthProImageProcessor.from_pretrained(model_id)
            model = DepthProForDepthEstimation.from_pretrained(model_id).to(PIPELINE_DEVICE).eval()
            _models_cache["depth_model"] = (model, processor, "depth_pro")
        except Exception as e:
            logger.warning(f"Depth Pro unavailable ({e}), falling back to DA-V2 Metric Indoor...")
            model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id).to(PIPELINE_DEVICE).eval()
            _models_cache["depth_model"] = (model, processor, "dav2")
    return _models_cache["depth_model"]

def cleanup_gpu(aggressive: bool = False):

    if aggressive:
        _models_cache["birefnet"] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def remove_bg_biref(image: Image.Image) -> Image.Image:

    model = get_birefnet()
    try:
        model.to(PIPELINE_DEVICE)
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

    finally:
        if 'inp' in locals(): del inp
        if 'mask_logits' in locals(): del mask_logits
        model.to("cpu")
        torch.cuda.empty_cache()

    alpha = (mask * 255).astype(np.uint8)
    alpha_im = Image.fromarray(alpha).resize(image.size)
    out = image.convert("RGBA")
    out.putalpha(alpha_im)

    return out

MIN_DETECTION_CONFIDENCE = 0.30

INDOOR_FURNITURE_CLASSES = {
    2,
    6,
    9,
    12,
    13,
    14,
    16,
    24,
    25,
    28,
    30,
    35,
    37,
    40,
    44,
    47,
    52,
    56,
    58,
    59,
    63,
    64,
    65,
    44,
    47,
    56,
    64,
    67,
    68,
    69,
    77,
    89,
    101,
    106,
    107,
    108,
    111,
    118,
    124,
    128,
    134,
    135,
    136,
    137,
    138,
    140,
    143,
    144,
    145,
    146,
    148,
    149,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
}

MIN_BBOX_SIZE = 0
MAX_BBOX_ASPECT_RATIO = 999.0
MIN_MASK_COVERAGE = 0.10

def filter_detections(
    detections: List[Tuple[str, List[int], float, int]],
    scene_id: str,
) -> List[Tuple[str, List[int], float, int]]:

    if not detections:
        return []

    filtered = []

    for crop_path, box, score, label in detections:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0

        if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
            logger.info(f"[{scene_id}] REJECT size: {w}x{h}px < {MIN_BBOX_SIZE}px — too small for 3D generation")
            continue

        aspect = max(w, h) / max(min(w, h), 1)

        filtered.append((crop_path, box, score, label))

    logger.info(f"[{scene_id}] Pre-generation filter: kept {len(filtered)} detections (UNFILTERED mode)")
    return filtered

def validate_mesh(mesh, scene_id: str, obj_idx: int) -> bool:

    try:
        if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
            total_faces = sum(g.faces.shape[0] for g in mesh.geometry.values() if hasattr(g, 'faces'))
        elif hasattr(mesh, 'faces'):
            total_faces = mesh.faces.shape[0]
        else:
            logger.warning(f"[{scene_id}] obj{obj_idx}: REJECT mesh — no face data")
            return False

        if total_faces < 50:
            logger.warning(f"[{scene_id}] obj{obj_idx}: REJECT mesh — only {total_faces} faces (degenerate)")
            return False

        extents = mesh.extents
        if extents is None or len(extents) < 3:
            logger.warning(f"[{scene_id}] obj{obj_idx}: REJECT mesh — no valid extents")
            return False

        min_ext = min(extents)
        max_ext = max(extents)
        if min_ext > 1e-6 and max_ext / min_ext > 20:
            logger.warning(
                f"[{scene_id}] obj{obj_idx}: REJECT mesh — degenerate proportions "
                f"(extents=[{extents[0]:.4f},{extents[1]:.4f},{extents[2]:.4f}], ratio={max_ext/min_ext:.1f})"
            )
            return False

        bbox_vol = extents[0] * extents[1] * extents[2]
        if bbox_vol < 1e-6:
            logger.warning(f"[{scene_id}] obj{obj_idx}: REJECT mesh — near-zero volume ({bbox_vol:.2e})")
            return False

        logger.info(
            f"[{scene_id}] obj{obj_idx}: mesh OK — {total_faces} faces, "
            f"extents=[{extents[0]:.3f},{extents[1]:.3f},{extents[2]:.3f}]"
        )
        return True
    except Exception as e:
        logger.warning(f"[{scene_id}] obj{obj_idx}: REJECT mesh — validation error: {e}")
        return False

def detect_objects(image_path: str, scene_folder: str, scene_id: str) -> List[Tuple[str, List[int], float, int]]:

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

    pattern = re.compile(r"crop(\d+)\.(?:png|jpg|jpeg)$")
    crops_data = []
    for ext in ("png", "jpg", "jpeg"):
        for p in crop_dir.rglob(f"*crop*.{ext}"):
            m = pattern.search(p.name)
            if m:
                crop_idx = int(m.group(1))
                box_txt = p.with_name(f"{p.stem}_box.txt")
                box = [0, 0, 0, 0]
                score = 1.0
                label = -1
                if box_txt.exists():
                    try:
                        with open(box_txt, "r") as f:
                            parts = f.read().strip().split(",")
                        box = list(map(int, parts[:4]))
                        if len(parts) >= 5:
                            score = float(parts[4])
                        if len(parts) >= 6:
                            label = int(parts[5])
                    except Exception:
                        pass
                crops_data.append((crop_idx, str(p), box, score, label))

    if not crops_data:
        logger.warning(f"[{scene_id}] No objects detected by D-FINE.")
        return []

    crops_data.sort(key=lambda x: x[0])

    before = len(crops_data)
    crops_data = [c for c in crops_data if c[3] >= MIN_DETECTION_CONFIDENCE]
    dropped = before - len(crops_data)
    if dropped > 0:
        logger.info(f"[{scene_id}] Filtered out {dropped} low-confidence detections (threshold={MIN_DETECTION_CONFIDENCE})")

    return [(c[1], c[2], c[3], c[4]) for c in crops_data]

def build_mesh(crop_path: str, scene_folder: str, scene_id: str, obj_idx: int = 0) -> str:

    base = Path(crop_path).stem
    out_dir = Path(scene_folder) / "meshes" / base
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(crop_path).convert("RGB")

    with ThesisProfiler("BG_Removal_BiRefNet", scene_id):
        image_no_bg = remove_bg_biref(image)

        alpha = np.array(image_no_bg.split()[-1])
        fg_pixels = np.count_nonzero(alpha > 127)
        total_pixels = alpha.shape[0] * alpha.shape[1]
        coverage = fg_pixels / total_pixels if total_pixels > 0 else 0
        if coverage < MIN_MASK_COVERAGE:
            raise RuntimeError(
                f"Mask coverage too low ({coverage:.1%} < {MIN_MASK_COVERAGE:.0%}) — "
                f"object likely occluded or segmentation failed"
            )
        logger.info(f"[{scene_id}] obj{obj_idx}: mask coverage={coverage:.1%} (threshold={MIN_MASK_COVERAGE:.0%})")

        w, h = image_no_bg.size
        sz = max(w, h)
        sq_img = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
        sq_img.paste(image_no_bg, ((sz - w) // 2, (sz - h) // 2))

        sq_img.putpixel((0, 0), (0, 0, 0, 1))
        sq_img.putpixel((sz - 1, 0), (0, 0, 0, 1))
        sq_img.putpixel((0, sz - 1), (0, 0, 0, 1))
        sq_img.putpixel((sz - 1, sz - 1), (0, 0, 0, 1))

        image_no_bg = sq_img

        no_bg_path = Path(crop_path).with_name(f"{base}_no_bg.png")
        image_no_bg.save(no_bg_path)

    with ThesisProfiler("3D_ShapeGen_Hunyuan", scene_id):
        shape_pipe = get_hunyuan_shape()
        shape_pipe.to(PIPELINE_DEVICE)
        mesh = shape_pipe(
            image=image_no_bg,
            num_inference_steps=50,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type="trimesh",
        )[0]
        mesh.export(out_dir / f"{base}_raw.glb")
        shape_pipe.to("cpu")
        cleanup_gpu(False)

    with ThesisProfiler("Mesh_Optimization", scene_id):
        for cleaner in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
            mesh = cleaner(mesh)

    if not validate_mesh(mesh, scene_id, obj_idx):
        raise RuntimeError(f"Mesh quality validation failed for obj{obj_idx}")

    with ThesisProfiler("3D_TexGen_Hunyuan", scene_id):
        paint_pipe = get_hunyuan_paint()
        if hasattr(paint_pipe, "to"):
            paint_pipe.to(PIPELINE_DEVICE)

        painted = paint_pipe(mesh, image=image_no_bg)

        if hasattr(paint_pipe, "to"):
            paint_pipe.to("cpu")
        cleanup_gpu(False)

    out_path = out_dir / f"{base}_textured.glb"
    painted.export(out_path)
    return str(out_path)

ROOM_MIN_DIM = 1.5
ROOM_MAX_DIM = 15.0

OBJ365_NAMES: Dict[int, str] = {
    0: 'Person', 1: 'Sneakers', 2: 'Chair', 3: 'Other Shoes', 4: 'Hat',
    5: 'Car', 6: 'Lamp', 7: 'Glasses', 8: 'Bottle', 9: 'Desk',
    10: 'Cup', 11: 'Street Lights', 12: 'Cabinet/shelf', 13: 'Handbag',
    14: 'Bracelet', 15: 'Plate', 16: 'Picture/Frame', 17: 'Helmet',
    18: 'Book', 19: 'Gloves', 20: 'Storage box', 21: 'Boat',
    22: 'Leather Shoes', 23: 'Flower', 24: 'Bench', 25: 'Potted Plant',
    26: 'Bowl/Basin', 27: 'Flag', 28: 'Pillow', 29: 'Boots', 30: 'Vase',
    31: 'Microphone', 32: 'Necklace', 33: 'Ring', 34: 'SUV', 35: 'Wine Glass',
    36: 'Belt', 37: 'Monitor/TV', 38: 'Backpack', 39: 'Umbrella',
    40: 'Traffic Light', 41: 'Speaker', 42: 'Watch', 43: 'Tie',
    44: 'Trash bin', 45: 'Slippers', 46: 'Bicycle', 47: 'Stool',
    48: 'Barrel/bucket', 49: 'Van', 50: 'Couch', 51: 'Sandals',
    52: 'Basket', 53: 'Drum', 54: 'Pen/Pencil', 55: 'Bus',
    56: 'Wild Bird', 57: 'High Heels', 58: 'Motorcycle', 59: 'Guitar',
    60: 'Carpet', 61: 'Cell Phone', 62: 'Bread', 63: 'Camera',
    64: 'Canned', 65: 'Truck', 66: 'Traffic cone', 67: 'Cymbal',
    68: 'Lifesaver', 69: 'Towel', 70: 'Stuffed Toy', 71: 'Candle',
    72: 'Sailboat', 73: 'Laptop', 74: 'Awning', 75: 'Bed',
    76: 'Faucet', 77: 'Tent', 78: 'Horse', 79: 'Mirror',
    80: 'Power outlet', 81: 'Sink', 82: 'Apple', 83: 'Air Conditioner',
    84: 'Knife', 85: 'Hockey Stick', 86: 'Paddle', 87: 'Pickup Truck',
    88: 'Fork', 89: 'Traffic Sign', 90: 'Balloon', 91: 'Tripod',
    92: 'Dog', 93: 'Spoon', 94: 'Clock', 95: 'Pot', 96: 'Cow',
    97: 'Cake', 98: 'Dining Table', 99: 'Sheep', 100: 'Hanger',
    101: 'Blackboard/Whiteboard', 102: 'Napkin', 103: 'Other Fish',
    104: 'Orange', 105: 'Toiletry', 106: 'Keyboard', 107: 'Tomato',
    108: 'Lantern', 109: 'Machinery Vehicle', 110: 'Fan',
    111: 'Green Vegetables', 112: 'Banana', 113: 'Baseball Glove',
    114: 'Airplane', 115: 'Mouse', 116: 'Train', 117: 'Pumpkin',
    118: 'Soccer', 119: 'Skiboard', 120: 'Luggage', 121: 'Nightstand',
    122: 'Tea pot', 123: 'Telephone', 124: 'Trolley', 125: 'Head Phone',
    126: 'Sports Car', 127: 'Stop Sign', 128: 'Dessert', 129: 'Scooter',
    130: 'Stroller', 131: 'Crane', 132: 'Remote', 133: 'Refrigerator',
    134: 'Oven', 135: 'Lemon', 136: 'Duck', 137: 'Baseball Bat',
    138: 'Surveillance Camera', 139: 'Cat', 140: 'Jug', 141: 'Broccoli',
    142: 'Piano', 143: 'Pizza', 144: 'Elephant', 145: 'Skateboard',
    146: 'Surfboard', 147: 'Gun', 148: 'Skating Shoes', 149: 'Gas Stove',
    150: 'Donut', 151: 'Bow Tie', 152: 'Carrot', 153: 'Toilet',
    154: 'Kite', 155: 'Strawberry', 156: 'Other Balls', 157: 'Shovel',
    158: 'Pepper', 159: 'Computer Box', 160: 'Toilet Paper',
    161: 'Cleaning Products', 162: 'Chopsticks', 163: 'Microwave',
    164: 'Pigeon', 165: 'Baseball', 166: 'Cutting Board',
    167: 'Coffee Table', 168: 'Side Table', 169: 'Scissors',
    170: 'Marker', 171: 'Pie', 172: 'Ladder', 173: 'Snowboard',
    174: 'Cookies', 175: 'Radiator', 176: 'Fire Hydrant', 177: 'Basketball',
    178: 'Zebra', 179: 'Grape', 180: 'Giraffe', 181: 'Potato',
    182: 'Sausage', 183: 'Tricycle', 184: 'Violin', 185: 'Egg',
    186: 'Fire Extinguisher', 187: 'Candy', 188: 'Fire Truck',
    189: 'Billiards', 190: 'Converter', 191: 'Bathtub', 192: 'Wheelchair',
    193: 'Golf Club', 194: 'Briefcase', 195: 'Cucumber', 196: 'Cigar',
    197: 'Paint Brush', 198: 'Pear', 199: 'Heavy Truck', 200: 'Hamburger',
    201: 'Extractor', 202: 'Extension Cord', 203: 'Tong',
    204: 'Tennis Racket', 205: 'Folder', 206: 'American Football',
    207: 'Earphone', 208: 'Mask', 209: 'Kettle', 210: 'Tennis',
    211: 'Ship', 212: 'Swing', 213: 'Coffee Machine', 214: 'Slide',
    215: 'Carriage', 216: 'Onion', 217: 'Green Beans', 218: 'Projector',
    219: 'Frisbee', 220: 'Washing Machine', 221: 'Chicken', 222: 'Printer',
    223: 'Watermelon', 224: 'Saxophone', 225: 'Tissue', 226: 'Toothbrush',
    227: 'Ice Cream', 228: 'Hot Air Balloon', 229: 'Cello',
    230: 'French Fries', 231: 'Scale', 232: 'Trophy', 233: 'Cabbage',
    234: 'Hot Dog', 235: 'Blender', 236: 'Peach', 237: 'Rice',
    238: 'Wallet/Purse', 239: 'Volleyball', 240: 'Deer', 241: 'Goose',
    242: 'Tape', 243: 'Tablet', 244: 'Cosmetics', 245: 'Trumpet',
    246: 'Pineapple', 247: 'Golf Ball', 248: 'Ambulance',
    249: 'Parking Meter', 250: 'Mango', 251: 'Key', 252: 'Hurdle',
    253: 'Fishing Rod', 254: 'Medal', 255: 'Flute', 256: 'Brush',
    257: 'Penguin', 258: 'Megaphone', 259: 'Corn', 260: 'Lettuce',
    261: 'Garlic', 262: 'Swan', 263: 'Helicopter', 264: 'Green Onion',
    265: 'Sandwich', 266: 'Nuts', 267: 'Speed Limit Sign',
    268: 'Induction Cooker', 269: 'Broom', 270: 'Trombone', 271: 'Plum',
    272: 'Rickshaw', 273: 'Goldfish', 274: 'Kiwi', 275: 'Router/Modem',
    276: 'Poker Card', 277: 'Toaster', 278: 'Shrimp', 279: 'Sushi',
    280: 'Cheese', 281: 'Notepaper', 282: 'Cherry', 283: 'Pliers',
    284: 'CD', 285: 'Pasta', 286: 'Hammer', 287: 'Cue', 288: 'Avocado',
    289: 'Hami Melon', 290: 'Flask', 291: 'Mushroom', 292: 'Screwdriver',
    293: 'Soap', 294: 'Recorder', 295: 'Bear', 296: 'Eggplant',
    297: 'Board Eraser', 298: 'Coconut', 299: 'Tape Measure/Ruler',
    300: 'Pig', 301: 'Showerhead', 302: 'Globe', 303: 'Chips',
    304: 'Steak', 305: 'Crosswalk Sign', 306: 'Stapler', 307: 'Camel',
    308: 'Formula 1', 309: 'Pomegranate', 310: 'Dishwasher', 311: 'Crab',
    312: 'Hoverboard', 313: 'Meatball', 314: 'Rice Cooker', 315: 'Tuba',
    316: 'Calculator', 317: 'Papaya', 318: 'Antelope', 319: 'Parrot',
    320: 'Seal', 321: 'Butterfly', 322: 'Dumbbell', 323: 'Donkey',
    324: 'Lion', 325: 'Urinal', 326: 'Dolphin', 327: 'Electric Drill',
    328: 'Hair Dryer', 329: 'Egg Tart', 330: 'Jellyfish', 331: 'Treadmill',
    332: 'Lighter', 333: 'Grapefruit', 334: 'Game Board', 335: 'Mop',
    336: 'Radish', 337: 'Baozi', 338: 'Target', 339: 'French',
    340: 'Spring Rolls', 341: 'Monkey', 342: 'Rabbit', 343: 'Pencil Case',
    344: 'Yak', 345: 'Red Cabbage', 346: 'Binoculars', 347: 'Asparagus',
    348: 'Barbell', 349: 'Scallop', 350: 'Noodles', 351: 'Comb',
    352: 'Dumpling', 353: 'Oyster', 354: 'Table Tennis Paddle',
    355: 'Cosmetics Brush', 356: 'Chainsaw', 357: 'Eraser', 358: 'Lobster',
    359: 'Durian', 360: 'Okra', 361: 'Lipstick', 362: 'Cosmetics Mirror',
    363: 'Curling', 364: 'Table Tennis',
}

WALL_MOUNTED_LABELS = {
    16,
    37,
    79,
    80,
    83,
    94,
    101,
    108,
    175,
    218,
    301,
}

CEILING_LABELS = {
    6,
    138,
}

MAYBE_WALL_LABELS = {
    6,
    27,
    41,
    69,
    74,
    100,
}

LARGE_FURNITURE_LABELS = {
    2,
    9,
    12,
    20,
    50,
    75,
    116,
}

FURNITURE_SIZE_PRIORS = {
    2:   (0.45, 0.85, 0.45),
    9:   (1.20, 0.75, 0.60),
    12:  (0.80, 1.80, 0.40),
    24:  (1.20, 0.45, 0.40),
    25:  (0.40, 0.80, 0.40),
    37:  (0.90, 0.55, 0.10),
    44:  (0.30, 0.60, 0.30),
    47:  (0.40, 0.50, 0.40),
    50:  (2.00, 0.85, 0.90),
    56:  (0.40, 0.40, 0.65),
    67:  (1.40, 0.75, 0.80),
    75:  (2.00, 0.60, 1.80),
    94:  (0.30, 0.30, 0.08),
    98:  (1.40, 0.75, 0.80),
    107: (0.35, 0.25, 0.02),
    133: (0.70, 1.70, 0.70),
    134: (0.60, 0.45, 0.55),
    136: (0.70, 1.70, 0.70),
    140: (0.60, 0.20, 0.45),
    143: (2.00, 0.60, 1.80),
    144: (2.00, 0.85, 0.90),
    148: (1.40, 0.75, 0.80),
    153: (0.40, 0.40, 0.65),
    163: (0.60, 0.45, 0.55),
    167: (1.00, 0.45, 0.60),
    168: (0.50, 0.55, 0.50),
    174: (2.00, 0.85, 0.90),
    175: (0.85, 0.85, 0.85),
    176: (1.00, 0.45, 0.60),
    191: (1.50, 0.60, 0.70),
}

def classify_placement_geometric(
    xc: float, yc: float, w_px: float, h_px: float,
    Z: float, W: int, H: int, f: float,
    room_bounds: Dict[str, float],
    cx: float = None, cy: float = None,  # type: ignore[assignment]
) -> str:

    if cx is None: cx = W / 2.0
    if cy is None: cy = H / 2.0
    Y = -(yc - cy) * Z / f
    X = (xc - cx) * Z / f

    obj_h = (h_px * Z) / f
    y_bottom = Y - obj_h / 2.0
    y_top = Y + obj_h / 2.0

    rx_min = room_bounds.get('x_min', -3.0)
    rx_max = room_bounds.get('x_max', 3.0)
    ry_min = room_bounds.get('y_min', -1.5)
    ry_max = room_bounds.get('y_max', 1.5)
    rz_max = room_bounds.get('z_max', 6.0)

    room_h = ry_max - ry_min
    if room_h < 0.1: room_h = 2.5

    if y_top > ry_max - room_h * 0.15 or yc < H * 0.15:
        return 'ceiling'

    if y_bottom < ry_min + room_h * 0.15:
        return 'floor'

    dist_back = abs(Z - rz_max)
    dist_left = abs(X - rx_min)
    dist_right = abs(X - rx_max)
    min_wall_dist = min(dist_back, dist_left, dist_right)

    if y_bottom > ry_min + room_h * 0.15 and min_wall_dist < room_h * 0.3:
        return 'wall'

    return 'floor'

def _extract_wall_texture(image: Image.Image) -> Image.Image:

    W, H = image.size
    left = int(W * 0.20)
    right = int(W * 0.80)
    top = int(H * 0.05)
    bottom = int(H * 0.40)
    wall_crop = image.crop((left, top, right, bottom))
    return wall_crop

def _extract_floor_texture(image: Image.Image) -> Image.Image:

    W, H = image.size
    left = int(W * 0.20)
    right = int(W * 0.80)
    top = int(H * 0.75)
    bottom = H
    floor_crop = image.crop((left, top, right, bottom))
    return floor_crop

def _extract_ceiling_texture(image: Image.Image) -> Image.Image:

    W, H = image.size
    left = int(W * 0.20)
    right = int(W * 0.80)
    top = 0
    bottom = int(H * 0.10)
    ceil_crop = image.crop((left, top, right, bottom))
    return ceil_crop

def _apply_texture_to_box(mesh: trimesh.Trimesh, texture_img: Image.Image) -> trimesh.Trimesh:

    try:
        import PIL
        tex_array = np.array(texture_img.convert('RGB'))
        material = trimesh.visual.material.SimpleMaterial(
            image=texture_img.convert('RGB'),
            ambient=[1.0, 1.0, 1.0, 1.0],
            diffuse=[1.0, 1.0, 1.0, 1.0],
        )
        if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
            verts = mesh.vertices
            bb_min = verts.min(axis=0)
            bb_max = verts.max(axis=0)
            bb_range = bb_max - bb_min
            bb_range[bb_range < 1e-6] = 1.0
            uv = np.zeros((len(verts), 2))
            uv[:, 0] = (verts[:, 0] - bb_min[0]) / bb_range[0]
            uv[:, 1] = (verts[:, 1] - bb_min[1]) / bb_range[1]
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv,
                material=material,
            )
    except Exception as e:
        logger.warning(f"Could not apply texture to mesh: {e}")
    return mesh

def robust_depth(crop: np.ndarray) -> float:

    if crop.size == 0: return 0.0
    p25, p75 = np.percentile(crop, [25, 75])
    trimmed = crop[(crop >= p25) & (crop <= p75)]
    return float(np.median(trimmed)) if trimmed.size else float(np.median(crop))

def object_depth(
    depth_map: np.ndarray,
    box: List[int],
    scene_id: str,
    obj_idx: int,
    mask_path: str = None,  # type: ignore[assignment]
) -> float:

    x0, y0, x1, y1 = box
    h_map, w_map = depth_map.shape[:2]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w_map, x1), min(h_map, y1)
    full_crop = depth_map[y0c:y1c, x0c:x1c]
    if full_crop.size == 0:
        return 0.0

    if mask_path is not None:
        try:
            alpha = np.array(Image.open(mask_path).convert("RGBA").split()[-1])
            mask_pil = Image.fromarray(alpha).resize((x1c - x0c, y1c - y0c), Image.NEAREST)
            mask_region = np.array(mask_pil) > 10

            Z = sample_bottom_mask_depth(full_crop, mask_region, bottom_fraction=0.10)
            if Z > 0.01:
                logger.info(
                    f"[{scene_id}] obj{obj_idx} depth: bottom-10%% mask Z={Z:.3f}m "
                    f"(crop {full_crop.shape[1]}x{full_crop.shape[0]}px)"
                )
                return Z
        except Exception as e:
            logger.warning(f"[{scene_id}] obj{obj_idx}: mask depth failed ({e}), using bbox fallback")

    bh, bw = full_crop.shape
    margin_x = int(bw * 0.2)
    margin_y = int(bh * 0.2)
    inner = full_crop[margin_y:bh - margin_y, margin_x:bw - margin_x]

    if inner.size > 0 and inner.shape[0] > 1 and inner.shape[1] > 1:
        gy = np.linspace(-1, 1, inner.shape[0])
        gx = np.linspace(-1, 1, inner.shape[1])
        gxx, gyy = np.meshgrid(gx, gy)
        gauss = np.exp(-(gxx ** 2 + gyy ** 2) / 0.5)

        vals = inner.ravel()
        weights = gauss.ravel()
        valid = vals > 0.01
        vals, weights = vals[valid], weights[valid]
        if vals.size > 0:
            p25, p75 = np.percentile(vals, [25, 75])
            keep = (vals >= p25) & (vals <= p75)
            vals, weights = vals[keep], weights[keep]
            if vals.size > 0:
                order = np.argsort(vals)
                vals, weights = vals[order], weights[order]
                cum_w = np.cumsum(weights)
                median_idx = np.searchsorted(cum_w, cum_w[-1] / 2)
                Z = float(vals[min(median_idx, len(vals) - 1)])
                logger.info(
                    f"[{scene_id}] obj{obj_idx} depth: centre-weighted Z={Z:.3f}m "
                    f"(inner crop {inner.shape[1]}x{inner.shape[0]}px)"
                )
                return Z

    Z = robust_depth(full_crop)
    logger.info(
        f"[{scene_id}] obj{obj_idx} depth: FALLBACK robust Z={Z:.3f}m "
        f"(crop {full_crop.shape[1]}x{full_crop.shape[0]}px)"
    )
    return Z

def _debug_depth_map(
    depth_map: np.ndarray,
    image: Image.Image,
    scene_folder: str,
    scene_id: str,
) -> None:

    debug_dir = Path(scene_folder) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    valid = depth_map[(depth_map > 0.01) & (depth_map < 100.0)]
    if valid.size == 0:
        logger.warning(f"[{scene_id}] Depth map has no valid pixels!")
        return

    bins = [0, 1, 3, 6, 10, 100]
    counts, _ = np.histogram(valid, bins=bins)
    hist_str = " | ".join(f"{bins[i]}-{bins[i+1]}m:{counts[i]}" for i in range(len(counts)))
    logger.info(
        f"[{scene_id}] DEPTH STATS: min={valid.min():.3f}m max={valid.max():.3f}m "
        f"mean={valid.mean():.3f}m std={valid.std():.3f}m | histogram: {hist_str}"
    )

    d_norm = depth_map.copy()
    d_norm = np.clip(d_norm, 0, float(np.percentile(valid, 99)))
    d_min, d_max = d_norm.min(), d_norm.max()
    if d_max - d_min > 1e-6:
        d_norm = ((d_norm - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d_norm, dtype=np.uint8)
    Image.fromarray(d_norm).save(debug_dir / "depth_map.png")

    try:
        from matplotlib import cm
        colored = (cm.viridis(d_norm.astype(np.float32) / 255.0)[:, :, :3] * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(colored).resize(image.size)
        blended = Image.blend(image.convert("RGB"), overlay_pil, alpha=0.45)
        blended.save(debug_dir / "depth_overlay.png")
    except Exception as e:
        logger.warning(f"[{scene_id}] Could not save depth overlay: {e}")

    logger.info(f"[{scene_id}] Saved debug depth images to {debug_dir}")

def _estimate_room_dimensions(
    depth_map: np.ndarray,
    W: int,
    H: int,
    f: float,
    scene_id: str,
) -> Tuple[float, float, float]:

    valid_mask = (depth_map > 0.1) & (depth_map < 40.0)

    if np.count_nonzero(valid_mask) < 100:
        logger.warning(f"[{scene_id}] Insufficient depth points, using fallback dimensions.")
        return 5.0, 3.0, 5.0

    v, u = np.indices((H, W))

    v_valid = v[valid_mask]
    u_valid = u[valid_mask]
    Z = depth_map[valid_mask]

    X = (u_valid - W / 2) * Z / f
    Y = -(v_valid - H / 2) * Z / f

    room_width = float(np.percentile(X, 99.5) - np.percentile(X, 0.5))
    room_depth = float(np.percentile(Z, 98.0))

    top_pixels = Y[v_valid < H * 0.1]
    bottom_pixels = Y[v_valid > H * 0.9]

    y_floor = float(np.percentile(bottom_pixels, 5)) if bottom_pixels.size > 100 else float(np.percentile(Y, 5))
    y_ceil = float(np.percentile(top_pixels, 95)) if top_pixels.size > 100 else float(np.percentile(Y, 95))

    room_height = y_ceil - y_floor

    if room_height < 2.5:
        room_height = max(room_height, 2.8)

    for name, val in [("width", room_width), ("height", room_height), ("depth", room_depth)]:
        if val < ROOM_MIN_DIM or val > ROOM_MAX_DIM:
            logger.warning(f"[{scene_id}] Room {name} CLAMPED: {val:.2f}m → [{ROOM_MIN_DIM}, {ROOM_MAX_DIM}]m")

    room_width = max(ROOM_MIN_DIM, min(ROOM_MAX_DIM, room_width))
    room_height = max(ROOM_MIN_DIM, min(ROOM_MAX_DIM, room_height))
    room_depth = max(ROOM_MIN_DIM, min(ROOM_MAX_DIM, room_depth))

    logger.info(f"[{scene_id}] POINTCLOUD RESULT: room {room_width:.2f}×{room_height:.2f}×{room_depth:.2f}m (W×H×D)")
    return room_width, room_height, room_depth

def _floor_align_objects(scene: trimesh.Scene, scene_id: str) -> float:

    room_names = {"room_floor", "room_back_wall", "room_left_wall", "room_right_wall", "room_ceiling"}
    obj_names = [n for n in scene.geometry if n not in room_names]
    if not obj_names:
        return -1.0

    min_y = float("inf")
    for name in obj_names:
        g = scene.geometry[name]
        bounds = g.bounds if hasattr(g, 'bounds') else None
        if bounds is not None and len(bounds) == 2:
            tk = scene.graph.get(name)
            if tk is not None:
                transform = tk[0] if isinstance(tk, tuple) else tk
                verts = g.vertices.copy() if hasattr(g, 'vertices') else None
            y_min_local = float(bounds[0][1])
            if y_min_local < min_y:
                min_y = y_min_local

    if min_y == float("inf"):
        try:
            min_y = float(scene.bounds[0][1])
        except Exception:
            min_y = -1.0

    logger.info(f"[{scene_id}] Floor alignment: lowest object Y = {min_y:.3f}m")
    return min_y

def _self_check_scene(
    scene: trimesh.Scene,
    room_width: float,
    room_height: float,
    room_depth: float,
    floor_y: float,
    cx_env: float,
    cz_env: float,
    scene_folder: str,
    scene_id: str,
) -> Dict[str, Any]:

    room_names = {"room_floor", "room_back_wall", "room_left_wall", "room_right_wall", "room_ceiling"}
    obj_names = [n for n in scene.geometry if n not in room_names]

    report: Dict[str, Any] = {
        "scene_id": scene_id,
        "room_dims_m": {"width": round(float(room_width), 3), "height": round(float(room_height), 3), "depth": round(float(room_depth), 3)},
        "floor_y": round(float(floor_y), 3),
        "num_objects": len(obj_names),
        "objects": [],
        "warnings": [],
    }

    margin = 0.2
    rx_min = cx_env - room_width / 2 - room_width * margin
    rx_max = cx_env + room_width / 2 + room_width * margin
    ry_min = floor_y - room_height * margin
    ry_max = floor_y + room_height + room_height * margin
    rz_min = 0 - room_depth * margin
    rz_max = cz_env * 2 + room_depth * margin

    def _world_bounds(gname):
        g = scene.geometry[gname]
        try:
            transform, _geom_name = scene.graph[gname]
        except Exception:
            transform = np.eye(4)
        if hasattr(g, 'vertices') and len(g.vertices) > 0:
            verts = np.array(g.vertices)
            ones = np.ones((verts.shape[0], 1))
            verts_h = np.hstack([verts, ones])
            world_verts = (transform @ verts_h.T).T[:, :3]
            b_min = world_verts.min(axis=0)
            b_max = world_verts.max(axis=0)
        else:
            local_bounds = g.bounds if hasattr(g, 'bounds') else np.array([[0, 0, 0], [0, 0, 0]])
            translation = transform[:3, 3]
            b_min = local_bounds[0] + translation
            b_max = local_bounds[1] + translation
        return b_min, b_max, b_max - b_min, (b_min + b_max) / 2

    world_bounds_cache = {}

    for name in obj_names:
        try:
            b_min, b_max, ext, center = _world_bounds(name)
            world_bounds_cache[name] = (b_min, b_max)
            max_dim = float(max(ext))
        except Exception:
            report["objects"].append({"name": name, "error": "cannot read geometry"})
            continue

        obj_info: Dict[str, Any] = {
            "name": name,
            "extents_m": [round(float(e), 3) for e in ext],
            "max_dim_m": round(float(max_dim), 3),
            "center": [round(float(c), 3) for c in center],
        }

        if max_dim < 0.03:
            w = f"obj '{name}' is implausibly small ({max_dim:.3f}m)"
            report["warnings"].append(w)
            obj_info["scale_warning"] = w
        elif max_dim > 6.0:
            w = f"obj '{name}' is implausibly large ({max_dim:.3f}m)"
            report["warnings"].append(w)
            obj_info["scale_warning"] = w
        else:
            obj_info["scale_ok"] = True

        outside = False
        if float(b_min[0]) < rx_min or float(b_max[0]) > rx_max:
            outside = True
        if float(b_min[2]) < rz_min or float(b_max[2]) > rz_max:
            outside = True
        if outside:
            w = f"obj '{name}' extends outside room bounds"
            report["warnings"].append(w)
            obj_info["contained"] = False
        else:
            obj_info["contained"] = True

        report["objects"].append(obj_info)

    overlap_warnings = []
    for i in range(len(obj_names)):
        if obj_names[i] not in world_bounds_cache:
            continue
        for j in range(i + 1, len(obj_names)):
            if obj_names[j] not in world_bounds_cache:
                continue
            try:
                bi_min, bi_max = world_bounds_cache[obj_names[i]]
                bj_min, bj_max = world_bounds_cache[obj_names[j]]
                inter_min = np.maximum(bi_min, bj_min)
                inter_max = np.minimum(bi_max, bj_max)
                inter_ext = np.maximum(0, inter_max - inter_min)
                inter_vol = float(inter_ext[0] * inter_ext[1] * inter_ext[2])
                if inter_vol > 0:
                    vol_i = float(np.prod(bi_max - bi_min))
                    vol_j = float(np.prod(bj_max - bj_min))
                    union_vol = vol_i + vol_j - inter_vol
                    iou = inter_vol / union_vol if union_vol > 0 else 0
                    if iou > 0.15:
                        w = f"overlap: '{obj_names[i]}' ∩ '{obj_names[j]}' IoU={iou:.2f}"
                        overlap_warnings.append(w)
            except Exception:
                pass
    if overlap_warnings:
        report["warnings"].extend(overlap_warnings)

    n_warn = len(report["warnings"])
    status = "PASS" if n_warn == 0 else f"WARN ({n_warn} issues)"
    logger.info(f"[{scene_id}] === SELF-CHECK: {status} ===")
    for w in report["warnings"]:
        logger.warning(f"[{scene_id}]   ⚠ {w}")

    debug_dir = Path(scene_folder) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    json_path = debug_dir / "self_check.json"
    with open(json_path, "w") as fp:
        json.dump(report, fp, indent=2)
    logger.info(f"[{scene_id}] Self-check report saved to {json_path}")

    return report

def _load_structured3d_depth_and_fov(image_path: str) -> Tuple[np.ndarray, float]:

    img_dir = Path(image_path).parent

    depth_path = img_dir / "depth.png"
    camera_path = img_dir / "camera_pose.txt"

    if not depth_path.exists() or not camera_path.exists():
        raise FileNotFoundError(f"No GT depth/camera at {img_dir}")

    depth_pil = Image.open(depth_path)
    depth_raw = np.array(depth_pil).astype(np.float64)
    depth_meters = depth_raw / 1000.0
    depth_meters[depth_meters <= 0] = 50.0

    with open(camera_path, "r") as f:
        parts = f.read().strip().split()
    xfov = float(parts[-3])

    logger.info(f"[GT] Loaded Structured3D depth: shape={depth_meters.shape}, "
                f"range=[{depth_meters.min():.2f}, {depth_meters.max():.2f}]m, FOV={math.degrees(xfov):.1f}°")
    return depth_meters, xfov

def _compute_depth_pairwise_distances(
    depth_map: np.ndarray,
    boxes: List[List[int]],
    f: float,
    W: int,
    H: int,
) -> np.ndarray:

    n = len(boxes)
    centres_3d = np.zeros((n, 3))
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        xc = (x0 + x1) / 2.0
        yc = (y0 + y1) / 2.0
        h_map, w_map = depth_map.shape[:2]
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w_map, x1), min(h_map, y1)
        crop = depth_map[y0c:y1c, x0c:x1c]
        if crop.size > 0:
            valid = crop[(crop > 0.01) & (crop < 40.0)]
            Z = float(np.median(valid)) if valid.size > 0 else float(np.median(crop))
        else:
            Z = 3.0
        centres_3d[i] = [(xc - W / 2.0) * Z / f, -(yc - H / 2.0) * Z / f, Z]

    from scipy.spatial.distance import cdist
    return cdist(centres_3d, centres_3d)

def _refine_positions_pairwise(
    positions: np.ndarray,
    target_dists: np.ndarray,
    n_iter: int = 50,
    lr: float = 0.02,
    scene_id: str = "",
) -> np.ndarray:

    n = len(positions)
    if n < 2:
        return positions

    pos = positions.copy().astype(np.float64)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if target_dists[i, j] > 0.1:
                pairs.append((i, j, target_dists[i, j]))

    if not pairs:
        return positions

    initial_energy = 0.0
    for i, j, d_target in pairs:
        d_cur = np.linalg.norm(pos[i] - pos[j])
        initial_energy += (d_cur - d_target) ** 2 / d_target

    for iteration in range(n_iter):
        grad = np.zeros_like(pos)
        for i, j, d_target in pairs:
            diff = pos[i] - pos[j]
            d_cur = np.linalg.norm(diff)
            if d_cur < 1e-6:
                continue
            error = (d_cur - d_target) / d_target
            force = error * diff / d_cur
            grad[i] += force
            grad[j] -= force

        pos -= lr * grad

    final_energy = 0.0
    for i, j, d_target in pairs:
        d_cur = np.linalg.norm(pos[i] - pos[j])
        final_energy += (d_cur - d_target) ** 2 / d_target

    logger.info(
        f"[{scene_id}] Pairwise refinement: energy {initial_energy:.4f} → {final_energy:.4f} "
        f"({len(pairs)} pairs, {n_iter} iterations)"
    )
    return pos

def _wall_aware_scale_correction(
    obj_extents: np.ndarray,
    obj_position: np.ndarray,
    room_bounds: Dict[str, float],
    lbl: int,
    scene_id: str,
    obj_idx: int,
) -> float:

    rx_min = room_bounds.get('x_min', -5.0)
    rx_max = room_bounds.get('x_max', 5.0)
    ry_min = room_bounds.get('y_min', -2.0)
    ry_max = room_bounds.get('y_max', 2.0)
    rz_max = room_bounds.get('z_max', 8.0)

    correction = 1.0

    ox, oy, oz = obj_position
    ow, oh, od = obj_extents

    if ox - ow / 2 < rx_min:
        max_w = (ox - rx_min) * 2
        if max_w > 0.05 and ow > max_w:
            correction = min(correction, max_w / ow)
    if ox + ow / 2 > rx_max:
        max_w = (rx_max - ox) * 2
        if max_w > 0.05 and ow > max_w:
            correction = min(correction, max_w / ow)

    depth_factor = 1.0
    if oz > 8.0:
        depth_factor = 0.7
    elif oz > 5.0:
        depth_factor = 0.85
    correction *= depth_factor

    cur_max = max(obj_extents) * correction
    if cur_max > 3.0:
        correction *= (3.0 / cur_max)
    elif cur_max > 2.0:
        correction *= 0.95

    if correction < 0.99:
        logger.info(
            f"[{scene_id}] obj{obj_idx}: wall-aware scale correction={correction:.3f} "
            f"(extents=[{ow:.2f},{oh:.2f},{od:.2f}]m at pos=[{ox:.2f},{oy:.2f},{oz:.2f}]m)"
        )

    return max(0.1, correction)

def position_meshes(
    mesh_paths: List[str],
    image_path: str,
    scene_folder: str,
    scene_id: str,
    boxes: List[List[int]],
    labels: List[int] = None,
) -> str:

    with ThesisProfiler("Spatial_Assembly_ZoeDepth", scene_id):
        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        use_gt = False
        try:
            depth_map, xfov = _load_structured3d_depth_and_fov(image_path)
            f = (W / 2) / math.tan(xfov / 2)
            cx, cy = W / 2.0, H / 2.0
            use_gt = True
            logger.info(f"[{scene_id}] Using GT depth + FOV. Focal length f={f:.1f}px")
        except FileNotFoundError:
            logger.info(f"[{scene_id}] No GT depth found, running neural depth estimation...")
            depth_result = get_depth_model()
            model, processor = depth_result[0], depth_result[1]
            model_type = depth_result[2] if len(depth_result) > 2 else "dav2"

            inputs = processor(images=image, return_tensors="pt").to(PIPELINE_DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)

            if model_type == "depth_pro":
                post = processor.post_process_depth_estimation(
                    outputs, target_sizes=[(H, W)]
                )
                depth_map = post[0]["predicted_depth"].cpu().numpy().astype(np.float32)
                fov_deg = float(post[0].get("fov", 70.0))
                f = (W / 2) / math.tan(math.radians(fov_deg) / 2)
                logger.info(f"[{scene_id}] Depth Pro: f={f:.1f}px (estimated FOV={fov_deg:.1f}°)")
            else:
                depth_map_torch = outputs.predicted_depth
                depth_map_torch = torch.nn.functional.interpolate(
                    depth_map_torch.unsqueeze(1),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
                depth_map = depth_map_torch.cpu().numpy().astype(np.float32)
                model_fov_deg = getattr(getattr(model, 'config', None), 'fov', None)
                if model_fov_deg is None:
                    model_fov_deg = 70.0
                f = (W / 2) / math.tan(math.radians(model_fov_deg) / 2)
                logger.info(f"[{scene_id}] DA-V2: f={f:.1f}px (FOV={model_fov_deg:.1f}°)")

            model.to("cpu")
            del inputs
            torch.cuda.empty_cache()

            depth_map = median_filter(depth_map, size=5)
            if depth_map.mean() < 0.1:
                depth_map = np.clip(depth_map, 1e-4, 50.0)

        if not use_gt:
            image_np = np.array(image)
            f_vp, cx_vp, cy_vp = estimate_intrinsics_vp(image_np, W, H)
            if model_type == "depth_pro":
                cx, cy = cx_vp, cy_vp
                logger.info(f"[{scene_id}] VP intrinsics: keeping Depth Pro f={f:.1f}px, using VP cx={cx:.1f} cy={cy:.1f}")
            else:
                f, cx, cy = f_vp, cx_vp, cy_vp
                logger.info(f"[{scene_id}] VP intrinsics: f={f:.1f}px, cx={cx:.1f}, cy={cy:.1f}")

        floor_y_ransac, floor_normal = extract_floor_plane_ransac(
            depth_map, f, cx, cy, W, H, scene_id=scene_id
        )

        ceiling_y_est = estimate_ceiling_y(depth_map, f, cx, cy, W, H, scene_id=scene_id)

        boxes_np = np.array(boxes)

        logger.info(f"[{scene_id}] === SPATIAL ASSEMBLY (CAMERA-SPACE PINHOLE) ===")
        logger.info(f"[{scene_id}] Image: {W}x{H}px | Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]m | f={f:.1f}px cx={cx:.1f} cy={cy:.1f} | GT={use_gt}")
        logger.info(f"[{scene_id}] RANSAC floor Y={floor_y_ransac:.3f}m, ceiling Y={ceiling_y_est}")
        _debug_depth_map(depth_map, image, scene_folder, scene_id)

        if labels is None:
            labels = [-1] * len(mesh_paths)
        labels_np = labels

        obj_data = []
        for idx, (mp, bb) in enumerate(zip(mesh_paths, boxes_np)):
            x0, y0, x1, y1 = bb.astype(int)
            w_px, h_px = x1 - x0, y1 - y0
            xc, yc = (x0 + x1) / 2.0, (y0 + y1) / 2.0

            base_name = Path(mp).stem.replace("_textured", "")
            mask_path_candidate = Path(scene_folder) / "crops" / f"{base_name}_no_bg.png"
            mask_for_depth = str(mask_path_candidate) if mask_path_candidate.exists() else None

            Z = object_depth(depth_map, [x0, y0, x1, y1], scene_id, idx, mask_path=mask_for_depth)
            if Z <= 0 or w_px <= 0 or h_px <= 0:
                logger.warning(f"[{scene_id}] obj{idx}: SKIPPED (Z={Z:.3f}, w={w_px}, h={h_px})")
                continue
            lbl = labels_np[idx] if idx < len(labels_np) else -1
            obj_data.append((idx, mp, [x0, y0, x1, y1], Z, xc, yc, w_px, h_px, lbl, 'floor'))

        if not obj_data:
            logger.warning(f"[{scene_id}] No valid objects to place.")
            all_depths = []
        else:
            all_depths = [d[3] for d in obj_data]
            z_min_scene = min(all_depths)
            z_max_scene = max(all_depths)
            z_range = z_max_scene - z_min_scene if z_max_scene > z_min_scene else 1.0
            logger.info(
                f"[{scene_id}] Object depth range: [{z_min_scene:.2f}, {z_max_scene:.2f}]m "
                f"(span={z_range:.2f}m across {len(obj_data)} objects)"
            )

        valid_mask = (depth_map > 0.1) & (depth_map < 40.0)
        if np.count_nonzero(valid_mask) > 100:
            v_grid, u_grid = np.indices((H, W))
            Z_pts = depth_map[valid_mask]
            X_pts = (u_grid[valid_mask] - cx) * Z_pts / f
            Y_pts = -(v_grid[valid_mask] - cy) * Z_pts / f

            rx_min = float(np.percentile(X_pts, 0.5))
            rx_max = float(np.percentile(X_pts, 99.5))

            ry_min = floor_y_ransac
            ry_max = float(np.percentile(Y_pts, 99.5))
            if ceiling_y_est is not None:
                ry_max = ceiling_y_est

            raw_z_max = float(np.percentile(Z_pts, 95.0))
            if all_depths:
                max_obj_z = max(all_depths)
                rz_max = min(max(max_obj_z + 0.5, min(raw_z_max, max_obj_z + 2.0)), 10.0)
            else:
                rz_max = min(max(raw_z_max, 3.5), 8.0)
        else:
            rx_min, rx_max = -3.0, 3.0
            ry_min = floor_y_ransac
            ry_max = ceiling_y_est if ceiling_y_est is not None else (floor_y_ransac + 2.7)
            rz_max = 6.0

        room_bounds_dict = {'x_min': rx_min, 'x_max': rx_max, 'y_min': ry_min, 'y_max': ry_max, 'z_max': rz_max}
        logger.info(f"[{scene_id}] Room Bounds (RANSAC floor + VP intrinsics): {room_bounds_dict}")

        refined_obj_data = []
        for (idx, mp, bb, Z, xc, yc, wp, hp, lbl, _) in obj_data:
            placement = classify_placement_geometric(xc, yc, wp, hp, Z, W, H, f, room_bounds_dict, cx=cx, cy=cy)
            class_name = "Object"
            logger.info(f"[{scene_id}] obj{idx} GEOMETRIC: {class_name} -> {placement}")
            refined_obj_data.append((idx, mp, bb, Z, xc, yc, wp, hp, lbl, placement))

        obj_data = refined_obj_data

        if obj_data:
            n_wall = sum(1 for d in obj_data if d[9] == 'wall')
            n_floor = sum(1 for d in obj_data if d[9] == 'floor')
            n_ceil = sum(1 for d in obj_data if d[9] == 'ceiling')
            logger.info(f"[{scene_id}] Final Placement: {n_floor} floor + {n_wall} wall + {n_ceil} ceiling")

        scene = trimesh.Scene()
        placed_info = []
        pre_placed = []

        for idx, mp, box, Z, xc, yc, w_px, h_px, lbl, placement in obj_data:
            x0, y0, x1, y1 = box

            mesh = trimesh.load(mp, force="scene")
            raw_extents = mesh.extents.copy()

            mask_w_px, mask_h_px = w_px, h_px
            base_name = Path(mp).stem.replace("_textured", "")
            no_bg_path = Path(scene_folder) / "crops" / f"{base_name}_no_bg.png"
            if no_bg_path.exists():
                try:
                    alpha = np.array(Image.open(no_bg_path).convert("RGBA").split()[-1])
                    mask = alpha > 10
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any() and cols.any():
                        r_min, r_max = np.where(rows)[0][[0, -1]]
                        c_min, c_max = np.where(cols)[0][[0, -1]]
                        mask_w_px = (c_max - c_min + 1) * (w_px / alpha.shape[1])
                        mask_h_px = (r_max - r_min + 1) * (h_px / alpha.shape[0])
                        logger.info(f"[{scene_id}] obj{idx}: mask-tight {mask_w_px:.0f}x{mask_h_px:.0f}px (bbox {w_px}x{h_px}px)")
                except Exception:
                    pass

            target_w = (mask_w_px * Z) / f
            target_h = (mask_h_px * Z) / f
            mesh_w = raw_extents[0] if raw_extents[0] > 1e-6 else 1e-6
            mesh_h = raw_extents[1] if raw_extents[1] > 1e-6 else 1e-6
            scale_w = target_w / mesh_w
            scale_h = target_h / mesh_h
            scale = min(scale_w, scale_h)

            if Z > 4.0:
                scale *= max(0.7, 1.0 - (Z - 4.0) * 0.05)

            mesh.apply_scale(scale)
            actual_extents = mesh.extents.copy()

            mesh_center = mesh.centroid if hasattr(mesh, 'centroid') else (mesh.bounds[0] + mesh.bounds[1]) / 2
            mesh.apply_translation(-mesh_center)

            X = (xc - cx) * Z / f
            Y = -(yc - cy) * Z / f

            D_half = actual_extents[2] / 2.0
            Z_pos = Z + D_half

            pos = np.array([X, Y, Z_pos])

            pre_placed.append({
                "mesh": mesh,
                "pos": pos,
                "extents": actual_extents.copy(),
                "name": Path(mp).stem,
                "class_name": "Object",
                "lbl": lbl,
                "box": box,
                "idx": idx,
                "placement": placement,
            })

        pre_placed.sort(key=lambda pp: pp["pos"][1] - pp["extents"][1] / 2.0)

        base_floor_y = floor_y_ransac
        base_ceil_y = room_bounds_dict['y_max']
        placed_surfaces = []

        for pp in pre_placed:
            mesh = pp["mesh"]
            pos = pp["pos"]
            extents = pp["extents"]
            lbl = pp["lbl"]
            placement = pp["placement"]

            scale_corr = _wall_aware_scale_correction(
                extents, pos, room_bounds_dict, lbl, scene_id, pp["idx"]
            )
            if scale_corr < 0.99 or scale_corr > 1.01:
                mesh.apply_scale(scale_corr)
                extents *= scale_corr

            obj_h = extents[1]
            ox, oy, oz = pos
            ow, oh, od = extents

            if placement == 'floor':
                x0, x1 = ox - ow/2, ox + ow/2
                z0, z1 = oz - od/2, oz + od/2

                drop_y = base_floor_y
                for surf in placed_surfaces:
                    if not (x1 < surf['x0'] or x0 > surf['x1'] or z1 < surf['z0'] or z0 > surf['z1']):
                        if surf['y_top'] > drop_y:
                            drop_y = surf['y_top']

                pos[1] = drop_y + (obj_h / 2.0)

                anchor_mesh_to_floor(mesh, floor_y=drop_y)

            elif placement == 'ceiling':
                pos[1] = base_ceil_y - (obj_h / 2.0)

            elif placement == 'wall':
                dist_left = abs(pos[0] - room_bounds_dict['x_min'])
                dist_right = abs(room_bounds_dict['x_max'] - pos[0])
                dist_back = abs(room_bounds_dict['z_max'] - pos[2])

                min_dist = min(dist_left, dist_right, dist_back)
                if min_dist == dist_back:
                    pos[2] = room_bounds_dict['z_max'] - (extents[2] / 2.0)
                elif min_dist == dist_left:
                    pos[0] = room_bounds_dict['x_min'] + (extents[0] / 2.0)
                else:
                    pos[0] = room_bounds_dict['x_max'] - (extents[0] / 2.0)

            if placement != 'floor':
                mesh.apply_translation(pos)
            else:
                current_center = mesh.centroid if hasattr(mesh, 'centroid') else (mesh.bounds[0] + mesh.bounds[1]) / 2
                mesh.apply_translation([pos[0] - current_center[0], 0, pos[2] - current_center[2]])

            actual_bounds = mesh.bounds if hasattr(mesh, 'bounds') and mesh.bounds is not None else None
            if actual_bounds is not None:
                placed_surfaces.append({
                    'x0': float(actual_bounds[0][0]),
                    'x1': float(actual_bounds[1][0]),
                    'z0': float(actual_bounds[0][2]),
                    'z1': float(actual_bounds[1][2]),
                    'y_top': float(actual_bounds[1][1]),
                })
            else:
                placed_surfaces.append({
                    'x0': pos[0] - extents[0]/2,
                    'x1': pos[0] + extents[0]/2,
                    'z0': pos[2] - extents[2]/2,
                    'z1': pos[2] + extents[2]/2,
                    'y_top': pos[1] + extents[1]/2,
                })

            buffer = 0.10
            rx_min = min(rx_min, pos[0] - (extents[0] / 2.0) - buffer)
            rx_max = max(rx_max, pos[0] + (extents[0] / 2.0) + buffer)
            ry_min = min(ry_min, pos[1] - (extents[1] / 2.0) - buffer)
            ry_max = max(ry_max, pos[1] + (extents[1] / 2.0) + buffer)
            rz_max = max(rz_max, pos[2] + (extents[2] / 2.0) + buffer)

            room_bounds_dict.update({'x_min': rx_min, 'x_max': rx_max, 'y_min': ry_min, 'y_max': ry_max, 'z_max': rz_max})

            name = pp["name"]
            class_name = pp["class_name"]
            idx = pp["idx"]
            x0, y0, x1, y1 = pp["box"]
            obj_w, obj_h, obj_d = float(extents[0]), float(extents[1]), float(extents[2])

            scene.add_geometry(mesh, node_name=name)
            placed_info.append({
                "idx": idx, "name": name, "class_name": class_name,
                "room_pos": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
                "extents": [round(obj_w, 3), round(obj_h, 3), round(obj_d, 3)],
                "depth_Z": round(float(pos[2]), 3),
            })

            logger.info(
                f"[{scene_id}] obj{idx}: {class_name} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m | "
                f"box=[{x0},{y0},{x1},{y1}] | size=[{obj_w:.3f}×{obj_h:.3f}×{obj_d:.3f}]m"
            )

        if len(placed_info) > 1:
            logger.info(f"[{scene_id}] --- INTER-OBJECT DISTANCES ---")
            for i in range(len(placed_info)):
                for j in range(i + 1, len(placed_info)):
                    pi = placed_info[i]["room_pos"]
                    pj = placed_info[j]["room_pos"]
                    dist = math.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2 + (pi[2]-pj[2])**2)

            for i in range(len(placed_info)):
                ds = []
                for j in range(len(placed_info)):
                    if i == j: continue
                    pi = placed_info[i]["room_pos"]
                    pj = placed_info[j]["room_pos"]
                    d = math.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2 + (pi[2]-pj[2])**2)
                    ds.append((d, placed_info[j]))

                ds.sort(key=lambda x: x[0])
                if ds:
                    closest_d, closest_obj = ds[0]
                    logger.info(
                        f"[{scene_id}] Distance: {placed_info[i]['class_name']} is {closest_d:.2f}m from nearest {closest_obj['class_name']}"
                    )

        room_names_set = {"room_floor", "room_back_wall", "room_left_wall", "room_right_wall", "room_ceiling"}
        rw = rx_max - rx_min

        rw = rx_max - rx_min
        rh = ry_max - ry_min
        cx_env = (rx_min + rx_max) / 2.0
        cy_env = (ry_min + ry_max) / 2.0
        cz_env = rz_max / 2.0

        floor_mesh = trimesh.creation.box(extents=[rw, 0.02, rz_max])
        floor_mesh.apply_translation([cx_env, ry_min - 0.01, cz_env])

        back_wall_mesh = trimesh.creation.box(extents=[rw, rh, 0.02])
        back_wall_mesh.apply_translation([cx_env, cy_env, rz_max + 0.01])

        left_wall_mesh = trimesh.creation.box(extents=[0.02, rh, rz_max])
        left_wall_mesh.apply_translation([rx_min - 0.01, cy_env, cz_env])

        right_wall_mesh = trimesh.creation.box(extents=[0.02, rh, rz_max])
        right_wall_mesh.apply_translation([rx_max + 0.01, cy_env, cz_env])

        ceiling_mesh = trimesh.creation.box(extents=[rw, 0.02, rz_max])
        ceiling_mesh.apply_translation([cx_env, ry_max + 0.01, cz_env])

        try:
            floor_mesh.visual.vertex_colors = [230, 228, 220, 255]
            back_wall_mesh.visual.vertex_colors = [240, 238, 235, 255]
            left_wall_mesh.visual.vertex_colors = [235, 235, 230, 255]
            right_wall_mesh.visual.vertex_colors = [235, 235, 230, 255]
            ceiling_mesh.visual.vertex_colors = [245, 245, 240, 255]
        except Exception:
            pass

        scene.add_geometry(floor_mesh, geom_name="room_floor", node_name="room_floor")
        scene.add_geometry(back_wall_mesh, geom_name="room_back_wall", node_name="room_back_wall")
        scene.add_geometry(left_wall_mesh, geom_name="room_left_wall", node_name="room_left_wall")
        scene.add_geometry(right_wall_mesh, geom_name="room_right_wall", node_name="room_right_wall")
        scene.add_geometry(ceiling_mesh, geom_name="room_ceiling", node_name="room_ceiling")

        num_objects_placed = len(placed_info)
        logger.info(f"[{scene_id}] Placed {num_objects_placed} objects in camera-space room.")
        logger.info(
            f"[{scene_id}] Room bounds: X=[{rx_min:.2f},{rx_max:.2f}] Y=[{ry_min:.2f},{ry_max:.2f}] Z=[0,{rz_max:.2f}]"
        )

        if not use_gt:
            manhattan_align(
                scene, depth_map, f, cx, cy, W, H,
                floor_normal=floor_normal,
                scene_id=scene_id,
            )
        else:
            logger.info(f"[{scene_id}] Manhattan Alignment skipped (using ground-truth depth)")

        metric_scale = 1.0
        if not use_gt:
            if ceiling_y_est is not None and abs(ceiling_y_est - floor_y_ransac) > 0.5:
                scene, metric_scale = scale_scene_metric(
                    scene, floor_y_ransac, ceiling_y_est,
                    target_height=2.7,
                    scene_id=scene_id,
                )
            else:
                logger.info(f"[{scene_id}] Metric scaling skipped (no reliable ceiling estimate)")
        else:
            logger.info(f"[{scene_id}] Metric scaling skipped (using metric ground-truth depth)")

        if len(placed_info) >= 2:
            obj_positions = np.array([p["room_pos"] for p in placed_info]) * metric_scale
            obj_extents = np.array([p["extents"] for p in placed_info]) * metric_scale
            adjusted_pos = resolve_overlaps(obj_positions, obj_extents, scene_id=scene_id)
            for k in range(len(placed_info)):
                delta = np.linalg.norm(adjusted_pos[k] - obj_positions[k])
                if delta > 0.01:
                    logger.info(f"[{scene_id}] obj{placed_info[k]['idx']}: overlap push {delta:.3f}m")

        _self_check_scene(
            scene, rw, rh, rz_max,
            ry_min, cx_env, cz_env, scene_folder, scene_id
        )

        yfov = 2.0 * math.atan((H / 2.0) / f)
        cam = trimesh.scene.cameras.Camera(
            name='photo_perspective',
            resolution=(W, H),
            fov=(math.degrees(yfov), math.degrees(yfov))
        )
        cam_transform = trimesh.transformations.rotation_matrix(math.pi, [0, 1, 0])

        y_shift = -floor_y_ransac
        for node in scene.graph.nodes_geometry:
            T, gname = scene.graph[node]
            T = T.copy()
            T[1, 3] += y_shift
            scene.graph.update(node, matrix=T)

        cam_transform[1, 3] = y_shift
        scene.camera = cam
        scene.camera_transform = cam_transform

        out_dir = Path(scene_folder) / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scene_positioned.glb"
        scene.export(str(out_path))

    return str(out_path)

def full_reconstruction(image_path: str, scene_folder: str) -> str:

    scene_id = Path(scene_folder).name
    logger.info(f"[{scene_id}] Starting Full Pipeline Reconstruction...")

    try:
        crops_and_boxes = detect_objects(image_path, scene_folder, scene_id)
    except Exception as e:
        logger.error(f"[{scene_id}] Detection failed: {e}")
        traceback.print_exc()
        raise

    if not crops_and_boxes:
        raise RuntimeError("Pipeline aborted: No objects detected in the input image.")

    crops_and_boxes = filter_detections(crops_and_boxes, scene_id)

    if not crops_and_boxes:
        raise RuntimeError("Pipeline aborted: All detections were rejected by pre-generation filters.")

    meshes = []
    valid_boxes = []
    valid_labels = []

    for idx, (crop_path, box, score, label) in enumerate(crops_and_boxes):
        logger.info(f"[{scene_id}] Processing crop {idx} (confidence={score:.2f}, label={label})")
        try:
            mesh_path = build_mesh(crop_path, scene_folder, scene_id, obj_idx=idx)
            meshes.append(mesh_path)
            valid_boxes.append(box)
            valid_labels.append(label)

            cleanup_gpu(aggressive=False)

        except Exception as e:
            logger.error(f"[{scene_id}] Mesh generation failed for crop {idx}: {e}")
            traceback.print_exc()

    if not meshes:
        logger.warning(f"[{scene_id}] No meshes could be built from the detected crops. Outputting an empty room.")

    final_scene_path = position_meshes(
        mesh_paths=meshes,
        image_path=image_path,
        scene_folder=scene_folder,
        scene_id=scene_id,
        boxes=valid_boxes,
        labels=valid_labels,
    )

    cleanup_gpu(aggressive=True)
    logger.info(f"[{scene_id}] Pipeline Complete! Final output saved to: {final_scene_path}")

    return final_scene_path

def full_reconstruction_panoramic(pano_folder: str, scene_folder: str) -> str:

    from .wall_pipeline import full_reconstruction_panoramic as _wall_recon
    return _wall_recon(pano_folder, scene_folder)