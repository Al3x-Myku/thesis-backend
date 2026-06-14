"""InstantMesh ablation arm — subprocess wrapper following the D-FINE pattern.

TencentARC/InstantMesh: single image → Zero123++ multi-view → LRM reconstruction
→ PBR texture baking (Albedo / Normal / AO UV maps).

Setup:
    git clone https://github.com/TencentARC/InstantMesh
    pip install -r InstantMesh/requirements.txt
    # weights download automatically from HF on first run

Controlled via MESH_ENGINE=instantmesh (default is 'hunyuan').
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ThesisPipeline")

INSTANTMESH_ROOT   = os.getenv("INSTANTMESH_ROOT", "./InstantMesh")
INSTANTMESH_CONFIG = os.getenv("INSTANTMESH_CONFIG", "configs/instant-mesh-large.yaml")
MESH_ENGINE        = os.getenv("MESH_ENGINE", "hunyuan")  # "hunyuan" | "instantmesh"


def run_instantmesh_inference(
    image_path: str,
    out_dir: str,
    obj_id: str,
    num_views: int = 6,
    export_texmap: bool = True,
) -> str:
    """Run InstantMesh on a pre-background-removed crop image.

    Background removal is handled upstream by BiRefNet (same as the Hunyuan path),
    so --no_rembg is passed here.

    Returns:
        Absolute path to the exported .obj mesh (with UV texture sidecar files).

    Raises:
        RuntimeError: if InstantMesh root not found, subprocess fails, or no mesh produced.
    """
    root = Path(INSTANTMESH_ROOT).resolve()
    if not root.exists():
        raise RuntimeError(
            f"InstantMesh root not found at {root}. "
            "Clone it with: git clone https://github.com/TencentARC/InstantMesh"
        )

    config = root / INSTANTMESH_CONFIG
    if not config.exists():
        raise RuntimeError(f"InstantMesh config not found: {config}")

    mesh_out = Path(out_dir) / f"{obj_id}_instantmesh"
    mesh_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(root / "run.py"),
        str(config),
        str(Path(image_path).resolve()),
        "--output_path", str(mesh_out),
        "--no_rembg",        # BG already removed by BiRefNet upstream
        "--num_views", str(num_views),
    ]
    if export_texmap:
        cmd.append("--export_texmap")

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) + (":" + existing_pp if existing_pp else "")

    logger.info(f"[InstantMesh] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(root))

    if result.returncode != 0:
        raise RuntimeError(
            f"InstantMesh subprocess failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout[-2000:]}\nSTDERR: {result.stderr[-2000:]}"
        )

    # InstantMesh writes: {output_path}/{stem}/{stem}.obj  (+ albedo/normal/ao .png)
    stem = Path(image_path).stem
    candidates = list(mesh_out.rglob("*.obj")) + list(mesh_out.rglob("*.glb"))
    if not candidates:
        raise RuntimeError(
            f"InstantMesh completed but no mesh found under {mesh_out}. "
            f"stdout: {result.stdout[-1000:]}"
        )

    # Prefer .obj with textures; fall back to .glb
    obj_files = [p for p in candidates if p.suffix == ".obj"]
    chosen = obj_files[0] if obj_files else candidates[0]
    logger.info(f"[InstantMesh] Mesh at: {chosen}")
    return str(chosen)
