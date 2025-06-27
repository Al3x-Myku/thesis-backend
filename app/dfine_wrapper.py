import sys
import subprocess
from pathlib import Path
from typing import Optional

def run_dfine_inference(
    dfine_root: str,
    config_path: str,
    checkpoint_path: str,
    input_image: str,
    device: str = "cuda:0",
    output_dir: Optional[str] = None,
) -> None:
    torch_inf_py = Path(dfine_root) / "tools" / "inference" / "torch_inf.py"
    if not torch_inf_py.exists():
        raise FileNotFoundError(f"{torch_inf_py} missing")

    # If config/checkpoint are relative, make them absolute under DFINE_ROOT
    cfg = Path(config_path)
    if not cfg.is_absolute():
        cfg = Path(dfine_root) / config_path
    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = Path(dfine_root) / checkpoint_path

    cmd = [
        sys.executable,
        str(torch_inf_py),
        "-c", str(cfg),
        "-r", str(ckpt),
        "-i", input_image,
        "-d", device,
    ]
    if output_dir:
        cmd += ["-o", output_dir]

    print("Running D-FINE:", " ".join(cmd))
    subprocess.run(cmd, check=True)
