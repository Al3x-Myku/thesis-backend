# app/dfine_wrapper.py
import os, sys, subprocess
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

    cmd = [
        sys.executable,
        str(torch_inf_py),
        "-c", str(config_path),
        "-r", str(checkpoint_path),
        "-i", str(input_image),
        "-d", device,
    ]

    if output_dir:
        # pass the target folder for crops
        cmd += ["-o", str(output_dir)]

    print("Running D-FINE:", " ".join(cmd))
    subprocess.run(cmd, check=True)
