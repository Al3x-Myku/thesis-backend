# app/dfine_wrapper.py

import os
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
    """
    Launches D-FINE’s torch_inf.py with an explicit absolute path,
    so that running from dfine_root (cwd) won’t prepend dfine_root twice.
    """
    # locate the inference script
    torch_inf_py = Path(dfine_root) / "tools" / "inference" / "torch_inf.py"
    if not torch_inf_py.exists():
        raise FileNotFoundError(f"Could not find torch_inf.py at: {torch_inf_py}")

    # turn it into an absolute path
    script = torch_inf_py.resolve()

    # build the command
    cmd = [
        sys.executable,
        str(script),
        "-c", str(config_path),
        "-r", str(checkpoint_path),
        "-i", str(input_image),
        "-d", device,
    ]

    # pass along the output directory for crops if provided
    if output_dir:
        cmd += ["-o", str(output_dir)]

    print("Running D-FINE:", " ".join(cmd))

    # run with cwd=dfine_root so imports like `from src.core import ...` work
    subprocess.run(cmd, check=True, cwd=dfine_root)
