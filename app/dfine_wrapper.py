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
    Spawn D-FINE’s torch_inf.py from inside its own repo directory,
    with PYTHONPATH pointing at the D-FINE root so that `import src.*`
    actually finds D-FINE/src.
    """
    # 1) locate the script absolutely
    torch_inf = (Path(dfine_root) / "tools" / "inference" / "torch_inf.py").resolve()
    if not torch_inf.exists():
        raise FileNotFoundError(f"Could not find `{torch_inf}`")

    # 2) build the exact same cmd you were using
    cmd = [
        sys.executable,
        str(torch_inf),
        "-c", str(config_path),
        "-r", str(checkpoint_path),
        "-i", str(input_image),
        "-d", device,
    ]
    if output_dir:
        cmd += ["-o", str(output_dir)]

    print("Running D-FINE:", " ".join(cmd))

    # 3) copy & extend the current env with PYTHONPATH=D-FINE root
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(dfine_root))

    # 4) cd into the D-FINE checkout so its internal sys.path hack still works
    subprocess.run(cmd, check=True, cwd=dfine_root, env=env)
