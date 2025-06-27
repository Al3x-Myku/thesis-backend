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
    dfine_root = Path(dfine_root).expanduser().resolve()
    torch_inf = dfine_root / "tools" / "inference" / "torch_inf.py"
    if not torch_inf.exists():
        raise FileNotFoundError(f"Could not find `{torch_inf}`")

    # Only prepend dfine_root if the path is relative
    def resolve_under_root(p: str) -> Path:
        p = Path(p)
        return p.resolve() if p.is_absolute() else (dfine_root / p).resolve()

    config_abs     = resolve_under_root(config_path)
    checkpoint_abs = resolve_under_root(checkpoint_path)
    input_abs      = Path(input_image).resolve()

    cmd = [
        sys.executable,
        str(torch_inf),
        "-c", str(config_abs),
        "-r", str(checkpoint_abs),
        "-i", str(input_abs),
        "-d", device,
    ]
    if output_dir:
        cmd += ["-o", str(Path(output_dir).resolve())]

    print("→ Running D-FINE:", " ".join(cmd))

    # ensure D-FINE/src is importable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(dfine_root)

    # cd into dfine_root so torch_inf’s internal paths work
    subprocess.run(cmd, check=True, cwd=str(dfine_root), env=env)
