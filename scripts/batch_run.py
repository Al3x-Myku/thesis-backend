#!/usr/bin/env python3
"""Batch pipeline runner — room photo + moodboard pairs → 2D variants + 3D GLB.

Expected input layout:
    pairs/
        001/
            room.jpg          (or room.png — any image)
            moodboard/
                board_1.jpg
                board_2.jpg   (one or more moodboard images)
        002/
            room.jpg
            moodboard/
                board_1.jpg

Outputs are saved alongside each pair:
    pairs/
        001/
            out/
                variant_0.png
                variant_1.png
                variant_2.png
                variant_3.png
                scene.glb
                meta.json     (scene_id, design_session_id, variant_ids, status)

Usage:
    python scripts/batch_run.py \\
        --pairs-dir pairs/ \\
        --api http://localhost:8000 \\
        --email your@email.com \\
        --password yourpassword \\
        --variants 4 \\
        --commit-variant 0     (index of variant to commit to 3D; -1 to skip 3D)
        --engine hunyuan       (or instantmesh for the ablation arm)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch design pipeline runner")
    p.add_argument("--pairs-dir", required=True, help="Directory containing numbered pair subdirs")
    p.add_argument("--api", default="http://localhost:8000", help="API base URL")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--variants", type=int, default=4, help="Number of 2D variants to generate per pair")
    p.add_argument("--commit-variant", type=int, default=0,
                   help="Index (0-based) of variant to commit to 3D. Pass -1 to skip 3D commit.")
    p.add_argument("--engine", choices=["hunyuan", "instantmesh"], default="hunyuan",
                   help="Mesh generation engine for 3D commit (server must have MESH_ENGINE set accordingly)")
    p.add_argument("--mode", choices=["full", "palette_only"], default="full",
                   help=(
                       "full: palette swatch + moodboard furniture crops condition the diffusion. "
                       "palette_only: only colour palette applied — use when you want the room's "
                       "own furniture preserved with no style bleed from the moodboard."
                   ))
    p.add_argument("--poll-interval", type=int, default=10, help="Seconds between status polls")
    p.add_argument("--timeout", type=int, default=1800, help="Max seconds to wait per stage")
    p.add_argument("--skip-reconstruction", action="store_true",
                   help="Skip the initial 3D reconstruction (design zone only)")
    p.add_argument("--dry-run", action="store_true", help="Print plan without making API calls")
    return p.parse_args()


# ── HTTP helpers ──────────────────────────────────────────────────────────────

class Client:
    def __init__(self, base_url: str, token: str):
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    def get(self, path: str, **kw) -> requests.Response:
        r = requests.get(f"{self.base}{path}", headers=self.headers, **kw)
        r.raise_for_status()
        return r

    def post(self, path: str, **kw) -> requests.Response:
        r = requests.post(f"{self.base}{path}", headers=self.headers, **kw)
        r.raise_for_status()
        return r

    def download(self, path: str, dest: Path) -> None:
        r = requests.get(f"{self.base}{path}", headers=self.headers, stream=True)
        r.raise_for_status()
        dest.write_bytes(r.content)


def login(api: str, email: str, password: str) -> str:
    r = requests.post(
        f"{api}/login",
        data={"username": email, "password": password},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def poll(client: Client, path: str, done_statuses: set, fail_statuses: set,
         interval: int, timeout: int, label: str) -> dict:
    """Poll GET path until response['status'] is in done_statuses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(path).json()
        status = data.get("status", "")
        progress = data.get("progress", 0.0)
        print(f"  [{label}] status={status} progress={progress:.0%}", flush=True)
        if status in done_statuses:
            return data
        if status in fail_statuses:
            raise RuntimeError(f"{label} failed: {data}")
        time.sleep(interval)
    raise TimeoutError(f"{label} did not complete within {timeout}s")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def run_pair(client: Client, pair_dir: Path, args) -> dict:
    out_dir = pair_dir / "out"
    out_dir.mkdir(exist_ok=True)
    meta_path = out_dir / "meta.json"

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    room_candidates = sorted(
        p for p in pair_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.stem.startswith("room")
    )
    if not room_candidates:
        room_candidates = sorted(
            p for p in pair_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
        )
    if not room_candidates:
        raise FileNotFoundError(f"No room image found in {pair_dir}")
    room_photo = room_candidates[0]

    moodboard_dir = pair_dir / "moodboard"
    if not moodboard_dir.is_dir():
        raise FileNotFoundError(f"No moodboard/ subdirectory in {pair_dir}")
    board_images = sorted(
        p for p in moodboard_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )
    if not board_images:
        raise FileNotFoundError(f"No moodboard images found in {moodboard_dir}")

    print(f"\n{'='*60}")
    print(f"  Pair: {pair_dir.name}")
    print(f"  Room photo:  {room_photo.name}")
    print(f"  Moodboard:   {[p.name for p in board_images]}")
    print(f"{'='*60}")

    # ── Step 1: Upload room photo + trigger reconstruction ────────────────────
    if "scene_id" not in meta:
        if args.skip_reconstruction:
            raise RuntimeError(
                "--skip-reconstruction requires a prior scene_id in out/meta.json"
            )
        print("  [1/5] Uploading room photo...")
        with open(room_photo, "rb") as f:
            r = client.post("/scenes/", files={"file": (room_photo.name, f, "image/jpeg")})
        scene = r.json()
        meta["scene_id"] = scene["id"]
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"        scene_id={meta['scene_id']}")
    else:
        print(f"  [1/5] Using existing scene_id={meta['scene_id']}")

    scene_id = meta["scene_id"]

    # ── Step 2: Wait for reconstruction ───────────────────────────────────────
    if not args.skip_reconstruction and meta.get("reconstruction_status") != "COMPLETED":
        print("  [2/5] Waiting for 3D reconstruction...")
        scene_data = poll(
            client, f"/scenes/{scene_id}",
            done_statuses={"COMPLETED"},
            fail_statuses={"FAILED"},
            interval=args.poll_interval,
            timeout=args.timeout,
            label="reconstruction",
        )
        meta["reconstruction_status"] = scene_data["status"]
        meta_path.write_text(json.dumps(meta, indent=2))
    else:
        print(f"  [2/5] Reconstruction: {'skipped' if args.skip_reconstruction else 'already done'}")

    # ── Step 3: Upload moodboard + trigger analysis ───────────────────────────
    if "design_session_id" not in meta:
        print(f"  [3/5] Uploading {len(board_images)} moodboard image(s)...")
        files = [
            ("files", (p.name, open(p, "rb"), "image/jpeg"))
            for p in board_images
        ]
        try:
            r = client.post(f"/scenes/{scene_id}/design/moodboard", files=files)
        finally:
            for _, (_, fh, _) in files:
                fh.close()
        ds = r.json()
        meta["design_session_id"] = ds.get("id") or ds.get("session_id")
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"        design_session_id={meta['design_session_id']}")
    else:
        print(f"  [3/5] Using existing design_session_id={meta['design_session_id']}")

    # ── Step 4: Wait for moodboard analysis ───────────────────────────────────
    if meta.get("analysis_status") != "READY":
        print("  [4/5] Waiting for moodboard analysis...")
        ds_data = poll(
            client, f"/scenes/{scene_id}/design/",
            done_statuses={"READY", "COMMITTED"},
            fail_statuses={"FAILED"},
            interval=args.poll_interval,
            timeout=args.timeout,
            label="moodboard analysis",
        )
        meta["analysis_status"] = ds_data.get("status")
        meta_path.write_text(json.dumps(meta, indent=2))
    else:
        print("  [4/5] Moodboard analysis: already done")

    # ── Step 5: Generate variants ─────────────────────────────────────────────
    if not meta.get("variant_ids"):
        print(f"  [5/5] Requesting {args.variants} variants (mode={args.mode})...")
        client.post(f"/scenes/{scene_id}/design/variants?n={args.variants}&mode={args.mode}")

        # Poll until READY again (task sets GENERATING then back to READY)
        poll(
            client, f"/scenes/{scene_id}/design/",
            done_statuses={"READY", "COMMITTED"},
            fail_statuses={"FAILED"},
            interval=args.poll_interval,
            timeout=args.timeout,
            label="variant generation",
        )

        variants = client.get(f"/scenes/{scene_id}/design/variants").json()
        meta["variant_ids"] = [v["id"] for v in variants]
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"        variants: {meta['variant_ids']}")
    else:
        print(f"  [5/5] Variants already generated: {meta['variant_ids']}")

    # ── Download variant images ───────────────────────────────────────────────
    print("  Downloading variant images...")
    for i, vid in enumerate(meta["variant_ids"]):
        dest = out_dir / f"variant_{i}.png"
        if not dest.exists():
            client.download(f"/scenes/{scene_id}/design/variants/{vid}/image", dest)
            print(f"    saved {dest.name}")
        else:
            print(f"    {dest.name} already exists, skipping")

    # ── Step 6 (optional): Commit a variant to 3D ────────────────────────────
    if args.commit_variant >= 0 and not meta.get("glb_path"):
        if args.commit_variant >= len(meta["variant_ids"]):
            print(f"  [6/6] Commit index {args.commit_variant} out of range — skipping 3D")
        else:
            commit_vid = meta["variant_ids"][args.commit_variant]
            print(f"  [6/6] Committing variant {commit_vid} to 3D (engine={args.engine})...")
            client.post(f"/scenes/{scene_id}/design/variants/{commit_vid}/commit")

            poll(
                client, f"/scenes/{scene_id}/design/",
                done_statuses={"COMMITTED"},
                fail_statuses={"FAILED"},
                interval=args.poll_interval,
                timeout=args.timeout,
                label="3D commit",
            )

            glb_dest = out_dir / "scene.glb"
            client.download(f"/scenes/{scene_id}/design/commit/download", glb_dest)
            meta["glb_path"] = str(glb_dest)
            meta["committed_variant_id"] = commit_vid
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"    saved {glb_dest.name}")
    elif args.commit_variant < 0:
        print("  [6/6] 3D commit skipped (--commit-variant -1)")
    else:
        print(f"  [6/6] 3D already committed: {meta.get('glb_path')}")

    meta["done"] = True
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pairs_dir = Path(args.pairs_dir)

    pair_dirs = sorted(
        p for p in pairs_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if not pair_dirs:
        print(f"No subdirectories found in {pairs_dir}")
        sys.exit(1)

    print(f"Found {len(pair_dirs)} pair(s): {[p.name for p in pair_dirs]}")

    if args.dry_run:
        print("\nDry run — would process:")
        for pd in pair_dirs:
            room = next((p for p in pd.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
            boards = list((pd / "moodboard").glob("*")) if (pd / "moodboard").is_dir() else []
            print(f"  {pd.name}: room={room and room.name}, boards={len(boards)}")
        return

    print(f"Logging in to {args.api} ...")
    token = login(args.api, args.email, args.password)
    client = Client(args.api, token)
    print("Logged in.\n")

    results = {}
    for pd in pair_dirs:
        try:
            meta = run_pair(client, pd, args)
            results[pd.name] = "ok"
        except Exception as e:
            print(f"\nERROR on {pd.name}: {e}")
            results[pd.name] = f"error: {e}"

    print("\n" + "="*60)
    print("Batch complete:")
    for name, status in results.items():
        print(f"  {name}: {status}")

    failed = [n for n, s in results.items() if s.startswith("error")]
    if failed:
        print(f"\n{len(failed)} pair(s) failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
