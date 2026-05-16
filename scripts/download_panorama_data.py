"""
Download Structured3D PANORAMA data to match existing perspective data.

The Structured3D panorama data includes per-room:
  - panorama/full/rgb_rawlight.png  (equirectangular 360° RGB)
  - panorama/full/depth.png         (16-bit depth in mm)
  - panorama/full/semantic.png      (NYUv2 40-class labels)
  - panorama/layout.txt             (ordered corner positions)
  - panorama/camera_xyz.txt         (camera location in mm)

This is REQUIRED for the wall-first panoramic reconstruction pipeline.

URL pattern (Azure Kujiale mirror):
  https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/Structured3D_panorama_XX.zip

Each zip covers ~200 scenes and is ~10-12 GB.
The full dataset (18 zips) is ~180 GB total.

Usage:
  # Download ALL panorama zips (match perspective dataset):
  python scripts/download_panorama_data.py

  # Download specific zip indices only (e.g., just zip 05 for scenes ~1000-1199):
  python scripts/download_panorama_data.py --indices 5 6 7

  # Download a small subset for testing (zip 05 covers scene_01058):
  python scripts/download_panorama_data.py --indices 5
"""

import os
import zipfile
import urllib.request
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://zju-kjl-jointlab-azure.kujiale.com/Structured3D"


def download_and_extract(url: str, output_dir: Path):
    """Download a zip file and extract it, with corruption recovery."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    zip_path = output_dir / filename

    def try_extract():
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            logger.info(f"Extracted {filename} into {output_dir}")
            return True
        except zipfile.BadZipFile:
            logger.warning(f"Corrupted zip: {filename}. Deleting and redownloading...")
            zip_path.unlink()
            return False

    if zip_path.exists():
        logger.info(f"{filename} already exists, attempting extraction...")
        if try_extract():
            return

    logger.info(f"Downloading {filename} from Azure Mirror... (This is a ~10 GB file)")

    # Download with progress reporting
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            gb_done = downloaded / (1024**3)
            gb_total = total_size / (1024**3)
            sys.stdout.write(f"\r  {filename}: {gb_done:.1f}/{gb_total:.1f} GB ({pct:.0f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    print()  # newline after progress
    logger.info(f"Downloaded {filename} successfully.")

    logger.info(f"Extracting {filename}...")
    try_extract()


def verify_panorama_data(data_dir: Path) -> dict:
    """Check which panorama data is available after download."""
    stats = {
        "scenes_with_rgb": 0,
        "scenes_with_layout": 0,
        "scenes_with_depth": 0,
        "total_panoramas": 0,
    }

    for rgb_path in data_dir.rglob("rgb_rawlight.png"):
        if "panorama" in str(rgb_path):
            stats["total_panoramas"] += 1

    for layout_path in data_dir.rglob("layout.txt"):
        stats["scenes_with_layout"] += 1

    for depth_path in data_dir.rglob("depth.png"):
        if "panorama" in str(depth_path):
            stats["scenes_with_depth"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download Structured3D panorama data for wall-first reconstruction"
    )
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None,
        help="Specific zip indices to download (0-17). Default: downloads ALL."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory. Default: data/Structured3D/"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only check what panorama data is available, don't download."
    )
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else Path(os.getcwd()) / "data" / "Structured3D"

    if args.verify_only:
        logger.info("Verifying existing panorama data...")
        stats = verify_panorama_data(out_dir)
        logger.info(f"Panorama RGB images: {stats['total_panoramas']}")
        logger.info(f"Layout files: {stats['scenes_with_layout']}")
        logger.info(f"Depth maps: {stats['scenes_with_depth']}")
        if stats['total_panoramas'] == 0:
            logger.warning("No panorama data found! Run this script without --verify-only to download.")
        return

    # Determine which zips to download
    if args.indices is not None:
        indices = args.indices
    else:
        # Download all 18 panorama zips (same as perspective_full)
        indices = list(range(18))

    logger.info(f"Will download {len(indices)} panorama zip(s): indices {indices}")
    total_estimated_gb = len(indices) * 10
    logger.info(f"Estimated total download: ~{total_estimated_gb} GB")

    for i in indices:
        url = f"{BASE_URL}/Structured3D_panorama_{i:02d}.zip"
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading panorama zip {i:02d} ({i+1}/{len(indices)})")
        logger.info(f"URL: {url}")
        logger.info(f"{'='*60}")

        try:
            download_and_extract(url, out_dir)
        except Exception as e:
            logger.error(f"Failed to download panorama_{i:02d}.zip: {e}")
            logger.error("Continuing with next zip...")
            continue

    # Verify
    logger.info("\n" + "="*60)
    logger.info("Download complete. Verifying panorama data...")
    stats = verify_panorama_data(out_dir)
    logger.info(f"Panorama RGB images: {stats['total_panoramas']}")
    logger.info(f"Layout files: {stats['scenes_with_layout']}")
    logger.info(f"Depth maps: {stats['scenes_with_depth']}")

    if stats['total_panoramas'] > 0:
        logger.info("SUCCESS: Panorama data is available for wall-first reconstruction!")
        logger.info("You can now run: python scripts/run_structured3d_benchmark.py --mode panoramic")
    else:
        logger.warning("WARNING: No panorama RGB images found after extraction.")
        logger.warning("The zips may use a different naming convention. Check extracted contents.")


if __name__ == "__main__":
    main()
