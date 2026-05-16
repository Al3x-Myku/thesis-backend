import os
import zipfile
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_and_extract(url: str, output_dir: Path):
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
            logger.warning(f"Failed to extract {filename}. Corrupted Zip. Deleting and redownloading...")
            zip_path.unlink()
            return False

    if zip_path.exists():
        if try_extract():
            return
            
    logger.info(f"Downloading {filename} from Azure Mirrors... (This is a multi-GB file)")
    urllib.request.urlretrieve(url, zip_path)
    logger.info(f"Successfully downloaded {filename}.")
    
    logger.info(f"Extracting {filename}...")
    try_extract()

if __name__ == "__main__":
    OUT_DIR = Path(os.getcwd()) / "data" / "Structured3D"
    
    # 1. Structure Annotations (Contains Ground Truth Layouts, Corners, and Scales)
    ANNOTATIONS_URL = "https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/Structured3D_annotation_3d.zip"
    download_and_extract(ANNOTATIONS_URL, OUT_DIR)
    
    # 2. 3D Bounding Boxes
    BBOX_URL = "https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/Structured3D_bbox.zip"
    download_and_extract(BBOX_URL, OUT_DIR)
    
    # 3. Perspective Renders Full
    # Iterate and download everything up to index 17, skipping corrupted 09 per issue 30
    for i in range(14, 18):
        url = f"https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/Structured3D_perspective_full_{i:02d}.zip"
        download_and_extract(url, OUT_DIR)
    
    logger.info("Structured3D Datasets have been fully synchronized with the local environment.")
