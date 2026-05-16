#!/usr/bin/env python3
import os
import tarfile
from huggingface_hub import hf_hub_download

print("==> Downloading 3D-FRONT-TEST Subset...")
try:
    render_tar = hf_hub_download(repo_id="huanngzh/3D-Front", filename="3D-FRONT-TEST-RENDER.tar.gz", repo_type="dataset")
    print(f"Render downloaded to: {render_tar}")
    
    scene_tar = hf_hub_download(repo_id="huanngzh/3D-Front", filename="3D-FRONT-TEST-SCENE.tar.gz", repo_type="dataset")
    print(f"Scene downloaded to: {scene_tar}")
    
    print("==> Extracting to data/3D-FRONT...")
    with tarfile.open(render_tar, "r:gz") as t:
        t.extractall(path="data/3D-FRONT/")
        
    with tarfile.open(scene_tar, "r:gz") as t:
        t.extractall(path="data/3D-FRONT/")
        
    print("Dataset extracted successfully.")
except Exception as e:
    print(f"Error fetching dataset subset: {e}")
