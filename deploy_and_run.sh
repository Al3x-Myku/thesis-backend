#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Get the directory where the script is located
ROOT_DIR=$(pwd)
DFINE_DIR="$ROOT_DIR/D-FINE"

echo "==> Pasul 1: Pregătirea sistemului de operare (Actualizare și instalare pachete de bază)..."
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv ffmpeg libsm6 libxext6 unzip wget cmake build-essential

echo "==> Pasul 2: Crearea și activarea mediului virtual Python..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip

echo "==> Pasul 3: Instalarea PyTorch (CUDA 12.4 pentru arhitectura Blackwell / RTX 5080)..."
# The RTX 5080 (sm_120) strictly requires newer CUDA versions. 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "==> Pasul 4: Pregătirea dataset-ului COCO 2017..."
# Create the directory the D-FINE config expects and take ownership
sudo mkdir -p /data/COCO2017
sudo chown -R $USER:$USER /data/COCO2017

cd /data/COCO2017

# Download Validation Images (~1GB) and Annotations (~241MB) if they don't exist
if [ ! -f "val2017.zip" ]; then
    echo "    -> Descărcare imagini de validare COCO..."
    wget -c http://images.cocodataset.org/zips/val2017.zip
fi
if [ ! -d "val2017" ]; then
    echo "    -> Extragere imagini de validare..."
    unzip -q -n val2017.zip
fi

if [ ! -f "annotations_trainval2017.zip" ]; then
    echo "    -> Descărcare adnotări COCO..."
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi
if [ ! -d "annotations" ]; then
    echo "    -> Extragere adnotări..."
    unzip -q -n annotations_trainval2017.zip
fi

# Create a dummy train2017 folder just in case the dataset builder checks for its existence 
# (avoids downloading the massive 18GB training set for a validation-only benchmark)
mkdir -p train2017 

# Go back to the root directory
cd "$ROOT_DIR"

echo "==> Pasul 5: Instalarea dependențelor adiționale (ONNX, D-FINE etc.)..."
cd "$DFINE_DIR"
pip install -r requirements.txt thop onnx onnxsim onnxruntime-gpu

echo "==> Pasul 6: Setarea variabilelor de mediu și rularea benchmark-ului..."
cd "$ROOT_DIR"
# Assuming your master benchmark script is named run_benchmark.py
python run_benchmark.py

echo "==> 🎉 Implementarea și Benchmark-ul s-au finalizat cu succes! 🎉"