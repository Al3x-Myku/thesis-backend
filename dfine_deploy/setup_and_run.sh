#!/bin/bash

# Oprim execuția la prima eroare
set -e

# Definim culorile pentru mesaje
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # Fără culoare

echo -e "${YELLOW}>>> [1/4] Actualizare sistem și instalare dependențe critice...${NC}"
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv ffmpeg libsm6 libxext6 unzip wget cmake build-essential
echo -e "${GREEN}✔ Dependențele de sistem au fost instalate.${NC}"

echo -e "${YELLOW}>>> [2/4] Configurare Virtual Environment...${NC}"
# Creăm venv doar dacă nu există deja pentru a evita rescrierea lui accidentală la rulări succesive
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}✔ Virtual Environment (venv) a fost creat și activat.${NC}"

echo -e "${YELLOW}>>> [3/4] Instalare PyTorch (CUDA 11.8) și pachete ML...${NC}"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnx onnxsim onnxruntime-gpu pycocotools matplotlib pyyaml scipy

if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}>>> Găsit requirements.txt! Instalez...${NC}"
    pip install -r requirements.txt
fi
echo -e "${GREEN}✔ Toate pachetele Python au fost instalate cu succes.${NC}"

echo -e "${YELLOW}>>> [4/4] Configurare variabile de mediu și lansare Benchmark...${NC}"
export USE_LIBUV=0
export CUDA_VISIBLE_DEVICES=0

echo -e "${YELLOW}>>> Rulez python scripts/run_benchmark.py...${NC}"
python scripts/run_benchmark.py

echo -e "${GREEN}✔ Workflow finalizat cu succes!${NC}"
