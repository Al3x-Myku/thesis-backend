#!/bin/bash
# Replicable Build Script for Hunyuan3D-2 Custom CUDA Operations
# This ensures forward-compatibility for Blackwell (RTX 5000 series) GPUs.

set -e # Exit on error

echo "[INFO] Setting Compiler to gcc-12 for CUDA 12.0 compatibility..."
export CC=gcc-12
export CXX=g++-12

echo "[INFO] Forcing PTX embedding for forward compatibility (e.g. RTX 5080)..."
export TORCH_CUDA_ARCH_LIST="9.0+PTX"

echo "[INFO] Cleaning old broken builds..."
cd Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
rm -rf build temp *.egg-info dist
echo "[INFO] Compiling Custom Rasterizer..."
python3 setup.py install

echo "[INFO] Cleaning old broken builds..."
cd ../differentiable_renderer
rm -rf build temp *.egg-info dist
echo "[INFO] Compiling Differentiable Renderer..."
python3 setup.py install

echo "[INFO] ✅ Custom 3D CUDA operations compiled successfully!"
