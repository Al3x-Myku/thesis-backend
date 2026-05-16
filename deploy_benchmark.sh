#!/bin/bash
set -e

echo "==> Deploying dependencies for Structured3D Automated Benchmark..."
ROOT_DIR=$(pwd)

if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "--> Instalarea pachetelor matematice avansate (Open3D, Chamfer, ICP)..."
pip install -r scripts/requirements_benchmark.txt

echo "==> Setting up Structured3D Ground Truth annotations (bypassing academic lockout)..."
python scripts/download_structured3d_subset.py

echo "--> Rularea suitei de evaluare pe Structured3D..."
python scripts/run_structured3d_benchmark.py

echo "==> Evaluare finalizată! Găsește rezultatele și graficele in benchmark_results/"
