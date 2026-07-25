#!/usr/bin/env bash
# install_python.sh – create a virtualenv and install PyTorch (CUDA 12.6) + Diffusers
# Safe to run as normal user (no sudo needed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

echo "==> Creating virtual environment in ${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip / setuptools / wheel ..."
pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch with CUDA 12.6 support (includes sm_52 / Maxwell kernels) ..."
# Official PyTorch cu126 wheels still support compute capability 5.2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "==> Installing Diffusers + supporting libraries ..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo " Python environment ready."
echo ""
echo " Activate it with:"
echo "   source ${VENV_DIR}/bin/activate"
echo ""
echo " Then run the smoke test:"
echo "   python smoke_test.py"
echo "============================================================"
