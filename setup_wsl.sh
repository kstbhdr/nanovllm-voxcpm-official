#!/bin/bash
# VoxCPM2 - WSL'de tek komutta kurulum ve calistirma
# Kullanim: bash setup_wsl.sh
set -e
echo "=== VoxCPM2 WSL Kurulum ==="
cd "$(dirname "$0")"

# Sanal ortam (Python >=3.10,<3.13 gerekli)
PYTHON=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || echo "python3")
echo "Kullanilan Python: $($PYTHON --version)"
[ ! -d venv ] && $PYTHON -m venv venv
source venv/bin/activate

# Bagimliliklar
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
pip install flash-attn --no-build-isolation -q
pip install -r requirements.txt -q
pip install -e . --no-deps -q

# Calistir
python voxcpm2_run.py
