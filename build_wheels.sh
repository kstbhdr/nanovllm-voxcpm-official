#!/bin/bash
# ======================================================
# VoxCPM2 - Wheel Builder
# WSL'de çalıştır:  bash build_wheels.sh
# Çıktı: ./dist/nanovllm_voxcpm-*.whl  (Colab'a upload et)
# ======================================================
set -e

echo "=========================================="
echo "  VoxCPM2 Wheel Builder"
echo "=========================================="

cd "$(dirname "$0")"

# 1. Kendi paketimizin wheel'ini üret
echo ""
echo "[1/3] 🏗️  nanovllm_voxcpm wheel'i üretiliyor..."
pip install build wheel --quiet
python -m build --wheel --outdir dist/ .
echo "   ✅  dist/ klasöründe:"
ls -lh dist/*.whl 2>/dev/null || echo "   (wheel oluşmadı, kontrol et)"

# 2. Zor/native bağımlılıkların wheel'lerini indir
echo ""
echo "[2/3] 📥 Native bağımlılık wheelleri indiriliyor..."
mkdir -p wheels
pip download \
  --only-binary=:all: \
  --dest wheels/ \
  soundfile>=0.13.1 \
  librosa \
  pydantic \
  xxhash \
  tqdm \
  numpy \
  torchcodec \
  psutil \
  2>&1 | tail -5
echo "   ✅  wheels/ klasöründe $(ls wheels/*.whl 2>/dev/null | wc -l) dosya"

# 3. Toplu çıktı
echo ""
echo "[3/3] 📦 Özet"
echo "   dist/nanovllm_voxcpm-*.whl  → Colab'a upload et"
echo "   wheels/*.whl                → Opsiyonel, yedek"
echo ""
echo "Colab'da kurulum:"
echo "  from google.colab import files"
echo "  uploaded = files.upload()  # .whl dosyasını seç"
echo "  !pip install nanovllm_voxcpm-*.whl"
echo "  !pip install wheels/*.whl"
echo "=========================================="
