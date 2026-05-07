#!/bin/bash
# VoxCPM2 Docker build & run script (WSL/Linux)
# Kullanım:
#   ./docker_build_and_run.sh              # build + run
#   ./docker_build_and_run.sh --push       # build + push to Docker Hub
#   ./docker_build_and_run.sh --no-cache   # clean build

set -euo pipefail

IMAGE_NAME="voxcpm2-inference"
DOCKER_HUB_USER="${DOCKER_HUB_USER:-}"  # export DOCKER_HUB_USER=kullaniciadin

echo "=========================================="
echo "  VoxCPM2 Docker Build"
echo "=========================================="

# 1. Build
echo ""
echo "[1/3] Docker image build ediliyor..."
BUILD_ARGS=""
if [[ "$*" == *"--no-cache"* ]]; then
    BUILD_ARGS="--no-cache"
fi
docker build -f Dockerfile.inference -t ${IMAGE_NAME}:latest ${BUILD_ARGS} .
echo "  ✅ Build tamam: ${IMAGE_NAME}:latest"

# 2. Push (opsiyonel)
if [[ "$*" == *"--push"* ]]; then
    if [ -z "$DOCKER_HUB_USER" ]; then
        echo "  ❌ HATA: DOCKER_HUB_USER tanimli degil"
        echo "  export DOCKER_HUB_USER=kullaniciadin"
        exit 1
    fi
    echo ""
    echo "[2/3] Docker Hub'a pushlaniyor..."
    docker tag ${IMAGE_NAME}:latest ${DOCKER_HUB_USER}/${IMAGE_NAME}:latest
    docker push ${DOCKER_HUB_USER}/${IMAGE_NAME}:latest
    echo "  ✅ Push tamam: ${DOCKER_HUB_USER}/${IMAGE_NAME}:latest"
fi

# 3. Run
echo ""
echo "[3/3] Container calistiriliyor..."
echo "  Model: openbmb/VoxCPM2 (HF'dan otomatik indirilir)"
echo ""

docker run --gpus all --rm -it \
    -v /mnt/d/models:/models \
    ${IMAGE_NAME}:latest

echo ""
echo "=========================================="
echo "  Tamamlandi!"
echo "=========================================="
