#!/bin/bash
# ==============================================
# VoxCPM2 - GitHub Release Publisher
# Kullanım:  bash publish_release.sh v1.0.0
# ==============================================
set -e

cd "$(dirname "$0")"

if [ -z "$1" ]; then
    echo "Kullanım: bash publish_release.sh <tag>"
    echo "Örnek:    bash publish_release.sh v1.0.0"
    exit 1
fi

TAG="$1"
REPO="kstbhdr/nanovllm-voxcpm-official"
WHEEL_FILE="voxcpm2_wheels.tar.gz"

# 1. Wheels'in hazır olduğundan emin ol
if [ ! -f "$WHEEL_FILE" ]; then
    echo "⏳ Wheels hazırlanıyor..."
    bash build_wheels.sh
fi

# 2. Git tag
echo "🏷️  Tag oluşturuluyor: $TAG"
git tag -a "$TAG" -m "VoxCPM2 Colab Wheels $TAG"

# 3. Push tag
echo "📤 Tag pushlanıyor..."
git push origin "$TAG"

# 4. GitHub Release oluştur (gh CLI gerekli)
if command -v gh &> /dev/null; then
    echo "📦 GitHub Release oluşturuluyor..."
    gh release create "$TAG" \
        --title "VoxCPM2 Colab Wheels $TAG" \
        --notes "Colab'da hızlı kurulum için önceden derlenmiş wheel'ler.

Kullanım:
1. Colab'da \`USE_PREBUILT_WHEELS = True\` yap
2. Notebook'u çalıştır" \
        "$WHEEL_FILE"
    echo "✅ Release yayında!"
else
    echo "⚠️  gh CLI bulunamadı. Elle release oluştur:"
    echo "   https://github.com/$REPO/releases/new?tag=$TAG"
    echo "   Dosyayı yükle: $WHEEL_FILE"
fi

echo ""
echo "✅ Tamamlandı!"
echo "Colab notebook: colab_voxcpm2_inference.ipynb"
