#!/usr/bin/env bash
# Download Phi-4-mini-instruct ONNX model for MauiOnnxSample
#
# The model can be placed in ONE of two locations:
#   1. Dev path (fastest - no rebuild needed):
#      ~/Documents/phi-4-mini/
#      ModelService checks this first at runtime.
#
#   2. Embedded as MauiAsset (production path, requires rebuild):
#      MauiOnnxSample/Resources/Raw/Models/phi-4-mini/
#      These files are bundled in the app and extracted to AppDataDirectory on first run.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Default: dev path (no rebuild required)
DEV_PATH="$HOME/Documents/phi-4-mini"
ASSET_PATH="$REPO_ROOT/MauiOnnxSample/Resources/Raw/Models/phi-4-mini"

TARGET="${1:-dev}"
case "$TARGET" in
  dev)    DEST="$DEV_PATH" ;;
  assets) DEST="$ASSET_PATH" ;;
  *)      echo "Usage: $0 [dev|assets]"; exit 1 ;;
esac

mkdir -p "$DEST"
echo "Downloading Phi-4-mini-instruct ONNX model (cpu-int4-rtn-block-32)..."
echo "Destination: $DEST"
echo ""

REPO="microsoft/Phi-4-mini-instruct-onnx"
SUBFOLDER="cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
TMP="/tmp/phi4-onnx-download"

if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download "$REPO" \
        --include "${SUBFOLDER}/*" \
        --local-dir "$TMP"
elif command -v python3 &> /dev/null; then
    echo "Using Python huggingface_hub..."
    python3 - << PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$REPO',
    allow_patterns=['${SUBFOLDER}/*'],
    local_dir='$TMP'
)
PYEOF
else
    echo "ERROR: Neither huggingface-cli nor python3 found."
    echo "Install with: pip install huggingface_hub"
    echo ""
    echo "Alternatively, manually download from:"
    echo "  https://huggingface.co/$REPO/tree/main/$SUBFOLDER"
    echo "  to: $DEST"
    exit 1
fi

echo ""
echo "Copying files to $DEST ..."
cp "$TMP/$SUBFOLDER/"* "$DEST/"
echo ""
echo "Model downloaded to: $DEST"

if [ "$TARGET" = "dev" ]; then
    echo ""
    echo "Dev path ready. ModelService will use this location automatically at runtime."
    echo "No rebuild needed."
else
    echo ""
    echo "Assets ready. Rebuild the app to bundle the model (adds ~2.8 GB to the app)."
    echo "  dotnet build MauiOnnxSample -f net11.0-maccatalyst -c Debug"
fi
