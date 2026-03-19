#!/usr/bin/env bash
# Download Phi-3.5-mini-instruct ONNX model for MauiOnnxSample

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$REPO_ROOT/MauiOnnxSample/Resources/Raw/Models/phi-3.5-mini"

echo "Downloading Phi-3.5-mini-instruct ONNX model..."
echo "Destination: $MODEL_DIR"
echo ""

# Check for huggingface-cli
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download microsoft/Phi-3.5-mini-instruct-onnx \
        --include "cpu-int4-rtn-block-32-acc-level-4/*" \
        --local-dir /tmp/phi35-onnx

    cp /tmp/phi35-onnx/cpu-int4-rtn-block-32-acc-level-4/* "$MODEL_DIR/"
elif command -v python3 &> /dev/null; then
    echo "Using Python huggingface_hub..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='microsoft/Phi-3.5-mini-instruct-onnx',
    allow_patterns=['cpu-int4-rtn-block-32-acc-level-4/*'],
    local_dir='/tmp/phi35-onnx'
)
import shutil, os
src = '/tmp/phi35-onnx/cpu-int4-rtn-block-32-acc-level-4'
dst = '$MODEL_DIR'
for f in os.listdir(src):
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
"
else
    echo "ERROR: Neither huggingface-cli nor python3 found."
    echo "Install with: pip install huggingface_hub"
    echo ""
    echo "Alternatively, manually download from:"
    echo "  https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx/tree/main/cpu-int4-rtn-block-32-acc-level-4"
    echo "  to: $MODEL_DIR"
    exit 1
fi

echo ""
echo "Model downloaded successfully to: $MODEL_DIR"
echo "You can now build and run the MauiOnnxSample app."
