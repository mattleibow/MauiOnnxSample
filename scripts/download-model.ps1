# Download Phi-4-mini-instruct ONNX model for MauiOnnxSample

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$ModelDir = Join-Path $RepoRoot "MauiOnnxSample\Resources\Raw\Models\phi-4-mini"

Write-Host "Downloading Phi-4-mini-instruct ONNX model..."
Write-Host "Destination: $ModelDir"
Write-Host ""

if (Get-Command "huggingface-cli" -ErrorAction SilentlyContinue) {
    Write-Host "Using huggingface-cli..."
    huggingface-cli download microsoft/Phi-4-mini-instruct-onnx `
        --include "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*" `
        --local-dir "$env:TEMP\phi4-onnx"
    
    $src = Join-Path "$env:TEMP\phi4-onnx" "cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4"
    Copy-Item "$src\*" -Destination $ModelDir -Force
} elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
    Write-Host "Using Python huggingface_hub..."
    python -c @"
from huggingface_hub import snapshot_download
import shutil, os, tempfile
tmp = tempfile.mkdtemp()
snapshot_download(
    repo_id='microsoft/Phi-4-mini-instruct-onnx',
    allow_patterns=['cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*'],
    local_dir=tmp
)
src = os.path.join(tmp, 'cpu_and_mobile', 'cpu-int4-rtn-block-32-acc-level-4')
dst = r'$ModelDir'
for f in os.listdir(src):
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
print('Done!')
"@
} else {
    Write-Error "Neither huggingface-cli nor python found. Install with: pip install huggingface_hub"
    Write-Host "Alternatively, manually download from:"
    Write-Host "  https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    Write-Host "  to: $ModelDir"
    exit 1
}

Write-Host ""
Write-Host "Model downloaded successfully to: $ModelDir"
Write-Host "You can now build and run the MauiOnnxSample app."
