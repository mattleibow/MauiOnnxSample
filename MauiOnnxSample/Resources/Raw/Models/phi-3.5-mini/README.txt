ONNX Model Files Required
=========================

This directory should contain the Phi-3.5-mini-instruct ONNX model files.
Run the download script from the repository root to obtain them:

  On macOS/Linux:
    chmod +x scripts/download-model.sh && ./scripts/download-model.sh

  On Windows (PowerShell):
    .\scripts\download-model.ps1

Required files:
  - model.onnx
  - model.onnx.data
  - tokenizer.json
  - tokenizer_config.json
  - special_tokens_map.json
  - genai_config.json
  - phi3-mini.gguf (optional, not used)

Model source:
  HuggingFace: microsoft/Phi-3.5-mini-instruct-onnx
  Variant: cpu-int4-rtn-block-32-acc-level-4
  Size: ~2.3 GB

Important: Do NOT commit model files to source control (they are gitignored).
