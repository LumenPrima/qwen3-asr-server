#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# Entrypoint: download model weights on first run, then start server.
# -------------------------------------------------------------------

MODEL_DIR="${MODEL_PATH:-/model}"
ASR_REPO="${ASR_MODEL_REPO:-AuggieActual/qwen3-asr-p25-0.6B}"
ALIGNER_DIR="${ALIGNER_PATH:-/aligner}"
ALIGNER_REPO="${ALIGNER_MODEL_REPO:-Qwen/Qwen3-ForcedAligner-0.6B}"
BACKEND="${INFERENCE_BACKEND:-python}"

# Download ASR model if not present
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Downloading ASR model: $ASR_REPO -> $MODEL_DIR"
    huggingface-cli download "$ASR_REPO" --local-dir "$MODEL_DIR"
    echo "ASR model downloaded."
else
    echo "ASR model found at $MODEL_DIR"
fi

# Download ForcedAligner if Python backend and not present
if [ "$BACKEND" = "python" ] && [ ! -f "$ALIGNER_DIR/model.safetensors" ]; then
    echo "Downloading ForcedAligner: $ALIGNER_REPO -> $ALIGNER_DIR"
    huggingface-cli download "$ALIGNER_REPO" --local-dir "$ALIGNER_DIR"
    echo "ForcedAligner downloaded."
elif [ "$BACKEND" = "python" ]; then
    echo "ForcedAligner found at $ALIGNER_DIR"
fi

echo "Starting server (backend=$BACKEND)..."
exec python server.py "$@"
