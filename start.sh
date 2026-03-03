#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODEL_PATH="${1:-$SCRIPT_DIR/qwen3-asr-p25-0.6B}"
PORT="${2:-8765}"

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
fi

# Install deps if needed
if ! "$VENV_DIR/bin/python3" -c "import qwen_asr" 2>/dev/null; then
    echo "Installing dependencies..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

echo "Starting Qwen3-ASR P25 server..."
echo "  Model: $MODEL_PATH"
echo "  Port:  $PORT"

MODEL_PATH="$MODEL_PATH" PORT="$PORT" exec "$VENV_DIR/bin/python3" "$SCRIPT_DIR/server.py"
