#!/usr/bin/env bash
# Qwen2.5-Coder-7B-Instruct — coding assistant (Q8_0)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

export LLAMA_CTX_SIZE=32768
export LLAMA_NGL=999
export LLAMA_THREADS=16

DEST="/mnt/data/models/qwen/Qwen2.5-Coder-7B-Instruct"
FILE="${DEST}/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf"

if [[ ! -f "$FILE" ]]; then
    _lib_info "Downloading Qwen2.5-Coder-7B-Instruct Q8_0..."
    mkdir -p "$DEST"
    hf download bartowski/Qwen2.5-Coder-7B-Instruct-GGUF \
        --include "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf" \
        --local-dir "$DEST"
fi

export MODEL_FLAG="-m"
export MODEL_VALUE="$FILE"
launch_server "qwen2.5-coder-7b"
wait_for_server