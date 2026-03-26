#!/usr/bin/env bash
# Llama-3.1-Nemotron-Nano-8B-v1 — bartowski Q6_K_L (~6.85 GB, single file, recommended)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

export LLAMA_CTX_SIZE=32768
export LLAMA_NGL=999
export LLAMA_THREADS=16

HF_REPO="bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF"
DEST="/mnt/data/models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
FILE="${DEST}/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q6_K_L.gguf"

if [[ ! -f "$FILE" ]]; then
    _lib_info "Downloading Nemotron Nano Q6_K_L from HF: ${HF_REPO}"
    mkdir -p "$DEST"
    hf download "${HF_REPO}" \
        --include "nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q6_K_L.gguf" \
        --local-dir "$DEST"
    [[ -f "$FILE" ]] || { _lib_fail "Download completed but expected file not found: $FILE"; exit 1; }
    _lib_ok "Download complete."
else
    _lib_ok "Model already present — skipping download."
fi

export MODEL_FLAG="-m"
export MODEL_VALUE="$FILE"

launch_server "nemotron-nano-8b"
wait_for_server