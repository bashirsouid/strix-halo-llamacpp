#!/usr/bin/env bash
# Qwen2.5-Coder-7B-Instruct — coding assistant (Q8_0)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

# Reset any speculative-decoding state from previous runs
clear_draft_config

export LLAMA_CTX_SIZE=32768
export LLAMA_NGL=999
export LLAMA_THREADS=1

DEST="/mnt/data/models/qwen/Qwen2.5-Coder-7B-Instruct"
FILE="${DEST}/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf"

# ── bench_all.py check mode ───────────────────────────────────────────────────
# When BENCH_MODE=check, only verify files are present — no download, no server.
# Exit 0 = files ready.  Exit 2 = not downloaded, skip this model.
if [[ "${BENCH_MODE:-}" == "check" ]]; then
    if [[ -f "$FILE" ]]; then
        exit 0
    else
        exit 2
    fi
fi
# ── end of BENCH_MODE=check logic ─────────────────────────────────────────────

if [[ ! -f "$FILE" ]]; then
    _lib_info "Downloading Qwen2.5-Coder-7B-Instruct Q8_0..."
    mkdir -p "$DEST"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download bartowski/Qwen2.5-Coder-7B-Instruct-GGUF \
        --include "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf" \
        --local-dir "$DEST"
fi

export MODEL_FLAG="-m"
export MODEL_VALUE="$FILE"
launch_server "qwen2.5-coder-7b"
wait_for_server