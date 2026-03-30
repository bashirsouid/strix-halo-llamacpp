#!/usr/bin/env bash
# Llama-3.1-Nemotron-Nano-8B-v1 — bartowski Q6_K_L (~6.85 GB, single file, recommended)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

export LLAMA_CTX_SIZE=131072
export LLAMA_NGL=999
export LLAMA_THREADS=1

HF_REPO="bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF"
DEST="/mnt/data/models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
FILE="${DEST}/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q6_K_L.gguf"

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
    _lib_info "Downloading Nemotron Nano Q6_K_L from HF: ${HF_REPO}"
    mkdir -p "$DEST"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "${HF_REPO}" \
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