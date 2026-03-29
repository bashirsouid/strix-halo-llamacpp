#!/usr/bin/env bash
# Mistral-Small-4-119B-2603 — MoE 119B (6.5B active), unsloth UD-Q3_K_M (~57 GB)
# Auto-downloads all 3 shards if missing, then launches via llama-server.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

# MoE 119B, 6.5B active params — KV cache scales with total layers, keep ctx moderate.
export LLAMA_CTX_SIZE=32768
export LLAMA_NGL=999
export LLAMA_THREADS=16

HF_REPO="unsloth/Mistral-Small-4-119B-2603-GGUF"
QUANT="UD-Q3_K_M"
DEST_DIR="/mnt/data/models/unsloth/Mistral-Small-4-119B-2603-GGUF/UD-Q3_K_M"
PART1="${DEST_DIR}/Mistral-Small-4-119B-2603-UD-Q3_K_M-00001-of-00003.gguf"
PART2="${DEST_DIR}/Mistral-Small-4-119B-2603-UD-Q3_K_M-00002-of-00003.gguf"
PART3="${DEST_DIR}/Mistral-Small-4-119B-2603-UD-Q3_K_M-00003-of-00003.gguf"

# ── Check / download missing shards ──────────────────────────────────
MISSING=0
[[ ! -f "$PART1" ]] && { _lib_warn "Missing shard 1/3"; MISSING=1; }
[[ ! -f "$PART2" ]] && { _lib_warn "Missing shard 2/3"; MISSING=1; }
[[ ! -f "$PART3" ]] && { _lib_warn "Missing shard 3/3"; MISSING=1; }

if (( MISSING )); then
    _lib_info "Downloading missing shards from HF: ${HF_REPO}"
    _lib_info "This will take a while (~57 GB total for all 3 parts)..."
    mkdir -p "$DEST_DIR"
    hf download "${HF_REPO}" \
        --include "${QUANT}/*.gguf" \
        --local-dir "/mnt/data/models/unsloth/Mistral-Small-4-119B-2603-GGUF/"
    _lib_ok "All shards downloaded."
else
    _lib_ok "All 3 shards present locally — skipping download."
fi

# ── Point llama-server at part 1; it auto-chains 2 and 3 ─────────────
export MODEL_FLAG="-m"
export MODEL_VALUE="$PART1"

launch_server "mistral-small-4-119b"
wait_for_server