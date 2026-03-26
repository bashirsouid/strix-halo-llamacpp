#!/usr/bin/env bash
# Nemotron-3-Super 120B-A12B MoE (unsloth UD-Q4_K_M, 3-part GGUF)
# Auto-downloads all 3 shards if any are missing, then launches via llama-server.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

# 120B MoE — 12B active params per token, so generation is fast.
# KV cache still scales with total layers so keep ctx moderate.
export LLAMA_CTX_SIZE=20480 # old: 16384
export LLAMA_NGL=999
export LLAMA_THREADS=16

HF_REPO="unsloth/Nemotron-3-Super-120B-A12B-GGUF"
QUANT="UD-Q4_K_M"
DEST_DIR="/mnt/data/models/nvidia/nemotron-3-super/UD-Q4_K_M"
PART1="${DEST_DIR}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf"
PART2="${DEST_DIR}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00002-of-00003.gguf"
PART3="${DEST_DIR}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00003-of-00003.gguf"

# ── Check / download missing shards ──────────────────────────────────
MISSING=0
[[ ! -f "$PART1" ]] && { _lib_warn "Missing shard 1/3"; MISSING=1; }
[[ ! -f "$PART2" ]] && { _lib_warn "Missing shard 2/3"; MISSING=1; }
[[ ! -f "$PART3" ]] && { _lib_warn "Missing shard 3/3"; MISSING=1; }

if (( MISSING )); then
    _lib_info "Downloading missing shards from HF: ${HF_REPO}"
    _lib_info "This will take a while (~63 GB total for all 3 parts)..."
    mkdir -p "$DEST_DIR"
    hf download "${HF_REPO}" \
        --include "${QUANT}/*.gguf" \
        --local-dir "/mnt/data/models/nvidia/nemotron-3-super/"
    _lib_ok "All shards downloaded."
else
    _lib_ok "All 3 shards present locally — skipping download."
fi

# ── Point llama-server at part 1; it auto-chains 2 and 3 ─────────────
export MODEL_FLAG="-m"
export MODEL_VALUE="$PART1"

launch_server "nemotron-3-super-120b"
wait_for_server