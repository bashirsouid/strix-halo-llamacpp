#!/usr/bin/env bash
# Nemotron-3-Super 120B-A12B MoE (unsloth UD-Q4_K_M, 3-part GGUF)
# Auto-downloads all 3 shards if any are missing, then launches via llama-server.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

# Reset any speculative-decoding state from previous runs
clear_draft_config

# 120B MoE — 12B active params per token, so generation is fast.
# KV cache still scales with total layers so keep ctx moderate.
export LLAMA_CTX_SIZE=16384
export LLAMA_NGL=999
export LLAMA_THREADS=1

HF_REPO="unsloth/Nemotron-3-Super-120B-A12B-GGUF"
QUANT="UD-Q4_K_M"
DEST_DIR="/mnt/data/models/nvidia/nemotron-3-super/UD-Q4_K_M"
PART1="${DEST_DIR}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf"
PART2="${DEST_DIR}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00002-of-00003.gguf"
PART3="${DEST_DIR}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00003-of-00003.gguf"

# ── bench_all.py check mode ───────────────────────────────────────────────────
# When BENCH_MODE=check, only verify files are present — no download, no server.
# Exit 0 = files ready.  Exit 2 = not downloaded, skip this model.
if [[ "${BENCH_MODE:-}" == "check" ]]; then
    if [[ -f "$PART1" ]]; then
        exit 0
    else
        exit 2
    fi
fi
# ── end of BENCH_MODE=check logic ─────────────────────────────────────────────

# ── Check / download missing shards ──────────────────────────────────
MISSING=0
[[ ! -f "$PART1" ]] && { _lib_warn "Missing shard 1/3"; MISSING=1; }
[[ ! -f "$PART2" ]] && { _lib_warn "Missing shard 2/3"; MISSING=1; }
[[ ! -f "$PART3" ]] && { _lib_warn "Missing shard 3/3"; MISSING=1; }

if (( MISSING )); then
    _lib_info "Downloading missing shards from HF: ${HF_REPO}"
    _lib_info "This will take a while (~63 GB total for all 3 parts)..."
    mkdir -p "$DEST_DIR"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "${HF_REPO}" \
        --include "${QUANT}/*.gguf" \
        --local-dir "/mnt/data/models/nvidia/nemotron-3-super/"
    _lib_ok "All shards downloaded."
else
    _lib_ok "All 3 shards present locally — skipping download."
fi

# -- Optional: Download draft model for speculative decoding (Nemotron Nano)
DRAFT_HF_REPO="bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF"
DRAFT_DEST="/mnt/data/models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
DRAFT_FILE="nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q6_K_L.gguf"
DRAFT_PATH="${DRAFT_DEST}/${DRAFT_FILE}"

if [[ ! -f "$DRAFT_PATH" ]]; then
    _lib_info "Downloading draft model for speculative decoding: ${DRAFT_HF_REPO}"
    mkdir -p "$DRAFT_DEST"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "${DRAFT_HF_REPO}" \
        --include "${DRAFT_FILE}" \
        --local-dir "$DRAFT_DEST"
fi

if [[ -f "$DRAFT_PATH" ]]; then
    export DRAFT_MODEL_PATH="$DRAFT_PATH"
    export DRAFT_MAX="8"
    export DRAFT_MIN="2"
    export LLAMA_COMPOSE_FILE="docker-compose.spec.yml"
    _lib_ok "Draft model ready for speculative decoding: ${DRAFT_PATH}"
else
    _lib_warn "Draft model not found after download — running without speculative decoding"
    clear_draft_config
fi
# -- END

# ── Point llama-server at part 1; it auto-chains 2 and 3 ─────────────
export MODEL_FLAG="-m"
export MODEL_VALUE="$PART1"

launch_server "nemotron-3-super-120b"
wait_for_server