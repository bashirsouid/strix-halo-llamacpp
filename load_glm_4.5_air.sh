#!/usr/bin/env bash
# GLM-4.5-Air — MoE 106B (32B active), unsloth IQ4_XS (~60.5 GB, 2 parts)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

export LLAMA_CTX_SIZE=32768
export LLAMA_NGL=999
export LLAMA_THREADS=1

HF_REPO="unsloth/GLM-4.5-Air-GGUF"
BASE_DIR="/mnt/data/models/zai/GLM-4.5-Air"
SHARD_DIR="${BASE_DIR}/IQ4_XS"
PART1="${SHARD_DIR}/GLM-4.5-Air-IQ4_XS-00001-of-00002.gguf"
PART2="${SHARD_DIR}/GLM-4.5-Air-IQ4_XS-00002-of-00002.gguf"

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

MISSING=0
[[ ! -f "$PART1" ]] && { _lib_warn "Missing shard 1/2"; MISSING=1; }
[[ ! -f "$PART2" ]] && { _lib_warn "Missing shard 2/2"; MISSING=1; }

if (( MISSING )); then
    _lib_info "Downloading GLM-4.5-Air IQ4_XS from HF: ${HF_REPO}"
    _lib_info "This will take a while (~60.5 GB total)..."
    mkdir -p "$SHARD_DIR"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "${HF_REPO}" \
        --include "IQ4_XS/*.gguf" \
        --local-dir "$BASE_DIR"
    [[ -f "$PART1" ]] || { _lib_fail "Download completed but shard 1 not found: $PART1"; exit 1; }
    _lib_ok "All shards downloaded."
else
    _lib_ok "Both shards present — skipping download."
fi

export LLAMA_CHAT_TEMPLATE_FILE="/mnt/data/models/llm-templates/glm4.5_chat_template.jinja"

export MODEL_FLAG="-m"
export MODEL_VALUE="$PART1"

launch_server "glm-4.5-air"
wait_for_server