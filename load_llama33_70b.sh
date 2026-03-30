#!/usr/bin/env bash
# Llama-3.3-70B-Instruct — bartowski Q6_K_L split (~58.4 GB, recommended)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

# Reset any speculative-decoding state from previous runs
clear_draft_config

export LLAMA_CTX_SIZE=16384
export LLAMA_NGL=999
export LLAMA_THREADS=1

HF_REPO="bartowski/Llama-3.3-70B-Instruct-GGUF"
BASE_DIR="/mnt/data/models/meta/Llama-3.3-70B-Instruct"
SHARD_DIR="${BASE_DIR}/Llama-3.3-70B-Instruct-Q6_K_L"

# ── bench_all.py check mode ───────────────────────────────────────────────────
# When BENCH_MODE=check, only verify files are present — no download, no server.
# Exit 0 = files ready.  Exit 2 = not downloaded, skip this model.
if [[ "${BENCH_MODE:-}" == "check" ]]; then
    if [[ -n "$(find "$SHARD_DIR" -name "*.gguf" 2>/dev/null | head -1)" ]]; then
        exit 0
    else
        exit 2
    fi
fi
# ── end of BENCH_MODE=check logic ─────────────────────────────────────────────

if [[ -z "$(find "$SHARD_DIR" -name "*.gguf" 2>/dev/null | head -1)" ]]; then
    _lib_info "Downloading Llama-3.3-70B Q6_K_L shards from HF: ${HF_REPO}"
    _lib_info "This will take a while (~58.4 GB total)..."
    mkdir -p "$SHARD_DIR"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "${HF_REPO}" \
        --include "Llama-3.3-70B-Instruct-Q6_K_L/*" \
        --local-dir "$BASE_DIR"
    [[ -n "$(find "$SHARD_DIR" -name "*.gguf" 2>/dev/null | head -1)" ]] \
        || { _lib_fail "Download completed but no shards found in: $SHARD_DIR"; exit 1; }
    _lib_ok "All shards downloaded."
else
    _lib_ok "Shards already present — skipping download."
fi

PART1=$(find "$SHARD_DIR" -name "*-00001-of-*.gguf" | head -1)
[[ -n "$PART1" ]] || { _lib_fail "Cannot find shard 1 in $SHARD_DIR"; exit 1; }

export MODEL_FLAG="-m"
export MODEL_VALUE="$PART1"

launch_server "llama-3.3-70b"
wait_for_server