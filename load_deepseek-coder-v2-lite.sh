#!/usr/bin/env bash
# DeepSeek-Coder-V2-Lite-Instruct — bartowski Q8_0_L (~17.09 GB, single file)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

# Reset any speculative-decoding state from previous runs
clear_draft_config

export LLAMA_CTX_SIZE=131072
export LLAMA_NGL=999
export LLAMA_THREADS=1

HF_REPO="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF"
DEST="/mnt/data/models/deepseek/DeepSeek-Coder-V2-Lite-Instruct"
FILE="${DEST}/DeepSeek-Coder-V2-Lite-Instruct-Q8_0_L.gguf"

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
    _lib_info "Downloading DeepSeek-Coder-V2-Lite Q8_0_L from HF: ${HF_REPO}"
    mkdir -p "$DEST"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "${HF_REPO}" \
        --include "DeepSeek-Coder-V2-Lite-Instruct-Q8_0_L.gguf" \
        --local-dir "$DEST"
    [[ -f "$FILE" ]] || { _lib_fail "Download completed but expected file not found: $FILE"; exit 1; }
    _lib_ok "Download complete."
else
    _lib_ok "Model already present — skipping download."
fi

export LLAMA_CHAT_TEMPLATE_FILE="/mnt/data/models/llm-templates/deepseek-coder-v2-lite.jinja"

export MODEL_FLAG="-m"
export MODEL_VALUE="$FILE"

launch_server "deepseek-coder-v2-lite"
wait_for_server