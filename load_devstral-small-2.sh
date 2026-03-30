#!/usr/bin/env bash
# Devstral-Small-2-24B-Instruct — Unsloth Q8_0 (~25 GB, single file)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

# Reset any speculative-decoding state from previous runs
clear_draft_config

export LLAMA_CTX_SIZE=262144
export LLAMA_NGL=999
export LLAMA_THREADS=1

HF_REPO="unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF"
DEST="/mnt/data/models/devstral/Devstral-Small-2-24B-Instruct-2512"
FNAME="Devstral-Small-2-24B-Instruct-2512-Q8_0.gguf"
FILE="${DEST}/${FNAME}"

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
    _lib_info "Downloading Devstral-Small-2-24B Q8_0 from HF: ${HF_REPO}"
    mkdir -p "$DEST"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "${HF_REPO}" \
        --include "${FNAME}" \
        --local-dir "${DEST}"
    [[ -f "$FILE" ]] || { _lib_fail "Download completed but expected file not found: $FILE"; exit 1; }
    _lib_ok "Download complete."
else
    _lib_ok "Model already present — skipping download."
fi

export LLAMA_CHAT_TEMPLATE_FILE="/mnt/data/models/llm-templates/devstral-small-2-instruct.jinja"

# -- Optional: Download a small draft model for speculative decoding. Not required, but can improve latency if you have it.
DRAFT_HF_REPO="bartowski/alamios_Mistral-Small-3.1-DRAFT-0.5B-GGUF"
DRAFT_DEST="/mnt/data/models/bartowski/alamios_Mistral-Small-3.1-DRAFT-0.5B-GGUF"
DRAFT_FILE="alamios_Mistral-Small-3.1-DRAFT-0.5B-Q4_K_M.gguf"
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
# -- End of optional draft model logic --

export MODEL_FLAG="-m"
export MODEL_VALUE="$FILE"

launch_server "devstral-small-2-24b"
wait_for_server