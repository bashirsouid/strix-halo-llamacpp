#!/usr/bin/env bash
# Devstral-Small-2-24B-Instruct — Unsloth Q8_0 (~25 GB, single file)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

export LLAMA_CTX_SIZE=262144
export LLAMA_NGL=999
export LLAMA_THREADS=16

HF_REPO="unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF"
DEST="/mnt/data/models/devstral/Devstral-Small-2-24B-Instruct-2512"
FNAME="Devstral-Small-2-24B-Instruct-2512-Q8_0.gguf"
FILE="${DEST}/${FNAME}"

if [[ ! -f "$FILE" ]]; then
    _lib_info "Downloading Devstral-Small-2-24B Q8_0 from HF: ${HF_REPO}"
    mkdir -p "$DEST"
    hf download "${HF_REPO}" \
        --include "${FNAME}" \
        --local-dir "${DEST}"
    [[ -f "$FILE" ]] || { _lib_fail "Download completed but expected file not found: $FILE"; exit 1; }
    _lib_ok "Download complete."
else
    _lib_ok "Model already present — skipping download."
fi

export LLAMA_CHAT_TEMPLATE_FILE="/mnt/data/models/llm-templates/devstral-small-2-instruct.jinja"

export MODEL_FLAG="-m"
export MODEL_VALUE="$FILE"

launch_server "devstral-small-2-24b"
wait_for_server