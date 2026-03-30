#!/usr/bin/env bash
# lib.sh — shared helpers for strix-llamacpp scripts
# Source this at the top of every script: source "$SCRIPT_DIR/lib.sh"
# =============================================================================

# ── Colour codes ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; YLW='\033[1;33m'; GRN='\033[0;32m'
CYN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

# ── Logging helpers ───────────────────────────────────────────────────────────
_lib_info() { echo -e "${CYN}[INFO]${NC}  $*"; }
_lib_ok()   { echo -e "${GRN}[ OK ]${NC}  $*"; }
_lib_warn() { echo -e "${YLW}[WARN]${NC}  $*"; }
_lib_fail() { echo -e "${RED}[FAIL]${NC}  $*"; }

_SEARCH_DIRS=()   # callers populate this; default is empty (find_gguf will skip search)

# ── Model search directories ──────────────────────────────────────────────────
# These are checked in order before attempting any download.
# Add your own paths to EXTRA_MODEL_SEARCH_DIRS in config.env (colon-separated).
#
# Covers: llama-server -hf cache, HF CLI default cache, ComfyUI models,
#         Ollama blobs, text-gen-webui, LM Studio, Open WebUI, manual downloads.
_build_search_dirs() {
  local hf_home="${HF_HOME:-${HOME}/.cache/huggingface}"
  local hf_hub="${hf_home}/hub"
  local llama_cache="${HOST_LLAMA_CACHE:-/mnt/data/llama.cpp-cache}"

  _SEARCH_DIRS=(
    # llama-server -hf download cache (our primary store)
    "${llama_cache}/hf/hub"
    "${llama_cache}"

    # HF CLI default cache (host-side)
    "${hf_hub}"
    "${HOME}/.cache/huggingface/hub"

    # Common model root
    "${MODELS_DIR:-/mnt/data/models}"

    # ComfyUI — frequently has GGUF downloads
    "/mnt/data/ComfyUI/models/llm"
    "/mnt/data/ComfyUI/models/gguf"
    "${HOME}/ComfyUI/models/llm"
    "${HOME}/ComfyUI/models/gguf"

    # LM Studio
    "${HOME}/.lmstudio/models"
    "${HOME}/.cache/lm-studio/models"

    # Ollama (blobs store raw model bytes)
    "${HOME}/.ollama/models"
    "/usr/share/ollama/.ollama/models"

    # text-generation-webui
    "${HOME}/text-generation-webui/models"
    "/mnt/data/text-generation-webui/models"

    # Open WebUI / anything on /mnt/data
    "/mnt/data"
  )

  # Append user-supplied extra dirs from config.env
  if [[ -n "${EXTRA_MODEL_SEARCH_DIRS:-}" ]]; then
    IFS=: read -ra _extra <<< "$EXTRA_MODEL_SEARCH_DIRS"
    _SEARCH_DIRS+=("${_extra[@]}")
  fi

  # Filter to only existing directories
  local filtered=()
  for d in "${_SEARCH_DIRS[@]}"; do
    [[ -d "$d" ]] && filtered+=("$d")
  done
  _SEARCH_DIRS=("${filtered[@]}")
}

# ── find_gguf KEYWORD [PREF_QUANT] ───────────────────────────────────────────
# Search all model directories for a GGUF matching KEYWORD.
# Sets FOUND_GGUF_PATH on success; returns 0 on hit, 1 on miss.
find_gguf() {
  local keyword="${1,,}"   # lower-case
  local pref_quant="${2:-Q4_K_M}"

  _build_search_dirs
  FOUND_GGUF_PATH=""

  local -a candidates=()

  for dir in "${_SEARCH_DIRS[@]}"; do
    while IFS= read -r -d '' f; do
      local base
      base=$(basename "$f" | tr '[:upper:]' '[:lower:]')
      # Match if the filename contains the keyword (spaces→dashes normalised)
      local norm_key="${keyword// /-}"
      if [[ "$base" == *"${norm_key}"* ]]; then
        candidates+=("$f")
      fi
    done < <(find "$dir" -maxdepth 10 -name "*.gguf" -print0 2>/dev/null)
  done

  (( ${#candidates[@]} == 0 )) && return 1

  # Prefer the requested quant
  local pref_lower="${pref_quant,,}"
  for f in "${candidates[@]}"; do
    local base
    base=$(basename "$f" | tr '[:upper:]' '[:lower:]')
    if [[ "$base" == *"${pref_lower}"* ]]; then
      FOUND_GGUF_PATH="$f"
      return 0
    fi
  done

  # Fall back to first match
  FOUND_GGUF_PATH="${candidates[0]}"
  return 0
}

# ── resolve_model KEYWORD HF_REPO [PREF_QUANT] ───────────────────────────────
# 1. Search locally; if found, sets MODEL_FLAG=-m  MODEL_VALUE=<path>
# 2. If not found, fall back to HF download: MODEL_FLAG=-hf  MODEL_VALUE=<repo>
# Exports MODEL_FLAG, MODEL_VALUE for docker-compose command substitution.
resolve_model() {
  local keyword="$1"
  local hf_repo="$2"
  local pref_quant="${3:-Q4_K_M}"

  echo -e "${BOLD}── Model Resolution ─────────────────────────────────────────────${NC}"
  echo -e "  Looking for: ${CYN}${keyword}${NC}  (preferred quant: ${pref_quant})"
  echo -e "  Searching ${#_SEARCH_DIRS[@]}+ directories..."

  if find_gguf "$keyword" "$pref_quant"; then
    local fsize
    fsize=$(du -sh "$FOUND_GGUF_PATH" 2>/dev/null | cut -f1 || echo "?")
    _lib_ok "Found local GGUF (${fsize}): ${FOUND_GGUF_PATH}"
    export MODEL_FLAG="-m"
    export MODEL_VALUE="$FOUND_GGUF_PATH"
  else
    _lib_warn "Not found locally — will download from HF: ${hf_repo}"
    export MODEL_FLAG="-hf"
    export MODEL_VALUE="$hf_repo"
  fi
  echo ""
}

# ── launch_server [ALIAS] ─────────────────────────────────────────────────────
# Starts the Vulkan container via docker-compose.yml.
launch_server() {
  local alias="${1:-model}"
  export MODEL_ALIAS="$alias"

  echo -e "${BOLD}── Launching llama-server ────────────────────────────────────────${NC}"
  echo -e "  →  Model flag : ${MODEL_FLAG}  ${MODEL_VALUE}"
  echo -e "  →  Alias      : ${alias}"
  echo -e "  →  Context    : ${LLAMA_CTX_SIZE} tokens"
  echo -e "  →  GPU layers : ${LLAMA_NGL}"
  echo -e "  →  Threads    : ${LLAMA_THREADS}"
  echo ""

  docker compose     --env-file "${SCRIPT_DIR}/config.env"     -f "${SCRIPT_DIR}/docker-compose.yml"     up -d --force-recreate
}

# ── wait_for_server ───────────────────────────────────────────────────────────
wait_for_server() {
  local port="${SERVER_PORT:-8000}"
  local timeout=600
  local elapsed=0

  echo -e "${BOLD}── Waiting for server ───────────────────────────────────────────${NC}"
  _lib_info "Local model — should be ready in under 60s."
  _lib_info "Ctrl+C stops waiting but leaves container running."
  echo ""
  echo -e "${BOLD}── Live Container Logs ───────────────────────────────────────────${NC}"

  while true; do
    docker logs --tail 5 strix-llamacpp 2>&1 \
    | while IFS= read -r _line; do
        printf "  \033[0;36m│\033[0m %s\n" "$_line"
      done || true

    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo ""; _lib_ok "Server is ready after ${elapsed}s"; break
    fi

    if ! docker ps -q --filter "name=strix-llamacpp$" --filter "status=running" | grep -q .; then
      echo ""; _lib_fail "Container exited before becoming ready."
      echo ""
      echo -e "${BOLD}── Container Logs (last 80 lines) ───────────────────────────────${NC}"
      docker logs strix-llamacpp --tail 80 2>&1 | sed 's/^/  /'
      exit 1
    fi

    sleep 5; elapsed=$((elapsed + 5))
    if (( elapsed >= timeout )); then
      echo ""; _lib_fail "Timeout (${timeout}s). Logs: docker logs strix-llamacpp --tail 80"
      exit 1
    fi
  done

  local lan_ip
  lan_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

  echo ""
  echo -e "${BOLD}── Access Info ──────────────────────────────────────────────────${NC}"
  echo -e "  OpenAI API      : ${CYN}http://localhost:${port}/v1${NC}"
  echo -e "                    ${CYN}http://${lan_ip}:${port}/v1${NC}  (LAN)"
  echo ""
  echo -e "  Quick test:"
  printf '  curl http://localhost:%s/v1/chat/completions \\
' "${port}"
  printf '    -H "Content-Type: application/json" \\
'
  printf '    -d ''{"model":"%s","messages":[{"role":"user","content":"hello"}],"max_tokens":32}''
' "${MODEL_ALIAS:-model}"
  echo ""
}
