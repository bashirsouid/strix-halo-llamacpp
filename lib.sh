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

# ── Draft / speculative decoding helpers ─────────────────────────────────────
# Loaders should call clear_draft_config first, then optionally export
# DRAFT_MODEL_PATH / DRAFT_MAX / DRAFT_MIN and LLAMA_COMPOSE_FILE when they
# actually want speculative decoding enabled.
clear_draft_config() {
  unset DRAFT_MODEL_PATH
  unset DRAFT_MAX
  unset DRAFT_MIN
  unset LLAMA_COMPOSE_FILE
}

_SEARCH_DIRS=()   # callers populate this; default is empty

# ── Model search directories ──────────────────────────────────────────────────
# Used by scan_models.sh to discover all GGUFs on disk.
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

# ── launch_server [ALIAS] ─────────────────────────────────────────────────────
# Starts the Vulkan container via the selected docker compose file.
#   LLAMA_COMPOSE_FILE (optional) selects an alternate compose YAML
#   (e.g., docker-compose.spec.yml for speculative decoding configs).
launch_server() {
  local alias="${1:-model}"
  export MODEL_ALIAS="$alias"

  local compose_file="${SCRIPT_DIR}/${LLAMA_COMPOSE_FILE:-docker-compose.yml}"

  echo -e "${BOLD}── Launching llama-server ────────────────────────────────────────${NC}"
  echo -e "  →  Compose YML : ${compose_file##*/}"
  echo -e "  →  Model flag  : ${MODEL_FLAG}  ${MODEL_VALUE}"
  echo -e "  →  Alias       : ${alias}"
  echo -e "  →  Context     : ${LLAMA_CTX_SIZE} tokens"
  echo -e "  →  GPU layers  : ${LLAMA_NGL}"
  echo -e "  →  Threads     : ${LLAMA_THREADS}"
  if [[ -n "${DRAFT_MODEL_PATH:-}" ]]; then
    echo -e "  →  Draft model : ${DRAFT_MODEL_PATH}"
  fi
  echo ""

  docker compose \
    --env-file "${SCRIPT_DIR}/config.env" \
    -f "${compose_file}" \
    up -d --force-recreate
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
  printf '  curl http://localhost:%s/v1/chat/completions \\\n' "${port}"
  printf '    -H "Content-Type: application/json" \\\n'
  printf '    -d ''{"model":"%s","messages":[{"role":"user","content":"hello"}],"max_tokens":32}''\n' "${MODEL_ALIAS:-model}"
  echo ""
}