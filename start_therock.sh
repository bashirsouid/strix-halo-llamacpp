#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

echo -e "${BOLD}"
cat <<'BANNER'
╔══════════════════════════════════════════════════════════════════╗
║   AMD Strix Halo (gfx1151) — llama.cpp (TheRock HIP native)     ║
╚══════════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

if ! docker image inspect strix-llamacpp-therock:local >/dev/null 2>&1; then
  _lib_fail "Image not found. Run ./build_therock.sh first."
  exit 1
fi

mkdir -p "${HOST_LLAMA_CACHE:-/mnt/data/llama.cpp-cache}/hf"

# Stop Vulkan container if running (same port)
if docker ps -q --filter "name=strix-llamacpp$" | grep -q .; then
  _lib_warn "Stopping Vulkan container (port ${SERVER_PORT:-8000} conflict)..."
  docker compose --env-file "$SCRIPT_DIR/config.env" -f "$SCRIPT_DIR/docker-compose.yml" down || true
fi

resolve_model \
  "${MODEL_KEYWORD:-nemotron-nano-8b}" \
  "${MODEL_HF_ID:-unsloth/Llama-3.1-Nemotron-Nano-8B-v1-GGUF}" \
  "${PREF_QUANT:-Q4_K_M}"

echo -e "${BOLD}── Launching llama-server (TheRock HIP) ─────────────────────────${NC}"
echo -e "  →  Model  : ${MODEL_FLAG:--hf}  ${MODEL_VALUE:-}"
echo -e "  →  Alias  : ${MODEL_ALIAS:-model}"
echo -e "  →  Context: ${LLAMA_CTX_SIZE:-32768} tokens  GPU layers: ${LLAMA_NGL:-999}"
echo ""

docker compose \
  --env-file "$SCRIPT_DIR/config.env" \
  -f "$SCRIPT_DIR/docker-compose.therock.yml" \
  up -d --force-recreate

PORT="${SERVER_PORT:-8000}"
TIMEOUT=600; ELAPSED=0

echo -e "${BOLD}── Waiting for server ───────────────────────────────────────────${NC}"
_lib_info "HIP: first generation may be slow (shader compile cache is cold)."
_lib_info "Ctrl+C stops waiting but leaves container running."
echo ""

while true; do
  docker logs --tail 4 strix-llamacpp-therock 2>&1 \
    | while IFS= read -r _line; do
        printf "  \033[0;36m│\033[0m %s\n" "$_line"
      done || true

  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo ""; _lib_ok "Server ready after ${ELAPSED}s"; break
  fi

  if ! docker ps -q --filter "name=strix-llamacpp-therock" --filter "status=running" | grep -q .; then
    echo ""; _lib_fail "Container exited before becoming ready."
    docker logs strix-llamacpp-therock --tail 80 2>&1 | sed 's/^/  /'
    exit 1
  fi

  sleep 5; ELAPSED=$((ELAPSED+5))
  (( ELAPSED >= TIMEOUT )) && { _lib_fail "Timeout."; exit 1; }
done

LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo ""
echo -e "${BOLD}── Access ───────────────────────────────────────────────────────${NC}"
echo -e "  OpenAI API : ${CYN}http://localhost:${PORT}/v1${NC}"
echo -e "               ${CYN}http://${LAN_IP}:${PORT}/v1${NC}  (LAN)"
echo -e "  Benchmark  : ${CYN}./bench_current.sh${NC}"
echo ""
