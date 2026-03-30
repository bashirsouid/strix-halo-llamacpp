#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.llamacpp.yml"

# Defaults (override via env if you like)
export MODEL_PATH="${MODEL_PATH:-/mnt/data/llama.cpp-cache/hf/hub/models--unsloth--Llama-3.1-Nemotron-Nano-8B-v1-GGUF/snapshots/d4b185dad118067300efd1b12498dad3a3358496/Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf}"
export SERVER_PORT="${SERVER_PORT:-8000}"
export LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-32768}"
export LLAMA_THREADS="${LLAMA_THREADS:-16}"
export LLAMA_NGL="${LLAMA_NGL:-999}"
export MODEL_ALIAS="${MODEL_ALIAS:-nemotron-nano-8b}"

echo "Starting llama.cpp (ROCm 6.4.4 + ROCWMMA) on Strix Halo..."
echo "  Model : ${MODEL_PATH}"
echo "  Port  : ${SERVER_PORT}"
echo "  Ctx   : ${LLAMA_CTX_SIZE} tokens, GPU layers: ${LLAMA_NGL}"

docker compose -f "${COMPOSE_FILE}" up -d

echo "Waiting for server to be ready..."
TRIES=0
MAX_TRIES=90

while (( TRIES < MAX_TRIES )); do
  if curl -sf "http://localhost:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
    echo
    echo "llama.cpp server ready on http://localhost:${SERVER_PORT}/v1"
    exit 0
  fi

  sleep 2
  TRIES=$((TRIES + 1))
  printf "  ...%ss elapsed\r" $((TRIES * 2))
done

echo
echo "Server did not become ready in time. Check logs:"
echo "  docker logs strix-llamacpp-rocm --tail 80"
exit 1