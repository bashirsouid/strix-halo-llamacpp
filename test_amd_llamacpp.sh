#!/usr/bin/env bash
set -euo pipefail

# AMD official llama.cpp Docker image tag (ROCm 7.1.1, Ubuntu 24.04)
# From AMD docs for gfx1150/gfx1151 prebuilt binaries.[web:326][web:273]
ROCM_LLAMA_TAG_DEFAULT="llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04"
ROCM_LLAMA_TAG="${ROCM_LLAMA_TAG:-$ROCM_LLAMA_TAG_DEFAULT}"

# Your Nemotron Nano Q4_K_M model path (adjust if you moved it)
MODEL_PATH_DEFAULT="/mnt/data/llama.cpp-cache/hf/hub/models--unsloth--Llama-3.1-Nemotron-Nano-8B-v1-GGUF/snapshots/d4b185dad118067300efd1b12498dad3a3358496/Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf"
MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"

SERVER_PORT="${SERVER_PORT:-8000}"
CONTAINER_NAME="strix-llamacpp-amd"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist:"
  echo "  $MODEL_PATH"
  echo "Set MODEL_PATH env to your GGUF file and retry."
  exit 1
fi

MODEL_DIR="$(dirname "$MODEL_PATH")"
MODEL_FILE="$(basename "$MODEL_PATH")"

echo "=== AMD official llama.cpp test ==="
echo "Image tag : rocm/llama.cpp:${ROCM_LLAMA_TAG}_server"
echo "Model     : $MODEL_PATH"
echo "Port      : $SERVER_PORT"
echo

echo "[1/3] Pulling image (if needed)..."
docker pull "rocm/llama.cpp:${ROCM_LLAMA_TAG}_server"

echo "[2/3] Stopping any existing container..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "[3/3] Starting llama-server..."
docker run -d --name "$CONTAINER_NAME" \
  --privileged \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 16G \
  -v "$MODEL_DIR:/data" \
  "rocm/llama.cpp:${ROCM_LLAMA_TAG}_server" \
  -m "/data/$MODEL_FILE" \
  --port "$SERVER_PORT" \
  --host "0.0.0.0" \
  -ngl 999 \
  -fa on

echo
echo "Waiting for server on http://localhost:${SERVER_PORT}/v1/models ..."
TRIES=0
MAX_TRIES=120

while (( TRIES < MAX_TRIES )); do
  if curl -sf "http://localhost:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
    echo
    echo "Server is UP on port ${SERVER_PORT} using AMD official llama.cpp."
    echo
    echo "Now run your benchmark in this repo (from the same directory):"
    echo "  ./bench_current.sh"
    echo
    echo "Compare the reported tok/s to your previous runs."
    echo "When done, stop the container with:"
    echo "  docker rm -f $CONTAINER_NAME"
    exit 0
  fi
  sleep 2
  TRIES=$((TRIES + 1))
  printf "  ...%ss elapsed\r" $((TRIES * 2))
done

echo
echo "ERROR: Server did not become ready in time. Recent logs:"
docker logs --tail 80 "$CONTAINER_NAME" || true
exit 1