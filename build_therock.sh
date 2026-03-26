#!/usr/bin/env bash
# build_therock.sh — one-time build of the TheRock llama.cpp image (~15-20 min).
# Re-run to update; bump LLAMA_REF in Dockerfile.therock to force a full rebuild.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; YLW='\033[1;33m'; GRN='\033[0;32m'
CYN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}"
cat <<'BANNER'
╔══════════════════════════════════════════════════════════════════╗
║   Building TheRock llama.cpp image (HIP native, gfx1151)         ║
║   Base: kyuz0/vllm-therock-gfx1151  — takes ~15-20 min          ║
╚══════════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

echo -e "${CYN}[INFO]${NC}  Pulling base image (skips if already cached)..."
docker pull docker.io/kyuz0/vllm-therock-gfx1151:20251130-175119 || true

echo -e "${CYN}[INFO]${NC}  Building strix-llamacpp-therock:local ..."
docker build \
  --progress=plain \
  -f Dockerfile.therock \
  -t strix-llamacpp-therock:local \
  .

echo -e "${GRN}[ OK ]${NC}  Build complete: strix-llamacpp-therock:local"
echo ""
echo -e "  Start : ${CYN}./start_therock.sh${NC}"
echo -e "  Bench : ${CYN}./bench_current.sh${NC}"
echo ""
