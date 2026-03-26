#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

echo -e "${BOLD}"
cat <<'BANNER'
╔══════════════════════════════════════════════════════════════════╗
║   AMD Strix Halo (gfx1151) — llama.cpp (Vulkan RADV)            ║
╚══════════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

mkdir -p "${HOST_LLAMA_CACHE:-/mnt/data/llama.cpp-cache}/hf"

# Delegate to the default model's loader — all download/launch logic lives there.
exec "$SCRIPT_DIR/${DEFAULT_LOADER:-load_nemotron_nano.sh}"