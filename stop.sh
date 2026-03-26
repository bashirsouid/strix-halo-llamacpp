#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

echo -e "${BOLD}── Stopping llama.cpp Container ─────────────────────────────────${NC}"
docker compose \
  --env-file "${SCRIPT_DIR}/config.env" \
  -f "${SCRIPT_DIR}/docker-compose.yml" \
  down || true
_lib_ok "Container stopped."
