#!/usr/bin/env bash
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
cd "$SCRIPT_DIR"
python3 server.py bench-all "$@"
