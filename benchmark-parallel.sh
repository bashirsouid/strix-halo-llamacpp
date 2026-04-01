#!/usr/bin/env bash
# Sweep --parallel values for a model to find the throughput sweet spot.
# Usage:
#   ./benchmark-parallel.sh                  # interactive model picker
#   ./benchmark-parallel.sh nemotron-nano-q4  # specific model
#   ./benchmark-parallel.sh nemotron-nano-q4 --max-np 10 --max-tokens 512
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
cd "$SCRIPT_DIR"
python3 server.py bench-parallel "$@"
