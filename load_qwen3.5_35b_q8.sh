#!/usr/bin/env bash
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
pushd "$SCRIPT_DIR" > /dev/null
python3 ./server.py serve qwen3.5-35b-q8 --backend rocm
popd > /dev/null
