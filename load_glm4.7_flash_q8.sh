#!/usr/bin/env bash
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
pushd "$SCRIPT_DIR" > /dev/null
python3 ./server.py serve glm-4.7-flash-q8 --backend rocm
popd > /dev/null
