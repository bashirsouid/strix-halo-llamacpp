#!/usr/bin/env bash
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
pushd "$SCRIPT_DIR" > /dev/null
python3 ./server.py serve nemotron-nano-q4
popd > /dev/null
