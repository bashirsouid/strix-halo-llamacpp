#!/usr/bin/env bash
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
python3 ./eval_viewer.py
popd > /dev/null
