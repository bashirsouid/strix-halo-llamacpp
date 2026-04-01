#!/usr/bin/env bash
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
python3 ./bench_viewer.py
popd > /dev/null
