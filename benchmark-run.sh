#!/usr/bin/env bash
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
python3 server.py bench
popd > /dev/null
