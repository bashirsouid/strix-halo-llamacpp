#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

# Auto-activate venv for the lifetime of this script if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        echo "  ✗  No virtualenv found at $VENV_DIR"
        echo "     Run:  source source-me.sh   (once, to set it up)"
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
fi

python3 server.py eval "$@"