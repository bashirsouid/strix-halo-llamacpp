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

mode="single"
args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            mode="all"
            shift
            ;;
        -h|--help)
            cat <<'USAGE'
Usage:
  ./evaluate.sh [MODEL] [server.py eval args]
  ./evaluate.sh --all [server.py eval-all args]

Examples:
  ./evaluate.sh qwen3-coder-next-q6 --suite humaneval
  ./evaluate.sh --all --suite mbpp
USAGE
            exit 0
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

case "$mode" in
    single)
        python3 server.py eval "${args[@]}"
        ;;
    all)
        python3 server.py eval-all "${args[@]}"
        ;;
esac

python3 tools/eval_viewer.py results/eval/eval_results.jsonl
