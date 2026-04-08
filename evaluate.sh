#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"

cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
# Auto-activate venv for the lifetime of this script if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo " ✗ No virtualenv found at $VENV_DIR"
    echo " Run: source source-me.sh (once, to set it up)"
    exit 1
  fi

  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
fi

mode="single"
args=()
suite="humaneval"
profile_set=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      mode="all"
      shift
      ;;
    --reanalyze)
      mode="reanalyze"
      shift
      ;;
    --quick)
      args+=("--profile" "quick")
      profile_set=1
      shift
      ;;
    --mini)
      args+=("--profile" "mini")
      profile_set=1
      shift
      ;;
    --full)
      args+=("--profile" "full")
      profile_set=1
      shift
      ;;
    --profile)
      args+=("$1")
      shift
      if [[ $# -eq 0 ]]; then
        echo " ✗ Missing value for --profile" >&2
        exit 1
      fi
      args+=("$1")
      profile_set=1
      shift
      ;;
    --suite)
      args+=("$1")
      shift
      if [[ $# -eq 0 ]]; then
        echo " ✗ Missing value for --suite" >&2
        exit 1
      fi
      suite="$1"
      args+=("$1")
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  ./evaluate.sh [MODEL] [server.py eval args]
  ./evaluate.sh --all [server.py eval-all args]
  ./evaluate.sh --reanalyze [server.py eval-reanalyze args]

Default behavior:
  * Humaneval runs default to --profile quick (fast curated subset)
  * Other suites keep server.py's default profile unless you override it

Convenience flags:
  --quick   shorthand for --profile quick
  --mini    shorthand for --profile mini
  --full    shorthand for --profile full

Examples:
  ./evaluate.sh qwen3-coder-next-q6 --suite humaneval
  ./evaluate.sh qwen3-coder-next-q6 --full --label baseline
  ./evaluate.sh --all --suite humaneval
  ./evaluate.sh --reanalyze
USAGE
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ "$mode" != "reanalyze" && "$profile_set" -eq 0 && "$suite" == "humaneval" ]]; then
  args+=("--profile" "quick")
fi

case "$mode" in
  single)
    python3 server.py eval "${args[@]}"
    ;;
  all)
    python3 server.py eval-all "${args[@]}"
    ;;
  reanalyze)
    python3 server.py eval-reanalyze "${args[@]}"
    ;;
esac

python3 tools/eval_viewer.py results/eval/eval_results.jsonl
