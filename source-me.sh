#!/usr/bin/env bash
# Usage: source source-me.sh
#
# Creates the project venv at ./.venv if it doesn't exist, then activates it
# and installs all eval dependencies.  Must be sourced, not executed, so the
# activation carries through to your shell session.
#
# Idempotent — safe to re-run; skips creation/install if already up to date.

# ── Guard: must be sourced, not executed ─────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "  ✗  source-me.sh must be sourced, not executed."
    echo "     Run:  source source-me.sh"
    echo "     or:   . source-me.sh"
    exit 1
fi

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── Create venv if it doesn't exist ──────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "  ℹ  Creating virtualenv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "  ✓  Virtualenv created."
fi

# ── Activate ─────────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "  ✓  Activated: $VIRTUAL_ENV"

# ── Install / upgrade dependencies ───────────────────────────────────────────
REQUIREMENTS=(
    "evalplus[vllm]"
    # Add more here as the eval suite grows, e.g.:
    # "bigcodebench"
    # "openai"
)

echo "  ℹ  Checking eval dependencies ..."
pip install --quiet --upgrade "${REQUIREMENTS[@]}"
echo "  ✓  Dependencies up to date."
echo
echo "  ✓  Eval environment ready.  Run ./evaluate.sh or ./evaluate-all.sh"