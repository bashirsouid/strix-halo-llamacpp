#!/usr/bin/env bash
# Unified automated test runner for strix-halo-llamacpp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
fi

PORT="${STRIX_TEST_PORT:-}"
if [[ -z "$PORT" && -f "$SCRIPT_DIR/.server.json" ]]; then
    PORT="$(python3 - <<'PY_PORT'
import json
from pathlib import Path
state = Path('.server.json')
try:
    data = json.loads(state.read_text())
    print(data.get('port', ''))
except Exception:
    print('')
PY_PORT
)"
fi
PORT="${PORT:-8000}"
BACKEND="${STRIX_TEST_BACKEND:-radv}"
TEST_MODEL_ALIAS="${STRIX_TEST_MODEL_ALIAS:-smollm2-135m-test-q4}"
STARTED_SERVER=0

server_is_up() {
    python3 - "$1" <<'PY'
import sys
import urllib.request

port = int(sys.argv[1])
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
        raise SystemExit(0 if resp.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

cleanup() {
    if [[ "$STARTED_SERVER" -eq 1 ]]; then
        echo "Stopping temporary test server..."
        python3 server.py stop >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ -f requirements-test.txt ]]; then
    echo "Installing test dependencies..."
    python3 -m pip install -q -r requirements-test.txt
fi

echo "Running shell entrypoint tests..."
bash tests/test_start.sh
bash tests/test_bash_entrypoints.sh

echo "Running Python test suite..."
pytest -q tests "$@"

if server_is_up "$PORT"; then
    echo "Reusing already-running server on port $PORT for live inference smoke test..."
    export STRIX_TEST_MODEL=auto
else
    echo "Starting temporary smoke-test server on port $PORT using $TEST_MODEL_ALIAS..."
    python3 server.py serve "$TEST_MODEL_ALIAS" --port "$PORT" --backend "$BACKEND"
    STARTED_SERVER=1
    export STRIX_TEST_MODEL="$TEST_MODEL_ALIAS"
fi

export STRIX_RUN_LIVE_INFERENCE=1
export STRIX_TEST_PORT="$PORT"
export STRIX_TEST_TIMEOUT="${STRIX_TEST_TIMEOUT:-30}"

echo "Running live inference smoke test..."
pytest -q tests/test_inference.py

echo ""
echo "All tests completed."
