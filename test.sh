#!/usr/bin/env bash
# Unified automated test runner for strix-halo-llamacpp.
#
# This helper keeps running independent phases even if an earlier phase fails,
# then emits a single FINAL RESULT line as the very last line of output.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
fi

PYTEST_TARGETS=(test_models.py test_entrypoints.py tests)
FAILED_STEPS=()
SKIPPED_STEPS=()
FINAL_EXIT=0

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

join_by() {
    local sep="$1"
    shift || true

    local out=""
    local item
    for item in "$@"; do
        if [[ -n "$out" ]]; then
            out+="$sep"
        fi
        out+="$item"
    done
    printf '%s' "$out"
}

record_failure() {
    local name="$1"
    local status="${2:-1}"
    FAILED_STEPS+=("${name} (exit ${status})")
    FINAL_EXIT=1
}

run_step() {
    local name="$1"
    shift

    echo "Running: $name"
    "$@"
    local status=$?
    if [[ "$status" -eq 0 ]]; then
        echo "PASS: $name"
        return 0
    fi

    echo "FAIL: $name (exit $status)" >&2
    record_failure "$name" "$status"
    return "$status"
}

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

print_final_result() {
    echo
    if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
        echo "FINAL RESULT: PASS"
        return
    fi

    local message="FINAL RESULT: FAIL (failed: $(join_by '; ' "${FAILED_STEPS[@]}")"
    if [[ ${#SKIPPED_STEPS[@]} -gt 0 ]]; then
        message+="; skipped: $(join_by '; ' "${SKIPPED_STEPS[@]}")"
    fi
    message+=")"
    echo "$message"
}

handle_signal() {
    FINAL_EXIT=130
    cleanup
    print_final_result
    exit "$FINAL_EXIT"
}

trap handle_signal INT TERM

main() {
    if [[ -f requirements-test.txt ]]; then
        run_step "install test dependencies" python3 -m pip install -q -r requirements-test.txt || true
    fi

    run_step "bash smoke test: tests/test_start.sh" bash tests/test_start.sh || true
    run_step "bash smoke test: tests/test_bash_entrypoints.sh" bash tests/test_bash_entrypoints.sh || true

    run_step "pytest suite" pytest -q "${PYTEST_TARGETS[@]}" "$@" || true

    if server_is_up "$PORT"; then
        echo "Reusing already-running server on port $PORT for live inference smoke test..."
        export STRIX_TEST_MODEL=auto
    else
        echo "Starting temporary smoke-test server on port $PORT using $TEST_MODEL_ALIAS..."
        if run_step "temporary smoke-test server startup" python3 server.py serve "$TEST_MODEL_ALIAS" --port "$PORT" --backend "$BACKEND"; then
            STARTED_SERVER=1
            export STRIX_TEST_MODEL="$TEST_MODEL_ALIAS"
        else
            SKIPPED_STEPS+=("live inference smoke test")
        fi
    fi

    if [[ ${#SKIPPED_STEPS[@]} -eq 0 ]]; then
        export STRIX_RUN_LIVE_INFERENCE=1
        export STRIX_TEST_PORT="$PORT"
        export STRIX_TEST_TIMEOUT="${STRIX_TEST_TIMEOUT:-30}"
        run_step "live inference smoke test" pytest -q tests/test_inference.py || true
    else
        echo "Skipping live inference smoke test because the server is unavailable."
    fi
}

main "$@"
cleanup
print_final_result
exit "$FINAL_EXIT"
