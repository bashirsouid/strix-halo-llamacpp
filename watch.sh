#!/usr/bin/env bash
# watch.sh — monitor and auto-restart the llama-server.
#
# Usage:
#   ./watch.sh                          # Watch with picker for backend and model
#   ./watch.sh --backend rocm           # Watch with ROCm backend (picker for model)
#   ./watch.sh nemotron-nano-q4         # Watch specific model (picker for backend)
#   ./watch.sh --backend radv nemotron-nano-q4   # Watch specific backend and model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH_ARGS=()
LAST_MODEL=""
LAST_BACKEND=""
WATCH_INTERVAL="${WATCH_INTERVAL:-30}"
WATCH_ONCE="${WATCH_ONCE:-0}"

_info()  { printf '\033[36m  ℹ  %s\033[0m\n' "$*"; }
_ok()    { printf '\033[32m  ✓  %s\033[0m\n' "$*"; }
_warn()  { printf '\033[33m  ⚠  %s\033[0m\n' "$*" >&2; }
_fail()  { printf '\033[31m  ✗  %s\033[0m\n' "$*" >&2; }

container_name_for_backend() {
    case "$1" in
        vulkan|radv) echo "strix-llama-vulkan" ;;
        amdvlk) echo "strix-llama-amdvlk" ;;
        rocm) echo "strix-llama-rocm" ;;
        rocm6) echo "strix-llama-rocm6" ;;
        rocm7) echo "strix-llama-rocm7" ;;
        rocm7-nightly) echo "strix-llama-rocm7-nightly" ;;
        *) return 1 ;;
    esac
}

read_state_field() {
    local field="$1"
    python3 -c "import json; data=json.load(open('$SCRIPT_DIR/.server.json')); print(data.get('$field', ''))" 2>/dev/null || true
}

capture_last_selection() {
    if [[ ! -f "$SCRIPT_DIR/.server.json" ]]; then
        return 0
    fi

    LAST_MODEL="$(read_state_field model)"
    LAST_BACKEND="$(read_state_field backend)"
}

is_server_running() {
    if [[ ! -f "$SCRIPT_DIR/.server.json" ]]; then
        return 1
    fi

    local backend
    backend="$(read_state_field backend)"
    [[ -z "$backend" ]] && return 1

    local container_name
    container_name="$(read_state_field container)"
    if [[ -z "$container_name" ]]; then
        container_name="$(container_name_for_backend "$backend" 2>/dev/null || true)"
    fi

    local rt
    rt="$(command -v docker || true)"
    if [[ -n "$rt" ]] && [[ -n "$container_name" ]]; then
        if "$rt" container inspect "$container_name" --format='{{.State.Running}}' 2>/dev/null | grep -q true; then
            return 0
        fi
    fi

    # Legacy/native fallback.
    if [[ -f "$SCRIPT_DIR/.server.pid" ]]; then
        local pid
        pid="$(cat "$SCRIPT_DIR/.server.pid" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi

    return 1
}

start_server() {
    _info "Starting server with: ./start.sh ${WATCH_ARGS[*]:-<picker>}"
    ./start.sh "${WATCH_ARGS[@]}"
    capture_last_selection
}

cleanup() {
    _info "Watch loop stopped."
    exit 0
}

print_start_mode() {
    if [[ ${#WATCH_ARGS[@]} -eq 0 ]]; then
        _info "Starting with interactive backend and model selection..."
    elif [[ ${#WATCH_ARGS[@]} -eq 1 ]] && [[ "${WATCH_ARGS[0]}" != --* ]]; then
        _info "Starting with model '${WATCH_ARGS[0]}' and backend picker..."
    elif [[ "${WATCH_ARGS[0]:-}" == "--backend" ]] && [[ -n "${WATCH_ARGS[1]:-}" ]]; then
        _info "Starting with backend '${WATCH_ARGS[1]}' and model picker..."
    else
        _info "Starting with specified backend and model..."
    fi
}

watch_loop() {
    while true; do
        sleep "$WATCH_INTERVAL"

        if ! is_server_running; then
            _warn "Server is not running, restarting..."
            if [[ -n "$LAST_MODEL" ]] && [[ -n "$LAST_BACKEND" ]]; then
                _info "Restarting with last selected model: $LAST_MODEL"
                WATCH_ARGS=(--backend "$LAST_BACKEND" "$LAST_MODEL")
            fi
            start_server
        fi

        if [[ "$WATCH_ONCE" == "1" ]]; then
            break
        fi
    done
}

main() {
    cd "$SCRIPT_DIR"
    WATCH_ARGS=("$@")

    trap cleanup SIGINT SIGTERM

    _info "strix-halo-llamacpp — watch mode"
    echo
    print_start_mode
    echo

    start_server
    watch_loop
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
