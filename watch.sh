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
cd "$SCRIPT_DIR"

WATCH_ARGS=("$@")

_info()  { printf '\033[36m  ℹ  %s\033[0m\n' "$*"; }
_ok()    { printf '\033[32m  ✓  %s\033[0m\n' "$*"; }
_warn()  { printf '\033[33m  ⚠  %s\033[0m\n' "$*" >&2; }
_fail()  { printf '\033[31m  ✗  %s\033[0m\n' "$*" >&2; }

is_server_running() {
    if [[ -f "$SCRIPT_DIR/.server.json" ]]; then
        local backend
        backend=$(python3 -c "import json; print(json.load(open('$SCRIPT_DIR/.server.json')).get('backend', 'radv'))" 2>/dev/null || echo "radv")
        
        if [[ "$backend" == "rocm" ]] || [[ "$backend" == "rocm6" ]] || [[ "$backend" == "rocm7" ]] || [[ "$backend" == "rocm7-nightly" ]]; then
            local rt
            rt=$(command -v podman || command -v docker)
            if [[ -n "$rt" ]]; then
                local container_name="strix-llama-${backend}"
                if "$rt" container exists "$container_name" &>/dev/null; then
                    if "$rt" container inspect "$container_name" --format='{{.State.Running}}' 2>/dev/null | grep -q true; then
                        return 0
                    fi
                fi
                return 1
            fi
            return 1
        else
            if [[ -f "$SCRIPT_DIR/.server.pid" ]]; then
                local pid
                pid=$(cat "$SCRIPT_DIR/.server.pid" 2>/dev/null)
                if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                    return 0
                fi
            fi
            return 1
        fi
    fi
    return 1
}

start_server() {
    _info "Starting server with: ./start.sh ${WATCH_ARGS[*]:-<picker>}"
    ./start.sh "${WATCH_ARGS[@]}"
}

cleanup() {
    _info "Watch loop stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

_info "strix-halo-llamacpp — watch mode"
echo

# Check if we're using start.sh arguments
if [[ ${#WATCH_ARGS[@]} -eq 0 ]]; then
    _info "Starting with interactive backend and model selection..."
elif [[ ${#WATCH_ARGS[@]} -eq 1 ]] && [[ "${WATCH_ARGS[0]}" != --* ]]; then
    _info "Starting with model '${WATCH_ARGS[0]}' and backend picker..."
elif [[ "${WATCH_ARGS[0]:-}" == "--backend" ]]; then
    if [[ -n "${WATCH_ARGS[1]:-}" ]]; then
        _info "Starting with backend '${WATCH_ARGS[1]}' and model picker..."
    fi
else
    _info "Starting with specified backend and model..."
fi
echo

# Initial launch
start_server

while true; do
    sleep 30
    if ! is_server_running; then
        _warn "Server is not running, restarting..."
        start_server
    fi
done
