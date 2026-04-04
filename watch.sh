#!/usr/bin/env bash
# watch.sh — monitor and auto-restart the llama-server.
#
# Usage:
#   ./watch.sh                          # Watch with default backend
#   ./watch.sh --backend rocm           # Watch with ROCm backend
#   ./watch.sh MODEL [--backend radv]   # Watch specific model
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BACKEND="radv"
WATCH_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "--backend" ]]; then
        WATCH_ARGS+=("$arg")
    elif [[ "${WATCH_ARGS[*]:-}" == *"--backend" ]] && [[ -z "${BACKEND_SET:-}" ]]; then
        BACKEND="$arg"
        BACKEND_SET=1
        WATCH_ARGS+=("$arg")
    else
        WATCH_ARGS+=("$arg")
    fi
done

for i in "${!WATCH_ARGS[@]}"; do
    if [[ "${WATCH_ARGS[$i]}" == "--backend" ]] && [[ -n "${WATCH_ARGS[$((i+1))]:-}" ]]; then
        BACKEND="${WATCH_ARGS[$((i+1))]}"
        break
    fi
done

_info()  { printf '\033[36m  ℹ  %s\033[0m\n' "$*"; }
_ok()    { printf '\033[32m  ✓  %s\033[0m\n' "$*"; }
_warn()  { printf '\033[33m  ⚠  %s\033[0m\n' "$*" >&2; }

is_server_running() {
    if [[ "$BACKEND" == "rocm" ]]; then
        local rt
        rt=$(command -v podman || command -v docker)
        if [[ -n "$rt" ]]; then
            if "$rt" container exists strix-llama-rocm &>/dev/null; then
                "$rt" container inspect strix-llama-rocm --format='{{.State.Running}}' 2>/dev/null | grep -q true
                return $?
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
}

check_server() {
    if is_server_running; then
        return 0
    else
        return 1
    fi
}

load_model() {
    _info "Loading model (backend: $BACKEND)..."
    python3 "$SCRIPT_DIR/server.py" serve "${WATCH_ARGS[@]}"
}

cleanup() {
    _info "Watch loop stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

_info "strix-halo-llamacpp — watch mode (backend: $BACKEND)"
_info "Checking server status every 30s..."
echo

while true; do
    if check_server; then
        _ok "Server is running"
    else
        _warn "Server is not running, restarting..."
        load_model
    fi
    
    sleep 30
done
