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
    if [[ ! -f "$SCRIPT_DIR/.server.json" ]]; then
        return 1
    fi
    
    local backend
    # Read backend with explicit error handling (don't silently default)
    if ! backend=$(python3 -c "import json; data=json.load(open('$SCRIPT_DIR/.server.json')); print(data.get('backend', ''))" 2>/dev/null); then
        return 1
    fi
    
    # If backend is empty or invalid, the check failed
    [[ -z "$backend" ]] && return 1
    
    # Check ROCm containers
    if [[ "$backend" == "rocm" ]] || [[ "$backend" == "rocm6" ]] || [[ "$backend" == "rocm7" ]] || [[ "$backend" == "rocm7-nightly" ]]; then
        local rt
        rt=$(command -v docker)
        if [[ -z "$rt" ]]; then
            return 1
        fi
        
        local container_name="strix-llama-${backend}"
        # Check if container exists and is running
        if "$rt" container inspect "$container_name" --format='{{.State.Running}}' 2>/dev/null | grep -q true; then
            return 0
        fi
        return 1
    # Check Vulkan/non-ROCm processes
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

start_server() {
    _info "Starting server with: ./start.sh ${WATCH_ARGS[*]:-<picker>}"
    ./start.sh "${WATCH_ARGS[@]}"
    
    # Capture model from .server.json for subsequent restarts
    if [[ -f "$SCRIPT_DIR/.server.json" ]]; then
        LAST_MODEL=$(python3 -c "import json; data=json.load(open('$SCRIPT_DIR/.server.json')); print(data.get('model', ''))" 2>/dev/null || echo "")
        if [[ -n "$LAST_MODEL" ]]; then
            # Store for next restart so we don't reprompt for model
            LAST_BACKEND=$(python3 -c "import json; data=json.load(open('$SCRIPT_DIR/.server.json')); print(data.get('backend', ''))" 2>/dev/null || echo "")
        fi
    fi
}

# Reconstruct start.sh args for restarts, reusing the captured model
get_restart_args() {
    local args=("${WATCH_ARGS[@]}")
    
    # If we have a previously selected model and backend, use them
    if [[ -n "${LAST_MODEL:-}" ]] && [[ -n "${LAST_BACKEND:-}" ]]; then
        # Check if args already include the model (non-flag positional arg)
        local has_model=0
        for arg in "${args[@]}"; do
            if [[ "$arg" != --* ]]; then
                has_model=1
                break
            fi
        done
        
        # If model not already in args, add the last model we know about
        if [[ $has_model -eq 0 ]]; then
            args+=("$LAST_MODEL")
        fi
    fi
    
    echo "${args[@]}"
}

cleanup() {
    _info "Watch loop stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

_info "strix-halo-llamacpp — watch mode"
echo

# Initialize model/backend capture variables
LAST_MODEL=""
LAST_BACKEND=""

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
        # On restart, use the last captured model to avoid re-prompting
        if [[ -n "${LAST_MODEL:-}" ]]; then
            _info "Restarting with last selected model: $LAST_MODEL"
            WATCH_ARGS=(--backend "$LAST_BACKEND" "$LAST_MODEL")
        fi
        start_server
    fi
done
