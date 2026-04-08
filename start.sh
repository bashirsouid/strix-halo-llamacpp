#!/usr/bin/env bash
# start.sh — one-command bootstrap for strix-halo-llamacpp.
#
# Installs system + pip dependencies (skipping what's already present),
# ensures Docker + Python deps are present, then serves
# a model with the interactive picker.
#
# Usage:
#   ./start.sh                          # Interactive: backend + model picker
#   ./start.sh --backend rocm           # Backend specified, model picker
#   ./start.sh nemotron-nano-q4         # Model specified, backend picker
#   ./start.sh --backend radv nemotron-nano-q4   # specific model + backend
#
# Safe to run repeatedly — every step is idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND=""
MODEL=""
SERVE_ARGS=()

# ── Colours ──────────────────────────────────────────────────────────────────

_info()  { printf '\033[36m  ℹ  %s\033[0m\n' "$*"; }
_ok()    { printf '\033[32m  ✓  %s\033[0m\n' "$*"; }
_warn()  { printf '\033[33m  ⚠  %s\033[0m\n' "$*" >&2; }

# ── Argument parsing ────────────────────────────────────────────────────────

parse_args() {
    BACKEND=""
    MODEL=""
    SERVE_ARGS=()

    local args=("$@")
    local i=0
    local count=${#args[@]}

    while [[ $i -lt $count ]]; do
        local arg="${args[$i]}"

        case "$arg" in
            --backend)
                if [[ $((i + 1)) -ge $count ]]; then
                    _warn "--backend requires a value"
                    exit 1
                fi
                BACKEND="${args[$((i + 1))]}"
                i=$((i + 2))
                ;;
            --np|--ctx|--ctx-per-slot|--threads|-t|--host|--port)
                if [[ $((i + 1)) -ge $count ]]; then
                    _warn "$arg requires a value"
                    exit 1
                fi
                SERVE_ARGS+=("$arg" "${args[$((i + 1))]}")
                i=$((i + 2))
                ;;
            --extra)
                SERVE_ARGS+=("$arg")
                i=$((i + 1))
                while [[ $i -lt $count ]]; do
                    SERVE_ARGS+=("${args[$i]}")
                    i=$((i + 1))
                done
                ;;
            --*)
                SERVE_ARGS+=("$arg")
                i=$((i + 1))
                ;;
            *)
                if [[ -z "$MODEL" ]]; then
                    MODEL="$arg"
                else
                    SERVE_ARGS+=("$arg")
                fi
                i=$((i + 1))
                ;;
        esac
    done
}

# ── Backend picker ───────────────────────────────────────────────────────────

pick_backend() {
    local backends=("radv" "amdvlk" "rocm" "rocm6" "rocm7" "rocm7-nightly")
    local desc_radv="Vulkan with RADV (fast generation)"
    local desc_amdvlk="Vulkan with AMDVLK (prompt processing)"
    local desc_rocm="ROCm HIP (fast prefill)"
    local desc_rocm6="ROCm 6.4.4"
    local desc_rocm7="ROCm 7.2"
    local desc_rocm7night="ROCm 7.2 nightly"

    local img_radv="docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv"
    local img_amdvlk="docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-amdvlk"
    local img_rocm="docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-nightly"
    local img_rocm6="docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-6.4.4"
    local img_rocm7="docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2"
    local img_rocm7night="docker.io/kyuz0/amd-strix-halo-toolboxes:rocm7-nightlies"

    local rt
    rt=$(command -v docker || true)

    if [[ -z "$rt" ]]; then
        _warn "Docker not found."
        _warn "Install with: sudo apt install docker.io"
    fi

    printf '\n'
    printf '  Available backends:\n'
    printf '\n'

    for i in {0..5}; do
        local backend="${backends[$i]}"
        local num=$((i + 1))
        local check_mark="·"
        local img_name=""

        case "$backend" in
            radv) img_name="$img_radv" ;;
            amdvlk) img_name="$img_amdvlk" ;;
            rocm) img_name="$img_rocm" ;;
            rocm6) img_name="$img_rocm6" ;;
            rocm7) img_name="$img_rocm7" ;;
            rocm7-nightly) img_name="$img_rocm7night" ;;
        esac

        if [[ -n "$rt" ]] && "$rt" image inspect "$img_name" >/dev/null 2>&1; then
            check_mark="✓"
        fi

        printf "  %s %d) %-16s " "$check_mark" "$num" "$backend"

        case "$backend" in
            radv) printf "(%s)\n" "$desc_radv" ;;
            amdvlk) printf "(%s)\n" "$desc_amdvlk" ;;
            rocm) printf "(%s)\n" "$desc_rocm" ;;
            rocm6) printf "(%s)\n" "$desc_rocm6" ;;
            rocm7) printf "(%s)\n" "$desc_rocm7" ;;
            rocm7-nightly) printf "(%s)\n" "$desc_rocm7night" ;;
        esac
    done

    printf '\n'

    while true; do
        read -p "  Enter number (1-6): " -r raw
        case "$raw" in
            1) PICKED_BACKEND="radv"; printf '\n'; return ;;
            2) PICKED_BACKEND="amdvlk"; printf '\n'; return ;;
            3) PICKED_BACKEND="rocm"; printf '\n'; return ;;
            4) PICKED_BACKEND="rocm6"; printf '\n'; return ;;
            5) PICKED_BACKEND="rocm7"; printf '\n'; return ;;
            6) PICKED_BACKEND="rocm7-nightly"; printf '\n'; return ;;
            *) printf '    Invalid choice. Enter a number between 1 and 6.\n' ;;
        esac
    done
}

# ── Dependencies ─────────────────────────────────────────────────────────────

check_docker() {
    if ! command -v docker &>/dev/null; then
        _warn "Docker not found. Install with: sudo apt install docker.io"
        exit 1
    fi
    _ok "Docker found."
}

install_pip_deps() {
    local missing=()

    python3 -c "import huggingface_hub" 2>/dev/null || missing+=("huggingface_hub")
    python3 -c "import hf_transfer" 2>/dev/null || missing+=("hf_transfer")

    if [[ ${#missing[@]} -eq 0 ]]; then
        _ok "Python dependencies already installed."
        return
    fi

    _info "Installing Python packages: ${missing[*]} ..."
    pip install --quiet --break-system-packages "${missing[@]}" 2>/dev/null \
        || pip install --quiet "${missing[@]}"
    _ok "Python dependencies installed."
}

# ── Serve ────────────────────────────────────────────────────────────────────

resolve_requested_port() {
    local port="8000"
    local i=0
    local count=${#SERVE_ARGS[@]}

    while [[ $i -lt $count ]]; do
        if [[ "${SERVE_ARGS[$i]}" == "--port" && $((i + 1)) -lt $count ]]; then
            port="${SERVE_ARGS[$((i + 1))]}"
            break
        fi
        i=$((i + 1))
    done

    printf '%s
' "$port"
}

read_server_state() {
    local state_file="$SCRIPT_DIR/.server.json"
    [[ -f "$state_file" ]] || return 1

    python3 - "$state_file" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
try:
    payload = json.loads(state_path.read_text())
except Exception:
    raise SystemExit(1)

model = str(payload.get("model", "")).strip()
backend = str(payload.get("backend", "")).strip()
port = str(payload.get("port", "8000")).strip() or "8000"
print("".join([model, backend, port]))
PY
}

show_post_start_info() {
    local state_line
    local active_model="$MODEL"
    local active_backend="$BACKEND"
    local active_port
    active_port="$(resolve_requested_port)"

    if state_line="$(read_server_state 2>/dev/null)"; then
        IFS=$'' read -r active_model active_backend active_port <<< "$state_line"
    fi

    _ok "OpenAI-compatible API: http://localhost:${active_port}/v1"
    _ok "Official llama.cpp Web UI: http://localhost:${active_port}/"

    if [[ -n "$active_model" ]]; then
        _info "Recommended 30m benchmark: python3 server.py aider-bench ${active_model} --backend ${active_backend:-radv} --profile python-30m"
    fi
}

serve_model() {
    local serve_cmd=(python3 "$SCRIPT_DIR/server.py" serve)

    if [[ -n "$MODEL" ]]; then
        serve_cmd+=("$MODEL")
    fi
    if [[ -n "$BACKEND" ]]; then
        serve_cmd+=(--backend "$BACKEND")
    fi
    if [[ ${#SERVE_ARGS[@]} -gt 0 ]]; then
        serve_cmd+=("${SERVE_ARGS[@]}")
    fi

    if [[ -z "$MODEL" ]]; then
        _info "Model not specified, showing picker... (backend: ${BACKEND:-<default>})"
    else
        _info "Ready to serve! (backend: ${BACKEND:-<default>}, model: $MODEL)"
    fi
    echo

    if ! "${serve_cmd[@]}"; then
        return $?
    fi

    echo
    show_post_start_info
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    cd "$SCRIPT_DIR"
    parse_args "$@"

    local need_backend=0
    local need_model=0

    [[ -z "$BACKEND" ]] && need_backend=1
    [[ -z "$MODEL" ]] && need_model=1

    if [[ $need_backend -eq 1 || $need_model -eq 1 ]]; then
        echo
        _info "strix-halo-llamacpp — quick start (interactive mode)"
        echo
    fi

    check_docker
    install_pip_deps

    if [[ $need_backend -eq 1 ]]; then
        echo
        pick_backend
        BACKEND="$PICKED_BACKEND"
    fi

    echo
    serve_model
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
