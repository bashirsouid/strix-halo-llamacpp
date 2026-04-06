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
cd "$SCRIPT_DIR"

# ── Parse arguments ──────────────────────────────────────────────────────────

# Use a simple approach: collect all args, extract backend and model
BACKEND=""
MODEL=""
ALL_ARGS=("$@")

# Find --backend and its value
for i in "${!ALL_ARGS[@]}"; do
    if [[ "${ALL_ARGS[$i]}" == "--backend" ]]; then
        if [[ -n "${ALL_ARGS[$((i+1))]:-}" ]]; then
            BACKEND="${ALL_ARGS[$((i+1))]}"
        fi
    fi
done

# Model is first non-flag positional arg after --backend VALUE
# We need to track if previous arg was a flag that takes a value
previous_was_flag_with_value=0
skip_next=0

for arg in "${ALL_ARGS[@]}"; do
    if [[ $skip_next -eq 1 ]]; then
        skip_next=0
        continue
    fi
    
    # If previous arg was a flag that takes a value, skip this arg
    if [[ $previous_was_flag_with_value -eq 1 ]]; then
        previous_was_flag_with_value=0
        continue
    fi
    
    if [[ "$arg" == "--backend" ]]; then
        previous_was_flag_with_value=1
        continue
    fi
    
    if [[ "$arg" == --* ]]; then
        # Check if this flag typically takes a value
        case "$arg" in
            --np|--ctx|--ctx-per-slot|--threads|-t|--host|--port|--extra)
                previous_was_flag_with_value=1
                continue
                ;;
        esac
        # Other flags (like --no-spec, --verbose) don't take values
        continue
    fi
    
    # This is a positional arg - if we haven't found a model yet, use it
    if [[ -z "$MODEL" ]]; then
        MODEL="$arg"
    fi
done

# Reconstruct SERVE_ARGS - include backend value and other flags, exclude model
SERVE_ARGS=()
SKIP_NEXT=0
for i in "${!ALL_ARGS[@]}"; do
    if [[ $SKIP_NEXT -eq 1 ]]; then
        SKIP_NEXT=0
        continue
    fi
    
    arg="${ALL_ARGS[$i]}"
    
    # Skip the model argument
    if [[ "$arg" == "$MODEL" ]]; then
        continue
    fi
    
    # Handle --backend specially - include both --backend and its value
    if [[ "$arg" == "--backend" ]]; then
        SERVE_ARGS+=("--backend" "$BACKEND")
        SKIP_NEXT=1
        continue
    fi
    
    SERVE_ARGS+=("$arg")
    
    # If this is a flag that takes a value, skip the next arg
    case "$arg" in
        --np|--ctx|--ctx-per-slot|--threads|-t|--host|--port|--extra)
            if [[ $((i+1)) -lt ${#ALL_ARGS[@]} ]]; then
                # Don't include value in SERVE_ARGS since we already added it above
                SKIP_NEXT=1
            fi
            ;;
    esac
done

# ── Colours ──────────────────────────────────────────────────────────────────

_info()  { printf '\033[36m  ℹ  %s\033[0m\n' "$*"; }
_ok()    { printf '\033[32m  ✓  %s\033[0m\n' "$*"; }
_warn()  { printf '\033[33m  ⚠  %s\033[0m\n' "$*" >&2; }

# ── Backend picker ───────────────────────────────────────────────────────────

pick_backend() {
    local backends=("radv" "amdvlk" "rocm" "rocm6" "rocm7" "rocm7-nightly")
    local desc_radv="Vulkan with RADV (fast generation)"
    local desc_amdvlk="Vulkan with AMDVLK (prompt processing)"
    local desc_rocm="ROCm HIP (fast prefill)"
    local desc_rocm6="ROCm 6.4.4"
    local desc_rocm7="ROCm 7.2"
    local desc_rocm7night="ROCm 7.2 nightly"
    
    # Container image tags for each backend
    local img_radv="docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv"
    local img_amdvlk="docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-amdvlk"
    local img_rocm="docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-nightly"
    local img_rocm6="docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-6.4.4"
    local img_rocm7="docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2"
    local img_rocm7night="docker.io/kyuz0/amd-strix-halo-toolboxes:rocm7-nightlies"
    
    # Container name mapping
    local name_radv="strix-llama-vulkan"
    local name_amdvlk="strix-llama-amdvlk"
    local name_rocm="strix-llama-rocm"
    local name_rocm6="strix-llama-rocm6"
    local name_rocm7="strix-llama-rocm7"
    local name_rocm7night="strix-llama-rocm7-nightly"
    
    # Find container runtime
    local rt
    rt=$(command -v docker || true)
    
    # Check if container runtime is available
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
        
        # Check if image exists locally using docker image inspect
        # This is more reliable than docker images with format
        local img_name=""
        case "$backend" in
            radv) img_name="$img_radv" ;;
            amdvlk) img_name="$img_amdvlk" ;;
            rocm) img_name="$img_rocm" ;;
            rocm6) img_name="$img_rocm6" ;;
            rocm7) img_name="$img_rocm7" ;;
            rocm7-nightly) img_name="$img_rocm7night" ;;
        esac
        
        # Check if image exists
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
            *)
                printf '    Invalid choice. Enter a number between 1 and 6.\n'
                ;;
        esac
    done
}

# ── Model picker (via server.py) ─────────────────────────────────────────────

pick_model() {
    python3 "$SCRIPT_DIR/server.py" serve
}

# ── 1. Docker check ──────────────────────────────────────────────────────────

check_docker() {
    if ! command -v docker &>/dev/null; then
        _warn "Docker not found. Install with: sudo apt install docker.io"
        exit 1
    fi
    _ok "Docker found."
}

# ── 2. Python / pip dependencies ─────────────────────────────────────────────

install_pip_deps() {
    local missing=()

    python3 -c "import huggingface_hub" 2>/dev/null || missing+=("huggingface_hub")
    python3 -c "import hf_transfer"     2>/dev/null || missing+=("hf_transfer")

    if [ ${#missing[@]} -eq 0 ]; then
        _ok "Python dependencies already installed."
        return
    fi

    _info "Installing Python packages: ${missing[*]} ..."
    pip install --quiet --break-system-packages "${missing[@]}" 2>/dev/null \
        || pip install --quiet "${missing[@]}"
    _ok "Python dependencies installed."
}

# ── 4. Serve ─────────────────────────────────────────────────────────────────

serve_model() {
    # Always add --backend if it's set (from picker or CLI)
    local serve_cmd=(python3 "$SCRIPT_DIR/server.py" serve "${SERVE_ARGS[@]}")
    if [[ -n "$BACKEND" ]]; then
        serve_cmd+=(--backend "$BACKEND")
    fi
    
    if [[ -z "$MODEL" ]]; then
        _info "Model not specified, showing picker... (backend: ${BACKEND:-<default>})"
        echo
        "${serve_cmd[@]}"
    else
        _info "Ready to serve! (backend: $BACKEND, model: $MODEL)"
        exec "${serve_cmd[@]}"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

# Determine if we need to prompt
NEED_BACKEND=0
NEED_MODEL=0

[[ -z "$BACKEND" ]] && NEED_BACKEND=1
[[ -z "$MODEL" ]] && NEED_MODEL=1

if [[ $NEED_BACKEND -eq 1 ]] || [[ $NEED_MODEL -eq 1 ]]; then
    echo
    _info "strix-halo-llamacpp — quick start (interactive mode)"
    echo
fi

check_docker
install_pip_deps

# Prompt for backend if needed
if [[ $NEED_BACKEND -eq 1 ]]; then
    echo
    pick_backend
    # IMPORTANT: Assign picked backend to BACKEND so it gets passed to serve_model
    BACKEND="$PICKED_BACKEND"
fi

echo
serve_model