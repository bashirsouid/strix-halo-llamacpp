#!/usr/bin/env bash
# start.sh — one-command bootstrap for strix-halo-llamacpp.
#
# Installs system + pip dependencies (skipping what's already present),
# builds llama.cpp from source if this is the first run, then serves
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
    rt=$(command -v podman || command -v docker || true)
    
    # Check if container runtime is available
    if [[ -z "$rt" ]]; then
        _warn "No container runtime (podman or docker) found."
        _warn "Install with: sudo apt install podman"
    fi
    
    printf '\n'
    printf '  Available backends:\n'
    printf '\n'
    
    for i in {0..5}; do
        local backend="${backends[$i]}"
        local num=$((i + 1))
        local check_mark="·"
        
        # Get image ID to check if image exists (returns empty if not found)
        local img_id=""
        case "$backend" in
            radv) img_id=$("$rt" images --format '{{.ID}}' "$img_radv" 2>/dev/null || true) ;;
            amdvlk) img_id=$("$rt" images --format '{{.ID}}' "$img_amdvlk" 2>/dev/null || true) ;;
            rocm) img_id=$("$rt" images --format '{{.ID}}' "$img_rocm" 2>/dev/null || true) ;;
            rocm6) img_id=$("$rt" images --format '{{.ID}}' "$img_rocm6" 2>/dev/null || true) ;;
            rocm7) img_id=$("$rt" images --format '{{.ID}}' "$img_rocm7" 2>/dev/null || true) ;;
            rocm7-nightly) img_id=$("$rt" images --format '{{.ID}}' "$img_rocm7night" 2>/dev/null || true) ;;
        esac
        
        if [[ -n "$img_id" ]]; then
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

# ── 1. System dependencies ───────────────────────────────────────────────────

# Vulkan packages — only needed for Vulkan backend
DEPS=(
    "cmake|cmake|cmake"
    "ninja|ninja-build|ninja-build"
    "git|git|git"
    "g++|g++|gcc-c++"
    "/usr/lib/*/pkgconfig/vulkan.pc|libvulkan-dev|vulkan-loader-devel"
    "glslangValidator|glslang-tools|glslang"
    "glslc|glslc|shaderc"
    "spirv-val|spirv-tools|spirv-tools"
)

install_system_deps() {
    if [[ -z "$BACKEND" ]]; then
        _info "Backend not specified, showing picker..."
        pick_backend
        BACKEND="$PICKED_BACKEND"
        # Rebuild SERVE_ARGS with the chosen backend
        SERVE_ARGS=()
        SERVE_ARGS+=("--backend" "$BACKEND")
        if [[ -n "$MODEL" ]]; then
            SERVE_ARGS+=("$MODEL")
        fi
    fi
    
    # For ROCm backend, we only need git + container runtime — skip Vulkan deps
    if [[ "$BACKEND" == "rocm" ]] || [[ "$BACKEND" == "rocm6" ]] || [[ "$BACKEND" == "rocm7" ]] || [[ "$BACKEND" == "rocm7-nightly" ]]; then
        if command -v git &>/dev/null; then
            _ok "git installed (ROCm builds inside container)."
        else
            _warn "git is required. Install with: sudo apt install git"
            exit 1
        fi
        if command -v podman &>/dev/null || command -v docker &>/dev/null; then
            _ok "Container runtime found (for ROCm build)."
        elif command -v hipcc &>/dev/null; then
            _ok "Native hipcc found (for ROCm build)."
        else
            _warn "ROCm build needs podman or docker (hipcc not available on Debian Trixie)."
            _warn "Install with: sudo apt install podman"
            exit 1
        fi
        return
    fi

    local missing_apt=()
    local missing_dnf=()
    local need_install=false

    for entry in "${DEPS[@]}"; do
        IFS='|' read -r check apt_pkg dnf_pkg <<< "$entry"

        if [[ "$check" == */* ]]; then
            # shellcheck disable=SC2086
            if ! compgen -G $check >/dev/null 2>&1; then
                missing_apt+=("$apt_pkg")
                missing_dnf+=("$dnf_pkg")
                need_install=true
            fi
        else
            if ! command -v "$check" &>/dev/null; then
                missing_apt+=("$apt_pkg")
                missing_dnf+=("$dnf_pkg")
                need_install=true
            fi
        fi
    done

    if ! [ -f /usr/include/vulkan/vulkan.h ] 2>/dev/null; then
        missing_apt+=("vulkan-headers")
        missing_dnf+=("vulkan-headers")
        need_install=true
    fi

    if ! $need_install; then
        _ok "All system dependencies already installed."
        return
    fi

    _info "Missing packages detected — installing ..."

    if command -v apt-get &>/dev/null; then
        _info "Using apt (Debian/Ubuntu) ..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq "${missing_apt[@]}"
    elif command -v dnf &>/dev/null; then
        _info "Using dnf (Fedora) ..."
        sudo dnf install -y "${missing_dnf[@]}"
    else
        _warn "Unsupported package manager.  Install manually: cmake, ninja, g++, libvulkan-dev, glslang, spirv-tools, git"
        exit 1
    fi

    _ok "System dependencies installed."
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

# ── 3. Build llama.cpp (first time only) ─────────────────────────────────────

ensure_built() {
    local build_backend="vulkan"
    case "$BACKEND" in
        rocm|rocm6|rocm7|rocm7-nightly) build_backend="rocm" ;;
    esac

    if [[ "$BACKEND" == "rocm" ]] || [[ "$BACKEND" == "rocm6" ]] || [[ "$BACKEND" == "rocm7" ]] || [[ "$BACKEND" == "rocm7-nightly" ]]; then
        if [ -x "$SCRIPT_DIR/llama.cpp/build-rocm/bin/llama-server" ]; then
            _ok "llama.cpp (ROCm) already built."
            return
        fi
        _info "First run with ROCm — building llama.cpp ..."
        python3 "$SCRIPT_DIR/server.py" build --backend rocm
    else
        # Check new dir first, then legacy
        if [ -x "$SCRIPT_DIR/llama.cpp/build-vulkan/bin/llama-server" ]; then
            _ok "llama.cpp (Vulkan) already built."
            return
        fi
        if [ -x "$SCRIPT_DIR/llama.cpp/build/bin/llama-server" ]; then
            _ok "llama.cpp already built (legacy dir)."
            return
        fi
        _info "First run — building llama.cpp from source with Vulkan ..."
        python3 "$SCRIPT_DIR/server.py" build --backend vulkan
    fi
}

# ── 4. Serve ─────────────────────────────────────────────────────────────────

serve_model() {
    if [[ -z "$MODEL" ]]; then
        _info "Model not specified, showing picker..."
        echo
        python3 "$SCRIPT_DIR/server.py" serve "${SERVE_ARGS[@]}"
    else
        _info "Ready to serve! (backend: $BACKEND, model: $MODEL)"
        exec python3 "$SCRIPT_DIR/server.py" serve "${SERVE_ARGS[@]}"
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

install_system_deps
install_pip_deps
ensure_built

echo
serve_model
