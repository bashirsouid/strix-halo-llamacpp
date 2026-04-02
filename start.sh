#!/usr/bin/env bash
# start.sh — one-command bootstrap for strix-halo-llamacpp.
#
# Installs system + pip dependencies (skipping what's already present),
# builds llama.cpp from source if this is the first run, then serves
# a model with the interactive picker.
#
# Usage:
#   ./start.sh                          # Vulkan (default)
#   ./start.sh --backend rocm           # ROCm (builds in container if needed)
#   ./start.sh --backend radv nemotron-nano-q4   # specific model + backend
#
# Safe to run repeatedly — every step is idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse --backend from args (default: radv) ───────────────────────────────

BACKEND="radv"
SERVE_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "--backend" ]]; then
        # Next arg is the backend value — handled below
        SERVE_ARGS+=("$arg")
    elif [[ "${SERVE_ARGS[*]:-}" == *"--backend" ]] && [[ -z "${BACKEND_SET:-}" ]]; then
        BACKEND="$arg"
        BACKEND_SET=1
        SERVE_ARGS+=("$arg")
    else
        SERVE_ARGS+=("$arg")
    fi
done

# Simpler parsing: just look for --backend VALUE in the args
for i in "${!SERVE_ARGS[@]}"; do
    if [[ "${SERVE_ARGS[$i]}" == "--backend" ]] && [[ -n "${SERVE_ARGS[$((i+1))]:-}" ]]; then
        BACKEND="${SERVE_ARGS[$((i+1))]}"
        break
    fi
done

# ── Colours ──────────────────────────────────────────────────────────────────

_info()  { printf '\033[36m  ℹ  %s\033[0m\n' "$*"; }
_ok()    { printf '\033[32m  ✓  %s\033[0m\n' "$*"; }
_warn()  { printf '\033[33m  ⚠  %s\033[0m\n' "$*" >&2; }

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
    # For ROCm backend, we only need git + container runtime — skip Vulkan deps
    if [[ "$BACKEND" == "rocm" ]]; then
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
    [[ "$BACKEND" == "rocm" ]] && build_backend="rocm"

    if [[ "$BACKEND" == "rocm" ]]; then
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
    _info "Ready to serve! (backend: $BACKEND)"
    exec python3 "$SCRIPT_DIR/server.py" serve "${SERVE_ARGS[@]}"
}

# ── Main ─────────────────────────────────────────────────────────────────────

_info "strix-halo-llamacpp — quick start (backend: $BACKEND)"
echo

install_system_deps
install_pip_deps
ensure_built

echo
serve_model
