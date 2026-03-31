#!/usr/bin/env bash
# start.sh — one-command bootstrap for strix-halo-llamacpp.
#
# Installs system + pip dependencies (skipping what's already present),
# builds llama.cpp from source if this is the first run, then serves
# the default model (Mistral Small 4).
#
# Safe to run repeatedly — every step is idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────

_info()  { printf '\033[36m  ℹ  %s\033[0m\n' "$*"; }
_ok()    { printf '\033[32m  ✓  %s\033[0m\n' "$*"; }
_warn()  { printf '\033[33m  ⚠  %s\033[0m\n' "$*" >&2; }

# ── 1. System dependencies ───────────────────────────────────────────────────

# The packages we need, mapped to a binary or header we can check for.
# Format:  "check_path|apt_pkg|dnf_pkg"
DEPS=(
    "cmake|cmake|cmake"
    "ninja|ninja-build|ninja-build"
    "git|git|git"
    "g++|g++|gcc-c++"
    # Vulkan dev headers — check for the pkg-config file
    "/usr/lib/*/pkgconfig/vulkan.pc|libvulkan-dev|vulkan-loader-devel"
    "glslangValidator|glslang-tools|glslang"
    "glslc|glslc|shaderc"
    "spirv-val|spirv-tools|spirv-tools"
)

install_system_deps() {
    local missing_apt=()
    local missing_dnf=()
    local need_install=false

    for entry in "${DEPS[@]}"; do
        IFS='|' read -r check apt_pkg dnf_pkg <<< "$entry"

        # If check contains a '/', treat it as a glob path; otherwise as a command.
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

    # Also need vulkan-headers on apt (separate package from libvulkan-dev)
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
    elif command -v pacman &>/dev/null; then
        _warn "Arch detected — install manually: vulkan-devel glslang spirv-tools cmake ninja gcc git"
        exit 1
    else
        _warn "Unsupported package manager.  Install these manually:"
        _warn "  cmake, ninja, g++, libvulkan-dev, glslang, spirv-tools, git"
        exit 1
    fi

    _ok "System dependencies installed."
}

# ── 2. Python / pip dependencies ─────────────────────────────────────────────

install_pip_deps() {
    local missing=()

    # Check each package via importability (faster than pip show)
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
    if [ -x "$SCRIPT_DIR/llama.cpp/build/bin/llama-server" ]; then
        _ok "llama.cpp already built.  (Run 'python server.py build' to update.)"
        return
    fi

    _info "First run — building llama.cpp from source with Vulkan ..."
    python3 "$SCRIPT_DIR/server.py" build
}

# ── 4. Serve ─────────────────────────────────────────────────────────────────

serve_default() {
    _info "Launching default model ..."
    exec python3 "$SCRIPT_DIR/server.py" serve mistral-small-4
}

# ── Main ─────────────────────────────────────────────────────────────────────

_info "strix-halo-llamacpp — quick start"
echo

install_system_deps
install_pip_deps
ensure_built

echo
serve_default
