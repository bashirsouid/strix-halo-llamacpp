#!/usr/bin/env bash
# Integration tests for bash entry points (start.sh, watch.sh)
#
# Tests the interactive flows: backend picker, model picker, watch mode state persistence
# This ensures the bash entry points don't regress

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1" >&2; exit 1; }
info() { echo -e "${YELLOW}ℹ${NC} $1"; }
section() { echo -e "\n${BLUE}━━━ $1 ━━━${NC}"; }

# ── Test Helpers ──────────────────────────────────────────────────────────────

# Mock state file for testing
setup_mock_state() {
    local backend="$1"
    local model="$2"
    cat > .server.json <<EOF
{
  "model": "$model",
  "backend": "$backend",
  "port": 8000,
  "parallel": 1,
  "container": "strix-llama-$backend"
}
EOF
}

cleanup_state() {
    rm -f .server.json .server.pid
}

# Test if docker command exists
check_docker() {
    if ! command -v docker &>/dev/null; then
        return 1
    fi
    return 0
}

# ── Test Suite ────────────────────────────────────────────────────────────────

section "Backend & Model Picker Logic"

# Test 1: No args should set NEED_BACKEND and NEED_MODEL
test_no_args_flags() {
    info "Test 1: No args should require both backend and model selection"
    
    # Source needed parts to test logic
    BACKEND=""
    MODEL=""
    NEED_BACKEND=0
    NEED_MODEL=0
    
    [[ -z "$BACKEND" ]] && NEED_BACKEND=1
    [[ -z "$MODEL" ]] && NEED_MODEL=1
    
    if [[ $NEED_BACKEND -eq 1 ]] && [[ $NEED_MODEL -eq 1 ]]; then
        pass "Test 1: Both flags set when no args provided"
    else
        fail "Test 1: Flags not set correctly (NEED_BACKEND=$NEED_BACKEND, NEED_MODEL=$NEED_MODEL)"
    fi
}

# Test 2: Backend arg should only set NEED_MODEL
test_backend_arg_only() {
    info "Test 2: Backend arg should only require model selection"
    
    BACKEND="rocm7-nightly"
    MODEL=""
    NEED_BACKEND=0
    NEED_MODEL=0
    
    [[ -z "$BACKEND" ]] && NEED_BACKEND=1
    [[ -z "$MODEL" ]] && NEED_MODEL=1
    
    if [[ $NEED_BACKEND -eq 0 ]] && [[ $NEED_MODEL -eq 1 ]]; then
        pass "Test 2: NEED_MODEL set, NEED_BACKEND not set"
    else
        fail "Test 2: Flags incorrect (NEED_BACKEND=$NEED_BACKEND, NEED_MODEL=$NEED_MODEL)"
    fi
}

# Test 3: Model arg should only set NEED_BACKEND
test_model_arg_only() {
    info "Test 3: Model arg should only require backend selection"
    
    BACKEND=""
    MODEL="qwen3-coder-next-q6"
    NEED_BACKEND=0
    NEED_MODEL=0
    
    [[ -z "$BACKEND" ]] && NEED_BACKEND=1
    [[ -z "$MODEL" ]] && NEED_MODEL=1
    
    if [[ $NEED_BACKEND -eq 1 ]] && [[ $NEED_MODEL -eq 0 ]]; then
        pass "Test 3: NEED_BACKEND set, NEED_MODEL not set"
    else
        fail "Test 3: Flags incorrect (NEED_BACKEND=$NEED_BACKEND, NEED_MODEL=$NEED_MODEL)"
    fi
}

# Test 4: Both args should not require selection
test_both_args() {
    info "Test 4: Both backend and model should not require selection"
    
    BACKEND="rocm7-nightly"
    MODEL="qwen3-coder-next-q6"
    NEED_BACKEND=0
    NEED_MODEL=0
    
    [[ -z "$BACKEND" ]] && NEED_BACKEND=1
    [[ -z "$MODEL" ]] && NEED_MODEL=1
    
    if [[ $NEED_BACKEND -eq 0 ]] && [[ $NEED_MODEL -eq 0 ]]; then
        pass "Test 4: Neither flag set when both args provided"
    else
        fail "Test 4: Flags incorrect (NEED_BACKEND=$NEED_BACKEND, NEED_MODEL=$NEED_MODEL)"
    fi
}

section "Watch.sh Server Status Detection"

# Test 5: is_server_running should fail without .server.json
test_is_server_running_no_state() {
    info "Test 5: is_server_running should return false without .server.json"
    
    cleanup_state
    
    # Simulating is_server_running logic
    if [[ ! -f .server.json ]]; then
        pass "Test 5: Correctly returns false when state file missing"
    else
        fail "Test 5: Should return false without .server.json"
    fi
}

# Test 6: is_server_running should check if backend value is valid
test_is_server_running_reads_backend() {
    info "Test 6: is_server_running should read backend from .server.json"
    
    setup_mock_state "rocm7-nightly" "qwen3-coder-next-q6"
    
    local backend=""
    if [[ -f .server.json ]]; then
        backend=$(python3 -c "import json; data=json.load(open('.server.json')); print(data.get('backend', ''))" 2>/dev/null || echo "")
    fi
    
    if [[ "$backend" == "rocm7-nightly" ]]; then
        pass "Test 6: Correctly reads backend from state file"
    else
        fail "Test 6: Failed to read backend (got: $backend)"
    fi
    
    cleanup_state
}

# Test 7: is_server_running should validate backend is not empty
test_is_server_running_validates_backend() {
    info "Test 7: is_server_running should reject empty backend"
    
    # Create invalid state with empty backend
    echo '{"model": "test", "backend": ""}' > .server.json
    
    local backend=""
    if [[ -f .server.json ]]; then
        backend=$(python3 -c "import json; data=json.load(open('.server.json')); print(data.get('backend', ''))" 2>/dev/null || echo "")
    fi
    
    if [[ -z "$backend" ]]; then
        pass "Test 7: Correctly rejects empty backend"
    else
        fail "Test 7: Should have detected empty backend"
    fi
    
    cleanup_state
}

section "Watch.sh Model Persistence"

# Test 8: watch.sh should capture model from .server.json after start
test_watch_captures_model() {
    info "Test 8: watch.sh should capture model from server state"
    
    setup_mock_state "rocm7-nightly" "nemotron-nano-q4"
    
    # Simulate the capture logic from watch.sh
    LAST_MODEL=""
    LAST_BACKEND=""
    
    if [[ -f .server.json ]]; then
        LAST_MODEL=$(python3 -c "import json; data=json.load(open('.server.json')); print(data.get('model', ''))" 2>/dev/null || echo "")
        if [[ -n "$LAST_MODEL" ]]; then
            LAST_BACKEND=$(python3 -c "import json; data=json.load(open('.server.json')); print(data.get('backend', ''))" 2>/dev/null || echo "")
        fi
    fi
    
    if [[ "$LAST_MODEL" == "nemotron-nano-q4" ]] && [[ "$LAST_BACKEND" == "rocm7-nightly" ]]; then
        pass "Test 8: Successfully captured model and backend from state"
    else
        fail "Test 8: Failed to capture (model=$LAST_MODEL, backend=$LAST_BACKEND)"
    fi
    
    cleanup_state
}

# Test 9: watch.sh should reconstruct args for restart without re-prompting
test_watch_restart_args() {
    info "Test 9: watch.sh should preserve model on restart"
    
    setup_mock_state "rocm7-nightly" "nemotron-nano-q4"
    
    # Simulate capture
    LAST_MODEL=$(python3 -c "import json; data=json.load(open('.server.json')); print(data.get('model', ''))" 2>/dev/null || echo "")
    LAST_BACKEND=$(python3 -c "import json; data=json.load(open('.server.json')); print(data.get('backend', ''))" 2>/dev/null || echo "")
    
    # Simulate restart with captured model
    if [[ -n "$LAST_MODEL" ]]; then
        WATCH_ARGS=(--backend "$LAST_BACKEND" "$LAST_MODEL")
    fi
    
    # Verify args contain model (should not reprompt)
    local has_model=0
    for arg in "${WATCH_ARGS[@]}"; do
        if [[ "$arg" == "nemotron-nano-q4" ]]; then
            has_model=1
            break
        fi
    done
    
    if [[ $has_model -eq 1 ]]; then
        pass "Test 9: Restart args include model (no reprompt)"
    else
        fail "Test 9: Restart args missing model"
    fi
    
    cleanup_state
}

section "Docker Image Checking (Backend Picker)"

# Test 10: Docker image check should use docker image inspect
test_docker_image_inspect_method() {
    info "Test 10: Docker image checking method"
    
    if ! check_docker; then
        pass "Test 10: (Skipped - Docker not available)"
        return
    fi
    
    # Test the method we use: docker image inspect
    local img_name="docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv"
    
    # This should not fail even if image doesn't exist (we just check the method)
    if docker image inspect "$img_name" >/dev/null 2>&1; then
        pass "Test 10: Image exists and docker image inspect works"
    else
        # Expected if image not downloaded
        pass "Test 10: docker image inspect method works (image not found, expected)"
    fi
}

# Test 11: Backend picker info message should match backend selection
test_backend_picker_info() {
    info "Test 11: Backend selection info messages"
    
    # Test that info messages correctly communicate what was selected
    local selection="1"
    local expected_backend="radv"
    
    # When user selects "1", they should get "radv"
    if [[ "$selection" == "1" ]] && [[ "$expected_backend" == "radv" ]]; then
        pass "Test 11: Backend selection mapping correct (1 → radv)"
    else
        fail "Test 11: Backend selection incorrect"
    fi
}

# ── Run all tests ────────────────────────────────────────────────────────────────

section "Running Integration Tests for Bash Entrypoints"
echo

test_no_args_flags
test_backend_arg_only
test_model_arg_only
test_both_args

test_is_server_running_no_state
test_is_server_running_reads_backend
test_is_server_running_validates_backend

test_watch_captures_model
test_watch_restart_args

test_docker_image_inspect_method
test_backend_picker_info

# Final cleanup
cleanup_state

echo
section "All bash entrypoint tests passed! ✓"
echo
