#!/usr/bin/env bash
# Test suite for start.sh argument parsing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1" >&2; exit 1; }
info() { echo -e "${YELLOW}ℹ${NC} $1"; }

test_parse_backend() {
    info "Test 1: Backend and model both specified"
    
    set -- --backend rocm nemotron-nano-q4
    
    ALL_ARGS=("$@")
    BACKEND=""
    MODEL=""
    
    # Find --backend and its value
    for i in "${!ALL_ARGS[@]}"; do
        if [[ "${ALL_ARGS[$i]}" == "--backend" ]]; then
            if [[ -n "${ALL_ARGS[$((i+1))]:-}" ]]; then
                BACKEND="${ALL_ARGS[$((i+1))]}"
            fi
        fi
    done
    
    # Model parsing
    previous_was_flag_with_value=0
    skip_next=0
    
    for arg in "${ALL_ARGS[@]}"; do
        if [[ $skip_next -eq 1 ]]; then
            skip_next=0
            continue
        fi
        
        if [[ $previous_was_flag_with_value -eq 1 ]]; then
            previous_was_flag_with_value=0
            continue
        fi
        
        if [[ "$arg" == "--backend" ]]; then
            previous_was_flag_with_value=1
            continue
        fi
        
        if [[ "$arg" == --* ]]; then
            case "$arg" in
                --np|--ctx|--ctx-per-slot|--threads|-t|--host|--port|--extra)
                    previous_was_flag_with_value=1
                    continue
                    ;;
            esac
            continue
        fi
        
        if [[ -z "$MODEL" ]]; then
            MODEL="$arg"
        fi
    done
    
    if [[ "$BACKEND" == "rocm" ]] && [[ "$MODEL" == "nemotron-nano-q4" ]]; then
        pass "Test 1: Backend and model parsed correctly (backend=$BACKEND, model=$MODEL)"
    else
        fail "Test 1: Parsing failed (backend=$BACKEND, model=$MODEL)"
    fi
}

test_backend_only() {
    info "Test 2: Backend only (no model)"
    
    set -- --backend rocm
    
    ALL_ARGS=("$@")
    BACKEND=""
    MODEL=""
    
    for i in "${!ALL_ARGS[@]}"; do
        if [[ "${ALL_ARGS[$i]}" == "--backend" ]]; then
            if [[ -n "${ALL_ARGS[$((i+1))]:-}" ]]; then
                BACKEND="${ALL_ARGS[$((i+1))]}"
            fi
        fi
    done
    
    previous_was_flag_with_value=0
    skip_next=0
    
    for arg in "${ALL_ARGS[@]}"; do
        if [[ $skip_next -eq 1 ]]; then
            skip_next=0
            continue
        fi
        
        if [[ $previous_was_flag_with_value -eq 1 ]]; then
            previous_was_flag_with_value=0
            continue
        fi
        
        if [[ "$arg" == "--backend" ]]; then
            previous_was_flag_with_value=1
            continue
        fi
        
        if [[ "$arg" == --* ]]; then
            case "$arg" in
                --np|--ctx|--ctx-per-slot|--threads|-t|--host|--port|--extra)
                    previous_was_flag_with_value=1
                    continue
                    ;;
            esac
            continue
        fi
        
        if [[ -z "$MODEL" ]]; then
            MODEL="$arg"
        fi
    done
    
    if [[ "$BACKEND" == "rocm" ]] && [[ -z "$MODEL" ]]; then
        pass "Test 2: Backend only parsed correctly (backend=$BACKEND, model=$MODEL)"
    else
        fail "Test 2: Parsing failed (backend=$BACKEND, model=$MODEL)"
    fi
}

test_model_only() {
    info "Test 3: Model only (no backend)"
    
    set -- nemotron-nano-q4
    
    ALL_ARGS=("$@")
    BACKEND=""
    MODEL=""
    
    for i in "${!ALL_ARGS[@]}"; do
        if [[ "${ALL_ARGS[$i]}" == "--backend" ]]; then
            if [[ -n "${ALL_ARGS[$((i+1))]:-}" ]]; then
                BACKEND="${ALL_ARGS[$((i+1))]}"
            fi
        fi
    done
    
    previous_was_flag_with_value=0
    skip_next=0
    
    for arg in "${ALL_ARGS[@]}"; do
        if [[ $skip_next -eq 1 ]]; then
            skip_next=0
            continue
        fi
        
        if [[ $previous_was_flag_with_value -eq 1 ]]; then
            previous_was_flag_with_value=0
            continue
        fi
        
        if [[ "$arg" == "--backend" ]]; then
            previous_was_flag_with_value=1
            continue
        fi
        
        if [[ "$arg" == --* ]]; then
            case "$arg" in
                --np|--ctx|--ctx-per-slot|--threads|-t|--host|--port|--extra)
                    previous_was_flag_with_value=1
                    continue
                    ;;
            esac
            continue
        fi
        
        if [[ -z "$MODEL" ]]; then
            MODEL="$arg"
        fi
    done
    
    if [[ -z "$BACKEND" ]] && [[ "$MODEL" == "nemotron-nano-q4" ]]; then
        pass "Test 3: Model only parsed correctly (backend=$BACKEND, model=$MODEL)"
    else
        fail "Test 3: Parsing failed (backend=$BACKEND, model=$MODEL)"
    fi
}

test_no_args() {
    info "Test 4: No arguments"
    
    set --
    
    ALL_ARGS=("$@")
    BACKEND=""
    MODEL=""
    
    for i in "${!ALL_ARGS[@]}"; do
        if [[ "${ALL_ARGS[$i]}" == "--backend" ]]; then
            if [[ -n "${ALL_ARGS[$((i+1))]:-}" ]]; then
                BACKEND="${ALL_ARGS[$((i+1))]}"
            fi
        fi
    done
    
    previous_was_flag_with_value=0
    skip_next=0
    
    for arg in "${ALL_ARGS[@]}"; do
        if [[ $skip_next -eq 1 ]]; then
            skip_next=0
            continue
        fi
        
        if [[ $previous_was_flag_with_value -eq 1 ]]; then
            previous_was_flag_with_value=0
            continue
        fi
        
        if [[ "$arg" == "--backend" ]]; then
            previous_was_flag_with_value=1
            continue
        fi
        
        if [[ "$arg" == --* ]]; then
            case "$arg" in
                --np|--ctx|--ctx-per-slot|--threads|-t|--host|--port|--extra)
                    previous_was_flag_with_value=1
                    continue
                    ;;
            esac
            continue
        fi
        
        if [[ -z "$MODEL" ]]; then
            MODEL="$arg"
        fi
    done
    
    if [[ -z "$BACKEND" ]] && [[ -z "$MODEL" ]]; then
        pass "Test 4: No args parsed correctly (backend=$BACKEND, model=$MODEL)"
    else
        fail "Test 4: Parsing failed (backend=$BACKEND, model=$MODEL)"
    fi
}

test_with_flags() {
    info "Test 5: Backend, model, and extra flags"
    
    set -- --backend radv --np 4 --ctx 8192 nemotron-nano-q4 --verbose
    
    ALL_ARGS=("$@")
    BACKEND=""
    MODEL=""
    
    for i in "${!ALL_ARGS[@]}"; do
        if [[ "${ALL_ARGS[$i]}" == "--backend" ]]; then
            if [[ -n "${ALL_ARGS[$((i+1))]:-}" ]]; then
                BACKEND="${ALL_ARGS[$((i+1))]}"
            fi
        fi
    done
    
    previous_was_flag_with_value=0
    skip_next=0
    
    for arg in "${ALL_ARGS[@]}"; do
        if [[ $skip_next -eq 1 ]]; then
            skip_next=0
            continue
        fi
        
        if [[ $previous_was_flag_with_value -eq 1 ]]; then
            previous_was_flag_with_value=0
            continue
        fi
        
        if [[ "$arg" == "--backend" ]]; then
            previous_was_flag_with_value=1
            continue
        fi
        
        if [[ "$arg" == --* ]]; then
            case "$arg" in
                --np|--ctx|--ctx-per-slot|--threads|-t|--host|--port|--extra)
                    previous_was_flag_with_value=1
                    continue
                    ;;
            esac
            continue
        fi
        
        if [[ -z "$MODEL" ]]; then
            MODEL="$arg"
        fi
    done
    
    if [[ "$BACKEND" == "radv" ]] && [[ "$MODEL" == "nemotron-nano-q4" ]]; then
        pass "Test 5: Backend and model with flags (backend=$BACKEND, model=$MODEL)"
    else
        fail "Test 5: Parsing failed (backend=$BACKEND, model=$MODEL)"
    fi
}

# Run all tests
echo "Running start.sh argument parsing tests..."
echo

test_parse_backend
test_backend_only
test_model_only
test_no_args
test_with_flags

echo
pass "All argument parsing tests passed!"
