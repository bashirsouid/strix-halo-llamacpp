#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
cd "$SCRIPT_DIR"

mode="single"
args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            if [[ "$mode" != "single" ]]; then
                echo "Cannot combine --all with another benchmark mode" >&2
                exit 1
            fi
            mode="all"
            shift
            ;;
        --parallel)
            if [[ "$mode" != "single" ]]; then
                echo "Cannot combine --parallel with another benchmark mode" >&2
                exit 1
            fi
            mode="parallel"
            shift
            ;;
        -h|--help)
            cat <<'USAGE'
Usage:
  ./benchmark-run.sh [MODEL] [server.py bench args]
  ./benchmark-run.sh --all [server.py bench-all args]
  ./benchmark-run.sh --parallel [MODEL] [server.py bench-parallel args]

Examples:
  ./benchmark-run.sh nemotron-nano-q4 --backend radv
  ./benchmark-run.sh --all --backend rocm
  ./benchmark-run.sh --parallel nemotron-nano-q4 --max-np 10
USAGE
            exit 0
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

case "$mode" in
    single)
        python3 server.py bench "${args[@]}"
        python3 tools/bench_viewer.py results/benchmark/bench_results.jsonl
        ;;
    all)
        python3 server.py bench-all "${args[@]}"
        python3 tools/bench_viewer.py results/benchmark/bench_results.jsonl
        ;;
    parallel)
        python3 server.py bench-parallel "${args[@]}"
        python3 tools/parallel_viewer.py results/benchmark/bench_parallel_results.jsonl
        ;;
esac
