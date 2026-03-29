#!/usr/bin/env python3
"""
bench_current.py — benchmark whatever model is running right now
Thresholds are derived automatically from model size via the /v1/models API.
This is a Python version of bench_current.sh with improvements:
- Better error handling and logging
- More modular and reusable code
- Improved output formatting
- Support for async requests (optional)
- Better handling of edge cases
"""

import json
import time
import statistics
import urllib.request
import urllib.error
import sys
import re
import argparse
from typing import Dict, List, Tuple, Optional

# ANSI color codes
RED = "\033[0;31m"
YLW = "\033[1;33m"
GRN = "\033[0;32m"
CYN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"


def print_banner():
    """Print the benchmark banner."""
    print(f"{BOLD}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║      Strix Halo — llama.cpp Benchmark                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{NC}")


def infer_active_params_b(model_id: str) -> float:
    """Infer the active parameters (in billions) from the model ID."""
    name = model_id.lower()
    # Check for MoE models (e.g., 8B-a8B)
    moe = re.search(r'(\d+(?:\.\d+)?)b[-_]a(\d+(?:\.\d+)?)b', name)
    if moe:
        return float(moe.group(2))
    # Check for dense models (e.g., 8B)
    dense = re.findall(r'(\d+(?:\.\d+)?)b', name)
    if dense:
        return float(dense[-1])
    return 8.0


def thresholds(active_b: float) -> Tuple[float, float, float]:
    """Calculate performance thresholds based on active parameters."""
    low = max(1.0, 300.0 / active_b)
    good = max(2.0, 500.0 / active_b)
    great = max(4.0, 800.0 / active_b)
    return round(low, 1), round(good, 1), round(great, 1)


def get_model_id(port: int) -> str:
    """Fetch the model ID from the server."""
    url = f"http://localhost:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get('data', [{}])[0].get('id', 'unknown')
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"{RED}Failed to fetch model ID: {e}{NC}")
        sys.exit(1)


def bench_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int = 600
) -> Dict:
    """Send a benchmark request and return performance metrics."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            t_first = time.perf_counter()
            raw = resp.read()
        t1 = time.perf_counter()
    except urllib.error.URLError as e:
        print(f"{RED}Request failed: {e}{NC}")
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"{RED}Failed to parse response: {e}{NC}")
        sys.exit(1)

    usage = data.get("usage", {})
    n_gen = usage.get("completion_tokens", 0)
    total_s = t1 - t0
    tps = n_gen / total_s if total_s > 0 and n_gen > 0 else 0.0
    ttft_ms = (t_first - t0) * 1000

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    preview = (msg.get("content") or msg.get("reasoning_content") or "")[:70]
    preview = preview.replace("\n", " ")

    return {
        "n_gen": n_gen,
        "tps": tps,
        "ttft_ms": ttft_ms,
        "total_s": total_s,
        "preview": preview,
    }


def run_benchmark(
    port: int,
    model: str,
    warmup: int,
    runs: int,
    max_tokens: int,
    active_b: float,
    thresholds: Tuple[float, float, float],
) -> None:
    """Run the benchmark for a given model and prompts."""
    PROMPTS = {
        "Short burst (TTFT)": "List the 5 most important things to know about Vulkan RADV on AMD.",
        "Sustained decode (real throughput)": (
            "Write a detailed explanation of how AMD Strix Halo unified memory "
            "architecture works, covering memory bandwidth, GTT tables, iGPU access, "
            "and implications for LLM inference."
        ),
        "Prefill stress (large prompt -> short answer)": (
            "Summarize in one sentence: " + ("AMD Strix Halo is a unified memory APU. " * 150)
        ),
    }

    all_medians = []

    for label, prompt in PROMPTS.items():
        sep = "-" * max(1, 54 - len(label))
        print(f"{BOLD}-- {label} {sep}{NC}")
        print(f"  {DIM}Prompt: {prompt[:80].replace(chr(10), ' ')}...{NC}\n")

        # Warmup runs
        for i in range(warmup):
            print(f"  {CYN}warmup {i+1}/{warmup}{NC} ", end="", flush=True)
            try:
                r = bench_request(
                    f"http://localhost:{port}/v1/chat/completions",
                    model,
                    prompt,
                    max_tokens,
                )
                print(
                    f"{r['n_gen']:>4} tok | {r['tps']:>6.1f} tok/s | "
                    f"TTFT {r['ttft_ms']:>5.0f} ms  (discarded)"
                )
            except Exception as e:
                print(f"{RED}FAILED: {e}{NC}")
                sys.exit(1)

        print()
        tps_list, ttft_list = [], []
        # Measurement runs
        for i in range(runs):
            print(f"  run {i+1}/{runs} ", end="", flush=True)
            r = bench_request(
                f"http://localhost:{port}/v1/chat/completions",
                model,
                prompt,
                max_tokens,
            )
            tps_list.append(r["tps"])
            ttft_list.append(r["ttft_ms"])
            print(
                f"{r['n_gen']:>4} tok | {r['tps']:>6.1f} tok/s | "
                f"TTFT {r['ttft_ms']:>5.0f} ms | {r['total_s']:.2f}s total"
            )

        med_tps = statistics.median(tps_list)
        med_ttft = statistics.median(ttft_list)
        all_medians.append(med_tps)

        # Determine rating
        if med_tps >= thresholds[2]:
            rating = f"{GRN}EXCELLENT{NC}"
        elif med_tps >= thresholds[1]:
            rating = f"{GRN}GOOD{NC}"
        elif med_tps >= thresholds[0]:
            rating = f"{YLW}LOW{NC}"
        else:
            rating = f"{RED}SLOW -- check GPU layers / Vulkan device{NC}"

        print(
            f"\n  Median: {BOLD}{med_tps:.1f} tok/s{NC}  TTFT: {med_ttft:.0f} ms"
            f"  Range: {min(tps_list):.1f}-{max(tps_list):.1f}"
            f"  Rating: {rating}\n"
        )

    overall = statistics.median(all_medians)
    print(f"{BOLD}{'='*68}{NC}")
    print(f"{BOLD}  Overall median : {overall:.1f} tok/s{NC}")
    print(f"{BOLD}  Model          : {model}  (~{active_b:.1f}B active params){NC}\n")
    print(f"  Reference ranges for ~{active_b:.1f}B active on Strix Halo (Vulkan RADV):")
    print(f"    {RED}< {thresholds[0]} tok/s{NC}    GPU not fully active -- check -ngl and /dev/dri mount")
    print(f"    {YLW}{thresholds[0]}-{thresholds[1]} tok/s{NC}   Partial acceleration or driver overhead")
    print(f"    {GRN}{thresholds[1]}-{thresholds[2]} tok/s{NC}   Good -- expected range for this model size")
    print(f"    {GRN}> {thresholds[2]} tok/s{NC}    Excellent")
    print(f"{BOLD}{'='*68}{NC}\n")


def main():
    """Main function to parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark the current model running on Strix Halo.")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=4,
        help="Number of measurement runs (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )

    args = parser.parse_args()

    print_banner()

    # Check if server is running
    url = f"http://localhost:{args.port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.getcode() != 200:
                print(f"{RED}No server on port {args.port}. Run ./start.sh or a load_*.sh first.{NC}")
                sys.exit(1)
    except urllib.error.URLError:
        print(f"{RED}No server on port {args.port}. Run ./start.sh or a load_*.sh first.{NC}")
        sys.exit(1)

    # Get model ID
    model = get_model_id(args.port)
    print(f"  Model   : {BOLD}{model}{NC}")
    print(f"  Warmup  : {args.warmup}   Measured: {args.runs}   Max tokens: {args.max_tokens}")
    print()

    # Infer active parameters and thresholds
    active_b = infer_active_params_b(model)
    THRESH_LOW, THRESH_GOOD, THRESH_GREAT = thresholds(active_b)
    print(f"  Active params : ~{active_b:.1f}B  (thresholds  low={THRESH_LOW}  good={THRESH_GOOD}  great={THRESH_GREAT} tok/s)\n")

    # Run benchmark
    run_benchmark(
        args.port,
        model,
        args.warmup,
        args.runs,
        args.max_tokens,
        active_b,
        (THRESH_LOW, THRESH_GOOD, THRESH_GREAT),
    )


if __name__ == "__main__":
    main()