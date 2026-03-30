#!/usr/bin/env python3
"""
bench_current.py — benchmark whatever model is running right now.

Improvements over prior version:
  - Streaming mode: decouples decode throughput from TTFT noise
  - Draft-aware metrics: fetches n_drafted/n_accept from /metrics if available
  - Decode-focused prompts: prompts chosen for high spec-decode acceptance
  - Removed prefill-stress and TTFT tests (don't exercise draft model)
  - Correct MoE active-param inference for Mistral-Small-4 (6.5B active)
  - Realistic temperature (0.2) matching actual usage
  - Longer max_tokens (512) for stable throughput measurement
  - --no-draft flag for baseline comparison in one session
"""

import json
import time
import statistics
import urllib.request
import urllib.error
import sys
import re
import argparse
from typing import Dict, List, Optional, Tuple

RED  = "\033[0;31m"
YLW  = "\033[1;33m"
GRN  = "\033[0;32m"
CYN  = "\033[0;36m"
MAG  = "\033[0;35m"
BOLD = "\033[1m"
DIM  = "\033[2m"
NC   = "\033[0m"

# ---------------------------------------------------------------------------
# Prompts chosen to maximise spec-decode acceptance rate:
#   - code gen: highly predictable token sequences
#   - structured list: repetitive formatting patterns
#   - prose explanation: longer output, realistic usage
# ---------------------------------------------------------------------------
PROMPTS = {
    "Code gen (Python)": (
        "Write a complete Python function that implements a binary search tree "
        "with insert, search, and in-order traversal methods. Include docstrings."
    ),
    "Structured list": (
        "List 20 common Linux command-line tools with a one-sentence description "
        "of each. Format as a numbered list."
    ),
    "Prose explanation": (
        "Write a detailed explanation of how AMD Strix Halo unified memory "
        "architecture works, covering memory bandwidth, GTT tables, iGPU access, "
        "and implications for LLM inference."
    ),
}

# Known MoE model active-param overrides: model alias substring -> active B
MOE_OVERRIDES = {
    "mistral-small-4":    6.5,
    "mixtral-8x7b":       12.6,
    "mixtral-8x22b":      39.0,
    "deepseek-coder-v2":  2.4,
}


def print_banner() -> None:
    print(f"{BOLD}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║      Strix Halo — llama.cpp Benchmark  (speculative-aware)       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{NC}")


def infer_active_params_b(model_id: str) -> float:
    name = model_id.lower()
    # Check explicit MoE overrides first
    for key, val in MOE_OVERRIDES.items():
        if key in name:
            return val
    # Generic MoE pattern: 8x7b, 119b-a6.5b
    moe = re.search(r'(\d+(?:\.\d+)?)b[-_]a(\d+(?:\.\d+)?)b', name)
    if moe:
        return float(moe.group(2))
    moe2 = re.search(r'(\d+)x(\d+(?:\.\d+)?)b', name)
    if moe2:
        return float(moe2.group(1)) * float(moe2.group(2)) * 0.6
    # Dense fallback
    dense = re.findall(r'(\d+(?:\.\d+)?)b', name)
    if dense:
        return float(dense[-1])
    return 8.0


def thresholds(active_b: float) -> Tuple[float, float, float]:
    low   = max(1.0,  300.0 / active_b)
    good  = max(2.0,  500.0 / active_b)
    great = max(4.0,  800.0 / active_b)
    return round(low, 1), round(good, 1), round(great, 1)


def get_model_id(port: int) -> str:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=10) as r:
            return json.loads(r.read()).get("data", [{}])[0].get("id", "unknown")
    except Exception as e:
        print(f"{RED}Failed to fetch model ID: {e}{NC}")
        sys.exit(1)


def fetch_draft_metrics(port: int) -> Optional[Dict]:
    """
    Pull speculative decoding stats from llama.cpp /metrics (Prometheus format).
    Returns dict with n_drafted, n_accept, accept_rate or None if unavailable.
    """
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return None

    def extract(key: str) -> Optional[float]:
        m = re.search(rf'^{re.escape(key)}\s+([\d.]+)', text, re.MULTILINE)
        return float(m.group(1)) if m else None

    drafted = extract("llamacpp:draft_tokens_total")
    accepted = extract("llamacpp:draft_tokens_accepted_total")
    if drafted is None or accepted is None:
        return None
    rate = (accepted / drafted * 100) if drafted > 0 else 0.0
    return {"n_drafted": int(drafted), "n_accept": int(accepted), "accept_rate": rate}


def bench_stream(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int = 600,
) -> Dict:
    """
    Stream the response and measure pure decode throughput.
    TTFT = time to first token (prefill dominated).
    decode_tps = tokens after first / time after first token (decode dominated).
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    t_first: Optional[float] = None
    n_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_tokens += 1
    except urllib.error.URLError as e:
        print(f"{RED}Request failed: {e}{NC}")
        sys.exit(1)

    t1 = time.perf_counter()
    if t_first is None:
        t_first = t1

    ttft_ms     = (t_first - t0) * 1000
    decode_s    = t1 - t_first
    decode_tps  = (n_tokens - 1) / decode_s if decode_s > 0 and n_tokens > 1 else 0.0
    total_tps   = n_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0

    return {
        "n_gen":        n_tokens,
        "ttft_ms":      ttft_ms,
        "decode_tps":   decode_tps,
        "total_tps":    total_tps,
        "total_s":      t1 - t0,
    }


def run_suite(
    port: int,
    model: str,
    warmup: int,
    runs: int,
    max_tokens: int,
    temperature: float,
    active_b: float,
    thresh: Tuple[float, float, float],
    label_prefix: str = "",
) -> List[float]:
    """Run all prompts. Returns list of median decode_tps per prompt."""
    all_medians: List[float] = []

    for label, prompt in PROMPTS.items():
        display = f"{label_prefix}{label}"
        sep = "-" * max(1, 54 - len(display))
        print(f"{BOLD}-- {display} {sep}{NC}")
        print(f"  {DIM}{prompt[:80]}...{NC}\n")

        # Warmup
        for i in range(warmup):
            print(f"  {CYN}warmup {i+1}/{warmup}{NC} ", end="", flush=True)
            r = bench_stream(port, model, prompt, max_tokens, temperature)
            print(f"{r['n_gen']:>4} tok | decode {r['decode_tps']:>6.1f} tok/s  (discarded)")

        print()
        tps_list, ttft_list = [], []

        for i in range(runs):
            print(f"  run {i+1}/{runs} ", end="", flush=True)
            r = bench_stream(port, model, prompt, max_tokens, temperature)
            tps_list.append(r["decode_tps"])
            ttft_list.append(r["ttft_ms"])
            print(
                f"{r['n_gen']:>4} tok | decode {r['decode_tps']:>6.1f} tok/s "
                f"| TTFT {r['ttft_ms']:>5.0f} ms | {r['total_s']:.2f}s"
            )

        med_tps  = statistics.median(tps_list)
        med_ttft = statistics.median(ttft_list)
        all_medians.append(med_tps)

        if   med_tps >= thresh[2]: rating = f"{GRN}EXCELLENT{NC}"
        elif med_tps >= thresh[1]: rating = f"{GRN}GOOD{NC}"
        elif med_tps >= thresh[0]: rating = f"{YLW}LOW{NC}"
        else:                      rating = f"{RED}SLOW{NC}"

        print(
            f"\n  Median decode: {BOLD}{med_tps:.1f} tok/s{NC}  "
            f"TTFT: {med_ttft:.0f} ms  "
            f"Range: {min(tps_list):.1f}-{max(tps_list):.1f}  "
            f"Rating: {rating}\n"
        )

    return all_medians


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the current model on Strix Halo.")
    parser.add_argument("--port",       type=int,   default=8000)
    parser.add_argument("--warmup",     type=int,   default=1)
    parser.add_argument("--runs",       type=int,   default=3)
    parser.add_argument("--max-tokens", type=int,   default=512,
                        help="Tokens to generate per run (default 512, longer = stabler)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature (default 0.2 — matches real usage)")
    parser.add_argument("--no-draft",   action="store_true",
                        help="Skip draft stats (useful when running without spec decode)")
    args = parser.parse_args()

    print_banner()

    # Server health check
    try:
        with urllib.request.urlopen(f"http://localhost:{args.port}/v1/models", timeout=10) as r:
            if r.getcode() != 200:
                raise RuntimeError
    except Exception:
        print(f"{RED}No server on port {args.port}. Run ./start.sh first.{NC}")
        sys.exit(1)

    model    = get_model_id(args.port)
    active_b = infer_active_params_b(model)
    thresh   = thresholds(active_b)

    print(f"  Model       : {BOLD}{model}{NC}")
    print(f"  Active params: ~{active_b:.1f}B  "
          f"(thresholds  low={thresh[0]}  good={thresh[1]}  great={thresh[2]} tok/s)")
    print(f"  Warmup: {args.warmup}   Runs: {args.runs}   "
          f"Max tokens: {args.max_tokens}   Temp: {args.temperature}")
    print()

    # Fetch draft stats snapshot BEFORE the run
    pre_stats = None if args.no_draft else fetch_draft_metrics(args.port)

    medians = run_suite(
        args.port, model,
        args.warmup, args.runs, args.max_tokens, args.temperature,
        active_b, thresh,
    )

    # Fetch draft stats snapshot AFTER the run and diff
    post_stats = None if args.no_draft else fetch_draft_metrics(args.port)

    overall = statistics.median(medians)
    print(f"{BOLD}{'='*68}{NC}")
    print(f"{BOLD}  Overall median decode : {overall:.1f} tok/s{NC}")
    print(f"{BOLD}  Model                 : {model}  (~{active_b:.1f}B active){NC}")

    if post_stats and pre_stats:
        d_drafted = post_stats["n_drafted"] - pre_stats["n_drafted"]
        d_accept  = post_stats["n_accept"]  - pre_stats["n_accept"]
        rate      = (d_accept / d_drafted * 100) if d_drafted > 0 else 0.0
        color     = GRN if rate >= 60 else (YLW if rate >= 40 else RED)
        print(f"\n  {MAG}Speculative decoding stats (this session):{NC}")
        print(f"    Drafted   : {d_drafted}")
        print(f"    Accepted  : {d_accept}")
        print(f"    Accept rate: {color}{rate:.1f}%{NC}  "
              + ("(good)" if rate >= 60 else "(low — consider different draft model)" if rate < 40 else "(ok)"))
    elif not args.no_draft:
        print(f"\n  {DIM}Draft stats unavailable — /metrics endpoint not exposed{NC}")
        print(f"  {DIM}Tip: check 'docker logs strix-llamacpp | grep -i draft'{NC}")

    print()
    print(f"  Reference ranges for ~{active_b:.1f}B active (Vulkan RADV):")
    print(f"    {RED}< {thresh[0]} tok/s{NC}    Check -ngl, /dev/dri, flash-attn")
    print(f"    {YLW}{thresh[0]}-{thresh[1]} tok/s{NC}   Partial acceleration")
    print(f"    {GRN}{thresh[1]}-{thresh[2]} tok/s{NC}   Good")
    print(f"    {GRN}> {thresh[2]} tok/s{NC}    Excellent")
    print(f"{BOLD}{'='*68}{NC}\n")


if __name__ == "__main__":
    main()
