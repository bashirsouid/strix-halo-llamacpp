#!/usr/bin/env python3
"""
bench_all.py — load each model via its loader script, run a single benchmark
pass per model, then print a summary table.

Usage:
    python bench_all_loaders.py ./load_*.sh
    python bench_all_loaders.py ./load_mistral-small-4_q4.sh ./load_qwen2.5-7b.sh
    python bench_all_loaders.py --port 8000 --max-tokens 256 ./load_*.sh
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED  = "\033[0;31m"
YLW  = "\033[1;33m"
GRN  = "\033[0;32m"
CYN  = "\033[0;36m"
MAG  = "\033[0;35m"
BOLD = "\033[1m"
DIM  = "\033[2m"
NC   = "\033[0m"

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS: Dict[str, str] = {
    "Short burst":     "List the 5 most important things to know about Vulkan RADV on AMD.",
    "Sustained decode": (
        "Write a detailed explanation of how AMD Strix Halo unified memory "
        "architecture works, covering memory bandwidth, GTT tables, iGPU access, "
        "and implications for LLM inference."
    ),
    "Prefill stress": (
        "Summarize in one sentence: " + ("AMD Strix Halo is a unified memory APU. " * 150)
    ),
}

PROMPT_LABELS = list(PROMPTS.keys())

# ── Data ─────────────────────────────────────────────────────────────────────
@dataclass
class PromptResult:
    label:    str
    tps:      float
    ttft_ms:  float
    n_gen:    int
    total_s:  float

@dataclass
class ModelResult:
    loader:       str
    model_id:     str
    active_b:     float
    prompts:      List[PromptResult] = field(default_factory=list)
    error:        Optional[str] = None

    @property
    def overall_tps(self) -> float:
        if not self.prompts:
            return 0.0
        return sum(p.tps for p in self.prompts) / len(self.prompts)

# ── Helpers ───────────────────────────────────────────────────────────────────
def print_banner(loaders: List[str]) -> None:
    print(f"\n{BOLD}╔══════════════════════════════════════════════════════════════════╗{NC}")
    print(f"{BOLD}║      Strix Halo — llama.cpp Multi-Model Benchmark                ║{NC}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════════════╝{NC}")
    print(f"  Loaders  : {len(loaders)}")
    for ldr in loaders:
        print(f"    {DIM}{ldr}{NC}")
    print()


def infer_active_params_b(model_id: str) -> float:
    name = model_id.lower()
    moe = re.search(r'(\d+(?:\.\d+)?)b[-_]a(\d+(?:\.\d+)?)b', name)
    if moe:
        return float(moe.group(2))
    dense = re.findall(r'(\d+(?:\.\d+)?)b', name)
    if dense:
        return float(dense[-1])
    return 8.0


def thresholds(active_b: float) -> Tuple[float, float, float]:
    low   = max(1.0, 300.0 / active_b)
    good  = max(2.0, 500.0 / active_b)
    great = max(4.0, 800.0 / active_b)
    return round(low, 1), round(good, 1), round(great, 1)


def rating_str(tps: float, thresh: Tuple[float, float, float]) -> str:
    low, good, great = thresh
    if tps >= great:
        return f"{GRN}EXCELLENT{NC}"
    if tps >= good:
        return f"{GRN}GOOD{NC}"
    if tps >= low:
        return f"{YLW}LOW{NC}"
    return f"{RED}SLOW{NC}"


def rating_plain(tps: float, thresh: Tuple[float, float, float]) -> str:
    low, good, great = thresh
    if tps >= great: return "EXCELLENT"
    if tps >= good:  return "GOOD"
    if tps >= low:   return "LOW"
    return "SLOW"


def wait_for_server(port: int, timeout: int = 120) -> bool:
    """Poll /v1/models until the server responds or timeout expires."""
    url = f"http://localhost:{port}/v1/models"
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.getcode() == 200:
                    return True
        except Exception:
            pass
        attempt += 1
        delay = min(3.0, 0.5 * attempt)
        print(f"  {DIM}waiting for server… ({attempt}) sleeping {delay:.1f}s{NC}", end="\r", flush=True)
        time.sleep(delay)
    return False


def get_model_id(port: int) -> str:
    url = f"http://localhost:{port}/v1/models"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    return data.get("data", [{}])[0].get("id", "unknown")


def bench_request(port: int, model: str, prompt: str, max_tokens: int) -> PromptResult:
    url  = f"http://localhost:{port}/v1/chat/completions"
    body = json.dumps({
        "model":      model,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream":     False,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        t_first = time.perf_counter()
        raw = resp.read()
    t1 = time.perf_counter()

    data   = json.loads(raw)
    usage  = data.get("usage", {})
    n_gen  = usage.get("completion_tokens", 0)
    total_s = t1 - t0
    tps    = n_gen / total_s if total_s > 0 and n_gen > 0 else 0.0
    ttft_ms = (t_first - t0) * 1000

    return PromptResult(label="", tps=tps, ttft_ms=ttft_ms, n_gen=n_gen, total_s=total_s)



def check_model_present(loader: str) -> bool:
    """Run the loader with BENCH_MODE=check to test if model files exist.
    Returns True if present, False if not downloaded. Raises on unexpected error."""
    loader_abs = os.path.abspath(loader)
    env = os.environ.copy()
    env["BENCH_MODE"] = "check"
    result = subprocess.run(
        ["bash", loader_abs],
        env=env,
        capture_output=True,
        cwd=os.path.dirname(loader_abs) or ".",
    )
    if result.returncode == 0:
        return True
    if result.returncode == 2:
        return False
    # Unexpected exit code — loader doesn't support check mode or errored;
    # fall through and let the real run decide.
    return True

def run_loader(loader: str) -> bool:
    """Run the loader shell script; return True on success."""
    print(f"\n{BOLD}{'─'*68}{NC}")
    print(f"{CYN}▶ Loading model via: {loader}{NC}")
    loader_abs = os.path.abspath(loader)
    result = subprocess.run(
        ["bash", loader_abs],
        capture_output=False,
        cwd=os.path.dirname(loader_abs) or ".",
    )
    if result.returncode != 0:
        print(f"{RED}  Loader exited with code {result.returncode} — skipping.{NC}")
        return False
    return True


# ── Core benchmark loop ───────────────────────────────────────────────────────
def benchmark_model(port: int, max_tokens: int) -> Tuple[str, List[PromptResult]]:
    model_id = get_model_id(port)
    print(f"  Model : {BOLD}{model_id}{NC}")
    active_b = infer_active_params_b(model_id)
    thresh   = thresholds(active_b)
    print(f"  Params: ~{active_b:.1f}B active  "
          f"(low={thresh[0]}  good={thresh[1]}  great={thresh[2]} tok/s)\n")

    results: List[PromptResult] = []

    for label, prompt in PROMPTS.items():
        sep = "─" * max(1, 52 - len(label))
        print(f"  {BOLD}·· {label} {sep}{NC}")
        print(f"  {DIM}  Prompt: {prompt[:90].replace(chr(10),' ')}…{NC}")
        print(f"  {'─'*62}")

        try:
            pr = bench_request(port, model_id, prompt, max_tokens)
        except Exception as e:
            print(f"  {RED}  FAILED: {e}{NC}\n")
            continue

        pr.label = label
        results.append(pr)

        r_str = rating_str(pr.tps, thresh)
        print(
            f"  {BOLD}{pr.tps:>6.1f} tok/s{NC}  "
            f"TTFT {pr.ttft_ms:>5.0f} ms  "
            f"{pr.n_gen} tok  "
            f"{pr.total_s:.1f}s  "
            f"→ {r_str}"
        )
        print()

    if results:
        overall = sum(r.tps for r in results) / len(results)
        o_str   = rating_str(overall, thresh)
        print(f"  {BOLD}Overall avg: {overall:.1f} tok/s{NC}  {o_str}\n")

    return model_id, results


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(all_results: List[ModelResult]) -> None:
    if not all_results:
        print(f"{RED}No results to summarise.{NC}")
        return

    col_model  = max(len("Model"), max(len(r.model_id) for r in all_results))
    col_prompt = 10   # each prompt column width

    header_prompts = [lbl[:col_prompt] for lbl in PROMPT_LABELS]
    total_width = col_model + 2 + len(header_prompts) * (col_prompt + 3) + 10 + 11

    print(f"\n{BOLD}╔{'═'*total_width}╗{NC}")
    print(f"{BOLD}║  Summary — tokens per second{'':>{total_width-30}}║{NC}")
    print(f"{BOLD}╚{'═'*total_width}╝{NC}\n")

    # Header row
    header = f"  {'Model':<{col_model}}"
    for lbl in header_prompts:
        header += f"  {lbl:>{col_prompt}}"
    header += f"  {'Overall':>9}  {'Rating':<10}"
    print(f"{BOLD}{header}{NC}")
    print(f"  {'─'*col_model}" + (f"  {'─'*col_prompt}" * len(header_prompts)) + "  ─────────  ──────────")

    for mr in all_results:
        if mr.error:
            row = f"  {mr.model_id:<{col_model}}"
            for _ in header_prompts:
                row += f"  {'ERR':>{col_prompt}}"
            row += f"  {'─':>9}  {RED}{mr.error[:10]}{NC}"
            print(row)
            continue

        thresh = thresholds(mr.active_b)
        pmap   = {p.label: p.tps for p in mr.prompts}

        row = f"  {mr.model_id:<{col_model}}"
        for lbl in PROMPT_LABELS:
            tps = pmap.get(lbl, 0.0)
            row += f"  {tps:>{col_prompt}.1f}"

        o = mr.overall_tps
        rating = rating_plain(o, thresh)
        color  = GRN if rating in ("GOOD", "EXCELLENT") else (YLW if rating == "LOW" else RED)
        row   += f"  {o:>9.1f}  {color}{rating:<10}{NC}"
        print(row)

    print()

    # Reference ranges
    if all_results:
        # use the first successful model for reference ranges
        ref = next((r for r in all_results if not r.error), None)
        if ref:
            thresh = thresholds(ref.active_b)
            print(f"  Reference ranges (~{ref.active_b:.1f}B active on Strix Halo Vulkan RADV):")
            print(f"    {RED}< {thresh[0]:>5} tok/s{NC}  GPU not fully active — check -ngl / /dev/dri mount")
            print(f"    {YLW}{thresh[0]:>5}–{thresh[1]:<5} tok/s{NC}  Partial acceleration or driver overhead")
            print(f"    {GRN}{thresh[1]:>5}–{thresh[2]:<5} tok/s{NC}  Good — expected range for this model size")
            print(f"    {GRN}>{thresh[2]:>6} tok/s{NC}  Excellent")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load each model via its loader script and run a single benchmark pass."
    )
    parser.add_argument(
        "loaders",
        nargs="+",
        metavar="LOADER",
        help="One or more loader shell scripts (e.g. ./load_*.sh)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="llama-server port (default: 8000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per prompt (default: 256)",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=180,
        help="Seconds to wait for server to become ready after each loader (default: 180)",
    )
    args = parser.parse_args()

    # Expand globs / validate paths
    loaders: List[str] = []
    for ldr in args.loaders:
        if not os.path.isfile(ldr):
            print(f"{YLW}Warning: {ldr} not found — skipping.{NC}")
            continue
        loaders.append(ldr)

    if not loaders:
        print(f"{RED}No valid loader scripts found. Exiting.{NC}")
        sys.exit(1)

    print_banner(loaders)

    all_results: List[ModelResult] = []

    for i, loader in enumerate(loaders, 1):
        print(f"\n{BOLD}[{i}/{len(loaders)}] {loader}{NC}")

        # ── Check model files present (no download) ──────────────────────────
        print(f"  {DIM}Checking model files…{NC}", end=" ", flush=True)
        if not check_model_present(loader):
            print(f"{YLW}NOT DOWNLOADED — skipping.{NC}")
            mr = ModelResult(loader=loader, model_id=os.path.basename(loader),
                             active_b=8.0, error="not downloaded")
            all_results.append(mr)
            continue
        print(f"{GRN}present.{NC}")

        # ── Run loader ────────────────────────────────────────────────────────
        if not run_loader(loader):
            mr = ModelResult(loader=loader, model_id=os.path.basename(loader),
                             active_b=8.0, error="loader failed")
            all_results.append(mr)
            continue

        # ── Wait for server ───────────────────────────────────────────────────
        print(f"\n  {CYN}Waiting for server on port {args.port}…{NC}")
        if not wait_for_server(args.port, timeout=args.server_timeout):
            print(f"\n  {RED}Server did not come up within {args.server_timeout}s — skipping.{NC}")
            mr = ModelResult(loader=loader, model_id=os.path.basename(loader),
                             active_b=8.0, error="timeout")
            all_results.append(mr)
            continue

        print(f"\r  {GRN}Server ready.{' '*40}{NC}\n")

        # ── Benchmark ─────────────────────────────────────────────────────────
        try:
            model_id, prompt_results = benchmark_model(args.port, args.max_tokens)
            active_b = infer_active_params_b(model_id)
            mr = ModelResult(
                loader=loader,
                model_id=model_id,
                active_b=active_b,
                prompts=prompt_results,
            )
        except Exception as e:
            print(f"  {RED}Benchmark error: {e}{NC}")
            mr = ModelResult(loader=loader, model_id=os.path.basename(loader),
                             active_b=8.0, error=str(e)[:30])

        all_results.append(mr)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*68}{NC}")
    print(f"{BOLD}  ALL MODELS DONE{NC}")
    print(f"{BOLD}{'═'*68}{NC}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
