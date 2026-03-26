#!/usr/bin/env bash
# bench_current.sh — benchmark whatever model is running right now
# Thresholds are derived automatically from model size via the /v1/models API.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

PORT="${SERVER_PORT:-8000}"
WARMUP="${1:-2}"
RUNS="${2:-4}"
MAX_TOKENS="${3:-256}"

echo -e "${BOLD}"
cat <<'BANNER'
╔══════════════════════════════════════════════════════════════════╗
║      Strix Halo — llama.cpp Benchmark                            ║
╚══════════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  _lib_fail "No server on port ${PORT}. Run ./start.sh or a load_*.sh first."
  exit 1
fi

MODEL_ID=$(curl -sf "http://localhost:${PORT}/v1/models" \
  | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; \
print(d[0]['id'] if d else 'unknown')" 2>/dev/null || echo "unknown")

echo -e "  Model   : ${BOLD}${MODEL_ID}${NC}"
echo -e "  Warmup  : ${WARMUP}   Measured: ${RUNS}   Max tokens: ${MAX_TOKENS}"
echo ""

python3 - \
  "$MODEL_ID" \
  "$PORT" \
  "$WARMUP" \
  "$RUNS" \
  "$MAX_TOKENS" \
<<'PYEOF'
import json, time, statistics, urllib.request, sys, re

MODEL      = sys.argv[1]
PORT       = int(sys.argv[2])
WARMUP     = int(sys.argv[3])
RUNS       = int(sys.argv[4])
MAX_TOKENS = int(sys.argv[5])

RED  = "\033[0;31m"; YLW = "\033[1;33m"; GRN = "\033[0;32m"
CYN  = "\033[0;36m"; BOLD = "\033[1m";   DIM = "\033[2m";  NC = "\033[0m"

def infer_active_params_b(model_id):
    name = model_id.lower()
    moe = re.search(r'(\d+(?:\.\d+)?)b[-_]a(\d+(?:\.\d+)?)b', name)
    if moe:
        return float(moe.group(2))
    dense = re.findall(r'(\d+(?:\.\d+)?)b', name)
    if dense:
        return float(dense[-1])
    return 8.0

def thresholds(active_b):
    low   = max(1.0,  300.0 / active_b)
    good  = max(2.0,  500.0 / active_b)
    great = max(4.0,  800.0 / active_b)
    return round(low, 1), round(good, 1), round(great, 1)

active_b = infer_active_params_b(MODEL)
THRESH_LOW, THRESH_GOOD, THRESH_GREAT = thresholds(active_b)

print(f"  Active params : ~{active_b:.1f}B  "
      f"(thresholds  low={THRESH_LOW}  good={THRESH_GOOD}  great={THRESH_GREAT} tok/s)\n")

PROMPTS = {
    "Short burst (TTFT)":
        "List the 5 most important things to know about Vulkan RADV on AMD.",
    "Sustained decode (real throughput)":
        ("Write a detailed explanation of how AMD Strix Halo unified memory "
         "architecture works, covering memory bandwidth, GTT tables, iGPU access, "
         "and implications for LLM inference."),
    "Prefill stress (large prompt -> short answer)":
        "Summarize in one sentence: " + ("AMD Strix Halo is a unified memory APU. " * 150),
}

def bench_request(prompt, max_tokens):
    url  = f"http://localhost:{PORT}/v1/chat/completions"
    body = json.dumps({
        "model":       MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "stream":      False,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        t_first = time.perf_counter()
        raw = resp.read()
    t1 = time.perf_counter()

    data    = json.loads(raw)
    usage   = data.get("usage", {})
    n_gen   = usage.get("completion_tokens", 0)
    total_s = t1 - t0
    tps     = n_gen / total_s if total_s > 0 and n_gen > 0 else 0.0
    ttft_ms = (t_first - t0) * 1000

    choice  = data.get("choices", [{}])[0]
    msg     = choice.get("message", {})
    preview = (msg.get("content") or msg.get("reasoning_content") or "")[:70]
    preview = preview.replace("\n", " ")

    return dict(n_gen=n_gen, tps=tps, ttft_ms=ttft_ms, total_s=total_s, preview=preview)

all_medians = []

for label, prompt in PROMPTS.items():
    sep = "-" * max(1, 54 - len(label))
    print(f"{BOLD}-- {label} {sep}{NC}")
    print(f"  {DIM}Prompt: {prompt[:80].replace(chr(10), ' ')}...{NC}\n")

    for i in range(WARMUP):
        print(f"  {CYN}warmup {i+1}/{WARMUP}{NC} ", end="", flush=True)
        try:
            r = bench_request(prompt, MAX_TOKENS)
            print(f"{r['n_gen']:>4} tok | {r['tps']:>6.1f} tok/s | "
                  f"TTFT {r['ttft_ms']:>5.0f} ms  (discarded)")
        except Exception as e:
            print(f"{RED}FAILED: {e}{NC}")
            sys.exit(1)

    print()
    tps_list, ttft_list = [], []
    for i in range(RUNS):
        print(f"  run {i+1}/{RUNS} ", end="", flush=True)
        r = bench_request(prompt, MAX_TOKENS)
        tps_list.append(r["tps"])
        ttft_list.append(r["ttft_ms"])
        print(f"{r['n_gen']:>4} tok | {r['tps']:>6.1f} tok/s | "
              f"TTFT {r['ttft_ms']:>5.0f} ms | {r['total_s']:.2f}s total")

    med_tps  = statistics.median(tps_list)
    med_ttft = statistics.median(ttft_list)
    all_medians.append(med_tps)

    if med_tps >= THRESH_GREAT:
        rating = f"{GRN}EXCELLENT{NC}"
    elif med_tps >= THRESH_GOOD:
        rating = f"{GRN}GOOD{NC}"
    elif med_tps >= THRESH_LOW:
        rating = f"{YLW}LOW{NC}"
    else:
        rating = f"{RED}SLOW -- check GPU layers / Vulkan device{NC}"

    print(f"\n  Median: {BOLD}{med_tps:.1f} tok/s{NC}  TTFT: {med_ttft:.0f} ms"
          f"  Range: {min(tps_list):.1f}-{max(tps_list):.1f}"
          f"  Rating: {rating}\n")

overall = statistics.median(all_medians)
print(f"{BOLD}{'='*68}{NC}")
print(f"{BOLD}  Overall median : {overall:.1f} tok/s{NC}")
print(f"{BOLD}  Model          : {MODEL}  (~{active_b:.1f}B active params){NC}\n")
print(f"  Reference ranges for ~{active_b:.1f}B active on Strix Halo (Vulkan RADV):")
print(f"    {RED}< {THRESH_LOW} tok/s{NC}    GPU not fully active -- check -ngl and /dev/dri mount")
print(f"    {YLW}{THRESH_LOW}-{THRESH_GOOD} tok/s{NC}   Partial acceleration or driver overhead")
print(f"    {GRN}{THRESH_GOOD}-{THRESH_GREAT} tok/s{NC}   Good -- expected range for this model size")
print(f"    {GRN}> {THRESH_GREAT} tok/s{NC}    Excellent")
print(f"{BOLD}{'='*68}{NC}\n")
PYEOF