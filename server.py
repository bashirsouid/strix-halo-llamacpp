#!/usr/bin/env python3
"""
Strix Halo llama.cpp launcher.

Builds llama.cpp from source with Vulkan, downloads models from HF,
and serves them with tuned flags for AMD Strix Halo (Ryzen AI Max / gfx1151).

Usage:
    python server.py build   [--rebuild]                 # build llama.cpp
    python server.py list                                 # list models
    python server.py serve   [MODEL] [--port N] ...      # download + serve
    python server.py stop                                 # stop server
    python server.py bench   [MODEL]                      # quick benchmark
    python server.py bench-all                            # benchmark all models
    python server.py bench-parallel [MODEL]               # sweep --np to find sweet spot
    python server.py download MODEL                       # download only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

from models import MODELS, get_model, ModelConfig

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR  = Path(__file__).resolve().parent
LLAMA_SRC    = PROJECT_DIR / "llama.cpp"
LLAMA_BUILD  = LLAMA_SRC / "build"
LLAMA_SERVER = LLAMA_BUILD / "bin" / "llama-server"
LLAMA_BENCH  = LLAMA_BUILD / "bin" / "llama-bench"
PID_FILE     = PROJECT_DIR / ".server.pid"
STATE_FILE   = PROJECT_DIR / ".server.json"

# ── Environment for Vulkan on Strix Halo ─────────────────────────────────────

STRIX_ENV = {
    "AMD_VULKAN_ICD":            "RADV",
    "HSA_ENABLE_SDMA":           "0",
    "GGML_VK_PREFER_HOST_MEMORY": "0",
}


# ── Colours ──────────────────────────────────────────────────────────────────

def _c(code: int, msg: str) -> str:
    return f"\033[{code}m{msg}\033[0m" if sys.stdout.isatty() else msg

def info(msg: str):  print(_c(36, f"  ℹ  {msg}"))
def ok(msg: str):    print(_c(32, f"  ✓  {msg}"))
def warn(msg: str):  print(_c(33, f"  ⚠  {msg}"), file=sys.stderr)
def fail(msg: str):  print(_c(31, f"  ✗  {msg}"), file=sys.stderr)


# ── Model picker TUI ────────────────────────────────────────────────────────

def pick_model(prompt_text: str = "Pick a model") -> ModelConfig:
    """Show a numbered list of models and let the user pick one."""
    print()
    print(f"  {prompt_text}:")
    print()
    for i, m in enumerate(MODELS, 1):
        dl = _c(32, "✓") if m.is_downloaded else _c(90, "·")
        spec = f"  [{m.spec.strategy}]" if m.spec.strategy else ""
        par = f"  np={m.parallel_slots}" if m.parallel_slots > 1 else ""
        ctx_k = m.ctx_per_slot // 1024
        print(f"  {dl} {i:>2d}) {m.name:<34s}{_c(90, spec)}{_c(33, par)}  {_c(90, f'{ctx_k}K/slot')}")
    print()

    while True:
        try:
            raw = input(f"  Enter number (1-{len(MODELS)}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not raw:
            continue
        try:
            idx = int(raw)
            if 1 <= idx <= len(MODELS):
                chosen = MODELS[idx - 1]
                print()
                return chosen
        except ValueError:
            pass
        print(f"    Invalid choice. Enter a number between 1 and {len(MODELS)}.")


def resolve_model(model_arg: str | None, prompt_text: str = "Pick a model") -> ModelConfig:
    """Resolve a model from CLI arg, or show picker if None."""
    if model_arg:
        return get_model(model_arg)
    return pick_model(prompt_text)


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD
# ═══════════════════════════════════════════════════════════════════════════════

BUILD_DEPS_FEDORA = "cmake ninja-build gcc-c++ vulkan-headers vulkan-loader-devel glslang shaderc spirv-tools"
BUILD_DEPS_UBUNTU = "cmake ninja-build build-essential libvulkan-dev glslang-tools glslc spirv-tools"


def check_build_deps():
    """Warn if cmake or ninja are missing."""
    missing = []
    for tool in ("cmake", "ninja", "git"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        fail(f"Missing build tools: {', '.join(missing)}")
        info(f"Fedora:  sudo dnf install {BUILD_DEPS_FEDORA}")
        info(f"Ubuntu:  sudo apt install {BUILD_DEPS_UBUNTU}")
        sys.exit(1)


def build_llamacpp(rebuild: bool = False):
    """Clone (or pull) llama.cpp and build with Vulkan."""
    check_build_deps()

    if LLAMA_SRC.exists() and not rebuild:
        info("Updating llama.cpp ...")
        subprocess.run(["git", "pull", "--ff-only"], cwd=LLAMA_SRC, check=True)
    elif LLAMA_SRC.exists() and rebuild:
        info("Rebuilding — cleaning build directory ...")
        if LLAMA_BUILD.exists():
            shutil.rmtree(LLAMA_BUILD)
        subprocess.run(["git", "pull", "--ff-only"], cwd=LLAMA_SRC, check=True)
    else:
        info("Cloning llama.cpp ...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggml-org/llama.cpp.git",
             str(LLAMA_SRC)],
            check=True,
        )

    info("Configuring with Vulkan ...")
    subprocess.run(
        [
            "cmake", "-B", str(LLAMA_BUILD),
            "-G", "Ninja",
            "-DGGML_VULKAN=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=LLAMA_SRC,
        check=True,
    )

    ncpu = os.cpu_count() or 4
    info(f"Building (using {ncpu} cores) ...")
    subprocess.run(
        [
            "cmake", "--build", str(LLAMA_BUILD),
            "--config", "Release",
            f"-j{ncpu}",
            "--target", "llama-server", "llama-bench",
        ],
        check=True,
    )

    if LLAMA_SERVER.exists():
        ok(f"Build complete: {LLAMA_SERVER}")
    else:
        fail("Build finished but llama-server binary not found.")
        sys.exit(1)


def ensure_built():
    """Make sure llama-server exists; build if it doesn't."""
    if not LLAMA_SERVER.exists():
        warn("llama-server not found — building from source ...")
        build_llamacpp()


# ═══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def _find_hf_cli() -> str | None:
    """Find the HuggingFace CLI binary name."""
    for name in ("hf", "huggingface-cli"):
        if shutil.which(name):
            return name
    return None


def _hf_download_cli(cli: str, repo: str, pattern: str, local_dir: str):
    """Download via the `hf` or `huggingface-cli` binary."""
    is_glob = any(c in pattern for c in "*?[")

    if is_glob:
        cmd = [cli, "download", repo, "--include", pattern, "--local-dir", local_dir]
    else:
        cmd = [cli, "download", repo, pattern, "--local-dir", local_dir]

    info(f"Running: {' '.join(cmd)}")
    subprocess.run(
        cmd,
        env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        check=True,
    )


def _hf_download_python(repo: str, pattern: str, local_dir: str):
    """Download via the huggingface_hub Python API (fallback)."""
    info(f"Using Python API fallback to download from {repo} ...")
    code = textwrap.dedent(f"""\
        import os
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import snapshot_download, hf_hub_download

        repo = {repo!r}
        pattern = {pattern!r}
        local_dir = {local_dir!r}

        is_glob = any(c in pattern for c in "*?[")

        if is_glob:
            snapshot_download(repo, allow_patterns=[pattern], local_dir=local_dir)
        else:
            hf_hub_download(repo, filename=pattern, local_dir=local_dir)
    """)
    subprocess.run([sys.executable, "-c", code], check=True)


def _hf_download(repo: str, pattern: str, local_dir: str):
    """Download files from HuggingFace, trying CLI first then Python API."""
    cli = _find_hf_cli()

    if cli:
        try:
            _hf_download_cli(cli, repo, pattern, local_dir)
            return
        except subprocess.CalledProcessError:
            warn(f"CLI download failed with '{cli}', trying Python API fallback ...")

    try:
        _hf_download_python(repo, pattern, local_dir)
    except subprocess.CalledProcessError:
        fail("Download failed with both CLI and Python API.")
        fail(f"  Repo:    {repo}")
        fail(f"  Pattern: {pattern}")
        fail(f"  Dest:    {local_dir}")
        info("Try manually:  hf download {repo} {pattern} --local-dir {local_dir}")
        sys.exit(1)


def download_model(cfg: ModelConfig):
    """Download model (and draft model if configured) from Hugging Face."""
    if cfg.is_downloaded:
        ok(f"Model already on disk: {cfg.name}")
    else:
        info(f"Downloading {cfg.name} from {cfg.hf_repo} ...")
        cfg.dest_dir.mkdir(parents=True, exist_ok=True)

        local_dir = str(cfg.dest_dir)
        if "/" in cfg.download_include:
            local_dir = str(cfg.dest_dir.parent)

        _hf_download(cfg.hf_repo, cfg.download_include, local_dir)

        if cfg.is_downloaded:
            ok(f"Download complete: {cfg.model_path}")
        else:
            fail(f"Download finished but shard not found in {cfg.dest_dir}")
            fail(f"  Expected glob: {cfg.shard_glob}")
            info(f"  Check what was actually downloaded: ls -la {cfg.dest_dir}")
            info("You may need to adjust dest_dir or shard_glob in models.py.")
            sys.exit(1)

    # Draft model
    draft = cfg.spec.draft
    if draft is not None and not draft.path.exists():
        info(f"Downloading draft model from {draft.hf_repo} ...")
        draft.dest_dir.mkdir(parents=True, exist_ok=True)
        _hf_download(draft.hf_repo, draft.filename, str(draft.dest_dir))
        if draft.path.exists():
            ok(f"Draft model ready: {draft.path}")
        else:
            warn("Draft model download finished but file not found — "
                 "speculation will fall back to ngram/none.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def stop_server():
    """Stop a running llama-server if we have a PID file."""
    if not PID_FILE.exists():
        info("No server PID file found — nothing to stop.")
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        ok(f"Sent SIGTERM to PID {pid}")
        for _ in range(20):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except OSError:
                break
        else:
            warn(f"PID {pid} still alive after 10s, sending SIGKILL")
            os.kill(pid, signal.SIGKILL)
    except OSError:
        info(f"PID {pid} already gone.")
    PID_FILE.unlink(missing_ok=True)
    STATE_FILE.unlink(missing_ok=True)


def wait_for_server(port: int = 8000, timeout: int = 360, verbose: bool = False) -> bool:
    """Poll /v1/models until the server is ready."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    if verbose:
        info(f"Waiting for server on port {port} (timeout {timeout}s) ...")

    dots = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    if not verbose:
                        print()
                    ok(f"Server ready — http://localhost:{port}/v1")
                    return True
        except Exception:
            pass
        if not verbose:
            print(".", end="", flush=True)
            dots += 1
            if dots % 60 == 0:
                print()
        time.sleep(2)

    if not verbose:
        print()
    fail(f"Server did not become ready within {timeout}s")
    return False


def launch_server(cfg: ModelConfig, port: int = 8000, backend: str = "radv",
                  extra_args: list[str] | None = None, verbose: bool = False,
                  parallel_override: int | None = None,
                  ctx_override: int | None = None):
    """Start llama-server as a background process."""
    ensure_built()
    download_model(cfg)

    # Stop any existing server
    stop_server()

    np = parallel_override if parallel_override is not None else cfg.parallel_slots

    # Override port in server args
    args = cfg.server_args(parallel_override=parallel_override, ctx_override=ctx_override)
    try:
        idx = args.index("--port")
        args[idx + 1] = str(port)
    except ValueError:
        args += ["--port", str(port)]

    if extra_args:
        args += extra_args

    # Build environment
    env = {**os.environ, **STRIX_ENV}
    if backend == "amdvlk":
        env["AMD_VULKAN_ICD"] = "AMDVLK"
    else:
        env["AMD_VULKAN_ICD"] = "RADV"

    cmd = [str(LLAMA_SERVER)] + args

    # Compute display values
    if ctx_override is not None:
        total_ctx = ctx_override
    else:
        total_ctx = cfg.ctx_per_slot * np
    ctx_per = total_ctx // np if np > 0 else total_ctx

    # Header
    spec_str = f"  spec={cfg.spec.strategy}" if cfg.spec.strategy else ""
    par_str = f"  np={np}" if np > 1 else ""
    print()
    info(f"{cfg.name}  ·  {backend.upper()}  ·  ctx={total_ctx} ({ctx_per}/slot × {np}){par_str}{spec_str}")

    if verbose:
        info(f"Command: {' '.join(cmd[:8])} ...")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    PID_FILE.write_text(str(proc.pid))
    STATE_FILE.write_text(json.dumps({
        "model": cfg.alias,
        "backend": backend,
        "port": port,
        "parallel": np,
    }))

    # Stream output in background
    import threading

    def _tail():
        assert proc.stdout is not None
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if verbose:
                print(f"  │ {decoded}")

    tailer = threading.Thread(target=_tail, daemon=True)
    tailer.start()

    if not verbose:
        print("  Loading ", end="", flush=True)

    if wait_for_server(port, verbose=verbose):
        if verbose:
            info(f"PID {proc.pid}.  Stop with: python server.py stop")
    else:
        fail("Server failed to start.  Re-run with --verbose to see output.")
        proc.terminate()
        PID_FILE.unlink(missing_ok=True)
        STATE_FILE.unlink(missing_ok=True)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCH — single-request throughput
# ═══════════════════════════════════════════════════════════════════════════════

def bench(port: int = 8000, model_alias: str | None = None,
          backend: str = "radv") -> float:
    """Quick throughput benchmark against a running server (single request)."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    if model_alias is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
                data = json.loads(resp.read())
                models = data.get("data", [])
                if models:
                    model_alias = models[0].get("id", "unknown")
        except Exception:
            model_alias = "unknown"

    prompts = [
        ("short",  "Write a haiku about silicon."),
        ("medium", "Explain how a CPU cache hierarchy works in 200 words."),
        ("long",   "Write a detailed guide to setting up a home lab for running "
                    "local LLMs.  Cover hardware selection, OS setup, model "
                    "selection, and performance tuning.  Be thorough."),
    ]

    info(f"Benchmarking server on port {port} (single-request) ...")
    print()

    results = []
    for label, prompt in prompts:
        payload = json.dumps({
            "model": "model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            fail(f"  [{label}]  Request failed: {e}")
            results.append({"label": label, "tok_s": 0, "comp_tok": 0,
                            "elapsed": 0, "prompt_tok": 0})
            continue
        elapsed = time.perf_counter() - t0

        usage = data.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        comp_tok   = usage.get("completion_tokens", 0)
        tok_s = comp_tok / elapsed if elapsed > 0 else 0

        print(f"  [{label:6s}]  {comp_tok:4d} tokens in {elapsed:6.1f}s  →  "
              f"{tok_s:5.1f} tok/s  (prompt: {prompt_tok} tok)")
        results.append({"label": label, "tok_s": tok_s, "comp_tok": comp_tok,
                        "elapsed": elapsed, "prompt_tok": prompt_tok})

    successful = [r for r in results if r["tok_s"] > 0]
    avg_tok_s = sum(r["tok_s"] for r in successful) / len(successful) if successful else 0

    print()
    info(f"Average: {avg_tok_s:.1f} tok/s across {len(successful)} runs")

    # Append to results log
    if avg_tok_s > 0 and model_alias and model_alias != "unknown":
        report_file = PROJECT_DIR / "bench_results.jsonl"
        timestamp = time.strftime("%Y-%m-%d %H:%M")

        quant = ""
        try:
            quant = get_model(model_alias).quant
        except ValueError:
            pass

        record = {
            "timestamp": timestamp,
            "backend": backend,
            "model": model_alias,
            "quant": quant,
            "avg_tok_s": round(avg_tok_s, 1),
        }
        with open(report_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        ok(f"Result appended to {report_file}")

    return avg_tok_s


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCH — concurrent throughput (measures aggregate tok/s with parallel load)
# ═══════════════════════════════════════════════════════════════════════════════

def _fire_one_request(port: int, prompt: str, max_tokens: int = 256) -> dict:
    """Send a single completion request and return timing info."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": time.perf_counter() - t0}

    elapsed = time.perf_counter() - t0
    usage = data.get("usage", {})
    comp_tok = usage.get("completion_tokens", 0)
    return {
        "ok": True,
        "comp_tok": comp_tok,
        "elapsed": elapsed,
        "tok_s": comp_tok / elapsed if elapsed > 0 else 0,
    }


def bench_concurrent(port: int = 8000, n_concurrent: int = 1,
                     max_tokens: int = 256) -> dict:
    """Fire n_concurrent requests simultaneously and measure aggregate throughput.

    Returns:
        dict with keys: n_concurrent, total_tokens, wall_time, aggregate_tok_s,
                        per_request_avg_tok_s, requests_ok, requests_failed
    """
    prompt = (
        "Explain the differences between TCP and UDP protocols.  "
        "Cover reliability, ordering, use cases, and performance trade-offs.  "
        "Be detailed."
    )

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(_fire_one_request, port, prompt, max_tokens)
                   for _ in range(n_concurrent)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall_time = time.perf_counter() - t0

    ok_results = [r for r in results if r.get("ok")]
    fail_results = [r for r in results if not r.get("ok")]

    total_tokens = sum(r["comp_tok"] for r in ok_results)
    aggregate_tok_s = total_tokens / wall_time if wall_time > 0 else 0
    per_req_avg = (sum(r["tok_s"] for r in ok_results) / len(ok_results)
                   if ok_results else 0)

    return {
        "n_concurrent": n_concurrent,
        "total_tokens": total_tokens,
        "wall_time": round(wall_time, 2),
        "aggregate_tok_s": round(aggregate_tok_s, 1),
        "per_request_avg_tok_s": round(per_req_avg, 1),
        "requests_ok": len(ok_results),
        "requests_failed": len(fail_results),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCH-PARALLEL — sweep --np values to find throughput sweet spot
# ═══════════════════════════════════════════════════════════════════════════════

def bench_parallel(cfg: ModelConfig, port: int = 8000, backend: str = "radv",
                   max_np: int | None = None, max_tokens: int = 256,
                   rounds: int = 2):
    """Sweep --parallel from 1 to max_np, benchmarking each.

    For each --np value:
      1. Start the server with that --parallel setting
      2. Fire a single-request bench (measures per-request latency)
      3. Fire N concurrent requests (where N = --np) to measure aggregate throughput
      4. Stop the server

    Prints a comparison table and recommends the optimal --np.
    """
    if not cfg.is_downloaded:
        fail(f"Model not downloaded: {cfg.name}.  Run 'python server.py download' first.")
        sys.exit(1)

    upper = max_np if max_np is not None else cfg.max_parallel

    print()
    info(f"╔══════════════════════════════════════════════════════════════╗")
    info(f"║  Parallel Sweep: {cfg.name:<42s} ║")
    info(f"║  Backend: {backend.upper():<8s}  ctx/slot: {cfg.ctx_per_slot:<8d}  max_np: {upper:<4d}  ║")
    info(f"╚══════════════════════════════════════════════════════════════╝")
    print()

    results: list[dict] = []
    report_file = PROJECT_DIR / "bench_parallel_results.jsonl"

    for np_val in range(1, upper + 1):
        print()
        info(f"── Testing --parallel {np_val} of {upper} ─────────────────────────")

        try:
            launch_server(cfg, port=port, backend=backend,
                          parallel_override=np_val)
        except SystemExit:
            warn(f"Server failed to start with --parallel {np_val}.  "
                 f"Likely OOM — stopping sweep here.")
            break

        time.sleep(1)  # settle

        # Single-request throughput
        single_results = []
        for _ in range(rounds):
            r = _fire_one_request(port,
                "Explain the differences between TCP and UDP.  "
                "Cover reliability, ordering, use cases, and trade-offs.",
                max_tokens=max_tokens)
            if r.get("ok"):
                single_results.append(r["tok_s"])
        single_avg = (sum(single_results) / len(single_results)
                      if single_results else 0)

        # Concurrent throughput (fire np_val requests at once)
        concurrent_result = bench_concurrent(port=port, n_concurrent=np_val,
                                             max_tokens=max_tokens)

        result = {
            "np": np_val,
            "single_tok_s": round(single_avg, 1),
            "concurrent_agg_tok_s": concurrent_result["aggregate_tok_s"],
            "concurrent_per_req_tok_s": concurrent_result["per_request_avg_tok_s"],
            "wall_time": concurrent_result["wall_time"],
            "total_tokens": concurrent_result["total_tokens"],
            "requests_ok": concurrent_result["requests_ok"],
            "requests_failed": concurrent_result["requests_failed"],
        }
        results.append(result)

        # Log
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "model": cfg.alias,
            "backend": backend,
            "ctx_per_slot": cfg.ctx_per_slot,
            **result,
        }
        with open(report_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"  np={np_val}:  single={single_avg:.1f} tok/s  "
              f"concurrent_agg={concurrent_result['aggregate_tok_s']:.1f} tok/s  "
              f"({concurrent_result['requests_ok']}/{np_val} ok)")

        stop_server()
        time.sleep(2)

    # ── Summary table ──────────────────────────────────────────────────────
    if not results:
        fail("No successful benchmark runs.")
        return

    print()
    print(f"  ╔═══════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  Parallel Sweep Results — {cfg.name:<42s} ║")
    print(f"  ╠═══════╦═══════════════╦═══════════════════╦═══════════════════╦═══════╣")
    print(f"  ║  --np  ║  Single tok/s ║  Concurrent (agg) ║  Concurrent/req  ║  OK   ║")
    print(f"  ╠═══════╬═══════════════╬═══════════════════╬═══════════════════╬═══════╣")

    best_agg = max(results, key=lambda r: r["concurrent_agg_tok_s"])
    best_single = max(results, key=lambda r: r["single_tok_s"])

    for r in results:
        agg_mark = " ★" if r == best_agg else "  "
        single_mark = " ★" if r == best_single else "  "
        print(f"  ║  {r['np']:>3d}  ║  {r['single_tok_s']:>9.1f}{single_mark} ║  "
              f"{r['concurrent_agg_tok_s']:>13.1f}{agg_mark} ║  "
              f"{r['concurrent_per_req_tok_s']:>13.1f}   ║  {r['requests_ok']:>3d}  ║")

    print(f"  ╚═══════╩═══════════════╩═══════════════════╩═══════════════════╩═══════╝")
    print()

    # Recommendation
    info(f"★ Best single-request latency:   --parallel {best_single['np']}  "
         f"({best_single['single_tok_s']:.1f} tok/s)")
    info(f"★ Best aggregate throughput:      --parallel {best_agg['np']}  "
         f"({best_agg['concurrent_agg_tok_s']:.1f} tok/s aggregate)")
    print()

    if best_single["np"] == best_agg["np"]:
        ok(f"Recommendation: --parallel {best_agg['np']}  (best at both single and concurrent)")
    else:
        info(f"For interactive single-user:   parallel_slots = {best_single['np']}")
        info(f"For API / multi-agent:         parallel_slots = {best_agg['np']}")

    print()
    info(f"Update models.py:  parallel_slots={best_agg['np']}  for {cfg.alias}")
    ok(f"Full results logged to {report_file}")


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCH-ALL
# ═══════════════════════════════════════════════════════════════════════════════

def bench_single(model_alias: str, port: int = 8000, backend: str = "radv") -> float:
    """Start a model, benchmark it, stop it, return avg tok/s."""
    cfg = get_model(model_alias)
    if not cfg.is_downloaded:
        warn(f"Skipping {cfg.name} — not downloaded.")
        return 0.0

    info(f"═══ Benchmarking: {cfg.name} ({cfg.alias})  np={cfg.parallel_slots} ═══")
    launch_server(cfg, port=port, backend=backend)

    try:
        avg = bench(port=port, model_alias=cfg.alias, backend=backend)
    except Exception as e:
        fail(f"Benchmark failed for {cfg.name}: {e}")
        avg = 0.0
    finally:
        stop_server()
        time.sleep(2)

    return avg


def bench_all(port: int = 8000, backend: str = "radv"):
    """Benchmark every downloaded model, then print a summary report."""
    downloaded = [m for m in MODELS if m.is_downloaded]
    if not downloaded:
        fail("No models downloaded.  Run 'python server.py download MODEL' first.")
        sys.exit(1)

    info(f"Found {len(downloaded)} downloaded models.  Benchmarking each ...")
    print()

    results: list[tuple[str, str, float, int]] = []
    for cfg in downloaded:
        avg = bench_single(cfg.alias, port=port, backend=backend)
        results.append((cfg.alias, cfg.name, avg, cfg.parallel_slots))
        print()

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print()
    print(f"  ══════════════════════════════════════════════════════════════════")
    print(f"  Benchmark Report — {timestamp}")
    print(f"  Backend: Vulkan ({backend.upper()})")
    print(f"  ──────────────────────────────────────────────────────────────────")
    print(f"  {'Model':<32s}  {'np':>3s}  {'Avg tok/s':>10s}")
    print(f"  {'─'*32}  {'─'*3}  {'─'*10}")

    for alias, name, avg, np_val in sorted(results, key=lambda x: -x[2]):
        bar = "█" * int(avg / 2)
        print(f"  {name:<32s}  {np_val:>3d}  {avg:>8.1f}  {bar}")

    print(f"  ══════════════════════════════════════════════════════════════════")
    print()

    report_file = PROJECT_DIR / "bench_results.jsonl"
    if report_file.exists():
        ok(f"All results logged to {report_file}")
    info("Run again after updating llama.cpp to track regressions.")


# ═══════════════════════════════════════════════════════════════════════════════
#  LIST
# ═══════════════════════════════════════════════════════════════════════════════

def list_models():
    """Print a table of available models."""
    print()
    print(f"  {'Alias':<28s} {'Quant':<14s} {'np':>3s} {'ctx/slot':>9s} {'Spec':<14s} {'DL':>3s}")
    print(f"  {'─'*28} {'─'*14} {'─'*3} {'─'*9} {'─'*14} {'─'*3}")
    for m in MODELS:
        spec = m.spec.strategy or "—"
        dl = "✓" if m.is_downloaded else "·"
        quant = m.quant or "—"
        ctx_k = f"{m.ctx_per_slot // 1024}K"
        print(f"  {m.alias:<28s} {quant:<14s} {m.parallel_slots:>3d} {ctx_k:>9s} {spec:<14s} {dl:>3s}")
    print()
    if any(m.notes for m in MODELS):
        print("  Notes:")
        for m in MODELS:
            if m.notes:
                wrapped = textwrap.fill(m.notes, width=72, initial_indent="    ",
                                        subsequent_indent="    ")
                print(f"  {m.alias}:")
                print(wrapped)
                print()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Strix Halo llama.cpp launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # build
    p_build = sub.add_parser("build", help="Build llama.cpp from source with Vulkan")
    p_build.add_argument("--rebuild", action="store_true",
                         help="Clean build directory and rebuild")

    # list
    sub.add_parser("list", help="List available models")

    # serve
    p_serve = sub.add_parser("serve", help="Download + launch a model")
    p_serve.add_argument("model", nargs="?", default=None,
                         help="Model alias or name (interactive picker if omitted)")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--ctx", type=int, default=None,
                         help="Override total context size")
    p_serve.add_argument("--ctx-per-slot", type=int, default=None,
                         help="Override context per slot (total = this × np)")
    p_serve.add_argument("--np", type=int, default=None, dest="parallel",
                         help="Override parallel slots")
    p_serve.add_argument("--threads", "-t", type=int, default=None,
                         help="Override thread count")
    p_serve.add_argument("--backend", choices=["radv", "amdvlk"], default="radv",
                         help="Vulkan driver (default: radv)")
    p_serve.add_argument("--no-spec", action="store_true",
                         help="Disable speculative decoding")
    p_serve.add_argument("--verbose", "-v", action="store_true",
                         help="Show full llama-server output while loading")
    p_serve.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                         help="Extra args passed to llama-server")

    # stop
    sub.add_parser("stop", help="Stop the running server")

    # bench
    p_bench = sub.add_parser("bench",
        help="Benchmark a model (starts/stops automatically, or tests running server)")
    p_bench.add_argument("model", nargs="?", default=None,
                         help="Model to benchmark (omit to test running server)")
    p_bench.add_argument("--port", type=int, default=8000)
    p_bench.add_argument("--backend", choices=["radv", "amdvlk"], default="radv")

    # bench-all
    p_ball = sub.add_parser("bench-all",
        help="Benchmark all downloaded models and print a comparison report")
    p_ball.add_argument("--port", type=int, default=8000)
    p_ball.add_argument("--backend", choices=["radv", "amdvlk"], default="radv")

    # bench-parallel  ← NEW
    p_bpar = sub.add_parser("bench-parallel",
        help="Sweep --parallel values to find optimal throughput")
    p_bpar.add_argument("model", nargs="?", default=None,
                        help="Model to sweep (interactive picker if omitted)")
    p_bpar.add_argument("--port", type=int, default=8000)
    p_bpar.add_argument("--backend", choices=["radv", "amdvlk"], default="radv")
    p_bpar.add_argument("--max-np", type=int, default=None,
                        help="Max --parallel value to test (default: model's max_parallel)")
    p_bpar.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens per request during benchmark (default: 256)")
    p_bpar.add_argument("--rounds", type=int, default=2,
                        help="Number of single-request rounds per --np value (default: 2)")

    # download
    p_dl = sub.add_parser("download", help="Download a model without serving")
    p_dl.add_argument("model", nargs="?", default=None,
                      help="Model alias or name (interactive picker if omitted)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "build":
        build_llamacpp(rebuild=args.rebuild)

    elif args.command == "list":
        list_models()

    elif args.command == "serve":
        cfg = resolve_model(args.model, "Pick a model to serve")

        # Apply overrides
        parallel_override = None
        ctx_override = None

        if args.parallel is not None:
            parallel_override = args.parallel
        if args.ctx is not None:
            ctx_override = args.ctx
        elif args.ctx_per_slot is not None:
            np = parallel_override if parallel_override is not None else cfg.parallel_slots
            ctx_override = args.ctx_per_slot * np
        if args.threads is not None:
            cfg.threads = args.threads
        if args.no_spec:
            cfg.spec.strategy = None

        launch_server(cfg, port=args.port, backend=args.backend,
                      extra_args=args.extra, verbose=args.verbose,
                      parallel_override=parallel_override,
                      ctx_override=ctx_override)

    elif args.command == "stop":
        stop_server()

    elif args.command == "bench":
        if args.model:
            bench_single(args.model, port=args.port, backend=args.backend)
        else:
            state_backend = "radv"
            state_model = None
            if STATE_FILE.exists():
                try:
                    state = json.loads(STATE_FILE.read_text())
                    state_backend = state.get("backend", "radv")
                    state_model = state.get("model")
                except (json.JSONDecodeError, KeyError):
                    pass
            bench(port=args.port, model_alias=state_model, backend=state_backend)

    elif args.command == "bench-all":
        bench_all(port=args.port, backend=args.backend)

    elif args.command == "bench-parallel":
        cfg = resolve_model(args.model, "Pick a model to sweep")
        bench_parallel(cfg, port=args.port, backend=args.backend,
                       max_np=args.max_np, max_tokens=args.max_tokens,
                       rounds=args.rounds)

    elif args.command == "download":
        cfg = resolve_model(args.model, "Pick a model to download")
        download_model(cfg)


if __name__ == "__main__":
    main()
