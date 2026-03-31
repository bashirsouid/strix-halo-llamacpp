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
    python server.py download MODEL                       # download only
"""

from __future__ import annotations

import argparse
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

from models import MODELS, DEFAULT_MODEL, get_model, ModelConfig

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR  = Path(__file__).resolve().parent
LLAMA_SRC    = PROJECT_DIR / "llama.cpp"
LLAMA_BUILD  = LLAMA_SRC / "build"
LLAMA_SERVER = LLAMA_BUILD / "bin" / "llama-server"
LLAMA_BENCH  = LLAMA_BUILD / "bin" / "llama-bench"
PID_FILE     = PROJECT_DIR / ".server.pid"

# ── Environment for Vulkan on Strix Halo ─────────────────────────────────────

STRIX_ENV = {
    "AMD_VULKAN_ICD":            "RADV",      # Mesa RADV — change to AMDVLK to test
    "HSA_ENABLE_SDMA":           "0",         # critical for ROCm perf, harmless for Vulkan
    "GGML_VK_PREFER_HOST_MEMORY": "0",
}


# ── Colours ──────────────────────────────────────────────────────────────────

def _c(code: int, msg: str) -> str:
    return f"\033[{code}m{msg}\033[0m" if sys.stdout.isatty() else msg

def info(msg: str):  print(_c(36, f"  ℹ  {msg}"))
def ok(msg: str):    print(_c(32, f"  ✓  {msg}"))
def warn(msg: str):  print(_c(33, f"  ⚠  {msg}"), file=sys.stderr)
def fail(msg: str):  print(_c(31, f"  ✗  {msg}"), file=sys.stderr)


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
        # Glob pattern → use --include
        cmd = [cli, "download", repo, "--include", pattern, "--local-dir", local_dir]
    else:
        # Exact filename → pass as positional arg (more reliable)
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
        import glob

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

    # Fallback: use the Python API directly
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

        # Download to dest_dir directly for single-file models,
        # or to the parent when the include pattern has a subdir.
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
        # Wait up to 10s for clean exit
        for _ in range(20):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)  # check if still alive
            except OSError:
                break
        else:
            warn(f"PID {pid} still alive after 10s, sending SIGKILL")
            os.kill(pid, signal.SIGKILL)
    except OSError:
        info(f"PID {pid} already gone.")
    PID_FILE.unlink(missing_ok=True)


def wait_for_server(port: int = 8000, timeout: int = 300) -> bool:
    """Poll /v1/models until the server is ready."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    info(f"Waiting for server on port {port} (timeout {timeout}s) ...")

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    ok("Server is ready!")
                    ok(f"  API:     http://localhost:{port}/v1")
                    ok(f"  Models:  http://localhost:{port}/v1/models")
                    return True
        except Exception:
            pass
        time.sleep(2)

    fail(f"Server did not become ready within {timeout}s")
    return False


def launch_server(cfg: ModelConfig, port: int = 8000, backend: str = "radv",
                  extra_args: list[str] | None = None):
    """Start llama-server as a background process."""
    ensure_built()
    download_model(cfg)

    # Stop any existing server
    stop_server()

    # Override port in server args
    args = cfg.server_args()
    # Replace the default port with the requested one
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

    info(f"Starting: {cfg.name} ({cfg.alias})")
    info(f"Backend:  Vulkan ({backend.upper()})")
    if cfg.spec.strategy:
        info(f"Speculation: {cfg.spec.strategy}")
    info(f"Command:  {' '.join(cmd[:6])} ...")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    PID_FILE.write_text(str(proc.pid))

    # Stream output in background while polling for readiness
    import threading

    def _tail():
        assert proc.stdout is not None
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace").rstrip()
            print(f"  │ {decoded}")

    tailer = threading.Thread(target=_tail, daemon=True)
    tailer.start()

    if wait_for_server(port):
        info(f"Server running as PID {proc.pid}.  Stop with: python server.py stop")
    else:
        fail("Server failed to start.  Check output above.")
        proc.terminate()
        PID_FILE.unlink(missing_ok=True)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCH
# ═══════════════════════════════════════════════════════════════════════════════

def bench(port: int = 8000, model_alias: str | None = None, backend: str = "radv") -> float:
    """Quick throughput benchmark against a running server."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    # Try to auto-detect the model alias from the running server
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

    info(f"Benchmarking server on port {port} ...")
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
            results.append({"label": label, "tok_s": 0, "comp_tok": 0, "elapsed": 0, "prompt_tok": 0})
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

    # Compute average tok/s across successful runs
    successful = [r for r in results if r["tok_s"] > 0]
    avg_tok_s = sum(r["tok_s"] for r in successful) / len(successful) if successful else 0

    print()
    info(f"Average: {avg_tok_s:.1f} tok/s across {len(successful)} runs")

    # Append to results log
    if avg_tok_s > 0 and model_alias and model_alias != "unknown":
        report_file = PROJECT_DIR / "bench_results.jsonl"
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        record = {
            "timestamp": timestamp,
            "backend": backend,
            "model": model_alias,
            "avg_tok_s": round(avg_tok_s, 1),
        }
        with open(report_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        ok(f"Result appended to {report_file}")

    return avg_tok_s


def bench_single(model_alias: str, port: int = 8000, backend: str = "radv") -> float:
    """Start a model, benchmark it, stop it, return avg tok/s."""
    cfg = get_model(model_alias)
    if not cfg.is_downloaded:
        warn(f"Skipping {cfg.name} — not downloaded.")
        return 0.0

    info(f"═══ Benchmarking: {cfg.name} ({cfg.alias}) ═══")
    launch_server(cfg, port=port, backend=backend)

    try:
        avg = bench(port=port, model_alias=cfg.alias, backend=backend)
    except Exception as e:
        fail(f"Benchmark failed for {cfg.name}: {e}")
        avg = 0.0
    finally:
        stop_server()
        time.sleep(2)  # let the port clear

    return avg


def bench_all(port: int = 8000, backend: str = "radv"):
    """Benchmark every downloaded model, then print a summary report."""
    downloaded = [m for m in MODELS if m.is_downloaded]
    if not downloaded:
        fail("No models downloaded.  Run 'python server.py download MODEL' first.")
        sys.exit(1)

    info(f"Found {len(downloaded)} downloaded models.  Benchmarking each ...")
    print()

    results: list[tuple[str, str, float]] = []
    for cfg in downloaded:
        avg = bench_single(cfg.alias, port=port, backend=backend)
        results.append((cfg.alias, cfg.name, avg))
        print()

    # ── Summary report ──────────────────────────────────────────────────────
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print()
    print(f"  ══════════════════════════════════════════════════════════════")
    print(f"  Benchmark Report — {timestamp}")
    print(f"  Backend: Vulkan ({backend.upper()})")
    print(f"  ──────────────────────────────────────────────────────────────")
    print(f"  {'Model':<32s}  {'Avg tok/s':>10s}")
    print(f"  {'─'*32}  {'─'*10}")

    for alias, name, avg in sorted(results, key=lambda x: -x[2]):
        bar = "█" * int(avg / 2)  # simple visual bar
        print(f"  {name:<32s}  {avg:>8.1f}  {bar}")

    print(f"  ══════════════════════════════════════════════════════════════")
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
    print(f"  {'Alias':<28s} {'Name':<26s} {'Spec':<14s} {'Downloaded':>10s}")
    print(f"  {'─'*28} {'─'*26} {'─'*14} {'─'*10}")
    for m in MODELS:
        spec = m.spec.strategy or "none"
        dl = "✓" if m.is_downloaded else "✗"
        default = " ◀" if m.alias == DEFAULT_MODEL else ""
        print(f"  {m.alias:<28s} {m.name:<26s} {spec:<14s} {dl:>10s}{default}")
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
    p_serve.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                         help=f"Model alias or name (default: {DEFAULT_MODEL})")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--ctx", type=int, default=None,
                         help="Override context size")
    p_serve.add_argument("--threads", "-t", type=int, default=None,
                         help="Override thread count")
    p_serve.add_argument("--backend", choices=["radv", "amdvlk"], default="radv",
                         help="Vulkan driver (default: radv)")
    p_serve.add_argument("--no-spec", action="store_true",
                         help="Disable speculative decoding")
    p_serve.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                         help="Extra args passed to llama-server")

    # stop
    sub.add_parser("stop", help="Stop the running server")

    # bench — single model (against running server, or auto-start one)
    p_bench = sub.add_parser("bench",
        help="Benchmark a model (starts/stops automatically, or tests running server)")
    p_bench.add_argument("model", nargs="?", default=None,
                         help="Model to benchmark (omit to test currently running server)")
    p_bench.add_argument("--port", type=int, default=8000)
    p_bench.add_argument("--backend", choices=["radv", "amdvlk"], default="radv")

    # bench-all — benchmark every downloaded model
    p_ball = sub.add_parser("bench-all",
        help="Benchmark all downloaded models and print a comparison report")
    p_ball.add_argument("--port", type=int, default=8000)
    p_ball.add_argument("--backend", choices=["radv", "amdvlk"], default="radv")

    # download
    p_dl = sub.add_parser("download", help="Download a model without serving")
    p_dl.add_argument("model", help="Model alias or name")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "build":
        build_llamacpp(rebuild=args.rebuild)

    elif args.command == "list":
        list_models()

    elif args.command == "serve":
        cfg = get_model(args.model)
        if args.ctx is not None:
            cfg.ctx_size = args.ctx
        if args.threads is not None:
            cfg.threads = args.threads
        if args.no_spec:
            cfg.spec.strategy = None
        launch_server(cfg, port=args.port, backend=args.backend,
                      extra_args=args.extra)

    elif args.command == "stop":
        stop_server()

    elif args.command == "bench":
        if args.model:
            bench_single(args.model, port=args.port, backend=args.backend)
        else:
            bench(port=args.port, backend=args.backend)

    elif args.command == "bench-all":
        bench_all(port=args.port, backend=args.backend)

    elif args.command == "download":
        cfg = get_model(args.model)
        download_model(cfg)


if __name__ == "__main__":
    main()
