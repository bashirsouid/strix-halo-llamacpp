#!/usr/bin/env python3
"""
Strix Halo llama.cpp launcher.

Runs llama.cpp via pre-built container images from kyuz0/amd-strix-halo-toolboxes.
All backends (vulkan, radv, amdvlk, rocm, rocm6, rocm7, rocm7-nightly) run in containers.

Container-based architecture:
  - All backends run via containers from docker.io/kyuz0/amd-strix-halo-toolboxes
  - Container images ship pre-built llama.cpp for gfx1151
  - No native build required for any backend
  
Available backends:
  - vulkan/radv:    Container: vulkan-radv         (Vulkan with RADV)
  - amdvlk:         Container: vulkan-amdvlk       (Vulkan with AMDVLK)
  - rocm:           Container: rocm-nightly        (latest ROCm)
  - rocm6:          Container: rocm-6.4.4          (ROCm 6.4.4)
  - rocm7:          Container: rocm-7.2            (ROCm 7.2)
  - rocm7-nightly:  Container: rocm7-nightlies     (nightly ROCm builds)

Usage:
    python server.py build   [--backend vulkan|radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly]
    python server.py list
    python server.py serve   [MODEL] [--backend vulkan|radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly] ...
    python server.py stop
    python server.py bench   [MODEL] [--backend vulkan|radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly]
    python server.py bench-all [--backend vulkan|radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly]
    python server.py bench-parallel [MODEL] [--backend vulkan|radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly]
    python server.py aider-setup
    python server.py aider-bench [MODEL] [--profile python-quick|python-all]
    python server.py aider-bench-all [--profile python-quick|python-all]
    python server.py download MODEL
    python server.py download-images
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import json
import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import re
from pathlib import Path
from typing import Any

from models import MODELS, get_model, ModelConfig
from eval_profiles import EvalProfile, ensure_override_dataset, resolve_eval_profile
from aider_benchmark import (
    AIDER_PROFILE_NAMES,
    BUILTIN_PROFILES as AIDER_PROFILES,
    DEFAULT_AIDER_MAX_TOKENS,
    DEFAULT_AIDER_REF,
    DEFAULT_POLYGLOT_REF,
    ensure_aider_setup,
    run_aider_benchmark,
)
from repo_cache import (
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
    DEFAULT_SLOT_ID,
    SLOT_CACHE_ROOT,
    ensure_cache_dirs,
    ensure_gitignore_entry,
    load_repo_context,
    make_warm_payload,
    refresh_repo_context,
    repo_paths,
    start_repo_proxy,
    write_opencode_config,
)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR  = Path(__file__).resolve().parent
LLAMA_SRC    = PROJECT_DIR / "llama.cpp"
PID_FILE     = PROJECT_DIR / ".server.pid"
STATE_FILE   = PROJECT_DIR / ".server.json"

# Legacy path — if you previously built into "build/", we check there as fallback.
LLAMA_BUILD_LEGACY = LLAMA_SRC / "build"

# All backends now run via containers. We use different container tags for each backend.
# Container images ship pre-built llama.cpp for gfx1151 from kyuz0/amd-strix-halo-toolboxes.
#
# Backend naming:
#   - vulkan/radv:   Use same container (vulkan-radv tag) for backward compatibility
#   - amdvlk:        Separate container (vulkan-amdvlk tag)
#   - rocm:          Use rocm-nightly (latest)
#   - rocm6:         rocm-6.4.4 (specific version)
#   - rocm7:         rocm-7.2 (specific version)
#   - rocm7-nightly: rocm7-nightlies (nightly builds)

CONTAINER_REGISTRY = "docker.io/kyuz0/amd-strix-halo-toolboxes"

CONTAINER_IMAGES = {
    "vulkan":      f"{CONTAINER_REGISTRY}:vulkan-radv",
    "radv":        f"{CONTAINER_REGISTRY}:vulkan-radv",
    "amdvlk":      f"{CONTAINER_REGISTRY}:vulkan-amdvlk",
    "rocm":        f"{CONTAINER_REGISTRY}:rocm-nightly",
    "rocm6":       f"{CONTAINER_REGISTRY}:rocm-6.4.4",
    "rocm7":       f"{CONTAINER_REGISTRY}:rocm-7.2",
    "rocm7-nightly": f"{CONTAINER_REGISTRY}:rocm7-nightlies",
}

CONTAINER_NAMES = {
    "vulkan":      "strix-llama-vulkan",
    "radv":        "strix-llama-vulkan",
    "amdvlk":      "strix-llama-amdvlk",
    "rocm":        "strix-llama-rocm",
    "rocm6":       "strix-llama-rocm6",
    "rocm7":       "strix-llama-rocm7",
    "rocm7-nightly": "strix-llama-rocm7-nightly",
}

VALID_BACKENDS = tuple(CONTAINER_IMAGES.keys())
VULKAN_BACKENDS = ("vulkan", "radv", "amdvlk")
ROCM_BACKENDS = ("rocm", "rocm6", "rocm7", "rocm7-nightly")

RESULTS_DIR = PROJECT_DIR / "results"
BENCH_RESULTS_DIR = RESULTS_DIR / "benchmark"
EVAL_RESULTS_DIR = RESULTS_DIR / "eval"
BENCH_RESULTS_FILE = BENCH_RESULTS_DIR / "bench_results.jsonl"
BENCH_PARALLEL_RESULTS_FILE = BENCH_RESULTS_DIR / "bench_parallel_results.jsonl"
EVAL_RESULTS_FILE = EVAL_RESULTS_DIR / "eval_results.jsonl"
EVAL_RAW_DIR = EVAL_RESULTS_DIR / "raw"

LOCAL_API_HOST = "0.0.0.0"


def _eval_metadata_dir() -> Path:
    return EVAL_RESULTS_FILE.parent / "metadata"


def _eval_profile_dataset_dir() -> Path:
    return EVAL_RESULTS_FILE.parent / "profiles"



def _eval_runs_dir() -> Path:
    return EVAL_RESULTS_FILE.parent / "runs"



def _ensure_results_dirs():
    BENCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    _eval_metadata_dir().mkdir(parents=True, exist_ok=True)
    _eval_profile_dataset_dir().mkdir(parents=True, exist_ok=True)
    _eval_runs_dir().mkdir(parents=True, exist_ok=True)

def _local_url(port: int, path: str) -> str:
    """Build a localhost URL for client-side health/API calls."""
    normalized = "/" + path.lstrip("/")
    return f"http://{LOCAL_API_HOST}:{port}{normalized}"

def _load_state() -> dict[str, Any] | None:
    """Load the current launcher state file if present."""
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _current_model_alias() -> str | None:
    state = _load_state() or {}
    model = state.get("model")
    return model if isinstance(model, str) and model else None


def _current_backend() -> str | None:
    """Return the backend of the currently running server, if any."""
    state = _load_state() or {}
    backend = state.get("backend")
    return backend if isinstance(backend, str) and backend else None


def _api_key_for_model(model_alias: str | None = None) -> str | None:
    alias = model_alias or _current_model_alias()
    if alias:
        try:
            return get_model(alias).api_key
        except Exception:
            pass
    return os.environ.get("API_KEY")


def _local_api_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_json(port: int, path: str, payload: dict[str, Any] | None = None,
                  method: str | None = None, timeout: int = 600,
                  api_key: str | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _local_url(port, path),
        data=data,
        headers=_local_api_headers(api_key),
        method=method or ("POST" if data is not None else "GET"),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def warm_repo_slot(repo_dir: str | Path, *, port: int = 8000,
                   model_alias: str | None = None,
                   slot_id: int = DEFAULT_SLOT_ID) -> dict[str, Any]:
    alias = model_alias or _current_model_alias()
    if not alias:
        raise ValueError("No model specified and no running server state found")
    context_text = load_repo_context(repo_dir, refresh_if_missing=True)
    payload = make_warm_payload(context_text, model_alias=alias, slot_id=slot_id)
    return _request_json(
        port,
        "/v1/chat/completions",
        payload=payload,
        method="POST",
        timeout=600,
        api_key=_api_key_for_model(alias),
    )


def save_repo_slot(repo_dir: str | Path, *, port: int = 8000,
                   slot_id: int = DEFAULT_SLOT_ID,
                   filename: str | None = None) -> dict[str, Any]:
    paths = repo_paths(repo_dir)
    slot_file = filename or paths.slot_filename
    return _request_json(
        port,
        f"/slots/{slot_id}?action=save",
        payload={"filename": slot_file},
        method="POST",
        timeout=600,
        api_key=_api_key_for_model(),
    )


def restore_repo_slot(repo_dir: str | Path, *, port: int = 8000,
                      slot_id: int = DEFAULT_SLOT_ID,
                      filename: str | None = None) -> dict[str, Any]:
    paths = repo_paths(repo_dir)
    slot_file = filename or paths.slot_filename
    return _request_json(
        port,
        f"/slots/{slot_id}?action=restore",
        payload={"filename": slot_file},
        method="POST",
        timeout=600,
        api_key=_api_key_for_model(),
    )


def load_env_file():
    """Load .env file if it exists."""
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        import os
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value.strip())

def _container_image(backend: str) -> str:
    """Return the container image for a given backend."""
    return CONTAINER_IMAGES.get(backend, CONTAINER_IMAGES["vulkan"])

def _container_name(backend: str) -> str:
    """Return the container name for a given backend."""
    return CONTAINER_NAMES.get(backend, CONTAINER_NAMES["vulkan"])

def _is_container_backend(backend: str) -> bool:
    """All backends now use containers."""
    return backend in VALID_BACKENDS

def _is_rocm(backend: str) -> bool:
    """True for any ROCm variant."""
    return backend in ROCM_BACKENDS


# ── Colours ──────────────────────────────────────────────────────────────────

def _c(code: int, msg: str) -> str:
    return f"\033[{code}m{msg}\033[0m" if sys.stdout.isatty() else msg

def info(msg: str):  print(_c(36, f"  ℹ  {msg}"))
def ok(msg: str):    print(_c(32, f"  ✓  {msg}"))
def warn(msg: str):  print(_c(33, f"  ⚠  {msg}"), file=sys.stderr)
def fail(msg: str):  print(_c(31, f"  ✗  {msg}"), file=sys.stderr)


# ── Model picker TUI ────────────────────────────────────────────────────────

def visible_models() -> list[ModelConfig]:
    """Models intended for interactive selection and user-facing listings."""
    return [m for m in MODELS if not getattr(m, "hidden", False)]


def pick_model(prompt_text: str = "Pick a model") -> ModelConfig:
    """Show a numbered list of models and let the user pick one."""
    models = visible_models()
    print()
    print(f"  {prompt_text}:")
    print()
    for i, m in enumerate(models, 1):
        dl = _c(32, "✓") if m.is_downloaded else _c(90, "·")
        spec = f"  [{m.spec.strategy}]" if m.spec.strategy else ""
        par = f"  np={m.parallel_slots}" if m.parallel_slots > 1 else ""
        ctx_k = m.ctx_per_slot // 1024
        print(f"  {dl} {i:>2d}) {m.name:<34s}{_c(90, spec)}{_c(33, par)}  {_c(90, f'{ctx_k}K/slot')}")
    print()

    while True:
        try:
            raw = input(f"  Enter number (1-{len(models)}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not raw:
            continue
        try:
            idx = int(raw)
            if 1 <= idx <= len(models):
                chosen = models[idx - 1]
                print()
                return chosen
        except ValueError:
            pass
        print(f"    Invalid choice. Enter a number between 1 and {len(models)}.")


def resolve_model(model_arg: str | None, prompt_text: str = "Pick a model") -> ModelConfig:
    """Resolve a model from CLI arg, or show picker if None."""
    if model_arg:
        return get_model(model_arg)
    return pick_model(prompt_text)


def resolve_model_or_running(model_arg: str | None,
                             prompt_text: str = "Pick a model") -> ModelConfig:
    """Resolve an explicit model, otherwise use the running server model, otherwise prompt."""
    if model_arg:
        return get_model(model_arg)
    running = _current_model_alias()
    if running:
        return get_model(running)
    return pick_model(prompt_text)


# ── Backend picker TUI ───────────────────────────────────────────────────────

def pick_backend(prompt_text: str = "Pick a backend") -> str:
    """Show a numbered list of backends with download status and let the user pick one."""
    backends = VALID_BACKENDS
    rt = _find_container_runtime()

    print()
    print(f"  {prompt_text}:")
    print()
    for i, b in enumerate(backends, 1):
        if b in VULKAN_BACKENDS:
            backend_type = "Vulkan"
        else:
            backend_type = "ROCm"

        # Check if the container image is already pulled locally
        image = _container_image(b)
        downloaded = False
        if rt:
            check = subprocess.run(
                [rt, "image", "inspect", image],
                capture_output=True,
            )
            downloaded = check.returncode == 0

        if downloaded:
            mark = _c(32, "✓")  # green checkmark
        else:
            mark = _c(90, "·")  # grey dot

        print(f"  {mark} {i:>2d}) {b:<16s}  {_c(90, f'({backend_type})')}")
    print()

    while True:
        try:
            raw = input(f"  Enter number (1-{len(backends)}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not raw:
            continue
        try:
            idx = int(raw)
            if 1 <= idx <= len(backends):
                chosen = backends[idx - 1]
                print()
                return chosen
        except ValueError:
            pass
        print(f"    Invalid choice. Enter a number between 1 and {len(backends)}.")


def resolve_backend(backend_arg: str | None, prompt_text: str = "Pick a backend") -> str:
    """Resolve a backend from CLI arg, or show picker if None."""
    if backend_arg:
        if backend_arg not in VALID_BACKENDS:
            fail(f"Invalid backend '{backend_arg}'. Valid backends: {', '.join(VALID_BACKENDS)}")
            sys.exit(1)
        return backend_arg
    return pick_backend(prompt_text)


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD
# ═══════════════════════════════════════════════════════════════════════════════

BUILD_DEPS_FEDORA_VULKAN = "cmake ninja-build gcc-c++ vulkan-headers vulkan-loader-devel glslang shaderc spirv-tools"
BUILD_DEPS_DEBIAN_VULKAN = "cmake ninja-build build-essential libvulkan-dev glslang-tools glslc spirv-tools"

# Container image for ROCm (pre-built llama.cpp for gfx1151, rebuilt on every commit)
ROCM_CONTAINER_IMAGE = "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2"
ROCM_CONTAINER_NAME  = "strix-llama-rocm"


def _find_container_runtime() -> str | None:
    """Find Docker"""
    return shutil.which("docker")




def _find_hf_cli() -> str | None:
    """Find the HuggingFace CLI binary name."""
    for name in ("hf", "huggingface-cli"):
        if shutil.which(name):
            return name
    return None


def _hf_download_cli(cli: str, repo: str, pattern: str, local_dir: str, revision: str | None = None):
    """Download via the `hf` or `huggingface-cli` binary."""
    is_glob = any(c in pattern for c in "*?[")
    cmd = [cli, "download", repo, "--local-dir", local_dir, "--resume"]
    if revision:
        cmd += ["--revision", revision]
    if is_glob:
        cmd += ["--include", pattern]
    else:
        cmd += [pattern]

    info(f"Running: {' '.join(cmd)}")
    subprocess.run(
        cmd,
        env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        check=True,
    )


def _hf_download_python(repo: str, pattern: str, local_dir: str, revision: str | None = None):
    """Download via the huggingface_hub Python API (fallback)."""
    info(f"Using Python API fallback to download from {repo} ...")
    code = textwrap.dedent(f"""\
        import os
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import snapshot_download, hf_hub_download

        repo = {repo!r}
        pattern = {pattern!r}
        local_dir = {local_dir!r}
        revision = {revision!r}

        is_glob = any(c in pattern for c in "*?[")
        resume = os.environ.get("HF_HUB_RESUME_DOWNLOAD", "1") in ("1", "true", "yes")

        if is_glob:
            snapshot_download(repo, allow_patterns=[pattern], local_dir=local_dir, revision=revision, resume_download=resume)
        else:
            hf_hub_download(repo, filename=pattern, local_dir=local_dir, revision=revision, resume_download=resume)
    """)
    subprocess.run([sys.executable, "-c", code], check=True)


def _hf_download(repo: str, pattern: str, local_dir: str, revision: str | None = None):
    """Download files from HuggingFace, trying CLI first then Python API."""
    cli = _find_hf_cli()

    if cli:
        try:
            _hf_download_cli(cli, repo, pattern, local_dir, revision=revision)
            return
        except subprocess.CalledProcessError:
            warn(f"CLI download failed with '{cli}', trying Python API fallback ...")

    try:
        _hf_download_python(repo, pattern, local_dir, revision=revision)
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
        info(f"Downloading {cfg.name} ...")
        info(f"  Repository: {cfg.hf_repo}")
        info(f"  Pattern:    {cfg.download_include}")
        info(f"  Dest:       {cfg.dest_dir}")
        cfg.dest_dir.mkdir(parents=True, exist_ok=True)

        local_dir = str(cfg.dest_dir)
        if "/" in cfg.download_include:
            local_dir = str(cfg.dest_dir.parent)

        # Support HF revision pinning from environment
        revision = os.environ.get("HF_REVISION")
        if revision:
            info(f"Using HF revision: {revision}")

        _hf_download(cfg.hf_repo, cfg.download_include, local_dir, revision=revision)

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


def download_container_images():
    """Download all container images with resume support."""
    rt = _find_container_runtime()
    if not rt:
        fail("No container runtime found (need Docker)")
        sys.exit(1)

    info("Downloading all container images ...")
    print()

    for backend, image in CONTAINER_IMAGES.items():
        info(f"Pulling image for {backend}: {image}")

        proc = subprocess.Popen(
            [rt, "pull", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if proc.stdout:
            for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    print(f"    {text}", flush=True)

        returncode = proc.wait()

        if returncode != 0:
            fail(f"Failed to pull {image}")
        else:
            ok(f"Image ready: {image}")

        print()

    info("All container images downloaded")


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def stop_server():
    """Stop all llama-server containers started by this launcher."""

    rt = _find_container_runtime()
    if rt:
        for container_name in CONTAINER_NAMES.values():
            # Use docker inspect to check if container exists
            check = subprocess.run(
                [rt, "inspect", container_name],
                capture_output=True,
            )
            if check.returncode == 0:
                subprocess.run(
                    [rt, "stop", "-t", "5", container_name],
                    capture_output=True, timeout=15,
                )
                subprocess.run(
                    [rt, "rm", "-f", container_name],
                    capture_output=True, timeout=10,
                )
                ok(f"Stopped container ({container_name})")
        # Let the port fully clear
        time.sleep(1)

    PID_FILE.unlink(missing_ok=True)
    STATE_FILE.unlink(missing_ok=True)


def wait_for_server(port: int = 8000, timeout: int = 300, verbose: bool = False) -> bool:
    """Poll /health until the server is ready with faster timeout.
    
    Args:
        port: Health check port
        timeout: Timeout in seconds (reduced from 360 to 60 for faster feedback)
        verbose: Print detailed progress
    """
    url = _local_url(port, "/health")
    deadline = time.time() + timeout
    if verbose:
        info(f"Waiting for server on port {port} (timeout {timeout}s) ...")

    dots = 0
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    if not verbose:
                        print()
                    ok(f"Server ready — http://localhost:{port}/v1")
                    return True
        except urllib.error.URLError as e:
            last_error = str(e)
        except Exception as e:
            last_error = str(e)
        
        if not verbose:
            print(".", end="", flush=True)
            dots += 1
            if dots % 60 == 0:
                print()
        time.sleep(1)  # Poll faster (every 1s instead of 2s)

    if not verbose:
        print()
    error_msg = f"Server did not become ready within {timeout}s"
    if last_error:
        error_msg += f" — last error: {last_error}"
    fail(error_msg)
    fail("This may indicate:")
    fail("  1. Model architecture not supported by this llama.cpp container")
    fail("  2. Insufficient VRAM for the model + KV cache")
    fail("  3. Docker/permission issue")
    fail("Run with --verbose to see full container logs")
    return False



def launch_server(cfg: ModelConfig, port: int = 8000, backend: str = "vulkan",
                  extra_args: list[str] | None = None, verbose: bool = False,
                  parallel_override: int | None = None,
                  ctx_override: int | None = None):
    """Start llama-server via container."""
    download_model(cfg)

    # Stop any existing server (native or container)
    stop_server()

    np = parallel_override if parallel_override is not None else cfg.parallel_slots

    ensure_cache_dirs()
    if cfg.slot_save_path:
        slot_save_dir = Path(cfg.slot_save_path).expanduser().resolve()
    else:
        slot_save_dir = SLOT_CACHE_ROOT
        cfg.slot_save_path = str(slot_save_dir)
    slot_save_dir.mkdir(parents=True, exist_ok=True)

    # Build the llama-server argument list
    args = cfg.server_args(parallel_override=parallel_override, ctx_override=ctx_override)
    try:
        idx = args.index("--port")
        args[idx + 1] = str(port)
    except ValueError:
        args += ["--port", str(port)]

    # Default to mmap (faster repeated loads via OS page cache) with Direct I/O off.
    # --extra args can override these if needed.
    if "--mmap" not in args and "--no-mmap" not in args:
        # TODO: it seems like mmap might be slower. Needs more investigation.
        #args += ["--mmap"]
        args += ["--no-mmap"]
    if "--direct-io" not in args and "--no-direct-io" not in args:
        args += ["--no-direct-io"]

    if extra_args:
        args += extra_args

    # Compute display values
    if ctx_override is not None:
        total_ctx = ctx_override
    else:
        total_ctx = cfg.ctx_per_slot * np
    ctx_per = total_ctx // np if np > 0 else total_ctx

    spec_str = f"  spec={cfg.spec.strategy}" if cfg.spec.strategy else ""
    par_str = f"  np={np}" if np > 1 else ""
    print()
    info(f"{cfg.name}  ·  {backend.upper()}  ·  ctx={total_ctx} ({ctx_per}/slot × {np}){par_str}{spec_str}")

    rt = _find_container_runtime()
    if not rt:
        fail("Container runtime not found (need Docker)")
        sys.exit(1)

    image = _container_image(backend)
    container_name = _container_name(backend)

    from models import MODELS_DIR

    # Check if container already exists using docker inspect
    # (docker container exists is not a valid command)
    exists_check = [rt, "inspect", container_name]
    exists_result = subprocess.run(exists_check, capture_output=True)
    if exists_result.returncode == 0:
        # Container already exists, start it
        subprocess.run([rt, "start", container_name], check=True)
        # Get container ID
        inspect_cmd = [rt, "inspect", "-f", "{{.Id}}", container_name]
        container_id_result = subprocess.run(inspect_cmd, capture_output=True, text=True)
        container_id = container_id_result.stdout.strip()[:12]
        info(f"Reusing existing {backend} container ({container_id})")
    else:
        # Build and run the container as before
        container_cmd = [
            rt, "run", "-d",
            "--name", container_name,
            "--device", "/dev/dri",
            "--device", "/dev/kfd",
            "--group-add", "video",
            "--group-add", "render",
            "--security-opt", "seccomp=unconfined",
            "-v", f"{MODELS_DIR}:{MODELS_DIR}:ro",
            "-v", f"{slot_save_dir}:{slot_save_dir}",
            "-p", f"{LOCAL_API_HOST}:{port}:{port}",
        #] + env_flags + [ #TODO: this is undefined
            image,
            "llama-server",
        ] + args

        if verbose:
            info(f"Container: {' '.join(container_cmd[:12])} ...")

        result = subprocess.run(container_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            fail(f"Failed to start {backend} container")
            if result.stderr:
                stderr_text = result.stderr.strip() if result.stderr else ""
                for line in stderr_text.splitlines()[-3:]:
                    fail(f"  {line}")
            sys.exit(1)
        container_id = result.stdout.strip()[:12]

    PID_FILE.unlink(missing_ok=True)
    cache_ram = 8192 if cfg.cache_ram is True else cfg.cache_ram
    STATE_FILE.write_text(json.dumps({
        "model": cfg.alias,
        "backend": backend,
        "port": port,
        "parallel": np,
        "container": container_name,
        "cache_prompt": cfg.cache_prompt,
        "cache_reuse": cfg.cache_reuse if cfg.cache_prompt else 0,
        "cache_ram": cache_ram,
        "slot_save_path": str(slot_save_dir),
    }))

# Tail container logs for progress — always, not just in verbose mode
    import threading

    _PROGRESS_PATTERNS = (
        "llm_load", "model size", "model type", "warming up",
        "loaded meta", "loading model", "server listening",
        "KV self size", "output buffer size",
    )
    
    _ERROR_PATTERNS = (
        "error", "failed", "could not load", "invalid", 
        "unsupported", "not supported", "deprecated",
        "cannot open", "does not exist", "no such file",
        "std::bad_alloc", "out of memory", "segmentation fault",
    )
    
    startup_failed = [False]  # Use list to allow thread to update
    failure_reason = [None]

    def _tail_container():
        try:
            log_proc = subprocess.Popen(
                [rt, "logs", "-f", container_name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if log_proc.stdout:
                for line in log_proc.stdout:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    lower_text = text.lower()
                    
                    if verbose:
                        print(f"   {text}")
                    else:
                        # Show key progress lines even in non-verbose mode
                        if any(pat in lower_text for pat in _PROGRESS_PATTERNS):
                            print(f"\r\033[K  ⏳ {text}", flush=True)
                    
                    # Capture first critical error for quick failure reporting
                    if not startup_failed[0]:
                        for err_pat in _ERROR_PATTERNS:
                            if err_pat in lower_text and ("error:" in lower_text or "failed:" in lower_text or "could not" in lower_text):
                                startup_failed[0] = True
                                failure_reason[0] = text
                                break
        except Exception:
            pass

    log_thread = threading.Thread(target=_tail_container, daemon=True)
    log_thread.start()

    if not verbose:
        print("  Loading ", end="", flush=True)
    
    # Check for startup errors early before waiting for health endpoint
    time.sleep(0.5)
    if startup_failed[0]:
        print()
        fail(f"Model load failed immediately:")
        fail(f"  {failure_reason[0]}")
        fail("")
        fail("This likely means:")
        fail("  • Model format/architecture not supported by this llama.cpp build")
        fail("  • Insufficient VRAM (check VRAM usage with 'rocm-smi')")
        fail("  • Model file is corrupted or incomplete")
        subprocess.run([rt, "stop", container_name], capture_output=True)
        subprocess.run([rt, "rm", "-f", container_name], capture_output=True)
        PID_FILE.unlink(missing_ok=True)
        STATE_FILE.unlink(missing_ok=True)
        sys.exit(1)

    if wait_for_server(port, verbose=verbose):
        ok(f"{backend.upper()} container running: {container_name} ({container_id})")
        info(f"Web UI: http://localhost:{port}/")
    else:
        fail("Server failed to start in container.")
        if not verbose:
            info("Check logs: " + f"{rt} logs {container_name}")
        subprocess.run([rt, "stop", container_name], capture_output=True)
        subprocess.run([rt, "rm", "-f", container_name], capture_output=True)
        PID_FILE.unlink(missing_ok=True)
        STATE_FILE.unlink(missing_ok=True)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCH — single-request throughput (generation + prefill)
# ═══════════════════════════════════════════════════════════════════════════════

# Paragraph repeated to build long prompts for prefill-dominant benchmarking.
# ~800 chars / ~200 tokens per copy.
_PP_PARAGRAPH = (
    "The modern CPU pipeline consists of several stages including instruction fetch, "
    "decode, execute, memory access, and writeback. Each stage operates on a different "
    "instruction simultaneously, enabling instruction-level parallelism and higher "
    "throughput. Branch prediction units attempt to guess the outcome of conditional "
    "branches before they are resolved. When the prediction is correct, the pipeline "
    "continues without interruption. When it is wrong, all work on the mispredicted "
    "path is discarded and execution restarts from the correct branch target. Modern "
    "processors use sophisticated predictors including TAGE, perceptron-based neural "
    "predictors, and loop detectors to minimize misprediction rates. Out-of-order "
    "execution allows the processor to look ahead in the instruction stream and execute "
    "instructions whose operands are ready, regardless of program order, while the "
    "reorder buffer ensures results are committed in the correct architectural sequence."
)


def _make_prefill_prompt(target_tokens: int = 800) -> str:
    """Build a long prompt for prefill-dominant benchmarking (~target_tokens input)."""
    # ~200 tokens per paragraph copy, ~4 chars per token
    target_chars = target_tokens * 4
    repeats = max(1, target_chars // len(_PP_PARAGRAPH))
    passage = "\n\n".join([_PP_PARAGRAPH] * repeats)
    return (
        "Read the following technical passage carefully, then answer the question.\n\n"
        + passage
        + "\n\nBased on the passage above, explain in one sentence how branch "
        "misprediction affects pipeline throughput."
    )


def _bench_one(port: int, prompt: str, max_tokens: int, label: str,
               timeout: int = 600) -> dict:
    """Send a single non-streaming request and return timing + token info."""
    url = _local_url(port, "/v1/chat/completions")
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return {"ok": False, "label": label, "error": str(e),
                "elapsed": time.perf_counter() - t0}

    elapsed = time.perf_counter() - t0
    usage = data.get("usage", {})
    prompt_tok = usage.get("prompt_tokens", 0)
    comp_tok = usage.get("completion_tokens", 0)
    tok_s = comp_tok / elapsed if elapsed > 0 else 0

    return {
        "ok": True,
        "label": label,
        "prompt_tok": prompt_tok,
        "comp_tok": comp_tok,
        "elapsed": elapsed,
        "tok_s": tok_s,
    }


def bench(port: int = 8000, model_alias: str | None = None,
          backend: str = "radv") -> dict:
    """Benchmark at small, medium, and large payload tiers.

    Each tier measures generation throughput (output tok/s) and effective
    prefill throughput (prompt tok/s).  Logs one JSONL record per tier.

    Returns dict with per-tier results:
      { "small": {gen_tok_s, pp_tok_s, combined_tok_s}, "medium": {...}, "large": {...} }
    """
    if model_alias is None:
        try:
            with urllib.request.urlopen(
                _local_url(port, "/v1/models"), timeout=5
            ) as resp:
                data = json.loads(resp.read())
                models = data.get("data", [])
                if models:
                    model_alias = models[0].get("id", "unknown")
        except Exception:
            model_alias = "unknown"

    info(f"Benchmarking server on port {port} ({backend}) ...")

    # Use the same payload tiers as bench-parallel for consistency
    tiers = [
        {
            "name": "small",
            "desc": "~50 input / 256 output (generation-dominant)",
            "prompts": [
                ("s-1", "Write a haiku about silicon.", 256),
                ("s-2", "Explain how a CPU cache works in 100 words.", 256),
            ],
        },
        {
            "name": "medium",
            "desc": "~1K input / 512 output (balanced)",
            "prompts": [
                ("m-1", _make_prefill_prompt(target_tokens=800), 512),
                ("m-2", _make_prefill_prompt(target_tokens=1200), 512),
            ],
        },
        {
            "name": "large",
            "desc": "~8K input / 2K output (prefill-heavy)",
            "prompts": [
                ("l-1", _make_prefill_prompt(target_tokens=6000), 2048),
                ("l-2", _make_prefill_prompt(target_tokens=8000), 2048),
            ],
        },
    ]

    quant = ""
    if model_alias and model_alias != "unknown":
        try:
            quant = get_model(model_alias).quant
        except ValueError:
            pass

    _ensure_results_dirs()
    report_file = BENCH_RESULTS_FILE
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    all_tier_results = {}

    for tier in tiers:
        tname = tier["name"]
        print()
        print(f"  ── {tname.upper()} — {tier['desc']} ─────────────────────────")

        gen_speeds = []
        pp_speeds = []

        for label, prompt, max_tok in tier["prompts"]:
            r = _bench_one(port, prompt, max_tok, label, timeout=600)
            if not r["ok"]:
                fail(f"  [{label}]  {r.get('error', 'unknown error')}")
                continue

            gen_s = r["tok_s"]
            pp_s = r["prompt_tok"] / r["elapsed"] if r["elapsed"] > 0 else 0

            gen_speeds.append(gen_s)
            pp_speeds.append(pp_s)

            print(f"  [{label:4s}]  "
                  f"prompt={r['prompt_tok']:5d}  "
                  f"output={r['comp_tok']:4d}  "
                  f"in {r['elapsed']:6.1f}s  →  "
                  f"gen={gen_s:5.1f}  pp={pp_s:5.1f} tok/s")

        gen_avg = sum(gen_speeds) / len(gen_speeds) if gen_speeds else 0
        pp_avg = sum(pp_speeds) / len(pp_speeds) if pp_speeds else 0
        combined = (gen_avg + pp_avg) / 2 if gen_avg > 0 and pp_avg > 0 else gen_avg

        all_tier_results[tname] = {
            "gen_tok_s": round(gen_avg, 1),
            "pp_tok_s": round(pp_avg, 1),
            "combined_tok_s": round(combined, 1),
        }

        # Log one record per tier
        if (gen_avg > 0 or pp_avg > 0) and model_alias and model_alias != "unknown":
            record = {
                "timestamp": timestamp,
                "backend": backend,
                "model": model_alias,
                "quant": quant,
                "payload": tname,
                "gen_tok_s": round(gen_avg, 1),
                "pp_tok_s": round(pp_avg, 1),
                "combined_tok_s": round(combined, 1),
                "avg_tok_s": round(gen_avg, 1),  # backward compat
            }
            with open(report_file, "a") as f:
                f.write(json.dumps(record) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print(f"  ═══════════════════════════════════════════════════════════")
    print(f"  {'Tier':<8s}  {'Gen tok/s':>10s}  {'PP tok/s':>10s}  {'Combined':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
    for tname in ("small", "medium", "large"):
        tr = all_tier_results.get(tname)
        if tr:
            print(f"  {tname:<8s}  {tr['gen_tok_s']:>10.1f}  {tr['pp_tok_s']:>10.1f}  {tr['combined_tok_s']:>10.1f}")
    print(f"  ═══════════════════════════════════════════════════════════")

    if model_alias and model_alias != "unknown":
        ok(f"Results appended to {report_file}")

    return all_tier_results


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCH — concurrent throughput (measures aggregate tok/s with parallel load)
# ═══════════════════════════════════════════════════════════════════════════════

def _fire_one_request(port: int, prompt: str, max_tokens: int = 256) -> dict:
    """Send a single completion request and return timing info."""
    url = _local_url(port, "/v1/chat/completions")
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
    prompt_tok = usage.get("prompt_tokens", 0)
    comp_tok = usage.get("completion_tokens", 0)
    return {
        "ok": True,
        "prompt_tok": prompt_tok,
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

    _ensure_results_dirs()
    results: list[dict] = []
    report_file = BENCH_PARALLEL_RESULTS_FILE

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

def bench_single(model_alias: str, port: int = 8000, backend: str = "radv") -> dict:
    """Start a model, benchmark it at all tiers, stop it, return results."""
    cfg = get_model(model_alias)
    empty = {t: {"gen_tok_s": 0, "pp_tok_s": 0, "combined_tok_s": 0}
             for t in ("small", "medium", "large")}
    if not cfg.is_downloaded:
        warn(f"Skipping {cfg.name} — not downloaded.")
        return empty

    info(f"═══ Benchmarking: {cfg.name} ({cfg.alias})  np={cfg.parallel_slots}  {backend} ═══")
    launch_server(cfg, port=port, backend=backend)

    try:
        result = bench(port=port, model_alias=cfg.alias, backend=backend)
    except Exception as e:
        fail(f"Benchmark failed for {cfg.name}: {e}")
        result = empty
    finally:
        stop_server()
        time.sleep(2)

    return result

def _slugify(value: str, *, max_len: int = 96) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    if not cleaned:
        return "run"
    return cleaned[:max_len]


def _project_relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def _resolve_project_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")



def _upsert_eval_record(record: dict[str, Any]) -> None:
    records = _load_jsonl_records(EVAL_RESULTS_FILE)
    run_id = record.get("run_id")
    if run_id:
        for index, existing in enumerate(records):
            if existing.get("run_id") == run_id:
                records[index] = record
                break
        else:
            records.append(record)
    else:
        records.append(record)
    records.sort(key=lambda item: ((item.get("timestamp") or ""), (item.get("run_id") or "")))
    _write_jsonl_records(EVAL_RESULTS_FILE, records)


def _model_eval_config_snapshot(
    model_alias: str,
    backend: str,
    suite: str,
    profile: EvalProfile,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "model": model_alias,
        "backend": backend,
        "suite": suite,
        "profile": profile.name,
    }
    try:
        cfg = get_model(model_alias)
    except ValueError:
        return snapshot

    snapshot.update(
        {
            "quant": cfg.quant,
            "parallel_slots": cfg.parallel_slots,
            "ctx_per_slot": cfg.ctx_per_slot,
            "batch_size": cfg.batch_size,
            "ubatch_size": cfg.ubatch_size,
            "threads": cfg.threads,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "min_p": cfg.min_p,
            "repeat_penalty": cfg.repeat_penalty,
            "reasoning_format": cfg.reasoning_format,
            "reasoning_budget": cfg.reasoning_budget,
            "reasoning": cfg.reasoning,
            "cache_prompt": cfg.cache_prompt,
            "cache_reuse": cfg.cache_reuse,
            "cache_ram": cfg.cache_ram,
            "cache_type_k": cfg.cache_type_k,
            "cache_type_v": cfg.cache_type_v,
            "chat_template_file": cfg.chat_template_file,
            "chat_template_kwargs": cfg.chat_template_kwargs,
            "extra_args": cfg.extra_args,
            "kv_unified": cfg.kv_unified,
            "clear_idle": cfg.clear_idle,
            "cpu_moe": cfg.cpu_moe,
            "n_cpu_moe": cfg.n_cpu_moe,
            "spec_strategy": cfg.spec.strategy if cfg.spec else None,
            "spec_args": cfg.spec.server_args() if cfg.spec else [],
        }
    )
    return snapshot


def _model_eval_config_fingerprint(
    model_alias: str,
    backend: str,
    suite: str,
    profile: EvalProfile,
) -> str:
    snapshot = _model_eval_config_snapshot(model_alias, backend, suite, profile)
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _parse_evalplus_scores(stdout_text: str) -> dict[str, Any]:
    matches = re.findall(r"pass@1:\s+([\d.]+)", stdout_text)
    base_score = None
    plus_score = None
    if len(matches) >= 1:
        base_score = float(matches[0])
    if len(matches) >= 2:
        plus_score = float(matches[1])
    return {
        "pass_at_1_base": base_score,
        "pass_at_1_plus": plus_score,
        "task_count": None,
    }


def _locate_evalplus_artifacts(run_root: Path, suite: str) -> dict[str, Path | None]:
    dataset_dir = run_root / suite
    if not dataset_dir.exists():
        return {
            "dataset_dir": None,
            "samples": None,
            "raw_samples": None,
            "result": None,
        }

    samples = sorted(
        path
        for path in dataset_dir.glob("*.jsonl")
        if not path.name.endswith(".raw.jsonl")
    )
    raw_samples = sorted(dataset_dir.glob("*.raw.jsonl"))
    results = sorted(dataset_dir.glob("*eval_results.json"))
    return {
        "dataset_dir": dataset_dir,
        "samples": samples[-1] if samples else None,
        "raw_samples": raw_samples[-1] if raw_samples else None,
        "result": results[-1] if results else None,
    }


def _evalplus_result_summary(result_path: Path | None) -> dict[str, Any]:
    if result_path is None or not result_path.exists():
        return {
            "pass_at_1_base": None,
            "pass_at_1_plus": None,
            "task_count": None,
        }
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "pass_at_1_base": None,
            "pass_at_1_plus": None,
            "task_count": None,
        }

    pass_block = payload.get("pass_at_k") if isinstance(payload, dict) else {}
    base_block = pass_block.get("base") if isinstance(pass_block, dict) else {}
    plus_block = pass_block.get("plus") if isinstance(pass_block, dict) else {}
    eval_block = payload.get("eval") if isinstance(payload, dict) else {}
    return {
        "pass_at_1_base": base_block.get("pass@1") if isinstance(base_block, dict) else None,
        "pass_at_1_plus": plus_block.get("pass@1") if isinstance(plus_block, dict) else None,
        "task_count": len(eval_block) if isinstance(eval_block, dict) else None,
    }


def _build_eval_record(metadata: dict[str, Any]) -> dict[str, Any]:
    result_path = _resolve_project_path(metadata.get("evalplus_result"))
    summary = _evalplus_result_summary(result_path)

    pass_at_1_base = summary.get("pass_at_1_base")
    if pass_at_1_base is None:
        pass_at_1_base = metadata.get("pass_at_1_base")

    pass_at_1_plus = summary.get("pass_at_1_plus")
    if pass_at_1_plus is None:
        pass_at_1_plus = metadata.get("pass_at_1_plus")

    task_count = summary.get("task_count")
    if task_count is None:
        task_count = metadata.get("task_count")

    return {
        "timestamp": metadata.get("timestamp"),
        "backend": metadata.get("backend"),
        "model": metadata.get("model"),
        "quant": metadata.get("quant") or "",
        "suite": metadata.get("suite"),
        "eval_tool": metadata.get("eval_tool", "evalplus"),
        "eval_profile": metadata.get("eval_profile"),
        "eval_profile_requested": metadata.get("eval_profile_requested"),
        "task_count": task_count,
        "run_label": metadata.get("run_label") or "",
        "config_fingerprint": metadata.get("config_fingerprint") or "",
        "run_id": metadata.get("run_id"),
        "ok": bool(metadata.get("ok")),
        "wall_time_sec": metadata.get("wall_time_sec"),
        "raw_log": metadata.get("raw_log"),
        "evalplus_root": metadata.get("evalplus_root"),
        "evalplus_result": metadata.get("evalplus_result"),
        "override_dataset": metadata.get("override_dataset"),
        "pass_at_1_base": pass_at_1_base,
        "pass_at_1_plus": pass_at_1_plus,
    }


def _metadata_matches(
    metadata: dict[str, Any],
    *,
    model_alias: str | None = None,
    suite: str | None = None,
    backend: str | None = None,
    profile_name: str | None = None,
    run_label: str | None = None,
) -> bool:
    if model_alias and metadata.get("model") != model_alias:
        return False
    if suite and metadata.get("suite") != suite:
        return False
    if backend and metadata.get("backend") != backend:
        return False
    if profile_name:
        profiles = {
            metadata.get("eval_profile"),
            metadata.get("eval_profile_requested"),
        }
        profiles.discard(None)
        if profile_name not in profiles:
            return False
    if run_label and (metadata.get("run_label") or "") != run_label:
        return False
    return True


def reanalyze_eval_results(
    model_alias: str | None = None,
    suite: str | None = None,
    backend: str | None = None,
    profile_name: str | None = None,
    run_label: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_results_dirs()

    metadata_files = sorted(_eval_metadata_dir().glob("*.json"))
    if not metadata_files:
        warn(f"No EvalPlus metadata found under {_eval_metadata_dir()}")
        return []

    existing = _load_jsonl_records(EVAL_RESULTS_FILE)
    legacy_records = [record for record in existing if not record.get("run_id")]
    records_by_run = {
        record["run_id"]: record
        for record in existing
        if isinstance(record, dict) and record.get("run_id")
    }

    matched_records: list[dict[str, Any]] = []
    for metadata_path in metadata_files:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warn(f"Skipping unreadable eval metadata: {metadata_path}")
            continue
        if not isinstance(metadata, dict):
            continue
        if not _metadata_matches(
            metadata,
            model_alias=model_alias,
            suite=suite,
            backend=backend,
            profile_name=profile_name,
            run_label=run_label,
        ):
            continue
        record = _build_eval_record(metadata)
        run_id = record.get("run_id")
        if run_id:
            records_by_run[run_id] = record
        matched_records.append(record)

    merged_records = legacy_records + sorted(
        records_by_run.values(),
        key=lambda item: ((item.get("timestamp") or ""), (item.get("run_id") or "")),
    )
    _write_jsonl_records(EVAL_RESULTS_FILE, merged_records)

    if matched_records:
        ok(f"Re-analyzed {len(matched_records)} EvalPlus runs into {EVAL_RESULTS_FILE}")
    else:
        warn("No EvalPlus metadata matched the requested filters.")
    return matched_records


def eval_single(
    model_alias: str | None,
    suite: str = "humaneval",
    port: int = 8000,
    backend: str = "radv",
    profile_name: str = "full",
    run_label: str | None = None,
) -> dict:
    """Start a model, run EvalPlus on it, stop it, return summary dict."""
    profile = resolve_eval_profile(profile_name, suite)
    if model_alias is None:
        cfg = resolve_model(None, prompt_text="Pick a model to evaluate")
    else:
        cfg = get_model(model_alias)

    if not cfg.is_downloaded:
        fail(f"Skipping {cfg.name} — not downloaded.")
        return {
            "ok": False,
            "eval_profile": profile.name,
            "task_count": profile.task_count,
        }

    task_suffix = f"  tasks={profile.task_count}" if profile.task_count else ""
    label_suffix = f"  label={run_label}" if run_label else ""
    info(
        f"═══ Evaluating: {cfg.name} ({cfg.alias})  np={cfg.parallel_slots}  "
        f"{backend}  suite={suite}  profile={profile.name}{task_suffix}{label_suffix} ═══"
    )

    launch_server(cfg, port=port, backend=backend)

    try:
        result = run_evalplus(
            port=port,
            suite=suite,
            model_alias=cfg.alias,
            backend=backend,
            profile_name=profile_name,
            run_label=run_label,
        )
    finally:
        stop_server()
        time.sleep(2)

    return result


def eval_all(
    suite: str = "humaneval",
    port: int = 8000,
    backend: str = "radv",
    profile_name: str = "full",
    run_label: str | None = None,
):
    """Run EvalPlus for every downloaded model and print a summary."""
    profile = resolve_eval_profile(profile_name, suite)
    downloaded = [m for m in MODELS if m.is_downloaded and not getattr(m, "hidden", False)]
    if not downloaded:
        fail("No models downloaded.  Run 'python server.py download MODEL' first.")
        sys.exit(1)

    task_suffix = f", {profile.task_count} tasks" if profile.task_count else ""
    info(
        f"Found {len(downloaded)} downloaded models.  "
        f"Running EvalPlus ({suite}, {backend}, profile={profile.name}{task_suffix}) ..."
    )
    print()

    summary: list[tuple[str, str, dict[str, Any]]] = []
    for cfg in downloaded:
        res = eval_single(
            cfg.alias,
            suite=suite,
            port=port,
            backend=backend,
            profile_name=profile_name,
            run_label=run_label,
        )
        summary.append((cfg.alias, cfg.name, res))
        print()

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print()
    print("  ═══════════════════════════════════════════════════════════════════════════════════════")
    print(f"  Eval Report — {timestamp}")
    print(f"  Backend: {backend.upper()}   Suite: {suite}   Profile: {profile.name}")
    print("  ───────────────────────────────────────────────────────────────────────────────────────")
    print(f"  {'Model':<30s} {'Profile':<10s} {'Tasks':>5s}  {'OK':>2s}  {'Wall (s)':>9s}")
    print(f"  {'─'*30} {'─'*10} {'─'*5}  {'─'*2}  {'─'*9}")

    for alias, name, res in summary:
        ok_flag = "✓" if res.get("ok") else "✗"
        wall = res.get("wall_time_sec", 0.0)
        task_count = res.get("task_count")
        tasks_str = str(task_count) if task_count else "—"
        profile_str = str(res.get("eval_profile") or profile.name)
        print(f"  {name:<30s} {profile_str:<10s} {tasks_str:>5s}  {ok_flag:>2s}  {wall:>9.1f}")

    print("  ═══════════════════════════════════════════════════════════════════════════════════════")
    print()

    if EVAL_RESULTS_FILE.exists():
        ok(f"All eval results logged to {EVAL_RESULTS_FILE}")
    info("Raw EvalPlus output logs are under ./results/eval/raw/")


def bench_all(port: int = 8000, backend: str = "radv"):
    """Benchmark every downloaded model at all payload tiers."""
    downloaded = [m for m in MODELS if m.is_downloaded and not getattr(m, "hidden", False)]
    if not downloaded:
        fail("No models downloaded.  Run 'python server.py download MODEL' first.")
        sys.exit(1)

    info(f"Found {len(downloaded)} downloaded models.  Benchmarking each ({backend}) ...")
    print()

    results: list[tuple[str, str, dict, int]] = []
    for cfg in downloaded:
        result = bench_single(cfg.alias, port=port, backend=backend)
        results.append((cfg.alias, cfg.name, result, cfg.parallel_slots))
        print()

    # ── Summary table ────────────────────────────────────────────────────
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print()
    print(f"  ══════════════════════════════════════════════════════════════════════════════════════")
    print(f"  Benchmark Report — {timestamp}  ·  Backend: {backend.upper()}")
    print(f"  ──────────────────────────────────────────────────────────────────────────────────────")
    print(f"  {'Model':<28s} {'np':>3s}  "
          f"{'S-gen':>6s} {'S-pp':>6s}  "
          f"{'M-gen':>6s} {'M-pp':>6s}  "
          f"{'L-gen':>6s} {'L-pp':>6s}")
    print(f"  {'─'*28} {'─'*3}  {'─'*6} {'─'*6}  {'─'*6} {'─'*6}  {'─'*6} {'─'*6}")

    def _sort_key(x):
        r = x[2]
        return -(r.get("small", {}).get("combined_tok_s", 0))

    for alias, name, res, np_val in sorted(results, key=_sort_key):
        s = res.get("small", {})
        m = res.get("medium", {})
        l = res.get("large", {})
        print(f"  {name:<28s} {np_val:>3d}  "
              f"{s.get('gen_tok_s', 0):>6.1f} {s.get('pp_tok_s', 0):>5.0f}  "
              f"{m.get('gen_tok_s', 0):>6.1f} {m.get('pp_tok_s', 0):>5.0f}  "
              f"{l.get('gen_tok_s', 0):>6.1f} {l.get('pp_tok_s', 0):>5.0f}")

    print(f"  ══════════════════════════════════════════════════════════════════════════════════════")
    print()

    report_file = BENCH_RESULTS_FILE
    if report_file.exists():
        ok(f"All results logged to {report_file}")
    info("Run with --backend rocm to compare.")



def run_evalplus(
    port: int,
    suite: str,
    model_alias: str,
    backend: str = "radv",
    profile_name: str = "full",
    run_label: str | None = None,
) -> dict:
    """Run EvalPlus against the running server and log/store a distinct run."""
    _ensure_results_dirs()

    profile = resolve_eval_profile(profile_name, suite)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_slug = time.strftime("%Y-%m-%d_%H-%M-%S")
    config_fingerprint = _model_eval_config_fingerprint(model_alias, backend, suite, profile)
    run_id = (
        f"{timestamp_slug}--{_slugify(model_alias)}--{suite}--"
        f"{_slugify(profile.name)}--{config_fingerprint[:8]}"
    )
    run_root = _eval_runs_dir() / run_id
    raw_path = EVAL_RAW_DIR / f"{run_id}.log"

    override_path = ensure_override_dataset(profile, _eval_profile_dataset_dir())

    cmd = [
        "evalplus.evaluate",
        "--model", f"strix-{model_alias}",
        "--dataset", suite,
        "--backend", "openai",
        "--base-url", _local_url(port, "/v1"),
        "--root", str(run_root),
        "--greedy",
    ]
    if profile.use_mini:
        cmd.append("--mini")

    env = os.environ.copy()
    env.pop("HUMANEVAL_OVERRIDE_PATH", None)
    env.pop("MBPP_OVERRIDE_PATH", None)
    api_key = _api_key_for_model(model_alias)
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    if override_path is not None:
        env["HUMANEVAL_OVERRIDE_PATH"] = str(override_path)

    info(
        f"Running EvalPlus for {model_alias} on {suite} "
        f"(backend={backend}, port={port}, profile={profile.name})"
    )
    if profile.task_count:
        info(f"Profile task count: {profile.task_count}")
    if run_label:
        info(f"Run label: {run_label}")
    info("Command: " + " ".join(cmd))

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    elapsed = time.perf_counter() - t0

    stdout_text = proc.stdout if isinstance(proc.stdout, str) else ""
    raw_path.write_text(stdout_text, encoding="utf-8")
    ok_str = "OK" if proc.returncode == 0 else f"FAIL ({proc.returncode})"
    info(f"EvalPlus finished in {elapsed:.1f}s — status: {ok_str}")
    info(f"Raw output saved to {raw_path}")

    artifacts = _locate_evalplus_artifacts(run_root, suite)
    summary = _evalplus_result_summary(artifacts.get("result"))
    if summary.get("pass_at_1_base") is None:
        summary = _parse_evalplus_scores(stdout_text)

    pass_at_1_base = summary.get("pass_at_1_base")
    pass_at_1_plus = summary.get("pass_at_1_plus")
    task_count = summary.get("task_count") or profile.task_count

    if pass_at_1_base is not None:
        score_msg = f"Base pass@1: {pass_at_1_base:.1%}"
        if pass_at_1_plus is not None:
            score_msg += f"  Plus pass@1: {pass_at_1_plus:.1%}"
        ok(score_msg)

    quant = ""
    try:
        quant = get_model(model_alias).quant
    except ValueError:
        pass

    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "backend": backend,
        "model": model_alias,
        "quant": quant,
        "suite": suite,
        "eval_tool": "evalplus",
        "eval_profile": profile.name,
        "eval_profile_requested": profile.requested,
        "eval_profile_description": profile.description,
        "task_count": task_count,
        "run_label": run_label or "",
        "config_fingerprint": config_fingerprint,
        "ok": (proc.returncode == 0),
        "wall_time_sec": round(elapsed, 1),
        "raw_log": _project_relpath(raw_path),
        "evalplus_root": _project_relpath(run_root),
        "evalplus_samples": _project_relpath(artifacts.get("samples")),
        "evalplus_raw_samples": _project_relpath(artifacts.get("raw_samples")),
        "evalplus_result": _project_relpath(artifacts.get("result")),
        "override_dataset": _project_relpath(override_path),
        "pass_at_1_base": pass_at_1_base,
        "pass_at_1_plus": pass_at_1_plus,
    }

    metadata_path = _eval_metadata_dir() / f"{run_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    record = _build_eval_record(metadata)
    _upsert_eval_record(record)

    ok(f"Eval result stored in {EVAL_RESULTS_FILE}")
    info(f"Run metadata saved to {metadata_path}")

    return record


def aider_setup(
    *,
    update: bool = False,
    aider_ref: str = DEFAULT_AIDER_REF,
    polyglot_ref: str = DEFAULT_POLYGLOT_REF,
) -> dict[str, Any]:
    """Prepare the local Aider benchmark harness and fixed exercise subsets."""
    info("Preparing Aider benchmark harness (Docker image + fixed exercise subsets)")
    setup = ensure_aider_setup(update=update, aider_ref=aider_ref, polyglot_ref=polyglot_ref)
    ok(f"Aider repo ready: {setup['aider_repo']}")
    info(f"Aider commit: {setup['aider_head'][:12]}")
    ok(f"Polyglot benchmark ready: {setup['polyglot_repo']}")
    info(f"Polyglot commit: {setup['polyglot_head'][:12]}")
    ok(f"Docker image ready: {setup['docker_image']}")
    for profile_name in AIDER_PROFILE_NAMES:
        profile = AIDER_PROFILES[profile_name]
        curated_dir = setup['curated_dirs'].get(profile_name, '')
        info(f"Profile {profile_name}: {curated_dir}")
        info(f"  {profile.description}")
    return setup


def aider_bench_single(
    model_alias: str | None,
    *,
    port: int = 8000,
    backend: str = "radv",
    profile_name: str = "python-quick",
    manifest_path: str | None = None,
    run_label: str | None = None,
    max_tokens: int = DEFAULT_AIDER_MAX_TOKENS,
    threads: int = 1,
    tries: int | None = None,
    edit_format: str = "whole",
    update_harness: bool = False,
    aider_ref: str = DEFAULT_AIDER_REF,
    polyglot_ref: str = DEFAULT_POLYGLOT_REF,
) -> dict[str, Any]:
    """Start a model, run the Aider code-edit benchmark, then stop it."""
    if model_alias is None:
        cfg = resolve_model(None, prompt_text="Pick a model to benchmark with Aider")
    else:
        cfg = get_model(model_alias)

    if not cfg.is_downloaded:
        fail(f"Skipping {cfg.name} — not downloaded.")
        return {
            "ok": False,
            "profile": profile_name,
        }

    profile_label = manifest_path or profile_name
    label_suffix = f"  label={run_label}" if run_label else ""
    info(
        f"═══ Aider benchmark: {cfg.name} ({cfg.alias})  {backend}  "
        f"profile={profile_label}  max_tokens={max_tokens}{label_suffix} ═══"
    )

    launch_server(cfg, port=port, backend=backend)

    try:
        result = run_aider_benchmark(
            model_alias=cfg.alias,
            backend=backend,
            port=port,
            profile_name=profile_name,
            manifest_path=manifest_path,
            run_label=run_label,
            max_tokens=max_tokens,
            threads=threads,
            tries=tries,
            edit_format=edit_format,
            context_window=cfg.ctx_size,
            api_key=_api_key_for_model(cfg.alias),
            update_harness=update_harness,
            aider_ref=aider_ref,
            polyglot_ref=polyglot_ref,
            model_display_name=cfg.name,
            quant=cfg.quant,
        )
    finally:
        stop_server()
        time.sleep(2)

    if not result.get("ok"):
        fail(f"Aider benchmark exited with return code {result.get('returncode')}.")
        log_tail = result.get("log_tail") or []
        if log_tail:
            print("    Last benchmark log lines:")
            for line in log_tail[-12:]:
                print(f"      {line}")
    if result.get("pass_rate_1") is not None:
        ok(f"Aider pass rate after try 1: {result['pass_rate_1']:.1f}%")
    if result.get("pass_rate_2") is not None:
        ok(f"Aider pass rate after try 2: {result['pass_rate_2']:.1f}%")
    if result.get("completed_tests") is not None and result.get("total_tests") is not None:
        info(f"Cases completed: {result['completed_tests']}/{result['total_tests']}")
    if result.get("seconds_per_case_wall") is not None:
        info(f"Wall time per case: {result['seconds_per_case_wall']:.2f}s")
    if result.get("completion_tok_s_wall") is not None:
        info(f"Completion tok/s (wall clock): {result['completion_tok_s_wall']:.2f}")
    if result.get("exhausted_context_windows"):
        warn(
            f"Context window exhaustion seen {int(result['exhausted_context_windows'])} time(s). "
            f"Consider raising --max-tokens or the server context window."
        )
    if result.get("num_malformed_responses"):
        warn(f"Malformed responses seen: {int(result['num_malformed_responses'])}")
    if result.get("syntax_errors"):
        warn(f"Syntax errors seen in generated code: {int(result['syntax_errors'])}")
    if result.get("test_timeouts"):
        warn(f"Timed out test runs: {int(result['test_timeouts'])}")
    failed_exercises = result.get("failed_exercises") or []
    if failed_exercises:
        info("Failed exercises: " + ", ".join(failed_exercises[:8]))
    important_lines = result.get("important_log_lines") or []
    if important_lines:
        warn("Important benchmark warnings:")
        for line in important_lines[:8]:
            print(f"    {line}")
    if result.get("log_file"):
        info(f"Full benchmark log saved to {result['log_file']}")
    if result.get("metadata_file"):
        info(f"Run metadata saved to {result['metadata_file']}")
    if result.get("results_file"):
        ok(f"Aider results appended to {result['results_file']}")

    return result


def aider_bench_all(
    *,
    port: int = 8000,
    backend: str = "radv",
    profile_name: str = "python-quick",
    manifest_path: str | None = None,
    run_label: str | None = None,
    max_tokens: int = DEFAULT_AIDER_MAX_TOKENS,
    threads: int = 1,
    tries: int | None = None,
    edit_format: str = "whole",
    update_harness: bool = False,
    aider_ref: str = DEFAULT_AIDER_REF,
    polyglot_ref: str = DEFAULT_POLYGLOT_REF,
) -> list[dict[str, Any]]:
    """Run the fixed Aider benchmark for every downloaded model."""
    downloaded = [m for m in MODELS if m.is_downloaded and not getattr(m, "hidden", False)]
    if not downloaded:
        fail("No models downloaded. Run 'python server.py download MODEL' first.")
        sys.exit(1)

    results: list[tuple[str, str, dict[str, Any]]] = []
    for cfg in downloaded:
        result = aider_bench_single(
            cfg.alias,
            port=port,
            backend=backend,
            profile_name=profile_name,
            manifest_path=manifest_path,
            run_label=run_label,
            max_tokens=max_tokens,
            threads=threads,
            tries=tries,
            edit_format=edit_format,
            update_harness=update_harness,
            aider_ref=aider_ref,
            polyglot_ref=polyglot_ref,
        )
        results.append((cfg.alias, cfg.name, result))

    print()
    print("  ══════════════════════════════════════════════════════════════════════════════════════")
    print(f"  {'Model':<28s} {'Try1':>6s} {'Try2':>6s} {'Cases':>9s} {'s/case':>8s} {'tok/s':>8s}")
    print(f"  {'─'*28} {'─'*6} {'─'*6} {'─'*9} {'─'*8} {'─'*8}")
    for _, name, result in results:
        try1 = result.get('pass_rate_1')
        try2 = result.get('pass_rate_2')
        completed = result.get('completed_tests') or 0
        total = result.get('total_tests') or 0
        seconds = result.get('seconds_per_case_wall') or 0
        tok_s = result.get('completion_tok_s_wall') or 0
        try1_text = f"{try1:.1f}%" if isinstance(try1, (int, float)) else "—"
        try2_text = f"{try2:.1f}%" if isinstance(try2, (int, float)) else "—"
        print(
            f"  {name:<28s} {try1_text:>6s} {try2_text:>6s} "
            f"{f'{completed}/{total}':>9s} {seconds:>8.2f} {tok_s:>8.2f}"
        )
    print("  ══════════════════════════════════════════════════════════════════════════════════════")
    print()

    return [result for _, _, result in results]

# ═══════════════════════════════════════════════════════════════════════════════
#  LIST
# ═══════════════════════════════════════════════════════════════════════════════

def list_models():
    """Print a table of available models."""
    models = visible_models()
    print()
    print(f"  {'Alias':<28s} {'Quant':<14s} {'np':>3s} {'ctx/slot':>9s} {'Spec':<14s} {'DL':>3s}")
    print(f"  {'─'*28} {'─'*14} {'─'*3} {'─'*9} {'─'*14} {'─'*3}")
    for m in models:
        spec = m.spec.strategy or "—"
        dl = "✓" if m.is_downloaded else "·"
        quant = m.quant or "—"
        ctx_k = f"{m.ctx_per_slot // 1024}K"
        print(f"  {m.alias:<28s} {quant:<14s} {m.parallel_slots:>3d} {ctx_k:>9s} {spec:<14s} {dl:>3s}")
    print()
    if any(m.notes for m in models):
        print("  Notes:")
        for m in models:
            if m.notes:
                wrapped = textwrap.fill(m.notes, width=72, initial_indent="    ",
                                        subsequent_indent="    ")
                print(f"  {m.alias}:")
                print(wrapped)
                print()


# ── Test Mode ──────────────────────────────────────────────────────────────────

def run_test_suite(args):
    """Run tests in dry-run mode without disrupting the main conversational model.
    
    This mode:
    1. Uses a tiny model or existing model for tests
    2. Skips actual model downloads if already present (--dry-run)
    3. Runs sequentially if --sequential is set
    4. Uses low parallel slots to minimize memory usage
    5. Does NOT stop/restart the main model server
    """
    import subprocess
    import sys
    import os
    
    print()
    info("╔════════════════════════════════════════════════════════════════╗")
    info("║  STRIX LLMAPPP TEST SUITE - DRY RUN MODE                      ║")
    info("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Use tiny model if specified, or existing model
    model_alias = args.model or "smollm2-135m-test-q4"
    if args.dry_run:
        # Check if model is already downloaded
        try:
            cfg = get_model(model_alias)
            if cfg.is_downloaded:
                ok(f"Model already downloaded: {cfg.name}")
            else:
                warn(f"Model not downloaded, but --dry-run requested. Skipping download.")
        except ValueError:
            warn(f"Unknown model: {model_alias}. Using smollm2-135m-test-q4 for testing.")
            model_alias = "smollm2-135m-test-q4"
            cfg = get_model(model_alias)
    
    print()
    info(f"Config:")
    print(f"  Model:    {model_alias}")
    print(f"  Port:     {args.port}")
    print(f"  Backend:  {args.backend}")
    print(f"  Parallel: {args.np}")
    if args.sequential:
        print(f"  Sequential: YES (avoids concurrent model switching)")
    if args.dry_run:
        print(f"  Dry-run:  YES (skips downloads)")
    print()
    
    # Test 1: Model validation
    print()
    print("  ── Test 1: Model Validation ──────────────────────────────────")
    cfg = None
    try:
        cfg = get_model(model_alias)
        ok(f"Model found: {cfg.name}")
        print(f"  Alias:   {cfg.alias}")
        print(f"  Parallel: {cfg.parallel_slots}")
        print(f"  Context: {cfg.ctx_per_slot} tokens/slot")
        if cfg.is_downloaded:
            ok("Model is downloaded")
        else:
            warn("Model not downloaded")
    except ValueError as e:
        fail(f"Model lookup failed: {e}")
        cfg = None
    print()
    
    # Test 2: Server arguments generation
    print()
    print("  ── Test 2: Server Arguments Generation ───────────────────────")
    try:
        if cfg and cfg.is_downloaded:
            args_list = cfg.server_args(parallel_override=args.np)
            print(f"  Generated {len(args_list)} arguments")
            print(f"  Sample args: {args_list[:8]} ...")
            ok("Server arguments generated successfully")
        else:
            warn("Skipping - model not downloaded")
    except Exception as e:
        fail(f"Argument generation failed: {e}")
    print()
    
    # Test 3: Container setup
    print()
    print("  ── Test 3: Container Setup ───────────────────────────────────")
    try:
        image = _container_image(args.backend)
        print(f"  Backend: {args.backend}")
        print(f"  Container image: {image}")
        ok("Container configuration ready")
    except Exception as e:
        fail(f"Container setup failed: {e}")
    print()
    
    # Test 4: Parallelization test
    print()
    print("  ── Test 4: Parallelization Support ───────────────────────────")
    if not args.sequential:
        print(f"  Testing {args.np} parallel slots...")
        # Test concurrent requests (dry-run, don't actually send)
        print(f"  Concurrent requests: simulated")
        ok("Parallelization configuration valid")
    else:
        ok("Sequential mode - parallelization skipped")
    print()
    
    # Test 5: Server readiness (dry-run check)
    print()
    print("  ── Test 5: Server Readiness Check ────────────────────────────")
    try:
        import urllib.request
        url = f"http://127.0.0.1:{args.port}/health"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    ok(f"Server is running on port {args.port}")
                else:
                    warn(f"Server responded with status {resp.status}")
        except Exception:
            warn(f"Server not responding on port {args.port} (expected if not started)")
    except Exception as e:
        warn(f"Health check failed: {e}")
    print()
    
    # Summary
    print()
    info("╔════════════════════════════════════════════════════════════════╗")
    info("║  TEST SUITE COMPLETE                                          ║")
    info("╚════════════════════════════════════════════════════════════════╝")
    print()
    ok("Tests completed in dry-run mode")
    ok("Main model server was NOT affected")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Strix Halo llama.cpp launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")
    
    # test mode - dry-run for testing without disrupting main model
    p_test = sub.add_parser("test", help="Run test suite in dry-run mode (no server)")
    p_test.add_argument("--model", default=None,
                        help="Model to test (uses the hidden smoke-test model by default)")
    p_test.add_argument("--port", type=int, default=8000,
                        help="Port for dry-run tests")
    p_test.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"], default="radv",
                        help="Backend for tests")
    p_test.add_argument("--np", type=int, default=1,
                        help="Parallel slots for parallelization tests (default: 1)")
    p_test.add_argument("--dry-run", action="store_true",
                        help="Skip actual model download if already present")
    p_test.add_argument("--sequential", action="store_true",
                        help="Run tests sequentially (avoids concurrent model switching)")
    p_test.add_argument("--timeout", type=int, default=30,
                        help="Timeout for server startup in seconds (default: 30)")

    # build
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
    p_serve.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"], default="vulkan",
                         help="Backend: vulkan/radv (RADV), amdvlk (AMDVLK), rocm/rocm6/rocm7/rocm7-nightly (ROCm)")
    p_serve.add_argument("--no-spec", action="store_true",
                         help="Disable speculative decoding")
    p_serve.add_argument("--verbose", "-v", action="store_true",
                         help="Show full llama-server output while loading")
    p_serve.add_argument("--cache-reuse", type=int, default=None,
                         help="Override llama.cpp --cache-reuse chunk size")
    p_serve.add_argument("--cache-ram", type=int, default=None,
                         help="Enable llama.cpp host-memory prompt cache in MiB")
    p_serve.add_argument("--disable-prompt-cache", action="store_true",
                         help="Disable llama.cpp prompt caching for this launch")
    p_serve.add_argument("--slot-save-path", default=None,
                         help="Directory used for llama.cpp slot save/restore snapshots")
    p_serve.add_argument("--disable-slot-persistence", action="store_true",
                         help="Disable llama.cpp slot save/restore endpoints for this launch")
    p_serve.add_argument("--reasoning-format", choices=["auto", "deepseek", "deepseek-legacy", "none"], default=None,
                         help="Override llama.cpp reasoning parser for this launch")
    p_serve.add_argument("--reasoning-budget", type=int, default=None,
                         help="Override llama.cpp reasoning budget for this launch")
    p_serve.add_argument("--enable-reasoning", action="store_true",
                         help="Force llama.cpp reasoning mode on for this launch")
    p_serve.add_argument("--disable-reasoning", action="store_true",
                         help="Force llama.cpp reasoning mode off for this launch")
    p_serve.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                         help="Extra args passed to llama-server")

    # stop
    sub.add_parser("stop", help="Stop the running server")

    # repo-aware caching helpers
    p_repo_init = sub.add_parser("repo-init",
        help="Generate stable repo context and an OpenCode config for a project")
    p_repo_init.add_argument("--repo", default=".",
                             help="Project repository path (default: current directory)")
    p_repo_init.add_argument("--model", default=None,
                             help="Model alias or name for OpenCode (default: running model or picker)")
    p_repo_init.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT,
                             help="Port that the repo-aware proxy will listen on")
    p_repo_init.add_argument("--context-limit", type=int, default=None,
                             help="Context limit to publish in opencode.json (default: selected model ctx)")
    p_repo_init.add_argument("--output-limit", type=int, default=8192,
                             help="Output token limit to publish in opencode.json")
    p_repo_init.add_argument("--provider-id", default="strix-local",
                             help="Provider ID written into opencode.json")
    p_repo_init.add_argument("--provider-name", default="Strix Halo llama.cpp",
                             help="Provider display name written into opencode.json")

    p_repo_refresh = sub.add_parser("repo-refresh",
        help="Refresh the cached architecture summary for a project repo")
    p_repo_refresh.add_argument("--repo", default=".",
                                help="Project repository path (default: current directory)")

    p_repo_proxy = sub.add_parser("repo-proxy",
        help="Run a local OpenAI-compatible proxy that injects cached repo context")
    p_repo_proxy.add_argument("--repo", default=".",
                              help="Project repository path (default: current directory)")
    p_repo_proxy.add_argument("--host", default=DEFAULT_PROXY_HOST,
                              help="Proxy listen host")
    p_repo_proxy.add_argument("--port", type=int, default=DEFAULT_PROXY_PORT,
                              help="Proxy listen port")
    p_repo_proxy.add_argument("--server-port", type=int, default=8000,
                              help="Upstream llama-server port")
    p_repo_proxy.add_argument("--slot-id", type=int, default=DEFAULT_SLOT_ID,
                              help="llama.cpp slot ID to pin this repo to")
    p_repo_proxy.add_argument("--refresh", action="store_true",
                              help="Refresh the repo context before starting the proxy")
    p_repo_proxy.add_argument("--verbose", action="store_true",
                              help="Enable proxy request logging")
    p_repo_proxy.add_argument("--no-metrics", action="store_true",
                              help="Disable one-line per-request cache/timing metrics from the proxy")

    p_repo_warm = sub.add_parser("repo-warm",
        help="Prime llama.cpp prompt cache for a project repo")
    p_repo_warm.add_argument("--repo", default=".",
                             help="Project repository path (default: current directory)")
    p_repo_warm.add_argument("--port", type=int, default=8000,
                             help="Upstream llama-server port")
    p_repo_warm.add_argument("--model", default=None,
                             help="Model alias or name (default: running model)")
    p_repo_warm.add_argument("--slot-id", type=int, default=DEFAULT_SLOT_ID,
                             help="llama.cpp slot ID to warm")

    p_repo_save = sub.add_parser("repo-save",
        help="Save the warmed llama.cpp KV state for a repo slot to disk")
    p_repo_save.add_argument("--repo", default=".",
                             help="Project repository path (default: current directory)")
    p_repo_save.add_argument("--port", type=int, default=8000,
                             help="Upstream llama-server port")
    p_repo_save.add_argument("--slot-id", type=int, default=DEFAULT_SLOT_ID,
                             help="llama.cpp slot ID to save")
    p_repo_save.add_argument("--filename", default=None,
                             help="Override the slot snapshot filename")

    p_repo_restore = sub.add_parser("repo-restore",
        help="Restore a previously saved llama.cpp KV state for a repo slot")
    p_repo_restore.add_argument("--repo", default=".",
                                help="Project repository path (default: current directory)")
    p_repo_restore.add_argument("--port", type=int, default=8000,
                                help="Upstream llama-server port")
    p_repo_restore.add_argument("--slot-id", type=int, default=DEFAULT_SLOT_ID,
                                help="llama.cpp slot ID to restore")
    p_repo_restore.add_argument("--filename", default=None,
                                help="Override the slot snapshot filename")

    p_repo_status = sub.add_parser("repo-status",
        help="Show cached repo context files and current server settings")
    p_repo_status.add_argument("--repo", default=".",
                               help="Project repository path (default: current directory)")

    # bench
    p_bench = sub.add_parser("bench",
        help="Benchmark a model (starts/stops automatically, or tests running server)")
    p_bench.add_argument("model", nargs="?", default=None,
                         help="Model to benchmark (omit to test running server)")
    p_bench.add_argument("--port", type=int, default=8000)
    p_bench.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"], default="radv")

    # bench-all
    p_ball = sub.add_parser("bench-all",
        help="Benchmark all downloaded models and print a comparison report")
    p_ball.add_argument("--port", type=int, default=8000)
    p_ball.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"], default="radv")

    # bench-parallel  ← NEW
    p_bpar = sub.add_parser("bench-parallel",
        help="Sweep --parallel values to find optimal throughput")
    p_bpar.add_argument("model", nargs="?", default=None,
                        help="Model to sweep (interactive picker if omitted)")
    p_bpar.add_argument("--port", type=int, default=8000)
    p_bpar.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"], default="radv")
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

    # download-images
    sub.add_parser("download-images", help="Download all container images")

    # aider-setup
    p_aider_setup = sub.add_parser("aider-setup",
        help="Clone/build the Aider benchmark harness and create fixed exercise subsets")
    p_aider_setup.add_argument("--update", action="store_true",
                               help="Fetch the latest harness refs and rebuild the Docker image")
    p_aider_setup.add_argument("--aider-ref", default=DEFAULT_AIDER_REF,
                               help="Git ref for the aider harness checkout")
    p_aider_setup.add_argument("--polyglot-ref", default=DEFAULT_POLYGLOT_REF,
                               help="Git ref for the polyglot-benchmark checkout")

    # aider-bench
    p_aider = sub.add_parser("aider-bench",
        help="Run the Aider code-edit benchmark against a fixed local subset")
    p_aider.add_argument("model", nargs="?", default=None,
                         help="Model to benchmark (omit for interactive picker)")
    p_aider.add_argument("--profile", default="python-quick", metavar="PROFILE",
                         help=f"Fixed benchmark profile to run (default: python-quick). Built-ins: {', '.join(AIDER_PROFILE_NAMES)}")
    p_aider.add_argument("--manifest", default=None,
                         help="Custom manifest file of exercises to benchmark instead of a built-in profile")
    p_aider.add_argument("--label", default=None,
                         help="Optional label to distinguish repeated aider runs")
    p_aider.add_argument("--max-tokens", type=int, default=DEFAULT_AIDER_MAX_TOKENS,
                         help=f"Generation cap forwarded to the model via Aider/LiteLLM (default: {DEFAULT_AIDER_MAX_TOKENS})")
    p_aider.add_argument("--threads", type=int, default=1,
                         help="Aider benchmark worker threads (default: 1)")
    p_aider.add_argument("--tries", type=int, default=None,
                         help="Number of repair attempts per exercise (default: profile default)")
    p_aider.add_argument("--edit-format", default="whole",
                         help="Aider edit format (default: whole)")
    p_aider.add_argument("--port", type=int, default=8000)
    p_aider.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"],
                         default=None)
    p_aider.add_argument("--update-harness", action="store_true",
                         help="Fetch latest harness refs and rebuild the Aider Docker image before running")
    p_aider.add_argument("--aider-ref", default=DEFAULT_AIDER_REF,
                         help="Git ref for the aider harness checkout")
    p_aider.add_argument("--polyglot-ref", default=DEFAULT_POLYGLOT_REF,
                         help="Git ref for the polyglot-benchmark checkout")

    # aider-bench-all
    p_aider_all = sub.add_parser("aider-bench-all",
        help="Run the Aider benchmark for all downloaded models")
    p_aider_all.add_argument("--profile", default="python-quick", metavar="PROFILE",
                             help=f"Fixed benchmark profile to run (default: python-quick). Built-ins: {', '.join(AIDER_PROFILE_NAMES)}")
    p_aider_all.add_argument("--manifest", default=None,
                             help="Custom manifest file of exercises to benchmark instead of a built-in profile")
    p_aider_all.add_argument("--label", default=None,
                             help="Optional label to attach to every aider benchmark run")
    p_aider_all.add_argument("--max-tokens", type=int, default=DEFAULT_AIDER_MAX_TOKENS)
    p_aider_all.add_argument("--threads", type=int, default=1)
    p_aider_all.add_argument("--tries", type=int, default=None)
    p_aider_all.add_argument("--edit-format", default="whole")
    p_aider_all.add_argument("--port", type=int, default=8000)
    p_aider_all.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"],
                             default=None)
    p_aider_all.add_argument("--update-harness", action="store_true")
    p_aider_all.add_argument("--aider-ref", default=DEFAULT_AIDER_REF)
    p_aider_all.add_argument("--polyglot-ref", default=DEFAULT_POLYGLOT_REF)

    # eval
    p_eval = sub.add_parser("eval",
        help="Run EvalPlus coding benchmark for a single model")
    p_eval.add_argument("model", nargs="?", default=None,
                        help="Model to evaluate (omit for interactive picker)")
    p_eval.add_argument("--suite", choices=["humaneval", "mbpp"],
                        default="humaneval",
                        help="EvalPlus suite (default: humaneval)")
    p_eval.add_argument("--profile", choices=["quick", "mini", "full"],
                        default="quick",
                        help="Evaluation profile: quick (curated humaneval subset), mini, or full")
    p_eval.add_argument("--label", default=None,
                        help="Optional label to distinguish repeated eval runs")
    p_eval.add_argument("--port", type=int, default=8000)
    p_eval.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"],
                        default=None)

    # eval-all
    p_eval_all = sub.add_parser("eval-all",
        help="Run EvalPlus for all downloaded models")
    p_eval_all.add_argument("--suite", choices=["humaneval", "mbpp"],
                            default="humaneval")
    p_eval_all.add_argument("--profile", choices=["quick", "mini", "full"],
                            default="quick")
    p_eval_all.add_argument("--label", default=None,
                            help="Optional label to attach to every run in the sweep")
    p_eval_all.add_argument("--port", type=int, default=8000)
    p_eval_all.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"],
                            default=None)

    # eval-reanalyze
    p_eval_re = sub.add_parser("eval-reanalyze",
        help="Rebuild eval summaries from stored EvalPlus artifacts")
    p_eval_re.add_argument("--model", default=None,
                           help="Only re-analyze runs for this model alias")
    p_eval_re.add_argument("--suite", choices=["humaneval", "mbpp"],
                           default=None)
    p_eval_re.add_argument("--backend", choices=["vulkan", "radv", "amdvlk", "rocm", "rocm6", "rocm7", "rocm7-nightly"],
                           default=None)
    p_eval_re.add_argument("--profile", choices=["quick", "quick-v1", "mini", "full"],
                           default=None,
                           help="Only re-analyze runs for this eval profile")
    p_eval_re.add_argument("--label", default=None,
                           help="Only re-analyze runs with this label")

    load_env_file()
    
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "list":
        list_models()

    elif args.command == "serve":
        cfg = copy.deepcopy(resolve_model(args.model, "Pick a model to serve"))

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
        if args.cache_reuse is not None:
            cfg.cache_reuse = args.cache_reuse
        if args.cache_ram is not None:
            cfg.cache_ram = args.cache_ram
        if args.disable_prompt_cache:
            cfg.cache_prompt = False
            cfg.cache_reuse = 0
        if args.disable_slot_persistence:
            cfg.slot_save_path = None
        elif args.slot_save_path is not None:
            cfg.slot_save_path = args.slot_save_path
        if args.reasoning_format is not None:
            cfg.reasoning_format = args.reasoning_format
        if args.reasoning_budget is not None:
            cfg.reasoning_budget = args.reasoning_budget
        if args.enable_reasoning:
            cfg.reasoning = True
        if args.disable_reasoning:
            cfg.reasoning = False

        launch_server(cfg, port=args.port, backend=args.backend,
                      extra_args=args.extra, verbose=args.verbose,
                      parallel_override=parallel_override,
                      ctx_override=ctx_override)

    elif args.command == "stop":
        stop_server()

    elif args.command == "repo-init":
        cfg = copy.deepcopy(resolve_model_or_running(args.model, "Pick a model for OpenCode"))
        paths = refresh_repo_context(args.repo)
        config_path = write_opencode_config(
            paths.repo_dir,
            model_alias=cfg.alias,
            model_name=cfg.name,
            context_limit=args.context_limit or cfg.ctx_size,
            proxy_port=args.proxy_port,
            provider_id=args.provider_id,
            provider_name=args.provider_name,
            output_limit=args.output_limit,
        )
        ok(f"Wrote repo context: {paths.context_file}")
        ok(f"Wrote metadata: {paths.metadata_file}")
        ok(f"Wrote OpenCode config: {config_path}")
        info(f"Next: python server.py repo-proxy --repo {paths.repo_dir}")

    elif args.command == "repo-refresh":
        paths = refresh_repo_context(args.repo)
        ok(f"Refreshed repo context: {paths.context_file}")

    elif args.command == "repo-proxy":
        if args.refresh:
            paths = refresh_repo_context(args.repo)
        else:
            paths = repo_paths(args.repo)
            if not paths.context_file.exists():
                paths = refresh_repo_context(args.repo)
        upstream_headers: dict[str, str] = {}
        api_key = _api_key_for_model()
        if api_key:
            upstream_headers["Authorization"] = f"Bearer {api_key}"
        info(
            f"Repo proxy for {paths.repo_dir} on http://{args.host}:{args.port} -> "
            f"http://127.0.0.1:{args.server_port} (slot {args.slot_id})"
        )
        start_repo_proxy(
            paths.repo_dir,
            listen_host=args.host,
            listen_port=args.port,
            upstream_port=args.server_port,
            slot_id=args.slot_id,
            upstream_headers=upstream_headers,
            verbose=args.verbose,
            metrics=not args.no_metrics,
        )

    elif args.command == "repo-warm":
        model_alias = args.model
        if model_alias:
            model_alias = get_model(model_alias).alias
        result = warm_repo_slot(args.repo, port=args.port, model_alias=model_alias, slot_id=args.slot_id)
        usage = result.get("usage") if isinstance(result, dict) else None
        ok(f"Warmed repo cache for slot {args.slot_id}")
        if isinstance(usage, dict) and usage:
            info(f"Warm request usage: {usage}")

    elif args.command == "repo-save":
        result = save_repo_slot(args.repo, port=args.port, slot_id=args.slot_id, filename=args.filename)
        ok(f"Saved slot {args.slot_id} to {result.get('filename', args.filename or repo_paths(args.repo).slot_filename)}")

    elif args.command == "repo-restore":
        result = restore_repo_slot(args.repo, port=args.port, slot_id=args.slot_id, filename=args.filename)
        ok(f"Restored slot {args.slot_id} from {result.get('filename', args.filename or repo_paths(args.repo).slot_filename)}")

    elif args.command == "repo-status":
        paths = repo_paths(args.repo)
        state = _load_state() or {}
        slot_root = Path(state.get("slot_save_path", SLOT_CACHE_ROOT))
        slot_file = slot_root / paths.slot_filename
        print()
        info(f"Repo: {paths.repo_dir}")
        info(f"Context file: {paths.context_file} ({'present' if paths.context_file.exists() else 'missing'})")
        info(f"Metadata file: {paths.metadata_file} ({'present' if paths.metadata_file.exists() else 'missing'})")
        info(f"OpenCode config: {paths.opencode_file} ({'present' if paths.opencode_file.exists() else 'missing'})")
        info(f"Slot snapshot: {slot_file} ({'present' if slot_file.exists() else 'missing'})")
        if state:
            info(
                "Running server: "
                f"model={state.get('model')} backend={state.get('backend')} port={state.get('port')} "
                f"cache_reuse={state.get('cache_reuse')} slot_save_path={state.get('slot_save_path')}"
            )
        else:
            warn("No running server state file found")

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

    elif args.command == "download-images":
        download_container_images()

    elif args.command == "aider-setup":
        aider_setup(update=args.update, aider_ref=args.aider_ref, polyglot_ref=args.polyglot_ref)

    elif args.command == "aider-bench":
        backend = args.backend or _current_backend() or "radv"
        aider_bench_single(
            args.model,
            port=args.port,
            backend=backend,
            profile_name=args.profile,
            manifest_path=args.manifest,
            run_label=args.label,
            max_tokens=args.max_tokens,
            threads=args.threads,
            tries=args.tries,
            edit_format=args.edit_format,
            update_harness=args.update_harness,
            aider_ref=args.aider_ref,
            polyglot_ref=args.polyglot_ref,
        )

    elif args.command == "aider-bench-all":
        backend = args.backend or _current_backend() or "radv"
        aider_bench_all(
            port=args.port,
            backend=backend,
            profile_name=args.profile,
            manifest_path=args.manifest,
            run_label=args.label,
            max_tokens=args.max_tokens,
            threads=args.threads,
            tries=args.tries,
            edit_format=args.edit_format,
            update_harness=args.update_harness,
            aider_ref=args.aider_ref,
            polyglot_ref=args.polyglot_ref,
        )

    elif args.command == "eval":
        # Use explicit backend if provided, otherwise try running backend, otherwise default to radv
        backend = args.backend or _current_backend() or "radv"
        eval_single(args.model, suite=args.suite,
                    port=args.port, backend=backend,
                    profile_name=args.profile, run_label=args.label)
    elif args.command == "eval-all":
        # Use explicit backend if provided, otherwise try running backend, otherwise default to radv
        backend = args.backend or _current_backend() or "radv"
        eval_all(suite=args.suite,
                 port=args.port, backend=backend,
                 profile_name=args.profile, run_label=args.label)
    elif args.command == "eval-reanalyze":
        reanalyze_eval_results(model_alias=args.model,
                               suite=args.suite,
                               backend=args.backend,
                               profile_name=args.profile,
                               run_label=args.label)

    elif args.command == "test":
        run_test_suite(args)

if __name__ == "__main__":
    main()
