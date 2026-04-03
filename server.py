#!/usr/bin/env python3
"""
Strix Halo llama.cpp launcher.

Builds llama.cpp from source (Vulkan or ROCm), downloads models from HF,
and serves them with tuned flags for AMD Strix Halo (Ryzen AI Max / gfx1151).

Supports side-by-side builds: both Vulkan and ROCm can coexist in separate
build directories.  Use --backend to select at build and serve time.

Usage:
    python server.py build   [--backend vulkan|rocm] [--rebuild]
    python server.py list
    python server.py serve   [MODEL] [--backend radv|amdvlk|rocm] ...
    python server.py stop
    python server.py bench   [MODEL] [--backend radv|amdvlk|rocm]
    python server.py bench-all
    python server.py bench-parallel [MODEL]               # sweep --np to find sweet spot
    python server.py download MODEL
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
PID_FILE     = PROJECT_DIR / ".server.pid"
STATE_FILE   = PROJECT_DIR / ".server.json"

# Side-by-side build directories so you can have both backends ready at once.
# `python server.py build --backend vulkan` → build-vulkan/
# `python server.py build --backend rocm`   → build-rocm/
LLAMA_BUILD_VULKAN = LLAMA_SRC / "build-vulkan"
LLAMA_BUILD_ROCM   = LLAMA_SRC / "build-rocm"

# Legacy path — if you previously built into "build/", we check there as fallback.
LLAMA_BUILD_LEGACY = LLAMA_SRC / "build"

VALID_BACKENDS = ("radv", "amdvlk", "rocm")
VULKAN_BACKENDS = ("radv", "amdvlk")

EVAL_RESULTS_FILE = PROJECT_DIR / "eval_results.jsonl"
EVAL_RAW_DIR      = PROJECT_DIR / "eval_raw"

def _build_dir(backend: str) -> Path:
    """Return the build directory for a given backend."""
    if backend == "rocm":
        return LLAMA_BUILD_ROCM
    return LLAMA_BUILD_VULKAN


def _server_bin(backend: str) -> Path:
    """Return path to llama-server for the given backend.

    Falls back to legacy build/ dir if the new per-backend dir doesn't exist.
    """
    primary = _build_dir(backend) / "bin" / "llama-server"
    if primary.exists():
        return primary
    # Fallback: old single build/ directory (before multi-backend support)
    legacy = LLAMA_BUILD_LEGACY / "bin" / "llama-server"
    if legacy.exists():
        return legacy
    return primary  # return expected path for error messages


def _bench_bin(backend: str) -> Path:
    """Return path to llama-bench for the given backend."""
    primary = _build_dir(backend) / "bin" / "llama-bench"
    if primary.exists():
        return primary
    legacy = LLAMA_BUILD_LEGACY / "bin" / "llama-bench"
    if legacy.exists():
        return legacy
    return primary


# ── Per-backend environment variables ────────────────────────────────────────
#
# Strix Halo (Ryzen AI Max 395) with GTT-mapped unified memory:
#   - VRAM aperture is kept minimal (512MB) for display only
#   - All model weight + KV cache lives in GTT (90GB mapped)
#   - This means RADV_PERFTEST=nogttspill is WRONG for our setup
#     (we *want* everything in GTT)

VULKAN_ENV = {
    "AMD_VULKAN_ICD":             "RADV",      # overridden to AMDVLK for --backend amdvlk
    "HSA_ENABLE_SDMA":            "0",          # critical for gfx1151 stability
    "GGML_VK_PREFER_HOST_MEMORY": "0",          # let Vulkan manage memory placement
}

ROCM_ENV = {
    "HSA_ENABLE_SDMA":            "0",          # critical for gfx1151 stability
    "ROCBLAS_USE_HIPBLASLT":      "1",          # use hipBLASLt for faster GEMM (prefill)
    "HIP_VISIBLE_DEVICES":        "0",          # target the iGPU
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

BUILD_DEPS_FEDORA_VULKAN = "cmake ninja-build gcc-c++ vulkan-headers vulkan-loader-devel glslang shaderc spirv-tools"
BUILD_DEPS_DEBIAN_VULKAN = "cmake ninja-build build-essential libvulkan-dev glslang-tools glslc spirv-tools"

# Container image for ROCm (pre-built llama.cpp for gfx1151, rebuilt on every commit)
ROCM_CONTAINER_IMAGE = "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2"
ROCM_CONTAINER_NAME  = "strix-llama-rocm"


def _find_container_runtime() -> str | None:
    """Find podman or docker."""
    for rt in ("podman", "docker"):
        if shutil.which(rt):
            return rt
    return None


def check_build_deps(backend: str = "vulkan"):
    """Check build dependencies.  For ROCm, returns 'native' or 'container'."""
    missing = []
    for tool in ("cmake", "ninja", "git"):
        if shutil.which(tool) is None:
            missing.append(tool)

    if backend == "rocm":
        if shutil.which("hipcc") is not None:
            # Native ROCm available
            if missing:
                fail(f"Missing build tools: {', '.join(missing)}")
                sys.exit(1)
            return "native"

        # hipcc not available — try container build
        rt = _find_container_runtime()
        if rt:
            info(f"hipcc not found — will build ROCm inside container ({rt})")
            info(f"Image: {ROCM_CONTAINER_IMAGE}")
            # For container build we only need git on host (cmake/ninja are inside)
            if not shutil.which("git"):
                fail("git is required even for container builds")
                sys.exit(1)
            return "container"

        # Neither native hipcc nor container runtime
        fail("ROCm build requires hipcc, but it's not installed.")
        info("AMD's ROCm packages have dependency issues on Debian Trixie.")
        info("Options:")
        info("  1. Install podman or docker, then re-run — builds inside a container")
        info("     sudo apt install podman")
        info(f"     Container image: {ROCM_CONTAINER_IMAGE}")
        info("  2. Use distrobox/toolbox with a ROCm container for full native ROCm")
        info("  3. On Fedora/Ubuntu: sudo dnf/apt install rocm-dev hipcc")
        sys.exit(1)

    else:
        # Vulkan
        if shutil.which("glslangValidator") is None:
            missing.append("glslangValidator")
        if missing:
            fail(f"Missing build tools: {', '.join(missing)}")
            info(f"Fedora:  sudo dnf install {BUILD_DEPS_FEDORA_VULKAN}")
            info(f"Debian:  sudo apt install {BUILD_DEPS_DEBIAN_VULKAN}")
            sys.exit(1)
        return "native"


def _clone_or_update_source(rebuild: bool = False, build_dir: Path | None = None):
    """Clone llama.cpp if missing, or pull latest.  Optionally clean build dir."""
    if LLAMA_SRC.exists() and not rebuild:
        info("Updating llama.cpp ...")
        subprocess.run(["git", "pull", "--ff-only"], cwd=LLAMA_SRC, check=True)
    elif LLAMA_SRC.exists() and rebuild:
        info("Rebuilding — cleaning build directory ...")
        if build_dir and build_dir.exists():
            shutil.rmtree(build_dir)
        subprocess.run(["git", "pull", "--ff-only"], cwd=LLAMA_SRC, check=True)
    else:
        info("Cloning llama.cpp ...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggml-org/llama.cpp.git",
             str(LLAMA_SRC)],
            check=True,
        )


def _build_rocm_in_container(build_dir: Path):
    """Pull the kyuz0 ROCm toolbox image for container-based serving.

    The kyuz0/amd-strix-halo-toolboxes images ship with llama.cpp pre-built
    for gfx1151, rebuilt on every upstream commit.  Since ROCm userspace libs
    aren't available on Debian Trixie, we run llama-server inside the
    container at serve time (with GPU passthrough and model dir mounted).
    """
    rt = _find_container_runtime()
    if not rt:
        fail("No container runtime found (need podman or docker)")
        sys.exit(1)

    info(f"Pulling ROCm container image ({rt}) ...")
    info(f"  Image: {ROCM_CONTAINER_IMAGE}")

    result = subprocess.run(
        [rt, "pull", ROCM_CONTAINER_IMAGE],
    )
    if result.returncode != 0:
        fail(f"Failed to pull {ROCM_CONTAINER_IMAGE}")
        sys.exit(1)

    # Verify llama-server exists inside the image
    info("Verifying llama-server is present in image ...")
    verify = subprocess.run(
        [rt, "run", "--rm", ROCM_CONTAINER_IMAGE, "llama-server", "--version"],
        capture_output=True, text=True, timeout=30,
    )
    if verify.returncode == 0:
        ver = verify.stdout.strip().splitlines()[0] if verify.stdout.strip() else "unknown"
        ok(f"ROCm image ready — llama-server {ver}")
    else:
        warn("Could not verify llama-server in image (may still work)")

    # Write a marker so ensure_built knows ROCm container mode is set up
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / ".container-mode").write_text(ROCM_CONTAINER_IMAGE)
    ok(f"ROCm backend ready (container mode via {rt})")


def _build_rocm_native(build_dir: Path):
    """Build llama.cpp with ROCm using native hipcc."""
    ncpu = os.cpu_count() or 4

    info("Configuring with ROCm (HIP) for gfx1151 ...")
    cmake_flags = [
        "cmake", "-B", str(build_dir),
        "-G", "Ninja",
        "-DGGML_HIP=ON",
        "-DAMDGPU_TARGETS=gfx1151",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    cmake_env = {**os.environ}
    hipcc = shutil.which("hipcc")
    if hipcc:
        hip_path_result = subprocess.run(
            ["hipconfig", "-R"], capture_output=True, text=True
        )
        if hip_path_result.returncode == 0:
            cmake_env["HIP_PATH"] = hip_path_result.stdout.strip()
        hipcc_dir = subprocess.run(
            ["hipconfig", "-l"], capture_output=True, text=True
        )
        if hipcc_dir.returncode == 0:
            cmake_env["HIPCXX"] = f"{hipcc_dir.stdout.strip()}/clang"

    subprocess.run(cmake_flags, cwd=LLAMA_SRC, check=True, env=cmake_env)

    info(f"Building ROCm native (using {ncpu} cores) ...")
    subprocess.run(
        [
            "cmake", "--build", str(build_dir),
            "--config", "Release",
            f"-j{ncpu}",
            "--target", "llama-server", "llama-bench",
        ],
        check=True,
        env=cmake_env,
    )

    server_bin = build_dir / "bin" / "llama-server"
    if server_bin.exists():
        ok(f"ROCm build complete (native): {server_bin}")
    else:
        fail("Build finished but llama-server binary not found.")
        sys.exit(1)


def build_llamacpp(rebuild: bool = False, backend: str = "vulkan"):
    """Build llama.cpp for the specified backend.

    For ROCm: uses native hipcc if available, otherwise extracts pre-built
    binaries from the kyuz0 Strix Halo toolbox container image.
    For Vulkan: builds from source (clones llama.cpp if needed).
    """
    build_mode = check_build_deps(backend)
    build_dir = _build_dir("rocm" if backend == "rocm" else "radv")

    if backend == "rocm" and build_mode == "container":
        # Container extraction doesn't need the source tree
        if rebuild and build_dir.exists():
            info("Cleaning ROCm build directory ...")
            shutil.rmtree(build_dir)
        _build_rocm_in_container(build_dir)
    elif backend == "rocm":
        _clone_or_update_source(rebuild, build_dir)
        _build_rocm_native(build_dir)
    else:
        _clone_or_update_source(rebuild, build_dir)

        ncpu = os.cpu_count() or 4
        info("Configuring with Vulkan ...")
        subprocess.run(
            [
                "cmake", "-B", str(build_dir),
                "-G", "Ninja",
                "-DGGML_VULKAN=ON",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            cwd=LLAMA_SRC,
            check=True,
        )

        info(f"Building Vulkan (using {ncpu} cores) ...")
        subprocess.run(
            [
                "cmake", "--build", str(build_dir),
                "--config", "Release",
                f"-j{ncpu}",
                "--target", "llama-server", "llama-bench",
            ],
            check=True,
        )

        server_bin = build_dir / "bin" / "llama-server"
        if server_bin.exists():
            ok(f"Build complete (vulkan): {server_bin}")
        else:
            fail("Build finished but llama-server binary not found.")
            sys.exit(1)


def ensure_built(backend: str = "radv"):
    """Make sure llama-server is available for the requested backend."""
    if backend == "rocm" and _rocm_uses_container():
        # Container mode — check for marker file from `build --backend rocm`
        marker = _build_dir("rocm") / ".container-mode"
        if marker.exists():
            return
        warn("ROCm container not set up — run: python server.py build --backend rocm")
        build_llamacpp(backend="rocm")
        return

    server = _server_bin(backend)
    if not server.exists():
        build_backend = "rocm" if backend == "rocm" else "vulkan"
        warn(f"llama-server not found for {backend} — building from source ...")
        build_llamacpp(backend=build_backend)


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
    """Stop a running llama-server (native process or ROCm container)."""

    # Read state to determine what's running
    is_container = False
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            is_container = "container" in state
        except (json.JSONDecodeError, KeyError):
            pass

    # Stop native process (only if NOT container mode)
    if not is_container and PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if pid > 0:
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
        except (OSError, ValueError):
            pass

    PID_FILE.unlink(missing_ok=True)

    # Stop ROCm container if running
    rt = _find_container_runtime()
    if rt:
        # Check if container exists before trying to stop it
        check = subprocess.run(
            [rt, "container", "exists", ROCM_CONTAINER_NAME],
            capture_output=True,
        )
        if check.returncode == 0:
            subprocess.run(
                [rt, "stop", "-t", "5", ROCM_CONTAINER_NAME],
                capture_output=True, timeout=15,
            )
            subprocess.run(
                [rt, "rm", "-f", ROCM_CONTAINER_NAME],
                capture_output=True, timeout=10,
            )
            ok(f"Stopped ROCm container ({ROCM_CONTAINER_NAME})")
            time.sleep(1)  # let the port fully clear

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


def _rocm_uses_container() -> bool:
    """Check if ROCm should run via container (no native hipcc available)."""
    return shutil.which("hipcc") is None


def launch_server(cfg: ModelConfig, port: int = 8000, backend: str = "radv",
                  extra_args: list[str] | None = None, verbose: bool = False,
                  parallel_override: int | None = None,
                  ctx_override: int | None = None):
    """Start llama-server as a background process (native or container)."""
    download_model(cfg)

    # Stop any existing server (native or container)
    stop_server()

    np = parallel_override if parallel_override is not None else cfg.parallel_slots

    # Build the llama-server argument list
    args = cfg.server_args(parallel_override=parallel_override, ctx_override=ctx_override)
    try:
        idx = args.index("--port")
        args[idx + 1] = str(port)
    except ValueError:
        args += ["--port", str(port)]

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

    # ── ROCm container mode ──────────────────────────────────────────────
    if backend == "rocm" and _rocm_uses_container():
        rt = _find_container_runtime()
        if not rt:
            fail("ROCm container mode requires podman or docker")
            sys.exit(1)

        from models import MODELS_DIR

        # Build the container run command
        env_flags = []
        for k, v in ROCM_ENV.items():
            env_flags += ["-e", f"{k}={v}"]

        container_cmd = [
            rt, "run", "-d",
            "--name", ROCM_CONTAINER_NAME,
            "--device", "/dev/dri",
            "--device", "/dev/kfd",
            "--group-add", "video",
            "--group-add", "render",
            "--security-opt", "seccomp=unconfined",
            "-v", f"{MODELS_DIR}:{MODELS_DIR}:ro",
            "-p", f"{port}:{port}",
        ] + env_flags + [
            ROCM_CONTAINER_IMAGE,
            "llama-server",
        ] + args

        if verbose:
            info(f"Container: {' '.join(container_cmd[:12])} ...")

        result = subprocess.run(container_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            fail(f"Failed to start ROCm container")
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-3:]:
                    fail(f"  {line}")
            sys.exit(1)

        container_id = result.stdout.strip()[:12]
        # Don't write a PID file for container mode — stop_server uses
        # the STATE_FILE to detect container mode and stops by container name.
        PID_FILE.unlink(missing_ok=True)
        STATE_FILE.write_text(json.dumps({
            "model": cfg.alias,
            "backend": backend,
            "port": port,
            "parallel": np,
            "container": ROCM_CONTAINER_NAME,
        }))

        if verbose:
            # Tail container logs in background
            import threading
            def _tail_container():
                try:
                    log_proc = subprocess.Popen(
                        [rt, "logs", "-f", ROCM_CONTAINER_NAME],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    )
                    for line in log_proc.stdout:
                        print(f"  │ {line.decode('utf-8', errors='replace').rstrip()}")
                except Exception:
                    pass
            threading.Thread(target=_tail_container, daemon=True).start()

        if not verbose:
            print("  Loading ", end="", flush=True)

        if wait_for_server(port, verbose=verbose):
            ok(f"ROCm container running: {ROCM_CONTAINER_NAME} ({container_id})")
        else:
            fail("Server failed to start in container.")
            if not verbose:
                info("Check logs: " + f"{rt} logs {ROCM_CONTAINER_NAME}")
            subprocess.run([rt, "stop", ROCM_CONTAINER_NAME], capture_output=True)
            subprocess.run([rt, "rm", "-f", ROCM_CONTAINER_NAME], capture_output=True)
            PID_FILE.unlink(missing_ok=True)
            STATE_FILE.unlink(missing_ok=True)
            sys.exit(1)

        return

    # ── Native binary mode (Vulkan or native ROCm) ───────────────────────
    ensure_built(backend)

    if backend == "rocm":
        env = {**os.environ, **ROCM_ENV}
    else:
        env = {**os.environ, **VULKAN_ENV}
        if backend == "amdvlk":
            env["AMD_VULKAN_ICD"] = "AMDVLK"

    server_bin = _server_bin(backend)
    cmd = [str(server_bin)] + args

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
                f"http://127.0.0.1:{port}/v1/models", timeout=5
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

    report_file = PROJECT_DIR / "bench_results.jsonl"
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

def eval_single(model_alias: str | None,
                suite: str = "humaneval",
                port: int = 8000,
                backend: str = "radv") -> dict:
    """Start a model, run EvalPlus on it, stop it, return summary dict."""
    if model_alias is None:
        cfg = resolve_model(None, prompt_text="Pick a model to evaluate")
    else:
        cfg = get_model(model_alias)

    if not cfg.is_downloaded:
        fail(f"Skipping {cfg.name} — not downloaded.")
        return {"ok": False}

    info(f"═══ Evaluating: {cfg.name} ({cfg.alias})  np={cfg.parallel_slots}  "
         f"{backend}  suite={suite} ═══")

    # Launch server with the model's preferred np (like bench_single)
    launch_server(cfg, port=port, backend=backend)

    try:
        result = run_evalplus(port=port, suite=suite,
                              model_alias=cfg.alias, backend=backend)
    finally:
        stop_server()
        time.sleep(2)

    return result

def eval_all(suite: str = "humaneval",
             port: int = 8000,
             backend: str = "radv"):
    """Run EvalPlus for every downloaded model and print a summary."""
    downloaded = [m for m in MODELS if m.is_downloaded]
    if not downloaded:
        fail("No models downloaded.  Run 'python server.py download MODEL' first.")
        sys.exit(1)

    info(f"Found {len(downloaded)} downloaded models.  "
         f"Running EvalPlus ({suite}, {backend}) ...")
    print()

    summary: list[tuple[str, str, dict]] = []
    for cfg in downloaded:
        res = eval_single(cfg.alias, suite=suite, port=port, backend=backend)
        summary.append((cfg.alias, cfg.name, res))
        print()

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print()
    print(f"  ══════════════════════════════════════════════════════════════════════════")
    print(f"  Eval Report — {timestamp}")
    print(f"  Backend: {backend.upper()}   Suite: {suite}")
    print(f"  ──────────────────────────────────────────────────────────────────────────")
    print(f"  {'Model':<32s}  {'OK':>2s}  {'Wall (s)':>9s}")
    print(f"  {'─'*32}  {'─'*2}  {'─'*9}")

    for alias, name, res in summary:
        ok_flag = "✓" if res.get("ok") else "✗"
        wall = res.get("wall_time_sec", 0.0)
        print(f"  {name:<32s}  {ok_flag:>2s}  {wall:>9.1f}")

    print(f"  ══════════════════════════════════════════════════════════════════════════")
    print()

    if EVAL_RESULTS_FILE.exists():
        ok(f"All eval results logged to {EVAL_RESULTS_FILE}")
    info("Raw EvalPlus output logs are under ./eval_raw/")

def bench_all(port: int = 8000, backend: str = "radv"):
    """Benchmark every downloaded model at all payload tiers."""
    downloaded = [m for m in MODELS if m.is_downloaded]
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

    report_file = PROJECT_DIR / "bench_results.jsonl"
    if report_file.exists():
        ok(f"All results logged to {report_file}")
    info("Run with --backend rocm to compare.")


def run_evalplus(port: int, suite: str, model_alias: str,
                 backend: str = "radv") -> dict:
    """Run EvalPlus (HumanEval/MBPP) against the running server and log output.

    Returns a dict with status + wall time. We don't try to parse pass@k yet;
    instead we store raw stdout in a file and reference it from JSONL.
    """
    EVAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    raw_fname = f"{timestamp}--{model_alias}--{suite}.log"
    raw_path = EVAL_RAW_DIR / raw_fname

    # EvalPlus CLI; assumes evalplus.evaluate is on PATH.
    # Uses llama.cpp's OpenAI-compatible API on localhost.
    cmd = [
        "evalplus.evaluate",
        "--model", f"strix-{model_alias}",
        "--dataset", suite,              # "humaneval" or "mbpp"
        "--backend", "openai",
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--greedy",
    ]

    info(f"Running EvalPlus for {model_alias} on {suite} "
         f"(backend={backend}, port={port})")
    info("Command: " + " ".join(cmd))

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    raw_path.write_text(proc.stdout)
    ok_str = "OK" if proc.returncode == 0 else f"FAIL ({proc.returncode})"
    info(f"EvalPlus finished in {elapsed:.1f}s — status: {ok_str}")
    info(f"Raw output saved to {raw_path}")

    # Attach quant if we know this model
    quant = ""
    try:
        quant = get_model(model_alias).quant
    except ValueError:
        pass

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "backend": backend,
        "model": model_alias,
        "quant": quant,
        "suite": suite,                  # humaneval / mbpp / etc.
        "eval_tool": "evalplus",
        "ok": (proc.returncode == 0),
        "wall_time_sec": round(elapsed, 1),
        "raw_log": str(raw_path.relative_to(PROJECT_DIR)),
        # Room to add parsed metrics later, e.g.:
        # "pass_at_1": ...,
        # "pass_at_10": ...,
    }

    with open(EVAL_RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    ok(f"Eval result appended to {EVAL_RESULTS_FILE}")

    return record

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
    p_build = sub.add_parser("build", help="Build llama.cpp from source")
    p_build.add_argument("--rebuild", action="store_true",
                         help="Clean build directory and rebuild")
    p_build.add_argument("--backend", choices=["vulkan", "rocm"], default="vulkan",
                         help="Build backend (default: vulkan).  Both can coexist.")

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
    p_serve.add_argument("--backend", choices=["radv", "amdvlk", "rocm"], default="radv",
                         help="Backend: radv (Vulkan RADV), amdvlk (Vulkan AMDVLK), rocm (ROCm HIP)")
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
    p_bench.add_argument("--backend", choices=["radv", "amdvlk", "rocm"], default="radv")

    # bench-all
    p_ball = sub.add_parser("bench-all",
        help="Benchmark all downloaded models and print a comparison report")
    p_ball.add_argument("--port", type=int, default=8000)
    p_ball.add_argument("--backend", choices=["radv", "amdvlk", "rocm"], default="radv")

    # bench-parallel  ← NEW
    p_bpar = sub.add_parser("bench-parallel",
        help="Sweep --parallel values to find optimal throughput")
    p_bpar.add_argument("model", nargs="?", default=None,
                        help="Model to sweep (interactive picker if omitted)")
    p_bpar.add_argument("--port", type=int, default=8000)
    p_bpar.add_argument("--backend", choices=["radv", "amdvlk", "rocm"], default="radv")
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

    # eval
    p_eval = sub.add_parser("eval",
        help="Run EvalPlus coding benchmark for a single model")
    p_eval.add_argument("model", nargs="?", default=None,
                        help="Model to evaluate (omit for interactive picker)")
    p_eval.add_argument("--suite", choices=["humaneval", "mbpp"],
                        default="humaneval",
                        help="EvalPlus suite (default: humaneval)")
    p_eval.add_argument("--port", type=int, default=8000)
    p_eval.add_argument("--backend", choices=["radv", "amdvlk", "rocm"],
                        default="radv")

    # eval-all
    p_eval_all = sub.add_parser("eval-all",
        help="Run EvalPlus for all downloaded models")
    p_eval_all.add_argument("--suite", choices=["humaneval", "mbpp"],
                            default="humaneval")
    p_eval_all.add_argument("--port", type=int, default=8000)
    p_eval_all.add_argument("--backend", choices=["radv", "amdvlk", "rocm"],
                            default="radv")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "build":
        build_llamacpp(rebuild=args.rebuild, backend=args.backend)

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

    elif args.command == "eval":
        eval_single(args.model, suite=args.suite,
                    port=args.port, backend=args.backend)
    elif args.command == "eval-all":
        eval_all(suite=args.suite,
                 port=args.port, backend=args.backend)

if __name__ == "__main__":
    main()
