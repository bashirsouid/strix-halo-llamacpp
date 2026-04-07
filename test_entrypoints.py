from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _read_logged_calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    calls: list[list[str]] = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        parts = [part for part in line.split("\x1f") if part]
        calls.append(parts)
    return calls


def _make_stub_bin(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "python3",
        """#!/usr/bin/env bash
set -euo pipefail
log_dir="${STRIX_STUB_LOG_DIR:?}"
{
  for arg in "$@"; do
    printf '%s\x1f' "$arg"
  done
  printf '\n'
} >> "$log_dir/python3_calls.log"

if [[ "${1:-}" == "-c" ]]; then
  exec /usr/bin/python3 "$@"
fi

script_dir="$(cd "$(dirname "${1:-.}")" && pwd)"
command="${2:-}"
backend="${STUB_DEFAULT_BACKEND:-vulkan}"
model=""
port="8000"

i=2
while [[ $i -le $# ]]; do
  arg="${!i:-}"
  case "$arg" in
    --backend)
      i=$((i + 1))
      backend="${!i:-$backend}"
      ;;
    --port)
      i=$((i + 1))
      port="${!i:-$port}"
      ;;
    serve)
      ;;
    --*)
      ;;
    *)
      if [[ -z "$model" ]]; then
        model="$arg"
      fi
      ;;
  esac
  i=$((i + 1))
done

container="strix-llama-vulkan"
case "$backend" in
  amdvlk) container="strix-llama-amdvlk" ;;
  rocm) container="strix-llama-rocm" ;;
  rocm6) container="strix-llama-rocm6" ;;
  rocm7) container="strix-llama-rocm7" ;;
  rocm7-nightly) container="strix-llama-rocm7-nightly" ;;
  vulkan|radv) container="strix-llama-vulkan" ;;
esac

if [[ "$command" == "serve" ]]; then
  if [[ -z "$model" ]]; then
    model="picker-model"
  fi
  cat > "$script_dir/.server.json" <<JSON
{"model": "$model", "backend": "$backend", "container": "$container", "port": $port}
JSON
fi

exit 0
""",
    )

    _write_executable(
        bin_dir / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
log_dir="${STRIX_STUB_LOG_DIR:?}"
{
  for arg in "$@"; do
    printf '%s\x1f' "$arg"
  done
  printf '\n'
} >> "$log_dir/docker_calls.log"

if [[ "${1:-}" == "image" && "${2:-}" == "inspect" ]]; then
  exit "${DOCKER_IMAGE_INSPECT_EXIT:-1}"
fi

if [[ "${1:-}" == "container" && "${2:-}" == "inspect" ]]; then
  if [[ "${DOCKER_CONTAINER_RUNNING:-false}" == "true" ]]; then
    printf 'true\n'
    exit 0
  fi
  exit 1
fi

exit 0
""",
    )

    _write_executable(
        bin_dir / "pip",
        """#!/usr/bin/env bash
set -euo pipefail
log_dir="${STRIX_STUB_LOG_DIR:?}"
{
  for arg in "$@"; do
    printf '%s\x1f' "$arg"
  done
  printf '\n'
} >> "$log_dir/pip_calls.log"
exit 0
""",
    )


def _make_sandbox(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    for filename in ("start.sh", "watch.sh", "stop.sh", "source-me.sh"):
        shutil.copy2(Path(filename), sandbox / filename)
        (sandbox / filename).chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub_bin(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STRIX_STUB_LOG_DIR"] = str(tmp_path)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return sandbox, env


def _last_python_call(tmp_path: Path) -> list[str]:
    calls = [call for call in _read_logged_calls(tmp_path / "python3_calls.log") if call and call[0] != "-c"]
    assert calls, "expected at least one python3 invocation"
    return calls[-1]


class TestStartEntrypoint:
    def test_start_sh_preserves_flag_values_and_backend(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path)
        result = subprocess.run(
            ["./start.sh", "--backend", "rocm", "dummy-model", "--np", "4", "--ctx", "8192", "--verbose"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert _last_python_call(tmp_path) == [
            str(sandbox / "server.py"),
            "serve",
            "dummy-model",
            "--backend",
            "rocm",
            "--np",
            "4",
            "--ctx",
            "8192",
            "--verbose",
        ]

    def test_start_sh_prompts_for_backend_when_only_model_is_given(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path)
        result = subprocess.run(
            ["./start.sh", "dummy-model"],
            cwd=sandbox,
            env=env,
            input="3\n",
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert _last_python_call(tmp_path) == [
            str(sandbox / "server.py"),
            "serve",
            "dummy-model",
            "--backend",
            "rocm",
        ]

    def test_start_sh_without_model_delegates_to_server_picker(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path)
        result = subprocess.run(
            ["./start.sh", "--backend", "amdvlk", "--port", "9000"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert _last_python_call(tmp_path) == [
            str(sandbox / "server.py"),
            "serve",
            "--backend",
            "amdvlk",
            "--port",
            "9000",
        ]


class TestWatchAndOtherEntrypoints:
    def test_watch_sh_uses_container_state_for_radv_without_restarting(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path)
        env["WATCH_INTERVAL"] = "0"
        env["WATCH_ONCE"] = "1"
        env["DOCKER_CONTAINER_RUNNING"] = "true"

        result = subprocess.run(
            ["./watch.sh", "--backend", "radv", "watch-model"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        serve_calls = [call for call in _read_logged_calls(tmp_path / "python3_calls.log") if len(call) > 1 and call[1] == "serve"]
        assert len(serve_calls) == 1
        assert serve_calls[0][2:] == ["watch-model", "--backend", "radv"]
        docker_calls = _read_logged_calls(tmp_path / "docker_calls.log")
        assert ["container", "inspect", "strix-llama-vulkan", "--format={{.State.Running}}"] in docker_calls

    def test_stop_sh_invokes_server_stop(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path)
        result = subprocess.run(
            ["./stop.sh"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert _last_python_call(tmp_path) == ["server.py", "stop"]

    def test_source_me_requires_sourcing(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path)
        result = subprocess.run(
            ["bash", "./source-me.sh"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "must be sourced" in result.stdout
