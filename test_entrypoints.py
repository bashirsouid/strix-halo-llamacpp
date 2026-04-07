from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


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

if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
  exit "${STUB_PYTHON_MODULE_PIP_EXIT:-0}"
fi

if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null || true
  if [[ $# -eq 1 ]]; then
    if [[ -n "${STUB_STATE_PORT:-}" ]]; then
      printf '%s\n' "$STUB_STATE_PORT"
    fi
    exit "${STUB_PYTHON_STDIN_EXIT:-0}"
  fi
  exit "${STUB_SERVER_IS_UP_EXIT:-1}"
fi

script_path="${1:-.}"
script_dir="$(cd "$(dirname "$script_path")" 2>/dev/null && pwd || pwd)"
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
    serve|bench|bench-all|bench-parallel|eval|eval-all|stop)
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

if [[ "$command" == "serve" && "${STUB_SERVE_EXIT:-0}" != "0" ]]; then
  exit "${STUB_SERVE_EXIT}"
fi

if [[ "$command" == "serve" ]]; then
  if [[ -z "$model" ]]; then
    model="picker-model"
  fi
  cat > "$script_dir/.server.json" <<JSON
{"model": "$model", "backend": "$backend", "container": "$container", "port": $port}
JSON
fi

exit "${STUB_PYTHON3_DEFAULT_EXIT:-0}"
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

    _write_executable(
        bin_dir / "pytest",
        """#!/usr/bin/env bash
set -euo pipefail
log_dir="${STRIX_STUB_LOG_DIR:?}"
{
  for arg in "$@"; do
    printf '%s\x1f' "$arg"
  done
  printf '\n'
} >> "$log_dir/pytest_calls.log"

live=0
for arg in "$@"; do
  if [[ "$arg" == "tests/test_inference.py" ]]; then
    live=1
    break
  fi
done

if [[ "$live" -eq 1 ]]; then
  exit "${STUB_PYTEST_LIVE_EXIT:-0}"
fi

exit "${STUB_PYTEST_EXIT:-0}"
""",
    )


def _make_sandbox(
    tmp_path: Path,
    *,
    files: tuple[str, ...] = ("start.sh", "watch.sh", "stop.sh", "source-me.sh"),
    with_tools: bool = False,
    with_venv: bool = False,
) -> tuple[Path, dict[str, str]]:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    for filename in files:
        destination = sandbox / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(filename), destination)
        destination.chmod(0o755)

    if with_tools:
        (sandbox / "tools").mkdir(exist_ok=True)

    if with_venv:
        activate = sandbox / ".venv" / "bin" / "activate"
        activate.parent.mkdir(parents=True, exist_ok=True)
        activate.write_text(
            f'VIRTUAL_ENV="{sandbox / ".venv"}"\nexport VIRTUAL_ENV\n'
        )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub_bin(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STRIX_STUB_LOG_DIR"] = str(tmp_path)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return sandbox, env


def _last_python_call(tmp_path: Path) -> list[str]:
    calls = [
        call
        for call in _read_logged_calls(tmp_path / "python3_calls.log")
        if call and call[0] not in {"-c", "-", "-m"}
    ]
    assert calls, "expected at least one python3 invocation"
    return calls[-1]


def _last_non_empty_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines, "expected at least one non-empty output line"
    return lines[-1]


def _write_runner_smoke_tests(sandbox: Path) -> None:
    tests_dir = sandbox / "tests"
    tests_dir.mkdir(exist_ok=True)

    _write_executable(
        tests_dir / "test_start.sh",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'tests/test_start.sh\n' >> "${STRIX_STUB_LOG_DIR:?}/runner_steps.log"
exit "${STUB_TEST_START_EXIT:-0}"
""",
    )

    _write_executable(
        tests_dir / "test_bash_entrypoints.sh",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'tests/test_bash_entrypoints.sh\n' >> "${STRIX_STUB_LOG_DIR:?}/runner_steps.log"
exit "${STUB_TEST_BASH_ENTRYPOINTS_EXIT:-0}"
""",
    )


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
        serve_calls = [
            call
            for call in _read_logged_calls(tmp_path / "python3_calls.log")
            if len(call) > 1 and call[1] == "serve"
        ]
        assert len(serve_calls) == 1
        assert serve_calls[0][2:] == ["watch-model", "--backend", "radv"]
        docker_calls = _read_logged_calls(tmp_path / "docker_calls.log")
        assert ["container", "inspect", "strix-llama-vulkan", "--format={{.State.Running}}"] in docker_calls

    def test_watch_sh_restarts_with_last_selection_when_container_is_down(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path)
        env["WATCH_INTERVAL"] = "0"
        env["WATCH_ONCE"] = "1"

        result = subprocess.run(
            ["./watch.sh", "--backend", "rocm", "watch-model"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        serve_calls = [
            call
            for call in _read_logged_calls(tmp_path / "python3_calls.log")
            if len(call) > 1 and call[1] == "serve"
        ]
        assert len(serve_calls) == 2
        assert serve_calls[0][2:] == ["watch-model", "--backend", "rocm"]
        assert serve_calls[1][2:] == ["watch-model", "--backend", "rocm"]
        assert "restarting" in result.stderr.lower()

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


class TestBenchmarkEntrypoint:
    @pytest.mark.parametrize(
        ("argv", "expected_calls"),
        [
            (
                ["./benchmark-run.sh", "dummy-model", "--backend", "radv"],
                [
                    ["server.py", "bench", "dummy-model", "--backend", "radv"],
                    ["tools/bench_viewer.py", "results/benchmark/bench_results.jsonl"],
                ],
            ),
            (
                ["./benchmark-run.sh", "--all", "--backend", "rocm"],
                [
                    ["server.py", "bench-all", "--backend", "rocm"],
                    ["tools/bench_viewer.py", "results/benchmark/bench_results.jsonl"],
                ],
            ),
            (
                ["./benchmark-run.sh", "--parallel", "dummy-model", "--max-np", "10"],
                [
                    ["server.py", "bench-parallel", "dummy-model", "--max-np", "10"],
                    ["tools/parallel_viewer.py", "results/benchmark/bench_parallel_results.jsonl"],
                ],
            ),
        ],
    )
    def test_benchmark_run_modes(self, tmp_path: Path, argv: list[str], expected_calls: list[list[str]]):
        sandbox, env = _make_sandbox(tmp_path, files=("benchmark-run.sh",), with_tools=True)
        result = subprocess.run(
            argv,
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert _read_logged_calls(tmp_path / "python3_calls.log") == expected_calls

    def test_benchmark_run_rejects_conflicting_modes(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path, files=("benchmark-run.sh",), with_tools=True)
        result = subprocess.run(
            ["./benchmark-run.sh", "--all", "--parallel"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "Cannot combine --parallel with another benchmark mode" in result.stderr


class TestEvaluateEntrypoint:
    @pytest.mark.parametrize(
        ("argv", "expected_calls"),
        [
            (
                ["./evaluate.sh", "dummy-model", "--suite", "humaneval"],
                [
                    ["server.py", "eval", "dummy-model", "--suite", "humaneval"],
                    ["tools/eval_viewer.py", "results/eval/eval_results.jsonl"],
                ],
            ),
            (
                ["./evaluate.sh", "--all", "--suite", "mbpp"],
                [
                    ["server.py", "eval-all", "--suite", "mbpp"],
                    ["tools/eval_viewer.py", "results/eval/eval_results.jsonl"],
                ],
            ),
        ],
    )
    def test_evaluate_modes(self, tmp_path: Path, argv: list[str], expected_calls: list[list[str]]):
        sandbox, env = _make_sandbox(tmp_path, files=("evaluate.sh",), with_tools=True, with_venv=True)
        env.pop("VIRTUAL_ENV", None)
        result = subprocess.run(
            argv,
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert _read_logged_calls(tmp_path / "python3_calls.log") == expected_calls

    def test_evaluate_requires_a_virtualenv_when_not_already_active(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path, files=("evaluate.sh",), with_tools=True)
        env.pop("VIRTUAL_ENV", None)
        result = subprocess.run(
            ["./evaluate.sh", "dummy-model"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "No virtualenv found" in result.stdout
        assert _read_logged_calls(tmp_path / "python3_calls.log") == []


class TestTopLevelTestRunner:
    def test_test_sh_reports_pass_on_last_line_after_all_steps_succeed(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path, files=("test.sh",))
        _write_runner_smoke_tests(sandbox)
        env["STUB_SERVER_IS_UP_EXIT"] = "0"

        result = subprocess.run(
            ["./test.sh"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert _last_non_empty_line(result.stdout) == "FINAL RESULT: PASS"
        assert (tmp_path / "runner_steps.log").read_text().splitlines() == [
            "tests/test_start.sh",
            "tests/test_bash_entrypoints.sh",
        ]
        assert _read_logged_calls(tmp_path / "pytest_calls.log") == [
            ["-q", "test_models.py", "test_entrypoints.py", "tests"],
            ["-q", "tests/test_inference.py"],
        ]

    def test_test_sh_reports_bash_subtest_failures_on_the_last_line(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path, files=("test.sh",))
        _write_runner_smoke_tests(sandbox)
        env["STUB_SERVER_IS_UP_EXIT"] = "0"
        env["STUB_TEST_BASH_ENTRYPOINTS_EXIT"] = "7"

        result = subprocess.run(
            ["./test.sh"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert _last_non_empty_line(result.stdout) == (
            "FINAL RESULT: FAIL (failed: bash smoke test: tests/test_bash_entrypoints.sh (exit 7))"
        )
        assert _read_logged_calls(tmp_path / "pytest_calls.log") == [
            ["-q", "test_models.py", "test_entrypoints.py", "tests"],
            ["-q", "tests/test_inference.py"],
        ]

    def test_test_sh_reports_live_inference_failures_on_the_last_line(self, tmp_path: Path):
        sandbox, env = _make_sandbox(tmp_path, files=("test.sh",))
        _write_runner_smoke_tests(sandbox)
        env["STUB_SERVER_IS_UP_EXIT"] = "0"
        env["STUB_PYTEST_LIVE_EXIT"] = "9"

        result = subprocess.run(
            ["./test.sh"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert _last_non_empty_line(result.stdout) == (
            "FINAL RESULT: FAIL (failed: live inference smoke test (exit 9))"
        )
        assert _read_logged_calls(tmp_path / "pytest_calls.log") == [
            ["-q", "test_models.py", "test_entrypoints.py", "tests"],
            ["-q", "tests/test_inference.py"],
        ]
