from __future__ import annotations

import datetime
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = Path.home() / ".cache" / "strix-halo-llamacpp"
AIDER_ROOT = CACHE_ROOT / "aider"
AIDER_REPO_DIR = AIDER_ROOT / "aider"
AIDER_BENCHMARK_ROOT = AIDER_ROOT / "benchmarks"
POLYGLOT_REPO_DIR = AIDER_BENCHMARK_ROOT / "polyglot-benchmark"
CURATED_ROOT = AIDER_BENCHMARK_ROOT / "curated"
MANIFEST_DIR = PROJECT_DIR / "benchmark_manifests"
RESULTS_DIR = PROJECT_DIR / "results" / "aider"
RESULTS_FILE = RESULTS_DIR / "aider_results.jsonl"
METADATA_DIR = RESULTS_DIR / "metadata"
LOG_DIR = RESULTS_DIR / "logs"

AIDER_IMAGE = os.environ.get("STRIX_AIDER_IMAGE", "strix-aider-benchmark")
AIDER_REPO_URL = os.environ.get("STRIX_AIDER_REPO_URL", "https://github.com/Aider-AI/aider.git")
POLYGLOT_REPO_URL = os.environ.get(
    "STRIX_POLYGLOT_REPO_URL", "https://github.com/Aider-AI/polyglot-benchmark.git"
)
DEFAULT_AIDER_REF = os.environ.get("STRIX_AIDER_REF", "main")
DEFAULT_POLYGLOT_REF = os.environ.get("STRIX_POLYGLOT_REF", "main")
DEFAULT_AIDER_MAX_TOKENS = int(os.environ.get("STRIX_AIDER_MAX_TOKENS", "24576"))
DEFAULT_AIDER_RANDOM_SEED = os.environ.get("STRIX_AIDER_RANDOM_SEED", "0")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


DEFAULT_AIDER_PROGRESS_POLL_SECONDS = max(0.25, _env_float("STRIX_AIDER_PROGRESS_POLL_SECONDS", 5.0))
DEFAULT_AIDER_PROGRESS_HEARTBEAT_SECONDS = max(
    0.0,
    _env_float("STRIX_AIDER_PROGRESS_HEARTBEAT_SECONDS", 60.0),
)


@dataclass(frozen=True)
class AiderProfile:
    name: str
    manifest_path: Path
    description: str
    tries: int = 2
    threads: int = 1
    edit_format: str = "whole"


@dataclass
class _ProgressState:
    last_completed: int
    last_heartbeat_at: float
    completion_announced: bool = False
    completion_announced_at: float | None = None
    completion_completed_tests: int = 0
    completion_total_tests: int = 0
    last_log_size: int = 0


@dataclass(frozen=True)
class FilteredRunResult:
    returncode: int
    all_results_written: bool = False
    all_results_written_at_sec: float | None = None
    post_completion_wait_sec: float | None = None


BUILTIN_PROFILES: dict[str, AiderProfile] = {
    "python-quick": AiderProfile(
        name="python-quick",
        manifest_path=MANIFEST_DIR / "aider-python-quick.txt",
        description=(
            "Fixed harder 9-exercise Python subset intended to finish in roughly 30 "
            "minutes on local reasoning models, while still separating models on parsing, "
            "stateful logic, graph/tree handling, and API-style edits."
        ),
    ),
    "python-all": AiderProfile(
        name="python-all",
        manifest_path=MANIFEST_DIR / "aider-python-all.txt",
        description=(
            "All 34 Python exercises from the Aider polyglot benchmark. Usually the most "
            "stable under-hour option for local code-model comparisons."
        ),
    ),
}
PROFILE_ALIASES = {
    "python-30m": "python-quick",
}
AIDER_PROFILE_NAMES = tuple(BUILTIN_PROFILES.keys())

SUMMARY_LINE_RE = re.compile(
    r"^(?:"
    r"- dirname:|"
    r"Copying |"
    r"Cleaning up and replacing|"
    r"Using pre-existing |"
    r"\.\.\.done$|"
    r"(?:test_cases|reasoning_effort|thinking_tokens|pass_rate_\d+|"
    r"percent_cases_well_formed|num_malformed_responses|num_with_malformed_responses|"
    r"syntax_errors|indentation_errors|exhausted_context_windows|test_timeouts|"
    r"total_tests|seconds_per_case):"
    r")"
)
NOISY_LINE_PREFIXES = (
    "fnames:",
    "Aider v",
    "Model:",
    "Git repo:",
    "Repo-map:",
    "Added ",
    "Removed ",
    "Tokens: ",
    "Cost: ",
    "Committing ",
    "Applying edits",
    "Search/replace",
    "Undoing",
    "Edit format:",
    "Warning: Input is not a terminal",
)
DIAGNOSTIC_LINE_RE = re.compile(
    r"^(?:Warning:|Tests failed:|Error loading model settings:|Traceback \(most recent call last\):|"
    r"Input tokens:|Total tokens:|Loaded model settings from:|No exercise directories found)"
)
DIAGNOSTIC_SUBSTRINGS = (
    "possibly exhausted context window",
    "context window exhausted",
    "token budget",
    "max token",
    "max_tokens",
    "out of tokens",
    "too many tokens",
    "truncated",
    "truncate",
    "timed out",
    "timeout",
    "rate limit",
    "failed to parse",
    "malformed response",
)
NOISY_TRACE_RE = re.compile(
    r"^(?:E\s+|>\s+|/usr/(?:lib|local/lib)/|[A-Za-z0-9_.-]+_test\.py:\d+:|[+]{1}\s|[-]{1}\s)"
)
AIDER_SITECUSTOMIZE = """
from __future__ import annotations

import os
import random
from pathlib import Path

try:
    from aider import models as _aider_models
except Exception:  # pragma: no cover
    _aider_models = None

if _aider_models is not None:
    _orig_register_litellm_models = _aider_models.register_litellm_models

    def _register_with_local_metadata(files):
        merged = list(files or [])
        seen = {str(item) for item in merged}
        for candidate in (
            Path.cwd() / ".aider.model.metadata.json",
            Path.home() / ".aider.model.metadata.json",
            Path("/aider/.aider.model.metadata.json"),
        ):
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in seen:
                merged.append(candidate_str)
                seen.add(candidate_str)
        return _orig_register_litellm_models(merged)

    if getattr(_aider_models.register_litellm_models, "__module__", "") != __name__:
        _aider_models.register_litellm_models = _register_with_local_metadata

_seed = os.environ.get("STRIX_AIDER_RANDOM_SEED")
if _seed not in (None, ""):
    try:
        random.seed(int(_seed))
    except ValueError:  # pragma: no cover
        random.seed(_seed)
"""


def _slugify(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = []
    last_dash = False
    for char in lowered:
        if char.isalnum():
            cleaned.append(char)
            last_dash = False
        else:
            if not last_dash:
                cleaned.append("-")
                last_dash = True
    slug = "".join(cleaned).strip("-")
    return slug or "run"


def _canonical_profile_name(profile_name: str) -> str:
    normalized = (profile_name or "python-quick").strip()
    return PROFILE_ALIASES.get(normalized, normalized)


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _ensure_dirs() -> None:
    AIDER_ROOT.mkdir(parents=True, exist_ok=True)
    AIDER_BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    CURATED_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _require_command(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"Required command not found on PATH: {name}")
    return resolved


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _looks_like_summary_line(line: str) -> bool:
    return bool(SUMMARY_LINE_RE.match(line.strip()))


def _is_diagnostic_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(stripped.startswith(prefix) for prefix in NOISY_LINE_PREFIXES):
        return False
    if _looks_like_summary_line(stripped):
        return True
    if stripped.startswith("Not a dir:") or stripped.startswith("No exercise directories found"):
        return True
    if NOISY_TRACE_RE.match(stripped) and not _looks_like_summary_line(stripped):
        return False
    if DIAGNOSTIC_LINE_RE.match(stripped):
        return True
    lowered = stripped.lower()
    return any(fragment in lowered for fragment in DIAGNOSTIC_SUBSTRINGS)


def _condense_aider_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("Tests failed:"):
        _, _, rhs = stripped.partition(":")
        exercise_path = rhs.strip()
        if exercise_path:
            return f"Tests failed: {Path(exercise_path).name}"
        return "Tests failed"
    return stripped


def _should_echo_aider_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _looks_like_summary_line(stripped):
        return False
    return _is_diagnostic_line(stripped)


def _collect_log_highlights(log_path: Path, *, limit: int = 20) -> list[str]:
    if not log_path.exists():
        return []
    highlights: list[str] = []
    seen: set[str] = set()
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not _is_diagnostic_line(raw_line):
            continue
        condensed = _condense_aider_line(raw_line)
        if not condensed or condensed in seen or _looks_like_summary_line(condensed):
            continue
        seen.add(condensed)
        highlights.append(condensed)
        if len(highlights) >= limit:
            break
    return highlights


def _collect_failed_exercises(log_path: Path, *, limit: int | None = None) -> list[str]:
    if not log_path.exists():
        return []
    failed: list[str] = []
    seen: set[str] = set()
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("Tests failed:"):
            continue
        _, _, rhs = stripped.partition(":")
        exercise = Path(rhs.strip()).name.strip()
        if not exercise or exercise in seen:
            continue
        seen.add(exercise)
        failed.append(exercise)
        if limit is not None and len(failed) >= limit:
            break
    return failed


def _write_sitecustomize(path: Path) -> Path:
    path.write_text(AIDER_SITECUSTOMIZE, encoding="utf-8")
    return path


def _tail_lines(path: Path, *, count: int = 40) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-count:]


def _file_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _run_filtered(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path,
    check: bool = True,
    progress_run_dir: Path | None = None,
    expected_total_tests: int | None = None,
    progress_poll_seconds: float = DEFAULT_AIDER_PROGRESS_POLL_SECONDS,
    progress_heartbeat_seconds: float = DEFAULT_AIDER_PROGRESS_HEARTBEAT_SECONDS,
) -> FilteredRunResult:
    def _reader_thread(stdout: Any, lines: "queue.Queue[str | None]") -> None:
        try:
            for raw_line in stdout:
                lines.put(raw_line)
        finally:
            try:
                stdout.close()
            finally:
                lines.put(None)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    poll_seconds = max(0.25, float(progress_poll_seconds or DEFAULT_AIDER_PROGRESS_POLL_SECONDS))
    heartbeat_seconds = max(0.0, float(progress_heartbeat_seconds or 0.0))
    progress_state = _ProgressState(last_completed=0, last_heartbeat_at=started_at)

    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None

        line_queue: queue.Queue[str | None] = queue.Queue()
        reader = threading.Thread(
            target=_reader_thread,
            args=(process.stdout, line_queue),
            name="aider-benchmark-log-reader",
            daemon=True,
        )
        reader.start()

        stdout_done = False

        while True:
            try:
                raw_line = line_queue.get(timeout=poll_seconds)
            except queue.Empty:
                if progress_run_dir is not None:
                    _maybe_emit_progress_update(
                        run_dir=progress_run_dir,
                        expected_total_tests=expected_total_tests,
                        started_at=started_at,
                        state=progress_state,
                        heartbeat_seconds=heartbeat_seconds,
                        log_path=log_path,
                    )
                if stdout_done and process.poll() is not None and line_queue.empty():
                    break
                continue

            if raw_line is None:
                stdout_done = True
            else:
                handle.write(raw_line)
                handle.flush()
                if _should_echo_aider_line(raw_line):
                    condensed = _condense_aider_line(raw_line)
                    if condensed:
                        print(condensed, flush=True)

            if progress_run_dir is not None:
                _maybe_emit_progress_update(
                    run_dir=progress_run_dir,
                    expected_total_tests=expected_total_tests,
                    started_at=started_at,
                    state=progress_state,
                    heartbeat_seconds=heartbeat_seconds,
                    log_path=log_path,
                )

            if stdout_done and process.poll() is not None and line_queue.empty():
                break

        reader.join(timeout=1.0)
        returncode = process.wait()

    if progress_run_dir is not None:
        _maybe_emit_progress_update(
            run_dir=progress_run_dir,
            expected_total_tests=expected_total_tests,
            started_at=started_at,
            state=progress_state,
            heartbeat_seconds=0.0,
            log_path=log_path,
            force_summary=True,
        )

    post_completion_wait_sec: float | None = None
    if progress_state.completion_announced_at is not None:
        post_completion_wait_sec = max(0.0, time.perf_counter() - progress_state.completion_announced_at)
        if post_completion_wait_sec >= max(1.0, poll_seconds):
            print(_format_post_completion_wait(wait_sec=post_completion_wait_sec), flush=True)

    result = FilteredRunResult(
        returncode=returncode,
        all_results_written=progress_state.completion_announced,
        all_results_written_at_sec=(
            round(progress_state.completion_announced_at - started_at, 2)
            if progress_state.completion_announced_at is not None
            else None
        ),
        post_completion_wait_sec=(
            round(post_completion_wait_sec, 2) if post_completion_wait_sec is not None else None
        ),
    )
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return result

def _git_head(repo_dir: Path) -> str:
    result = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True)
    return (result.stdout or "").strip()


def _docker_image_exists(image: str) -> bool:
    result = _run(["docker", "image", "inspect", image], capture_output=True, check=False)
    return result.returncode == 0


def _ensure_checkout(repo_url: str, dest: Path, ref: str, *, update: bool = False) -> str:
    _require_command("git")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not (dest / ".git").exists():
        raise RuntimeError(f"Existing path is not a git checkout: {dest}")

    if not dest.exists():
        _run(["git", "clone", repo_url, str(dest)])
        if ref:
            _run(["git", "checkout", ref], cwd=dest)
        return _git_head(dest)

    if update:
        _run(["git", "fetch", "--all", "--tags"], cwd=dest)
        if ref:
            _run(["git", "checkout", ref], cwd=dest)
            _run(["git", "pull", "--ff-only", "origin", ref], cwd=dest, check=False)
    return _git_head(dest)


def read_manifest_entries(manifest_path: str | Path) -> list[str]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")

    entries: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        normalized = line.lstrip("./").replace("\\", "/")
        if normalized not in seen:
            entries.append(normalized)
            seen.add(normalized)
    if not entries:
        raise ValueError(f"Benchmark manifest is empty: {path}")
    return entries


def resolve_profile(
    profile_name: str = "python-quick",
    manifest_path: str | Path | None = None,
) -> AiderProfile:
    if manifest_path:
        custom_path = Path(manifest_path).expanduser().resolve()
        return AiderProfile(
            name=f"custom-{_slugify(custom_path.stem)}",
            manifest_path=custom_path,
            description=f"Custom benchmark manifest from {custom_path}",
        )

    canonical = _canonical_profile_name(profile_name)
    try:
        return BUILTIN_PROFILES[canonical]
    except KeyError as exc:
        raise ValueError(
            f"Unknown aider benchmark profile: {profile_name}. Available: {', '.join(AIDER_PROFILE_NAMES)}"
        ) from exc


def _materialize_manifest(polyglot_root: Path, profile: AiderProfile) -> Path:
    entries = read_manifest_entries(profile.manifest_path)
    manifest_text = "\n".join(entries) + "\n"
    manifest_hash = _sha1_text(manifest_text)
    dest_root = CURATED_ROOT / profile.name
    meta_path = dest_root / ".manifest.json"
    expected_meta = {
        "profile": profile.name,
        "manifest": str(profile.manifest_path),
        "manifest_hash": manifest_hash,
        "entries": entries,
    }

    if dest_root.exists() and meta_path.exists():
        try:
            current_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current_meta = None
        if current_meta == expected_meta and all((dest_root / entry).exists() for entry in entries):
            return dest_root
        shutil.rmtree(dest_root)

    dest_root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        src = polyglot_root / entry
        if not src.exists():
            raise FileNotFoundError(f"Manifest entry does not exist in polyglot-benchmark: {src}")
        dst = dest_root / entry
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

    meta_path.write_text(json.dumps(expected_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest_root


def _write_model_settings(settings_path: Path, *, model_name: str, edit_format: str, max_tokens: int) -> Path:
    content = (
        f"- name: aider/extra_params\n"
        f"  extra_params:\n"
        f"    max_tokens: {int(max_tokens)}\n"
        f"\n"
        f"- name: {model_name}\n"
        f"  edit_format: {edit_format}\n"
        f"  use_repo_map: false\n"
        f"  streaming: false\n"
    )
    settings_path.write_text(content, encoding="utf-8")
    return settings_path


def _write_model_metadata(
    metadata_path: Path,
    *,
    model_name: str,
    context_window: int,
    max_tokens: int,
) -> Path:
    payload = {
        model_name: {
            "max_tokens": int(max_tokens),
            "max_input_tokens": int(context_window),
            "max_output_tokens": int(max_tokens),
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def _iter_result_paths(run_dir: Path) -> Iterable[Path]:
    return sorted(run_dir.glob("*/exercises/practice/*/.aider.results.json"))


def load_run_results(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    results: list[dict[str, Any]] = []
    for path in _iter_result_paths(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            results.append(payload)
    return results


def summarize_run_dir(run_dir: str | Path, *, wall_time_sec: float | None = None) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    results = load_run_results(root)
    total_tests = len(list(root.glob("*/exercises/practice/*")))
    completed_tests = len(results)

    summary: dict[str, Any] = {
        "run_dir": str(root),
        "total_tests": total_tests,
        "completed_tests": completed_tests,
        "wall_time_sec": round(wall_time_sec, 2) if wall_time_sec is not None else None,
    }
    if not results:
        return summary

    tries = max(len(result.get("tests_outcomes", [])) for result in results)
    pass_counts = [0] * tries
    duration_sec = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    syntax_errors = 0
    indentation_errors = 0
    exhausted_context_windows = 0
    num_malformed_responses = 0
    num_with_malformed_responses = 0
    lazy_comments = 0
    test_timeouts = 0
    error_outputs = 0
    user_asks = 0

    models: set[str] = set()
    edit_formats: set[str] = set()
    commit_hashes: set[str] = set()
    reasoning_efforts: set[str] = set()
    thinking_tokens_values: set[int] = set()

    for result in results:
        outcomes = result.get("tests_outcomes", [])
        if outcomes and outcomes[-1]:
            for index in range(len(outcomes) - 1, tries):
                pass_counts[index] += 1

        duration_sec += float(result.get("duration", 0) or 0)
        prompt_tokens += int(result.get("prompt_tokens", 0) or 0)
        completion_tokens += int(result.get("completion_tokens", 0) or 0)
        syntax_errors += int(result.get("syntax_errors", 0) or 0)
        indentation_errors += int(result.get("indentation_errors", 0) or 0)
        exhausted_context_windows += int(result.get("num_exhausted_context_windows", 0) or 0)
        malformed = int(result.get("num_malformed_responses", 0) or 0)
        num_malformed_responses += malformed
        if malformed:
            num_with_malformed_responses += 1
        lazy_comments += int(result.get("lazy_comments", 0) or 0)
        test_timeouts += int(result.get("test_timeouts", 0) or 0)
        error_outputs += int(result.get("num_error_outputs", 0) or 0)
        user_asks += int(result.get("num_user_asks", 0) or 0)

        model_name = result.get("model")
        if isinstance(model_name, str) and model_name:
            models.add(model_name)
        edit_format = result.get("edit_format")
        if isinstance(edit_format, str) and edit_format:
            edit_formats.add(edit_format)
        commit_hash = result.get("commit_hash")
        if isinstance(commit_hash, str) and commit_hash:
            commit_hashes.add(commit_hash)
        reasoning_effort = result.get("reasoning_effort")
        if isinstance(reasoning_effort, str) and reasoning_effort:
            reasoning_efforts.add(reasoning_effort)
        thinking_tokens = result.get("thinking_tokens")
        if isinstance(thinking_tokens, int):
            thinking_tokens_values.add(thinking_tokens)

    summary.update(
        {
            "duration_sec": round(duration_sec, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "syntax_errors": syntax_errors,
            "indentation_errors": indentation_errors,
            "exhausted_context_windows": exhausted_context_windows,
            "num_malformed_responses": num_malformed_responses,
            "num_with_malformed_responses": num_with_malformed_responses,
            "lazy_comments": lazy_comments,
            "test_timeouts": test_timeouts,
            "error_outputs": error_outputs,
            "user_asks": user_asks,
            "models": sorted(models),
            "edit_formats": sorted(edit_formats),
            "commit_hashes": sorted(commit_hashes),
            "reasoning_efforts": sorted(reasoning_efforts),
            "thinking_tokens_values": sorted(thinking_tokens_values),
        }
    )

    for index, passed in enumerate(pass_counts, start=1):
        summary[f"pass_count_{index}"] = passed
        summary[f"pass_rate_{index}"] = round((100.0 * passed / completed_tests), 1)

    if completed_tests:
        summary["percent_cases_well_formed"] = round(
            100.0 * (1.0 - (num_with_malformed_responses / completed_tests)), 1
        )
        summary["seconds_per_case_model"] = round(duration_sec / completed_tests, 2)
        if wall_time_sec is not None:
            summary["seconds_per_case_wall"] = round(float(wall_time_sec) / completed_tests, 2)
    if duration_sec > 0:
        summary["completion_tok_s_model"] = round(completion_tokens / duration_sec, 2)
        summary["prompt_tok_s_model"] = round(prompt_tokens / duration_sec, 2)
    if wall_time_sec and wall_time_sec > 0:
        summary["completion_tok_s_wall"] = round(completion_tokens / float(wall_time_sec), 2)
        summary["prompt_tok_s_wall"] = round(prompt_tokens / float(wall_time_sec), 2)
    return summary


def _format_progress_summary(
    summary: dict[str, Any],
    *,
    expected_total_tests: int | None = None,
) -> str | None:
    completed_tests = int(summary.get("completed_tests", 0) or 0)
    if completed_tests <= 0:
        return None

    run_dir = summary.get("run_dir")
    dirname = Path(str(run_dir)).name if run_dir else "run"
    total_tests = int(summary.get("total_tests", 0) or 0) or int(expected_total_tests or 0)

    lines = [
        f"- dirname: {dirname}",
        f"  test_cases: {completed_tests}",
    ]

    pass_rate_keys = sorted(
        (key for key in summary if key.startswith("pass_rate_")),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    for key in pass_rate_keys:
        value = summary.get(key)
        if value is None:
            continue
        lines.append(f"  {key}: {float(value):.1f}")

    ordered_metrics = [
        "percent_cases_well_formed",
        "error_outputs",
        "num_malformed_responses",
        "num_with_malformed_responses",
        "syntax_errors",
        "indentation_errors",
        "exhausted_context_windows",
        "test_timeouts",
    ]
    for key in ordered_metrics:
        value = summary.get(key)
        if value is None:
            value = 0
        lines.append(f"  {key}: {value}")

    lines.append(f"  total_tests: {total_tests}")

    seconds_per_case = summary.get("seconds_per_case_wall")
    if seconds_per_case is None:
        seconds_per_case = summary.get("seconds_per_case_model")
    if seconds_per_case is not None:
        lines.append(f"  seconds_per_case: {float(seconds_per_case):.1f}")

    return "\n".join(lines)


def _format_progress_heartbeat(
    *,
    completed_tests: int,
    total_tests: int | None,
    elapsed_sec: float,
) -> str:
    total_text = str(int(total_tests)) if total_tests else "?"
    elapsed_min = elapsed_sec / 60.0
    return f"Progress: {completed_tests}/{total_text} completed after {elapsed_min:.1f}m"


def _format_results_written_notice(*, completed_tests: int, total_tests: int | None) -> str:
    total_text = str(int(total_tests)) if total_tests else "?"
    return (
        f"All exercise result files written ({completed_tests}/{total_text}); "
        f"waiting for benchmark process to exit..."
    )


def _format_finalizing_heartbeat(
    *,
    completed_tests: int,
    total_tests: int | None,
    elapsed_since_completion_sec: float,
    saw_new_log_output: bool,
) -> str:
    total_text = str(int(total_tests)) if total_tests else "?"
    elapsed_min = elapsed_since_completion_sec / 60.0
    log_status = "new log output seen" if saw_new_log_output else "no new log output"
    return (
        f"Finalizing: {completed_tests}/{total_text} result files exist; "
        f"benchmark process still alive {elapsed_min:.1f}m later ({log_status})."
    )


def _format_post_completion_wait(*, wait_sec: float) -> str:
    return f"Benchmark process exited {wait_sec / 60.0:.1f}m after all exercise result files were written."


def _maybe_emit_progress_update(
    *,
    run_dir: Path,
    expected_total_tests: int | None,
    started_at: float,
    state: _ProgressState,
    heartbeat_seconds: float,
    log_path: Path,
    force_summary: bool = False,
) -> None:
    now = time.perf_counter()
    wall_time_sec = now - started_at
    summary = summarize_run_dir(run_dir, wall_time_sec=wall_time_sec)
    completed_tests = int(summary.get("completed_tests", 0) or 0)
    total_tests = int(summary.get("total_tests", 0) or 0) or int(expected_total_tests or 0)

    if completed_tests > state.last_completed or (force_summary and completed_tests > 0 and completed_tests != state.last_completed):
        formatted = _format_progress_summary(summary, expected_total_tests=total_tests)
        if formatted:
            print(formatted, flush=True)
        state.last_completed = completed_tests
        state.last_heartbeat_at = now

    all_results_written = total_tests > 0 and completed_tests >= total_tests
    if all_results_written and not state.completion_announced:
        print(
            _format_results_written_notice(
                completed_tests=completed_tests,
                total_tests=total_tests,
            ),
            flush=True,
        )
        state.completion_announced = True
        state.completion_announced_at = now
        state.completion_completed_tests = completed_tests
        state.completion_total_tests = total_tests
        state.last_log_size = _file_size_bytes(log_path)
        state.last_heartbeat_at = now
        return

    if state.completion_announced:
        if heartbeat_seconds > 0 and (now - state.last_heartbeat_at) >= heartbeat_seconds:
            current_log_size = _file_size_bytes(log_path)
            print(
                _format_finalizing_heartbeat(
                    completed_tests=state.completion_completed_tests or completed_tests,
                    total_tests=state.completion_total_tests or total_tests,
                    elapsed_since_completion_sec=max(0.0, now - (state.completion_announced_at or now)),
                    saw_new_log_output=current_log_size > state.last_log_size,
                ),
                flush=True,
            )
            state.last_log_size = current_log_size
            state.last_heartbeat_at = now
        return

    if heartbeat_seconds > 0 and (now - state.last_heartbeat_at) >= heartbeat_seconds:
        print(
            _format_progress_heartbeat(
                completed_tests=completed_tests,
                total_tests=total_tests,
                elapsed_sec=wall_time_sec,
            ),
            flush=True,
        )
        state.last_heartbeat_at = now

def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def ensure_aider_setup(
    *,
    update: bool = False,
    aider_ref: str = DEFAULT_AIDER_REF,
    polyglot_ref: str = DEFAULT_POLYGLOT_REF,
) -> dict[str, Any]:
    _ensure_dirs()
    _require_command("git")
    _require_command("docker")

    aider_head = _ensure_checkout(AIDER_REPO_URL, AIDER_REPO_DIR, aider_ref, update=update)
    polyglot_head = _ensure_checkout(POLYGLOT_REPO_URL, POLYGLOT_REPO_DIR, polyglot_ref, update=update)

    sitecustomize_path = _write_sitecustomize(AIDER_REPO_DIR / "sitecustomize.py")

    if update or not _docker_image_exists(AIDER_IMAGE):
        _run(["docker", "build", "--file", "benchmark/Dockerfile", "-t", AIDER_IMAGE, "."], cwd=AIDER_REPO_DIR)

    curated_dirs = {
        profile.name: str(_materialize_manifest(POLYGLOT_REPO_DIR, profile))
        for profile in BUILTIN_PROFILES.values()
    }
    return {
        "aider_repo": str(AIDER_REPO_DIR),
        "aider_head": aider_head,
        "polyglot_repo": str(POLYGLOT_REPO_DIR),
        "polyglot_head": polyglot_head,
        "benchmark_root": str(AIDER_BENCHMARK_ROOT),
        "curated_dirs": curated_dirs,
        "docker_image": AIDER_IMAGE,
        "sitecustomize": str(sitecustomize_path),
    }


def _build_run_id(
    *,
    model_alias: str,
    profile_name: str,
    backend: str,
    run_label: str | None = None,
) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    label_suffix = f"--{_slugify(run_label)}" if run_label else ""
    return (
        f"{timestamp}--{_slugify(model_alias)}--{_slugify(profile_name)}--{_slugify(backend)}"
        f"{label_suffix}"
    )


def run_aider_benchmark(
    *,
    model_alias: str,
    backend: str,
    port: int,
    profile_name: str = "python-quick",
    manifest_path: str | Path | None = None,
    run_label: str | None = None,
    max_tokens: int = DEFAULT_AIDER_MAX_TOKENS,
    threads: int = 1,
    tries: int | None = None,
    edit_format: str = "whole",
    context_window: int = 524288,
    api_key: str | None = None,
    update_harness: bool = False,
    aider_ref: str = DEFAULT_AIDER_REF,
    polyglot_ref: str = DEFAULT_POLYGLOT_REF,
    model_display_name: str | None = None,
    quant: str | None = None,
) -> dict[str, Any]:
    _ensure_dirs()
    setup = ensure_aider_setup(update=update_harness, aider_ref=aider_ref, polyglot_ref=polyglot_ref)
    profile = resolve_profile(profile_name, manifest_path)
    curated_dir = _materialize_manifest(POLYGLOT_REPO_DIR, profile)
    manifest_entries = read_manifest_entries(profile.manifest_path)

    model_name = f"openai/{model_alias}"
    settings_path = _write_model_settings(
        AIDER_REPO_DIR / ".aider.model.settings.yml",
        model_name=model_name,
        edit_format=edit_format,
        max_tokens=max_tokens,
    )
    metadata_path = _write_model_metadata(
        AIDER_REPO_DIR / ".aider.model.metadata.json",
        model_name=model_name,
        context_window=context_window,
        max_tokens=max_tokens,
    )

    run_id = _build_run_id(
        model_alias=model_alias,
        profile_name=profile.name,
        backend=backend,
        run_label=run_label,
    )
    run_dir = AIDER_BENCHMARK_ROOT / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    log_path = LOG_DIR / f"{run_id}.log"

    relative_exercises_dir = curated_dir.relative_to(AIDER_BENCHMARK_ROOT).as_posix()
    base_url = f"http://host.docker.internal:{port}/v1"

    inner_main = [
        "python3",
        "./benchmark/benchmark.py",
        run_id,
        "--model",
        model_name,
        "--edit-format",
        edit_format,
        "--threads",
        str(int(threads)),
        "--tries",
        str(int(tries if tries is not None else profile.tries)),
        "--exercises-dir",
        relative_exercises_dir,
        "--read-model-settings",
        "/aider/.aider.model.settings.yml",
        "--clean",
    ]
    shell_command = (
        "git config --global --add safe.directory /aider && "
        f"cd /aider && exec {shlex.join(inner_main)}"
    )

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-v",
        f"{AIDER_REPO_DIR}:/aider",
        "-v",
        f"{AIDER_BENCHMARK_ROOT}:/benchmarks",
        "-e",
        f"OPENAI_API_KEY={api_key or 'local'}",
        "-e",
        f"OPENAI_API_BASE={base_url}",
        "-e",
        f"STRIX_AIDER_RANDOM_SEED={DEFAULT_AIDER_RANDOM_SEED}",
        "-e",
        "AIDER_DOCKER=1",
        "-e",
        "AIDER_BENCHMARK_DIR=/benchmarks",
        "-e",
        "PYTHONPATH=/aider",
        "-e",
        "HOME=/aider",
        setup["docker_image"],
        "bash",
        "-lc",
        shell_command,
    ]

    started_at = time.perf_counter()
    run_proc = _run_filtered(
        docker_cmd,
        log_path=log_path,
        check=False,
        progress_run_dir=run_dir,
        expected_total_tests=len(manifest_entries),
    )
    wall_time_sec = time.perf_counter() - started_at

    summary = summarize_run_dir(run_dir, wall_time_sec=wall_time_sec)
    important_log_lines = _collect_log_highlights(log_path)
    failed_exercises = _collect_failed_exercises(log_path)
    summary.update(
        {
            "ok": run_proc.returncode == 0,
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_alias,
            "model_display_name": model_display_name or model_alias,
            "quant": quant,
            "backend": backend,
            "profile": profile.name,
            "profile_description": profile.description,
            "manifest": str(profile.manifest_path),
            "manifest_entries": manifest_entries,
            "max_tokens": int(max_tokens),
            "threads": int(threads),
            "tries": int(tries if tries is not None else profile.tries),
            "edit_format": edit_format,
            "context_window": int(context_window),
            "benchmark_dir": str(run_dir),
            "benchmark_exercises_dir": relative_exercises_dir,
            "aider_repo": setup["aider_repo"],
            "aider_head": setup["aider_head"],
            "polyglot_repo": setup["polyglot_repo"],
            "polyglot_head": setup["polyglot_head"],
            "docker_image": setup["docker_image"],
            "openai_base_url": base_url,
            "settings_file": str(settings_path),
            "model_metadata_file": str(metadata_path),
            "returncode": run_proc.returncode,
            "force_rerun": True,
            "all_results_written_before_exit": run_proc.all_results_written,
            "all_results_written_at_sec": run_proc.all_results_written_at_sec,
            "post_completion_wait_sec": run_proc.post_completion_wait_sec,
            "log_file": str(log_path),
            "important_log_lines": important_log_lines,
            "important_log_line_count": len(important_log_lines),
            "failed_exercises": failed_exercises,
            "failed_exercise_count": len(failed_exercises),
            "random_seed": DEFAULT_AIDER_RANDOM_SEED,
            "sitecustomize": setup.get("sitecustomize"),
            "progress_poll_seconds": DEFAULT_AIDER_PROGRESS_POLL_SECONDS,
            "progress_heartbeat_seconds": DEFAULT_AIDER_PROGRESS_HEARTBEAT_SECONDS,
        }
    )
    if run_proc.returncode != 0:
        summary["log_tail"] = _tail_lines(log_path, count=40)

    metadata_path_out = METADATA_DIR / f"{run_id}.json"
    summary["metadata_file"] = str(metadata_path_out)
    summary["results_file"] = str(RESULTS_FILE)
    metadata_path_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_jsonl(RESULTS_FILE, summary)
    return summary
