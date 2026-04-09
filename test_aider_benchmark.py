from __future__ import annotations

import contextlib
import importlib
import json
import sys
import types
from pathlib import Path

import aider_benchmark
from tools import eval_viewer


def _import_server_for_tests(monkeypatch):
    eval_profiles = types.ModuleType("eval_profiles")

    class EvalProfile:  # pragma: no cover - tiny import shim
        pass

    eval_profiles.EvalProfile = EvalProfile
    eval_profiles.ensure_override_dataset = lambda *args, **kwargs: None
    eval_profiles.resolve_eval_profile = lambda *args, **kwargs: None

    repo_cache = types.ModuleType("repo_cache")
    repo_cache.DEFAULT_PROXY_HOST = "127.0.0.1"
    repo_cache.DEFAULT_PROXY_PORT = 8001
    repo_cache.DEFAULT_SLOT_ID = 0
    repo_cache.SLOT_CACHE_ROOT = Path("/tmp/slots")
    repo_cache.ensure_cache_dirs = lambda *args, **kwargs: None
    repo_cache.ensure_gitignore_entry = lambda *args, **kwargs: None
    repo_cache.load_repo_context = lambda *args, **kwargs: {}
    repo_cache.make_warm_payload = lambda *args, **kwargs: {}
    repo_cache.refresh_repo_context = lambda *args, **kwargs: {}
    repo_cache.repo_paths = lambda *args, **kwargs: {}
    repo_cache.start_repo_proxy = lambda *args, **kwargs: None
    repo_cache.write_opencode_config = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "eval_profiles", eval_profiles)
    monkeypatch.setitem(sys.modules, "repo_cache", repo_cache)
    sys.modules.pop("server", None)
    return importlib.import_module("server")



def _make_exercise(root: Path, relpath: str) -> None:
    exercise_dir = root / relpath
    exercise_dir.mkdir(parents=True, exist_ok=True)
    (exercise_dir / ".meta").mkdir(exist_ok=True)
    (exercise_dir / ".docs").mkdir(exist_ok=True)
    (exercise_dir / ".meta" / "config.json").write_text("{}")


def test_read_manifest_entries_skips_comments_and_duplicates(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        """
        # comment
        python/exercises/practice/beer-song
        python/exercises/practice/beer-song

        ./python/exercises/practice/poker   # inline comment
        """
    )

    assert aider_benchmark.read_manifest_entries(manifest) == [
        "python/exercises/practice/beer-song",
        "python/exercises/practice/poker",
    ]


def test_materialize_manifest_copies_only_listed_exercises(tmp_path: Path) -> None:
    polyglot_root = tmp_path / "polyglot-benchmark"
    _make_exercise(polyglot_root, "python/exercises/practice/beer-song")
    _make_exercise(polyglot_root, "python/exercises/practice/poker")
    _make_exercise(polyglot_root, "python/exercises/practice/wordy")

    manifest = tmp_path / "subset.txt"
    manifest.write_text(
        "python/exercises/practice/beer-song\npython/exercises/practice/wordy\n"
    )
    profile = aider_benchmark.AiderProfile(
        name="test-profile",
        manifest_path=manifest,
        description="test profile",
    )

    original_curated_root = aider_benchmark.CURATED_ROOT
    try:
        aider_benchmark.CURATED_ROOT = tmp_path / "curated"
        out_dir = aider_benchmark._materialize_manifest(polyglot_root, profile)
    finally:
        aider_benchmark.CURATED_ROOT = original_curated_root

    assert (out_dir / "python/exercises/practice/beer-song").exists()
    assert (out_dir / "python/exercises/practice/wordy").exists()
    assert not (out_dir / "python/exercises/practice/poker").exists()

    meta = json.loads((out_dir / ".manifest.json").read_text())
    assert meta["profile"] == "test-profile"
    assert len(meta["entries"]) == 2


def test_summarize_run_dir_computes_pass_rates_and_token_rates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    case_one = run_dir / "python/exercises/practice/beer-song"
    case_two = run_dir / "python/exercises/practice/poker"
    case_one.mkdir(parents=True)
    case_two.mkdir(parents=True)

    (case_one / ".aider.results.json").write_text(
        json.dumps(
            {
                "model": "openai/test-model",
                "edit_format": "whole",
                "commit_hash": "abc1234",
                "tests_outcomes": [False, True],
                "duration": 12.0,
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "num_malformed_responses": 0,
                "num_exhausted_context_windows": 0,
                "syntax_errors": 0,
                "indentation_errors": 0,
                "lazy_comments": 0,
                "test_timeouts": 0,
                "num_error_outputs": 0,
                "num_user_asks": 0,
            }
        )
    )
    (case_two / ".aider.results.json").write_text(
        json.dumps(
            {
                "model": "openai/test-model",
                "edit_format": "whole",
                "commit_hash": "abc1234",
                "tests_outcomes": [False, False],
                "duration": 8.0,
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "num_malformed_responses": 2,
                "num_exhausted_context_windows": 1,
                "syntax_errors": 1,
                "indentation_errors": 0,
                "lazy_comments": 1,
                "test_timeouts": 0,
                "num_error_outputs": 1,
                "num_user_asks": 0,
            }
        )
    )

    summary = aider_benchmark.summarize_run_dir(run_dir, wall_time_sec=40.0)

    assert summary["total_tests"] == 2
    assert summary["completed_tests"] == 2
    assert summary["pass_count_1"] == 0
    assert summary["pass_count_2"] == 1
    assert summary["pass_rate_1"] == 0.0
    assert summary["pass_rate_2"] == 50.0
    assert summary["seconds_per_case_model"] == 10.0
    assert summary["seconds_per_case_wall"] == 20.0
    assert summary["completion_tok_s_model"] == 5.0
    assert summary["completion_tok_s_wall"] == 2.5
    assert summary["percent_cases_well_formed"] == 50.0
    assert summary["num_malformed_responses"] == 2
    assert summary["syntax_errors"] == 1
    assert summary["exhausted_context_windows"] == 1


def test_resolve_profile_accepts_python_quick_and_legacy_alias() -> None:
    assert aider_benchmark.resolve_profile("python-quick").name == "python-quick"
    assert aider_benchmark.resolve_profile("python-30m").name == "python-quick"



def test_should_echo_aider_line_keeps_warnings_but_hides_chatter() -> None:
    assert not aider_benchmark._should_echo_aider_line("fnames: beer_song.py")
    assert not aider_benchmark._should_echo_aider_line("E       AssertionError: 8.0 != 800")
    assert not aider_benchmark._should_echo_aider_line("grep_test.py:55: AssertionError")
    assert not aider_benchmark._should_echo_aider_line("- dirname: run-id")
    assert not aider_benchmark._should_echo_aider_line("  test_cases: 4")
    assert aider_benchmark._should_echo_aider_line("Warning: context window exhausted")
    assert aider_benchmark._should_echo_aider_line("Tests failed: /benchmarks/run/python/exercises/practice/book-store")
    assert aider_benchmark._condense_aider_line(
        "Tests failed: /benchmarks/run/python/exercises/practice/book-store"
    ) == "Tests failed: book-store"


def test_format_progress_summary_and_heartbeat() -> None:
    summary = {
        "run_dir": "/tmp/2026-04-08-run",
        "completed_tests": 4,
        "pass_rate_1": 0.0,
        "pass_rate_2": 75.0,
        "percent_cases_well_formed": 100.0,
        "error_outputs": 0,
        "num_malformed_responses": 0,
        "num_with_malformed_responses": 0,
        "syntax_errors": 0,
        "indentation_errors": 0,
        "exhausted_context_windows": 0,
        "test_timeouts": 0,
        "seconds_per_case_wall": 223.67,
    }

    formatted = aider_benchmark._format_progress_summary(summary, expected_total_tests=9)

    assert formatted is not None
    assert formatted.splitlines() == [
        "- dirname: 2026-04-08-run",
        "  test_cases: 4",
        "  pass_rate_1: 0.0",
        "  pass_rate_2: 75.0",
        "  percent_cases_well_formed: 100.0",
        "  error_outputs: 0",
        "  num_malformed_responses: 0",
        "  num_with_malformed_responses: 0",
        "  syntax_errors: 0",
        "  indentation_errors: 0",
        "  exhausted_context_windows: 0",
        "  test_timeouts: 0",
        "  total_tests: 9",
        "  seconds_per_case: 223.7",
    ]

    assert aider_benchmark._format_progress_heartbeat(
        completed_tests=4,
        total_tests=9,
        elapsed_sec=3600.0,
    ) == "Progress: 4/9 completed after 60.0m"
    assert aider_benchmark._format_results_written_notice(
        completed_tests=5,
        total_tests=5,
    ) == "All exercise result files written (5/5); waiting for benchmark process to exit..."
    assert aider_benchmark._format_results_written_notice(
        completed_tests=5,
        total_tests=5,
        active_requests=2,
    ) == "All exercise result files written (5/5); waiting for benchmark process to exit... active_llm_requests=2"
    assert aider_benchmark._format_finalizing_heartbeat(
        completed_tests=5,
        total_tests=5,
        elapsed_since_completion_sec=360.0,
        saw_new_log_output=False,
    ) == "Finalizing: 5/5 result files exist; benchmark process still alive 6.0m later (no new log output)."
    assert aider_benchmark._format_finalizing_heartbeat(
        completed_tests=5,
        total_tests=5,
        elapsed_since_completion_sec=360.0,
        saw_new_log_output=False,
        active_requests=1,
    ) == "Finalizing: 5/5 result files exist; benchmark process still alive 6.0m later (no new log output, active_llm_requests=1)."
    assert aider_benchmark._format_post_completion_wait(wait_sec=360.0) == (
        "Benchmark process exited 6.0m after all exercise result files were written."
    )



def test_collect_failed_exercises_and_write_sitecustomize(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "Tests failed: /benchmarks/run/python/exercises/practice/book-store\n"
        "Tests failed: /benchmarks/run/python/exercises/practice/grep\n"
        "Tests failed: /benchmarks/run/python/exercises/practice/book-store\n"
    )

    assert aider_benchmark._collect_failed_exercises(log_path) == ["book-store", "grep"]

    sitecustomize = tmp_path / "sitecustomize.py"
    aider_benchmark._write_sitecustomize(sitecustomize)
    contents = sitecustomize.read_text()
    assert "register_litellm_models" in contents
    assert "STRIX_AIDER_RANDOM_SEED" in contents


def test_run_aider_benchmark_uses_exec_and_host_side_stats(monkeypatch, tmp_path: Path) -> None:
    aider_repo = tmp_path / "aider"
    benchmark_root = tmp_path / "benchmarks"
    curated_dir = benchmark_root / "curated" / "python-quick"
    results_dir = tmp_path / "results" / "aider"
    metadata_dir = results_dir / "metadata"
    log_dir = results_dir / "logs"
    manifest = tmp_path / "aider-python-quick.txt"

    aider_repo.mkdir(parents=True)
    curated_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    manifest.write_text("python/exercises/practice/book-store\n")

    monkeypatch.setattr(aider_benchmark, "AIDER_REPO_DIR", aider_repo)
    monkeypatch.setattr(aider_benchmark, "AIDER_BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr(aider_benchmark, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(aider_benchmark, "RESULTS_FILE", results_dir / "aider_results.jsonl")
    monkeypatch.setattr(aider_benchmark, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(aider_benchmark, "LOG_DIR", log_dir)
    monkeypatch.setattr(aider_benchmark, "_ensure_dirs", lambda: None)
    monkeypatch.setattr(
        aider_benchmark,
        "ensure_aider_setup",
        lambda **kwargs: {
            "aider_repo": str(aider_repo),
            "aider_head": "a" * 40,
            "polyglot_repo": str(tmp_path / "polyglot-benchmark"),
            "polyglot_head": "b" * 40,
            "docker_image": "strix-aider-benchmark",
            "sitecustomize": str(aider_repo / "sitecustomize.py"),
        },
    )
    monkeypatch.setattr(
        aider_benchmark,
        "resolve_profile",
        lambda profile_name, manifest_path=None: aider_benchmark.AiderProfile(
            name="python-quick",
            manifest_path=manifest,
            description="quick",
        ),
    )
    monkeypatch.setattr(aider_benchmark, "_materialize_manifest", lambda polyglot_root, profile: curated_dir)

    captured: dict[str, object] = {}

    class FakeRequestMonitor:
        def __init__(self, base_url: str, log_path: Path | None = None) -> None:
            self.base_url = base_url
            self.log_path = log_path

        def active_request_count(self) -> int:
            return 0

    @contextlib.contextmanager
    def fake_request_monitor(*, verbose: bool, upstream_port: int, proxy_log_path: Path):
        captured["request_monitor"] = {
            "verbose": verbose,
            "upstream_port": upstream_port,
            "proxy_log_path": proxy_log_path,
        }
        yield FakeRequestMonitor(base_url=f"http://host.docker.internal:{upstream_port + 1}/v1", log_path=proxy_log_path)

    def fake_run_filtered(cmd, **kwargs):
        captured["cmd"] = cmd
        log_path = kwargs["log_path"]
        log_path.write_text("Tests failed: /benchmarks/run/python/exercises/practice/book-store\n")
        return aider_benchmark.FilteredRunResult(
            returncode=0,
            all_results_written=True,
            all_results_written_at_sec=123.0,
            post_completion_wait_sec=45.5,
        )

    monkeypatch.setattr(aider_benchmark, "_maybe_start_request_monitor", fake_request_monitor)
    monkeypatch.setattr(aider_benchmark, "_run_filtered", fake_run_filtered)
    monkeypatch.setattr(
        aider_benchmark,
        "summarize_run_dir",
        lambda run_dir, wall_time_sec=None: {
            "run_dir": str(run_dir),
            "completed_tests": 1,
            "total_tests": 1,
            "pass_rate_1": 100.0,
            "pass_rate_2": 100.0,
        },
    )

    result = aider_benchmark.run_aider_benchmark(
        model_alias="fake-model",
        backend="rocm7",
        port=8000,
        profile_name="python-quick",
        context_window=8192,
        verbose=True,
    )

    docker_cmd = captured["cmd"]
    assert docker_cmd[-2:] == ["-lc", docker_cmd[-1]]
    shell_command = docker_cmd[-1]
    assert "exec python3 ./benchmark/benchmark.py" in shell_command
    assert "--stats" not in shell_command
    assert "status=$?" not in shell_command
    assert result["all_results_written_before_exit"] is True
    assert result["all_results_written_at_sec"] == 123.0
    assert result["post_completion_wait_sec"] == 45.5
    assert result["proxy_enabled"] is True
    assert result["proxy_log_file"].endswith(".proxy.log")
    assert result["openai_base_url"] == "http://host.docker.internal:8001/v1"


class _FakeCfg:
    def __init__(self, parallel_slots: int, ctx_per_slot: int, ctx_size: int | None = None) -> None:
        self.parallel_slots = parallel_slots
        self.ctx_per_slot = ctx_per_slot
        self.ctx_size = ctx_size if ctx_size is not None else parallel_slots * ctx_per_slot


def test_server_resolve_aider_threads_and_context_defaults(monkeypatch) -> None:
    server = _import_server_for_tests(monkeypatch)

    cfg = _FakeCfg(parallel_slots=5, ctx_per_slot=4096)

    assert server._resolve_aider_threads(cfg, None) == (3, "models.py parallel_slots capped at 3 for eval")
    assert server._resolve_aider_threads(cfg, 2) == (2, "--threads")
    assert server._resolve_aider_context_window(cfg, 3) == 12288
    assert server._resolve_aider_context_window(cfg, 2) == 8192


def test_aider_bench_single_uses_effective_threads_for_server_and_metadata(monkeypatch) -> None:
    server = _import_server_for_tests(monkeypatch)

    calls: dict[str, object] = {}

    cfg = _FakeCfg(parallel_slots=4, ctx_per_slot=2048)
    cfg.is_downloaded = True
    cfg.name = "Fake Model"
    cfg.alias = "fake-model"
    cfg.quant = "Q8_0"

    monkeypatch.setattr(server, "get_model", lambda alias: cfg)
    monkeypatch.setattr(server, "stop_server", lambda: None)
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(server, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "ok", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "warn", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "fail", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_api_key_for_model", lambda _alias: "local")

    def fake_launch(model_cfg, **kwargs):
        calls["launch"] = {"cfg": model_cfg, **kwargs}

    def fake_run(**kwargs):
        calls["run"] = kwargs
        return {"ok": True, "threads": kwargs["threads"]}

    monkeypatch.setattr(server, "launch_server", fake_launch)
    monkeypatch.setattr(server, "run_aider_benchmark", fake_run)

    result = server.aider_bench_single("fake-model", backend="radv")

    assert result["threads"] == 3
    assert calls["launch"]["parallel_override"] == 3
    assert calls["run"]["threads"] == 3
    assert calls["run"]["context_window"] == 6144
    assert calls["run"]["verbose"] is False

    calls.clear()
    result = server.aider_bench_single("fake-model", backend="radv", threads=2, verbose=True)

    assert result["threads"] == 2
    assert calls["launch"]["parallel_override"] == 2
    assert calls["run"]["threads"] == 2
    assert calls["run"]["context_window"] == 4096
    assert calls["run"]["verbose"] is True


def test_eval_viewer_build_report_writes_html(tmp_path: Path) -> None:
    input_path = tmp_path / "aider_results.jsonl"
    output_path = tmp_path / "report.html"
    input_path.write_text(json.dumps({"model": "demo", "pass_rate_2": 50.0}) + "\n")

    written = eval_viewer.build_report(input_path=input_path, output_path=output_path, open_browser=False)

    assert written == output_path.resolve()
    contents = output_path.read_text(encoding="utf-8")
    assert "Strix Halo Aider Results" in contents
    assert "demo" in contents


def test_server_refreshes_aider_html_report_after_single_run(monkeypatch) -> None:
    server = _import_server_for_tests(monkeypatch)
    calls: dict[str, object] = {}
    cfg = _FakeCfg(parallel_slots=2, ctx_per_slot=2048)
    cfg.is_downloaded = True
    cfg.name = "Fake Model"
    cfg.alias = "fake-model"
    cfg.quant = "Q8_0"

    monkeypatch.setattr(server, "get_model", lambda alias: cfg)
    monkeypatch.setattr(server, "stop_server", lambda: None)
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(server, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "ok", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "warn", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "fail", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_api_key_for_model", lambda _alias: "local")

    def fake_launch(model_cfg, **kwargs):
        calls["launch"] = {"cfg": model_cfg, **kwargs}

    def fake_run(**kwargs):
        calls["run"] = kwargs
        return {"ok": True, "threads": kwargs["threads"], "results_file": "/tmp/aider_results.jsonl"}

    monkeypatch.setattr(server, "launch_server", fake_launch)
    monkeypatch.setattr(server, "run_aider_benchmark", fake_run)
    monkeypatch.setattr(server, "_refresh_aider_html_report", lambda: Path("/tmp/eval_report.html"))

    result = server.aider_bench_single("fake-model", backend="radv")

    assert result["threads"] == 2
    assert calls["launch"]["parallel_override"] == 2
    assert calls["run"]["threads"] == 2


def test_server_refreshes_aider_html_report_once_after_all_runs(monkeypatch) -> None:
    server = _import_server_for_tests(monkeypatch)

    cfg_one = types.SimpleNamespace(alias="m1", name="Model One", is_downloaded=True, hidden=False)
    cfg_two = types.SimpleNamespace(alias="m2", name="Model Two", is_downloaded=True, hidden=False)
    monkeypatch.setattr(server, "MODELS", [cfg_one, cfg_two])

    calls: dict[str, object] = {"single": [], "refresh": 0}

    def fake_single(model_alias, **kwargs):
        calls["single"].append((model_alias, kwargs))
        return {
            "threads": kwargs.get("threads") or 1,
            "pass_rate_1": 10.0,
            "pass_rate_2": 20.0,
            "completed_tests": 1,
            "total_tests": 1,
            "seconds_per_case_wall": 12.0,
            "completion_tok_s_wall": 3.0,
        }

    def fake_refresh():
        calls["refresh"] += 1
        return Path("/tmp/eval_report.html")

    monkeypatch.setattr(server, "aider_bench_single", fake_single)
    monkeypatch.setattr(server, "_refresh_aider_html_report", fake_refresh)
    monkeypatch.setattr(server, "ok", lambda *args, **kwargs: None)

    results = server.aider_bench_all(backend="radv", verbose=True)

    assert len(results) == 2
    assert calls["refresh"] == 1
    assert all(kwargs["refresh_report"] is False for _, kwargs in calls["single"])
    assert all(kwargs["verbose"] is True for _, kwargs in calls["single"])
