from __future__ import annotations

import json
from pathlib import Path

import aider_benchmark


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
    assert aider_benchmark._should_echo_aider_line("  exhausted_context_windows: 2")
    assert aider_benchmark._should_echo_aider_line("Warning: context window exhausted")
