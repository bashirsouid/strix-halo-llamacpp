from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import pytest

import server
from eval_profiles import HUMANEVAL_QUICK_V1_TASK_IDS, ensure_override_dataset, resolve_eval_profile


def _fake_humaneval_row(task_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "entry_point": "solve",
        "prompt": "def solve(x):\n    '''demo'''\n",
        "canonical_solution": "    return x\n",
        "base_input": [],
        "plus_input": [],
        "atol": 0,
    }


class TestEvalProfiles:
    def test_quick_profile_uses_curated_subset(self):
        profile = resolve_eval_profile("quick", "humaneval")
        assert profile.name == "quick-v1"
        assert profile.task_count == 48
        assert profile.task_ids == HUMANEVAL_QUICK_V1_TASK_IDS

    def test_quick_profile_rejects_mbpp(self):
        with pytest.raises(ValueError):
            resolve_eval_profile("quick", "mbpp")

    def test_ensure_override_dataset_writes_selected_rows(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_evalplus = types.ModuleType("evalplus")
        fake_data = types.ModuleType("evalplus.data")
        dataset = {
            task_id: _fake_humaneval_row(task_id)
            for task_id in HUMANEVAL_QUICK_V1_TASK_IDS
        }

        def fake_get_human_eval_plus(version: str = "default") -> dict[str, dict[str, object]]:
            assert version == "default"
            return dataset

        fake_data.get_human_eval_plus = fake_get_human_eval_plus  # type: ignore[attr-defined]
        fake_evalplus.data = fake_data  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "evalplus", fake_evalplus)
        monkeypatch.setitem(sys.modules, "evalplus.data", fake_data)

        profile = resolve_eval_profile("quick", "humaneval")
        output_path = ensure_override_dataset(profile, tmp_path)

        assert output_path is not None
        assert output_path.exists()

        with gzip.open(output_path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line)["task_id"] for line in handle if line.strip()]

        assert tuple(rows) == HUMANEVAL_QUICK_V1_TASK_IDS


class TestEvalReanalysis:
    def test_reanalyze_eval_results_rebuilds_summary_from_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        project_dir = tmp_path
        results_file = project_dir / "results" / "eval" / "eval_results.jsonl"
        raw_dir = project_dir / "results" / "eval" / "raw"

        monkeypatch.setattr(server, "PROJECT_DIR", project_dir)
        monkeypatch.setattr(server, "EVAL_RESULTS_FILE", results_file)
        monkeypatch.setattr(server, "EVAL_RAW_DIR", raw_dir)

        server._ensure_results_dirs()

        run_id = "2026-04-07_12-00-00--demo-model--humaneval--quick-v1--abc12345"
        result_path = server._eval_runs_dir() / run_id / "humaneval" / "strix-demo_openai_temp_0.0_eval_results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "pass_at_k": {
                        "base": {"pass@1": 0.5},
                        "plus": {"pass@1": 0.375},
                    },
                    "eval": {
                        "HumanEval/0": [],
                        "HumanEval/1": [],
                    },
                }
            ),
            encoding="utf-8",
        )

        metadata = {
            "run_id": run_id,
            "timestamp": "2026-04-07 12:00:00",
            "backend": "rocm",
            "model": "demo-model",
            "quant": "Q6_K",
            "suite": "humaneval",
            "eval_tool": "evalplus",
            "eval_profile": "quick-v1",
            "eval_profile_requested": "quick",
            "task_count": 48,
            "run_label": "reasoning-on",
            "config_fingerprint": "abc12345ef",
            "ok": True,
            "wall_time_sec": 123.4,
            "raw_log": "results/eval/raw/demo.log",
            "evalplus_root": server._project_relpath(result_path.parents[1]),
            "evalplus_result": server._project_relpath(result_path),
            "override_dataset": "results/eval/profiles/humaneval-quick-v1.jsonl.gz",
            "pass_at_1_base": None,
            "pass_at_1_plus": None,
        }
        metadata_path = server._eval_metadata_dir() / f"{run_id}.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        records = server.reanalyze_eval_results(model_alias="demo-model")

        assert len(records) == 1
        record = records[0]
        assert record["pass_at_1_base"] == 0.5
        assert record["pass_at_1_plus"] == 0.375
        assert record["task_count"] == 2
        assert record["eval_profile"] == "quick-v1"
        assert record["run_label"] == "reasoning-on"

        persisted = [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines()]
        assert persisted == records
