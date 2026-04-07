from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import server
from models import ModelConfig


def create_tiny_model(tmp_path: Path) -> ModelConfig:
    gguf = tmp_path / "tiny.Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF\x03\x00\x00\x00")
    return ModelConfig(
        name="Tiny Parallel Model",
        alias="tiny-parallel-model",
        hf_repo="owner/tiny-model",
        dest_dir=tmp_path,
        download_include="*.gguf",
        shard_glob="*.gguf",
        quant="Q4_K_M",
        parallel_slots=8,
        max_parallel=16,
        ctx_per_slot=1024,
        batch_size=512,
        ubatch_size=64,
        threads=2,
    )


class TestConcurrentBenchmarks:
    def test_bench_concurrent_aggregates_request_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        responses = [
            {"ok": True, "comp_tok": 20, "tok_s": 10.0},
            {"ok": True, "comp_tok": 30, "tok_s": 15.0},
            {"ok": False, "error": "boom"},
        ]

        def fake_fire(port: int, prompt: str, max_tokens: int = 256):
            return responses.pop(0)

        monkeypatch.setattr(server, "_fire_one_request", fake_fire)
        monkeypatch.setattr(server.time, "perf_counter", Mock(side_effect=[0.0, 2.0]))

        result = server.bench_concurrent(port=8123, n_concurrent=3, max_tokens=64)

        assert result == {
            "n_concurrent": 3,
            "total_tokens": 50,
            "wall_time": 2.0,
            "aggregate_tok_s": 25.0,
            "per_request_avg_tok_s": 12.5,
            "requests_ok": 2,
            "requests_failed": 1,
        }


class TestParallelSweep:
    def test_bench_parallel_logs_one_record_per_parallel_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = create_tiny_model(tmp_path)
        monkeypatch.setattr(server, "PROJECT_DIR", tmp_path)
        monkeypatch.setattr(server, "launch_server", lambda *args, **kwargs: None)
        monkeypatch.setattr(server, "stop_server", lambda: None)
        monkeypatch.setattr(server.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(server, "_fire_one_request", lambda *args, **kwargs: {"ok": True, "tok_s": 11.0})
        monkeypatch.setattr(
            server,
            "bench_concurrent",
            lambda port, n_concurrent, max_tokens: {
                "aggregate_tok_s": 20.0 * n_concurrent,
                "per_request_avg_tok_s": 10.0,
                "wall_time": 1.0,
                "total_tokens": 128 * n_concurrent,
                "requests_ok": n_concurrent,
                "requests_failed": 0,
            },
        )

        server.bench_parallel(cfg, port=8123, backend="radv", max_np=3, rounds=1)

        report_path = tmp_path / "bench_parallel_results.jsonl"
        lines = [json.loads(line) for line in report_path.read_text().splitlines()]
        assert [line["np"] for line in lines] == [1, 2, 3]
        assert all(line["backend"] == "radv" for line in lines)
        assert all(line["ctx_per_slot"] == 1024 for line in lines)

    def test_bench_parallel_requires_downloaded_model(self, undownloaded_model: ModelConfig):
        with pytest.raises(SystemExit):
            server.bench_parallel(undownloaded_model, max_np=1)
