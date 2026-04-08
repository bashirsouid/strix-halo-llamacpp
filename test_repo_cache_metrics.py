from __future__ import annotations

import repo_cache


def test_extract_completion_metrics_prefers_usage_and_timings() -> None:
    payload = {
        "usage": {"prompt_tokens": 1500, "completion_tokens": 320},
        "timings": {
            "cache_n": 1200,
            "prompt_n": 300,
            "prompt_ms": 50.0,
            "prompt_per_second": 6000.0,
            "predicted_n": 320,
            "predicted_ms": 4.0,
            "predicted_per_second": 80.0,
        },
    }

    metrics = repo_cache.extract_completion_metrics(payload)
    assert metrics["prompt_tokens"] == 1500
    assert metrics["completion_tokens"] == 320
    assert metrics["cache_tokens"] == 1200
    assert metrics["prompt_eval_tokens"] == 300
    assert metrics["prompt_tps"] == 6000.0
    assert metrics["completion_tps"] == 80.0


def test_format_proxy_metrics_line_includes_cache_and_token_rates() -> None:
    line = repo_cache.format_proxy_metrics_line(
        path="/v1/chat/completions",
        status=200,
        elapsed_sec=5.0,
        request_payload={"model": "qwen3-coder-next-q6", "id_slot": 0},
        response_payload={
            "timings": {
                "cache_n": 1200,
                "prompt_n": 300,
                "prompt_per_second": 6000.0,
                "predicted_n": 320,
                "predicted_per_second": 80.0,
            }
        },
    )

    assert "status=200" in line
    assert "model=qwen3-coder-next-q6" in line
    assert "slot=0" in line
    assert "ctx=1500" in line
    assert "cache=1200/1500(80.0%)" in line
    assert "pp=6000tok/s" in line
    assert "gen=320" in line
    assert "gen_tps=80.0" in line
