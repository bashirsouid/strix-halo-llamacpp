#!/usr/bin/env python3
"""Manual inference smoke test helpers.

This module keeps a small manual CLI entry point while making the pytest suite
safe to run without a live llama.cpp server.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

import pytest


class DummyResponse:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _inference_check(port: int = 8000, timeout: int = 30) -> bool:
    """Test inference with a simple prompt against a running server."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "nemotron-nano-q4",
        "messages": [{"role": "user", "content": "Hello."}],
        "max_tokens": 16,
        "temperature": 0.1,
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_data = json.loads(resp.read())

        choices = response_data.get("choices", [])
        if not choices:
            print(f"  ✗ Unexpected response structure: {response_data}")
            return False

        message = choices[0].get("message", {})
        content = message.get("content", "")
        print()
        print("  ── Inference Test ────────────────────────────────────────")
        print(f"  ✓ Server is responding on port {port}")
        print(f"  ✓ Model: {response_data.get('model', 'unknown')}")
        print(f"  ✓ Response: {content.strip()[:100]}")
        print(f"  ✓ Tokens used: {response_data.get('usage', {}).get('total_tokens', 'unknown')}")
        print("  ──────────────────────────────────────────────────────────")
        print()
        return True
    except urllib.error.URLError as exc:
        print(f"  ✗ Server not responding: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"  ✗ Test failed: {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test server inference")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    args = parser.parse_args(argv)
    return 0 if _inference_check(port=args.port, timeout=args.timeout) else 1


class TestInferenceCheck:
    def test_inference_check_success(self, monkeypatch: pytest.MonkeyPatch):
        open_calls: list[str] = []

        def fake_urlopen(req, timeout: int = 0):
            open_calls.append(req.full_url)
            return DummyResponse(
                {
                    "model": "nemotron-nano-q4",
                    "choices": [{"message": {"content": "Hello back."}}],
                    "usage": {"total_tokens": 12},
                }
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        assert _inference_check(port=8123, timeout=5) is True
        assert open_calls == ["http://127.0.0.1:8123/v1/chat/completions"]

    def test_inference_check_connection_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("boom")),
        )

        assert _inference_check(port=8123, timeout=5) is False


@pytest.mark.integration
def test_live_inference_smoke():
    if os.environ.get("STRIX_RUN_LIVE_INFERENCE") != "1":
        pytest.skip("set STRIX_RUN_LIVE_INFERENCE=1 to run against a live local server")
    assert _inference_check() is True


if __name__ == "__main__":
    raise SystemExit(main())
