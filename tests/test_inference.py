#!/usr/bin/env python3
"""Simple live inference smoke test.

When run under pytest, this test is skipped unless STRIX_RUN_LIVE_INFERENCE=1.
When a model alias is not supplied via STRIX_TEST_MODEL, the test queries
/v1/models and uses the first served model alias it finds.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest


def _detect_model(port: int, timeout: int = 5) -> str:
    requested = os.environ.get("STRIX_TEST_MODEL", "auto").strip()
    if requested and requested.lower() != "auto":
        return requested

    url = f"http://127.0.0.1:{port}/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    models = payload.get("data", [])
    if not models:
        raise RuntimeError("No served models returned from /v1/models")
    return models[0].get("id") or "unknown"


def _inference_check(port: int = 8000, timeout: int = 30) -> bool:
    """Test inference with a simple prompt."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    model = _detect_model(port)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello."},
        ],
        "max_tokens": 16,
        "temperature": 0.1,
    }

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_data = json.loads(resp.read())

        if "choices" in response_data and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message", {})
            content = message.get("content", "")
            print()
            print(" ── Inference Test ────────────────────────────────────────")
            print(f" ✓ Server is responding on port {port}")
            print(f" ✓ Model: {response_data.get('model', model)}")
            print(f" ✓ Response: {content.strip()[:100]}")
            print(f" ✓ Tokens used: {response_data.get('usage', {}).get('total_tokens', 'unknown')}")
            print(" ──────────────────────────────────────────────────────────")
            print()
            return True

        print(f" ✗ Unexpected response structure: {response_data}")
        return False
    except urllib.error.URLError as e:
        print(f" ✗ Server not responding: {e}")
        return False
    except Exception as e:
        print(f" ✗ Test failed: {e}")
        return False


def main() -> None:
    """Run inference test from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Test server inference")
    parser.add_argument("--port", type=int, default=int(os.environ.get("STRIX_TEST_PORT", "8000")), help="Server port")
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("STRIX_TEST_TIMEOUT", "30")), help="Request timeout")
    args = parser.parse_args()

    result = _inference_check(port=args.port, timeout=args.timeout)
    assert result is True, "Inference test failed"


if __name__ == "__main__":
    main()


def test_inference() -> None:
    """Pytest wrapper for the live inference smoke test."""
    if os.environ.get("STRIX_RUN_LIVE_INFERENCE") != "1":
        pytest.skip("Live inference smoke test only runs via ./test.sh or STRIX_RUN_LIVE_INFERENCE=1")

    port = int(os.environ.get("STRIX_TEST_PORT", "8000"))
    timeout = int(os.environ.get("STRIX_TEST_TIMEOUT", "30"))
    result = _inference_check(port=port, timeout=timeout)
    assert result is True, "Inference test failed"
