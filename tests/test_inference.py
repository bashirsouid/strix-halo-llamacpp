#!/usr/bin/env python3
"""Simple inference test to validate server is responding.

This test sends a real inference request to the running server
to verify it's operational. It uses a tiny prompt and short output.

Run when server is already running (e.g., main conversational model).
"""

import json
import urllib.request
import urllib.error
import sys

def test_inference(port: int = 8000, timeout: int = 30) -> bool:
    """Test inference with a simple prompt.
    
    Args:
        port: Server port (default 8000)
        timeout: Request timeout in seconds (default 30)
    
    Returns:
        True if inference succeeded, False otherwise
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    
    payload = {
        "model": "nemotron-nano-q4",  # Use the already-running model
        "messages": [
            {"role": "user", "content": "Hello."}
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
            
            # Check if response has expected structure
            if "choices" in response_data and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message", {})
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
            else:
                print(f"  ✗ Unexpected response structure: {response_data}")
                return False
                
    except urllib.error.URLError as e:
        print(f"  ✗ Server not responding: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def main():
    """Run inference test."""
    import argparse
    parser = argparse.ArgumentParser(description="Test server inference")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    args = parser.parse_args()
    
    success = test_inference(port=args.port, timeout=args.timeout)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


def test_inference_pytest():
    """Pytest wrapper for inference test."""
    assert test_inference() is True
