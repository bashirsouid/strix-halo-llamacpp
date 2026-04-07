#!/usr/bin/env bash
# Run the automated test suite for strix-halo-llamacpp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing test dependencies..."
pip install -q -r requirements-test.txt

echo "Running pytest..."
pytest -q "$@"

echo ""
echo "Automated tests completed."
echo "Dry-run launcher validation:"
echo "  python server.py test --dry-run --sequential"
echo ""
echo "Optional live inference smoke test (requires a running local server):"
echo "  STRIX_RUN_LIVE_INFERENCE=1 pytest tests/test_inference.py -m integration -v"
