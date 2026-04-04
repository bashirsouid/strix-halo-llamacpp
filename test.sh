#!/usr/bin/env bash
# Run the test suite for strix-llamacpp

set -e

echo "🔧 Installing test dependencies..."
pip install -q pytest pytest-xdist

echo "🚀 Running test suite..."
pytest tests/ -v --tb=short "$@"

echo "✅ Tests completed!"
echo ""
echo "For dry-run tests (won't disrupt main model):"
echo "  python server.py test --sequential --dry-run"
echo ""
echo "For inference test on running server:"
echo "  python tests/test_inference.py --port 8000"
