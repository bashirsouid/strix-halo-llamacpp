#!/usr/bin/env bash
# Run the test suite for strix-llamacpp

set -e

echo "🔧 Installing test dependencies..."
pip install -q pytest pytest-xdist

echo "🚀 Running test suite..."
pytest tests/ -v --tb=short "$@"

echo "✅ Tests completed!"
