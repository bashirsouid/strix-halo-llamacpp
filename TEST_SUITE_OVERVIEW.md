# Test Suite for Strix Halo Llama.cpp

## Overview

This test suite validates `server.py` functionality while protecting the main conversational model server. It provides two modes:

1. **pytest tests** - Unit tests with mocked dependencies (fast, isolated)
2. **Dry-run mode** - Real configuration tests without starting/stopping servers

## Why Two Modes?

When testing with the main conversational model running:
- **Dry-run mode** validates configuration without disrupting the main server
- pytest tests can be run independently when no server is active

## Dry-Run Mode Usage

```bash
# Basic validation (uses nemotron-nano-q4)
python server.py test --sequential --dry-run

# Test with parallel slots
python server.py test --np 4 --sequential

# Test different backend
python server.py test --backend rocm --sequential

# Test specific model
python server.py test --model nemotron-nano-q4 --sequential --dry-run

# Full help
python server.py test --help
```

### Dry-Run Features

- ✅ Validates model exists and is downloaded
- ✅ Generates server arguments
- ✅ Checks environment variables
- ✅ Validates parallelization settings
- ✅ Checks server health endpoint (if running)
- ❌ Does NOT start/stop servers
- ❌ Does NOT download models (unless explicitly requested)
- ❌ Does NOT affect main conversational model

## Python pytest Tests

### Setup
```bash
pip install -r requirements-test.txt
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# With parallelization (recommended for 32-core system)
pytest tests/ -n auto

# Specific test files
pytest tests/test_server.py -v
pytest tests/test_parallelization.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

### Test Categories

#### `test_server.py` - Core Server Tests
- **Model Schema Tests**: Validate all models conform to schema
- **Model Lookup Tests**: Test `get_model()` function
- **Server Configuration Tests**: Verify environment setup
- **Model Server Arguments**: Test argument generation
- **Server Management Tests**: Start/stop server (mocked)
- **Benchmark Tests**: Single and concurrent benchmarks (mocked)
- **Parallelization Tests**: Multi-slot scenarios
- **Performance Tests**: Fast path benchmarks

#### `test_parallelization.py` - Parallelization Tests
- **Model Config Tests**: Test parallel slot counts
- **Server Launch Tests**: Test different np values
- **Multi-Process Tests**: Thread/process pool usage
- **GPU Parallelization**: Memory and multi-GPU scenarios

## Creating Tests

### For Dry-Run Mode
Tests are integrated into `server.py` via the `run_test_suite()` function.

### For pytest
Add to `tests/test_server.py` or `tests/test_parallelization.py`:

```python
def test_example_feature():
    """Brief description of what's being tested."""
    # Test implementation
    pass
```

## Test Configuration

### pytest.ini
```ini
[pytest]
testpaths = tests
addopts = -v --tb=short --strict-markers
markers =
    parallel: Parallelization tests
    performance: Performance benchmarks
```

### pytest-xdist for Parallel Execution
```bash
# Auto-detect number of cores
pytest tests/ -n auto

# Specify number of workers
pytest tests/ -n 32

# Group by test class
pytest tests/ --dist loadscope
```

## Integration with CI/CD

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install pytest pytest-xdist
      - run: pytest tests/ -n auto
```

## Performance Targets

| Operation | Target |
|-----------|--------|
| Model lookup (100x) | < 100ms |
| Args generation (100x) | < 500ms |
| Full test suite | < 10 min (parallel) |
| Single test | < 30s |

## Troubleshooting

### Dry-Run Mode
- If server not running, health check will warn (expected)
- Model not downloaded: use `python server.py download MODEL` first
- Dry-run mode skips downloads automatically

### pytest Tests
- Install dependencies: `pip install -r requirements-test.txt`
- Tests require models to be downloaded
- Some tests use mocked dependencies (network, subprocess)
- Slow tests marked with `@pytest.mark.slow`

## Contributing Tests

1. Add tests for new features in appropriate file
2. Use existing models for real-world testing
3. Mark slow tests with `@pytest.mark.slow`
4. Ensure tests run in < 30 seconds each
5. Test both dry-run and pytest modes

## Files

- `tests/__init__.py` - Test package init
- `tests/test_server.py` - Core server tests (67 tests)
- `tests/test_parallelization.py` - Parallelization tests (14 tests)
- `pytest.ini` - pytest configuration
- `requirements-test.txt` - Test dependencies
- `TESTING.md` - This documentation
- `test.sh` - Test runner script

## Summary

This comprehensive test suite ensures the server configuration is correct while protecting the main conversational model. Use dry-run mode during active chat sessions, and pytest tests when the server is not running.
