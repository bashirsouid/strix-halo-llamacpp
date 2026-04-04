# Strix Halo Llama.cpp Test Suite

Comprehensive test suite for `server.py` and `models.py`.

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run with parallelization
pytest tests/ -n auto

# Run specific test file
pytest tests/test_server.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## Test Organization

### `test_server.py` - Core Server Tests
- **Model Schema Tests**: Validate all models in `models.py`
- **Model Lookup Tests**: Test `get_model()` function
- **Server Configuration Tests**: Verify environment setup
- **Model Server Arguments Tests**: Test argument generation
- **Model Download Tests**: Test download functionality
- **Server Management Tests**: Start/stop server
- **Benchmark Tests**: Single and concurrent benchmarks
- **Parallelization Tests**: Multi-slot scenarios
- **Performance Tests**: Fast path benchmarks

### `test_parallelization.py` - Parallelization Tests
- Tests using a "super tiny model" for real-world scenarios
- Tests for multi-core parallelization
- GPU parallelization scenarios
- Thread and process pool usage

## Running Tests

### Basic Usage
```bash
pytest tests/ -v
```

### Parallel Execution (Recommended)
With 32 CPU cores, use pytest-xdist:
```bash
pytest tests/ -n auto
```

Or specify number of workers:
```bash
pytest tests/ -n 32
```

### By Test Category
```bash
# Fast tests only (no downloads/builds)
pytest tests/ -m "not slow and not integration"

# Parallelization tests
pytest tests/test_parallelization.py -n auto

# Performance tests
pytest tests/ -m performance
```

### With Coverage
```bash
pytest tests/ --cov=. --cov-report=term-missing --cov-report=html
```

## Test Fixtures

### `tmp_project_dir`
Creates a temporary project directory with necessary files and build directories.

### `sample_model`
Returns a minimal test `ModelConfig` instance.

### `tiny_model`
Returns a "super tiny" model for fast parallelization tests:
- ~100KB GGUF file (minimal)
- Supports 8-16 parallel slots
- Low resource requirements
- Runs on CPU or GPU

### `mocked_subprocess`
Mocks subprocess calls to avoid actual builds/downloads.

## Writing New Tests

### Add to `test_server.py`
```python
class TestNewFeature:
    """Test new feature X."""
    
    def test_basic_case(self):
        """Basic functionality."""
        # Test implementation
        pass
    
    @pytest.mark.parametrize("input,expected", [...])
    def test_parametrized(self, input, expected):
        """Test with multiple inputs."""
        pass
```

### Add to `test_parallelization.py`
```python
class TestParallelScenario:
    """Test parallel scenarios with tiny model."""
    
    def test_high_parallelization(self, tmp_path: Path):
        """Test with high parallel slots."""
        model = self._create_tiny_model(tmp_path)
        # Test with np=16
        pass
```

## Performance Targets

- **Model lookup**: < 100ms for 100 lookups
- **Args generation**: < 500ms for 100 generations
- **Full test suite**: < 10 minutes with parallelization
- **Single test**: < 30 seconds (configurable)

## Continuous Integration

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

## Contributing Tests

1. Add tests for new features in appropriate file
2. Use `tiny_model` fixture for parallelization tests
3. Mark slow tests with `@pytest.mark.slow`
4. Ensure tests run in < 30 seconds each
5. Run `pytest tests/ -n auto` before committing
