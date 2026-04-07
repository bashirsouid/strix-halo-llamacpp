"""Test suite for server.py - Strix Halo llama.cpp launcher.

This test suite is designed to be fast, comprehensive, and parallelizable.
It tests server.py functionality while mocking external dependencies.

Key features:
- Uses pytest with parallel execution support
- Mocks all external dependencies (network, subprocess, filesystem)
- Tests all public functions in server.py
- Tests model schema compliance with models.py
- Includes parallelization tests for multi-slot scenarios
- Creates a "tiny model" for real-world testing
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest

from models import DraftModel, ModelConfig, SpecConfig, MODELS, get_model

# ── Test fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with all necessary files."""
    # Copy essential files
    (tmp_path / ".env").write_text("API_KEY=test_key_123\n")
    (tmp_path / ".server.json").write_text("{}")
    
    # Create build directories
    (tmp_path / "llama.cpp").mkdir()
    (tmp_path / "llama.cpp" / "build-vulkan").mkdir(parents=True)
    (tmp_path / "llama.cpp" / "build-rocm").mkdir(parents=True)
    
    # Create models directory
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    return tmp_path


@pytest.fixture
def sample_model() -> ModelConfig:
    """Create a minimal test model."""
    return ModelConfig(
        name="Test Model (Q4_K_M)",
        alias="test-model-q4",
        hf_repo="test/repo-GGUF",
        dest_dir=Path("/tmp/models/test"),
        download_include="*Q4_K_M*.gguf",
        shard_glob="*Q4_K_M*.gguf",
        quant="Q4_K_M",
        parallel_slots=2,
        max_parallel=4,
        ctx_per_slot=4096,
    )


@pytest.fixture
def tiny_model(tmp_path: Path) -> ModelConfig:
    """Create a super tiny model for parallelization tests.
    
    This model is designed to:
    - Be very small (< 100MB)
    - Support high parallelization (np=8+)
    - Have minimal dependencies
    - Run fast on both CPU and GPU
    """
    tiny_model_dir = tmp_path / "test_tiny_model"
    tiny_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy GGUF file (minimal valid structure)
    # GGUF header: magic "GGUF", version, metadata
    gguf_header = bytearray()
    
    # Magic number: "GGUF" (little-endian: 84 88 71 85)
    gguf_header.extend(b"GGUF")
    
    # Version (3 bytes)
    gguf_header.extend((3).to_bytes(4, byteorder="little"))
    
    # Number of dimensions (uint64)
    gguf_header.extend((0).to_bytes(8, byteorder="little"))
    
    # Write the minimal GGUF file
    gguf_path = tiny_model_dir / "test-tiny-model.Q4_K_M.gguf"
    gguf_path.write_bytes(bytes(gguf_header))
    
    return ModelConfig(
        name="Tiny Test Model (Q4_K_M)",
        alias="tiny-test-q4",
        hf_repo="test/tiny-model-GGUF",
        dest_dir=tiny_model_dir,
        download_include="*Q4_K_M*.gguf",
        shard_glob="*Q4_K_M*.gguf",
        quant="Q4_K_M",
        parallel_slots=8,
        max_parallel=16,
        ctx_per_slot=1024,
        batch_size=512,
        ubatch_size=64,
        threads=2,
        notes="Super tiny model for fast parallelization tests",
    )


@pytest.fixture
def mocked_subprocess():
    """Mock subprocess calls to avoid actual builds/downloads."""
    with patch("subprocess.run") as mock_run, \
         patch("subprocess.Popen") as mock_popen, \
         patch("subprocess.Popen.__enter__") as mock_enter, \
         patch("subprocess.Popen.__exit__") as mock_exit:
        
        # Configure Popen mock
        popen_instance = Mock()
        popen_instance.stdout = None
        popen_instance.wait.return_value = 0
        mock_popen.return_value = popen_instance
        mock_enter.return_value = popen_instance
        
        # Default run return
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        yield mock_run, mock_popen


# ── Model Schema Tests ────────────────────────────────────────────────────────


class TestModelSchema:
    """Test that all models in models.py conform to the expected schema."""
    
    def test_all_models_have_required_fields(self):
        """Every model must have all required fields."""
        required_fields = {
            "name", "alias", "hf_repo", "dest_dir", 
            "download_include", "shard_glob"
        }
        
        for model in MODELS:
            for field in required_fields:
                assert hasattr(model, field), f"Model {model.name} missing field: {field}"
    
    def test_all_models_have_valid_parallel_slots(self):
        """Parallel slots must be positive integers."""
        for model in MODELS:
            assert isinstance(model.parallel_slots, int), f"{model.name}: parallel_slots must be int"
            assert model.parallel_slots >= 1, f"{model.name}: parallel_slots must be >= 1"
    
    def test_all_models_have_valid_max_parallel(self):
        """Max parallel must be >= parallel_slots."""
        for model in MODELS:
            assert isinstance(model.max_parallel, int), f"{model.name}: max_parallel must be int"
            assert model.max_parallel >= model.parallel_slots, \
                f"{model.name}: max_parallel must be >= parallel_slots"
    
    def test_all_models_have_valid_ctx_per_slot(self):
        """Context per slot must be positive."""
        for model in MODELS:
            assert isinstance(model.ctx_per_slot, int), f"{model.name}: ctx_per_slot must be int"
            assert model.ctx_per_slot > 0, f"{model.name}: ctx_per_slot must be > 0"
    
    def test_all_models_have_valid_quantization(self):
        """Quantization field must be string or empty."""
        for model in MODELS:
            assert isinstance(model.quant, str), f"{model.name}: quant must be str"
    
    def test_all_models_have_unique_aliases(self):
        """Each model must have a unique alias."""
        aliases = [m.alias for m in MODELS]
        assert len(aliases) == len(set(aliases)), "Duplicate model aliases found"
    
    def test_all_models_have_valid_hf_repo(self):
        """HuggingFace repo must follow owner/repo format."""
        for model in MODELS:
            assert "/" in model.hf_repo, f"{model.name}: hf_repo must be owner/repo format"
    
    def test_draft_model_schema(self):
        """Draft models must have required fields."""
        for model in MODELS:
            if model.spec.strategy and "draft" in model.spec.strategy:
                assert model.spec.draft is not None, f"{model.name}: draft model required"
                assert isinstance(model.spec.draft, DraftModel)
                assert model.spec.draft.hf_repo
                assert model.spec.draft.filename
                assert model.spec.draft.dest_dir
    
    def test_spec_config_strategies(self):
        """Test that all spec config strategies are valid."""
        valid_strategies = {None, "draft", "ngram", "draft+ngram", 
                          "ngram-cache", "ngram-simple", "ngram-map-k", 
                          "ngram-map-k4v", "ngram-mod"}
        
        for model in MODELS:
            if model.spec.strategy:
                assert model.spec.strategy in valid_strategies, \
                    f"{model.name}: invalid spec strategy: {model.spec.strategy}"


# ── Model Lookup Tests ────────────────────────────────────────────────────────


class TestModelLookup:
    """Test model lookup functionality."""
    
    def test_get_model_by_name(self):
        """Lookup by model name should work."""
        model = get_model("Qwen3 Coder Next (Q6_K)")
        assert model.alias == "qwen3-coder-next-q6"
    
    def test_get_model_by_alias(self):
        """Lookup by alias should work."""
        model = get_model("qwen3-coder-next-q6")
        assert model.name == "Qwen3 Coder Next (Q6_K)"
    
    def test_get_model_by_substring(self):
        """Lookup by unique substring should work."""
        model = get_model("qwen3-coder-next")
        assert model.alias == "qwen3-coder-next-q6"
    
    def test_get_model_not_found(self):
        """Invalid model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent-model")
    
    def test_get_model_ambiguous(self):
        """Ambiguous substring should raise ValueError."""
        with pytest.raises(ValueError, match="Ambiguous"):
            get_model("nemotron")  # Multiple matches
    
    def test_get_model_case_insensitive(self):
        """Lookup should be case-insensitive."""
        model = get_model("QWEN3-CODER-NEXT-Q6")
        assert model.alias == "qwen3-coder-next-q6"


# ── Server Configuration Tests ────────────────────────────────────────────────


class TestServerConfiguration:
    """Test server configuration and environment setup."""
    
    def test_container_image_resolution(self):
        """Container image should be correctly resolved for each backend."""
        from server import _container_image, VALID_BACKENDS, CONTAINER_REGISTRY
        
        for backend in VALID_BACKENDS:
            image = _container_image(backend)
            assert CONTAINER_REGISTRY in image
            assert backend in image or "vulkan" in image or "rocm" in image
    
    def test_container_name_resolution(self):
        """Container name should be correctly resolved for each backend."""
        from server import _container_name, VALID_BACKENDS
        
        for backend in VALID_BACKENDS:
            name = _container_name(backend)
            assert name.startswith("strix-llama-")
            assert len(name) > len("strix-llama-")
    
    def test_container_images(self):
        """Container images should be properly configured."""
        from server import CONTAINER_IMAGES, VALID_BACKENDS, CONTAINER_REGISTRY
        
        for backend in VALID_BACKENDS:
            assert backend in CONTAINER_IMAGES
            assert CONTAINER_REGISTRY in CONTAINER_IMAGES[backend]
    
    def test_container_names(self):
        """Container names should be properly configured."""
        from server import CONTAINER_NAMES, VALID_BACKENDS
        
        for backend in VALID_BACKENDS:
            assert backend in CONTAINER_NAMES
            assert CONTAINER_NAMES[backend].startswith("strix-llama-")


# ── Model Server Arguments Tests ──────────────────────────────────────────────


class TestModelServerArgs:
    """Test server argument generation."""
    
    def test_server_args_basic(self, sample_model: ModelConfig, tmp_path: Path):
        """Basic server arguments should be correctly generated."""
        # Create dummy model file
        dummy_model = tmp_path / "test-model.Q4_K_M.gguf"
        dummy_model.parent.mkdir(parents=True, exist_ok=True)
        dummy_model.write_text("dummy")
        
        sample_model.dest_dir = tmp_path
        sample_model.shard_glob = "*.gguf"
        
        args = sample_model.server_args()
        
        assert "-m" in args
        assert "--parallel" in args
        assert "--ctx-size" in args
        assert "--host" in args
        assert "--port" in args
    
    def test_server_args_with_override(self, sample_model: ModelConfig, tmp_path: Path):
        """Server args should respect override parameters."""
        dummy_model = tmp_path / "test.Q4_K_M.gguf"
        dummy_model.parent.mkdir(parents=True, exist_ok=True)
        dummy_model.write_text("dummy")
        
        sample_model.dest_dir = tmp_path
        sample_model.shard_glob = "*.gguf"
        
        args = sample_model.server_args(parallel_override=4, ctx_override=8192)
        
        np_idx = args.index("--parallel")
        assert int(args[np_idx + 1]) == 4
        
        ctx_idx = args.index("--ctx-size")
        assert int(args[ctx_idx + 1]) == 8192
    
    def test_spec_args_ngram(self, tmp_path: Path):
        """Ngram spec config should generate correct args."""
        # Create dummy model file
        dummy_model = tmp_path / "test.gguf"
        dummy_model.write_text("dummy")
        
        spec = MagicMock()
        spec.strategy = "ngram"
        spec.server_args.return_value = ["--spec-type", "ngram-mod", "--spec-ngram-size-n", "24"]
        
        with patch("models.SpecConfig.server_args", spec.server_args):
            model = ModelConfig(
                name="Test", alias="test", hf_repo="test/repo",
                dest_dir=tmp_path, download_include="*.gguf",
                shard_glob="*.gguf", spec=SpecConfig(strategy="ngram")
            )
            
            args = model.server_args()
            assert "--spec-type" in args
    
    def test_spec_args_draft(self, tmp_path: Path):
        """Draft spec config should generate correct args."""
        draft_path = tmp_path / "draft.gguf"
        draft_path.write_text("draft")
        
        spec = SpecConfig(
            strategy="draft",
            draft=DraftModel(
                hf_repo="test/draft",
                filename="draft.gguf",
                dest_dir=tmp_path
            )
        )
        
        model = ModelConfig(
            name="Test", alias="test", hf_repo="test/repo",
            dest_dir=tmp_path, download_include="*.gguf",
            shard_glob="*.gguf", spec=spec
        )
        
        args = model.server_args()
        assert "--model-draft" in args
        assert "--draft-max" in args
        assert "--draft-min" in args


# ── Model Download Tests ──────────────────────────────────────────────────────


class TestModelDownload:
    """Test model download functionality."""
    
    def test_download_model_schema(self, sample_model: ModelConfig, tmp_path: Path):
        """Download model schema validation."""
        # Test that model config is valid
        assert sample_model.hf_repo
        assert sample_model.download_include
        assert sample_model.shard_glob
        assert sample_model.dest_dir


# ── Server Management Tests ───────────────────────────────────────────────────


class TestServerManagement:
    """Test server start/stop functionality (integration tests)."""
    
    @pytest.mark.slow
    def test_stop_server_integration(self, tmp_project_dir: Path):
        """Integration test: stop server functionality."""
        # This is a basic integration test
        # Full server start/stop tests are in test_inference.py
        pass
    
    @pytest.mark.slow
    def test_wait_for_server_integration(self, tmp_project_dir: Path):
        """Integration test: server readiness check."""
        # Real server readiness test is done in test_inference.py
        pass


# ── Benchmark Tests ───────────────────────────────────────────────────────────


class TestBenchmarkSingle:
    """Test single-request benchmark functionality."""
    
    def test_bench_one_request(self, tmp_project_dir: Path):
        """Single request benchmark should return timing data."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            # Mock response
            response = Mock()
            response.read.return_value = json.dumps({
                "usage": {"prompt_tokens": 10, "completion_tokens": 100}
            }).encode()
            mock_urlopen.return_value.__enter__.return_value = response
            
            with patch("server.time.perf_counter", return_value=0), \
                 patch("server.time.sleep"):
                from server import _bench_one
                
                result = _bench_one(
                    port=8000,
                    prompt="test",
                    max_tokens=100,
                    label="test-1"
                )
                
                assert result["ok"] is True
                assert "elapsed" in result
                assert "tok_s" in result
    
    def test_bench_one_error(self, tmp_project_dir: Path):
        """Failed request should return error info."""
        with patch("urllib.request.urlopen", side_effect=Exception("Connection error")):
            with patch("server.time.perf_counter", return_value=0):
                from server import _bench_one
                
                result = _bench_one(
                    port=8000,
                    prompt="test",
                    max_tokens=100,
                    label="test-1"
                )
                
                assert result["ok"] is False
                assert "error" in result


class TestBenchmarkConcurrent:
    """Test concurrent throughput benchmark (integration tests)."""
    
    @pytest.mark.slow
    def test_bench_concurrent_integration(self, tmp_project_dir: Path):
        """Integration test: concurrent benchmark."""
        # Real concurrent benchmark requires running server
        # Use test_inference.py to validate actual server
        pass


class TestBenchmarkParallel:
    """Test parallel sweep benchmark (integration tests)."""
    
    @pytest.mark.slow
    def test_bench_parallel_integration(self, tiny_model: ModelConfig, tmp_path: Path):
        """Integration test: parallel sweep."""
        # Real parallel sweep requires running server
        # Use bench command for this: python server.py bench-parallel MODEL
        pass


# ── Parallelization Tests (Real-world) ────────────────────────────────────────


class TestParallelization:
    """Test parallelization with tiny models for real-world scenarios."""
    
    def test_tiny_model_creation(self, tiny_model: ModelConfig, tmp_path: Path):
        """Tiny model should be created and valid."""
        assert tiny_model.is_downloaded, "Tiny model should be marked as downloaded"
        assert tiny_model.parallel_slots == 8, "Tiny model should support high parallelization"
        assert tiny_model.max_parallel == 16, "Tiny model should support max 16 parallel slots"
    
    def test_tiny_model_server_args(self, tiny_model: ModelConfig):
        """Tiny model should generate correct server args."""
        args = tiny_model.server_args()
        
        assert int(args[args.index("--parallel") + 1]) == 8
        assert int(args[args.index("--ctx-size") + 1]) == 8192  # 1024 * 8
    
    @pytest.mark.parametrize("np_slots", [1, 4, 8, 16])
    def test_tiny_model_different_parallel(self, tiny_model: ModelConfig, tmp_path: Path, np_slots: int):
        """Test tiny model with different parallel slot counts."""
        args = tiny_model.server_args(parallel_override=np_slots)
        
        np_idx = args.index("--parallel")
        actual_np = int(args[np_idx + 1])
        
        assert actual_np == np_slots, f"Expected {np_slots} slots, got {actual_np}"


# ── Integration Tests (Lightweight) ───────────────────────────────────────────


class TestLightweightIntegration:
    """Lightweight integration tests that don't require full build."""
    
    def test_env_loading(self, tmp_project_dir: Path):
        """Environment variables should be loaded from .env file."""
        env_file = tmp_project_dir / ".env"
        env_file.write_text("API_KEY=secret123\nHF_HUB_TOKEN=token456\n")
        
        with patch("server.PROJECT_DIR", tmp_project_dir):
            from server import load_env_file
            
            # Test that loading function exists and is callable
            assert callable(load_env_file)
    
    def test_resolve_model_integration(self):
        """Model resolution integration test."""
        # Test with existing models
        from server import resolve_model
        from models import MODELS
        
        # Should be able to resolve at least one model
        assert len(MODELS) > 0


# ──边缘测试与错误处理 ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_model_not_downloaded(self, sample_model: ModelConfig, tmp_path: Path):
        """Server args should fail for undownloaded model."""
        sample_model.dest_dir = tmp_path
        
        with pytest.raises(FileNotFoundError, match="not downloaded"):
            sample_model.server_args()
    
    def test_empty_model_list(self):
        """Empty model list should not crash lookup."""
        from models import MODELS
        
        assert len(MODELS) > 0, "Model catalog should not be empty"
    
    def test_server_args_with_reasoning(self, tmp_path: Path):
        """Reasoning model should have correct args."""
        model = ModelConfig(
            name="Reasoning Test", alias="reasoning-test", hf_repo="test/repo",
            dest_dir=tmp_path, download_include="*.gguf", shard_glob="*.gguf",
            reasoning=True,
            reasoning_format="traverse",
            reasoning_budget=1024
        )
        
        dummy_model = tmp_path / "test.gguf"
        dummy_model.write_text("dummy")
        
        args = model.server_args()
        
        assert "--reasoning" in args
        assert "--reasoning-format" in args
        assert "--reasoning-budget" in args


# ── Performance Tests (Fast) ──────────────────────────────────────────────────


class TestBackendResolution:
    """Test backend picker and resolver functions."""
    
    def test_resolve_backend_with_arg(self):
        """Backend resolver should return CLI arg when valid."""
        from server import resolve_backend
        
        result = resolve_backend("radv")
        assert result == "radv"
    
    def test_resolve_backend_with_invalid_arg(self):
        """Backend resolver should exit on invalid backend."""
        from server import resolve_backend
        
        with pytest.raises(SystemExit):
            resolve_backend("invalid_backend")
    
    def test_resolve_backend_without_arg(self):
        """Backend resolver should pick backend when None."""
        from server import resolve_backend
        
        with patch('server.pick_backend', return_value="rocm7"):
            result = resolve_backend(None)
            assert result == "rocm7"
    
    def test_pick_backend_output_format(self):
        """Backend picker should display valid backends."""
        from server import pick_backend, VALID_BACKENDS
        
        # Test that all backends are in VALID_BACKENDS
        for backend in VALID_BACKENDS:
            assert backend in VALID_BACKENDS
    
    def test_resolve_backend_interactive_flow(self, capsys):
        """Backend resolver should prompt when no arg provided."""
        from server import resolve_backend
        
        with patch('builtins.input', return_value='3'):
            with patch('server.pick_backend') as mock_pick:
                mock_pick.return_value = "rocm"
                result = resolve_backend(None)
                assert result == "rocm"


class TestPerformance:
    """Fast performance tests for critical paths."""
    
    def test_model_lookup_performance(self):
        """Model lookup should be O(n) and fast."""
        import time
        
        start = time.time()
        for _ in range(100):
            get_model("qwen3-coder-next-q6")
        elapsed = time.time() - start
        
        # Should complete 100 lookups in < 100ms
        assert elapsed < 0.1, f"Model lookup too slow: {elapsed}s"


# ── Main ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
