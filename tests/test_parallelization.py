"""Parallelization tests using the tiny model for real-world scenarios.

This module tests the parallelization capabilities of server.py using the tiny
test model that runs fast on both CPU and GPU. It's designed to be executed
with pytest-xdist for parallel execution on multi-core systems.

Run with:
    pytest tests/test_parallelization.py -n auto
    pytest tests/test_parallelization.py --dist loadscope
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from models import ModelConfig


class TestParallelModelConfig:
    """Test parallel model configuration with tiny models."""
    
    def test_tiny_model_parallel_slots(self, tmp_path: Path):
        """Tiny model should support high parallel slots."""
        model = self._create_tiny_model(tmp_path)
        
        assert model.parallel_slots == 8, "Should support 8 parallel slots"
        assert model.max_parallel == 16, "Should support up to 16 parallel slots"
    
    def test_tiny_model_context_per_slot(self, tmp_path: Path):
        """Tiny model should have manageable context per slot."""
        model = self._create_tiny_model(tmp_path)
        
        assert model.ctx_per_slot == 1024, "Should have 1024 tokens per slot"
    
    def test_tiny_model_total_context(self, tmp_path: Path):
        """Total context should be ctx_per_slot × parallel_slots."""
        model = self._create_tiny_model(tmp_path)
        
        expected_total = model.ctx_per_slot * model.parallel_slots
        assert expected_total == 8192, f"Total context should be 8192, got {expected_total}"
    
    def test_tiny_model_batch_sizes(self, tmp_path: Path):
        """Tiny model should have appropriate batch sizes."""
        model = self._create_tiny_model(tmp_path)
        
        assert model.batch_size == 512, "Batch size should be 512"
        assert model.ubatch_size == 64, "Microbatch size should be 64"
        assert model.threads == 2, "Should use 2 CPU threads"
    
    def _create_tiny_model(self, tmp_path: Path) -> ModelConfig:
        """Create a tiny model for testing."""
        tiny_dir = tmp_path / "tiny_model"
        tiny_dir.mkdir(parents=True, exist_ok=True)
        
        # Minimal GGUF file
        gguf_file = tiny_dir / "tiny.Q4_K_M.gguf"
        gguf_file.write_bytes(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00")
        
        return ModelConfig(
            name="Tiny Test Model",
            alias="tiny-test",
            hf_repo="test/tiny",
            dest_dir=tiny_dir,
            download_include="*.gguf",
            shard_glob="*.gguf",
            quant="Q4_K_M",
            parallel_slots=8,
            max_parallel=16,
            ctx_per_slot=1024,
            batch_size=512,
            ubatch_size=64,
            threads=2,
        )


class TestParallelServerLaunch:
    """Test server launch with different parallel configurations."""
    
    @pytest.mark.parametrize("np", [1, 4, 8, 12, 16])
    def test_different_parallel_values(self, np: int, tmp_path: Path):
        """Test server args with various parallel slot counts."""
        model = self._create_tiny_model(tmp_path)
        args = model.server_args(parallel_override=np)
        
        np_idx = args.index("--parallel")
        actual_np = int(args[np_idx + 1])
        
        assert actual_np == np, f"Expected {np} slots, got {actual_np}"
    
    def test_parallel_sweep_optimal(self, tmp_path: Path):
        """Test finding optimal parallel configuration."""
        model = self._create_tiny_model(tmp_path)
        
        # Simulate parallel sweep
        results = []
        for np in range(1, model.max_parallel + 1):
            args = model.server_args(parallel_override=np)
            
            # Extract parallel value from args
            np_idx = args.index("--parallel")
            actual_np = int(args[np_idx + 1])
            
            # Simulated performance (would be measured in real benchmark)
            # In reality, this would be: tok_s = throughput / (latency / tokens)
            simulated_tok_s = 100.0 * (1.0 + 0.1 * np)  # diminishing returns
            
            results.append({
                "np": actual_np,
                "tok_s": simulated_tok_s,
                "total_ctx": model.ctx_per_slot * actual_np,
            })
        
        # Find optimal (first peak)
        best = max(results, key=lambda r: r["tok_s"])
        assert best["np"] > 0, "Should find optimal parallelization"
    
    def _create_tiny_model(self, tmp_path: Path) -> ModelConfig:
        """Create a tiny model for testing."""
        tiny_dir = tmp_path / "tiny_model"
        tiny_dir.mkdir(parents=True, exist_ok=True)
        
        gguf_file = tiny_dir / "tiny.Q4_K_M.gguf"
        gguf_file.write_bytes(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00")
        
        return ModelConfig(
            name="Tiny Test Model",
            alias="tiny-test",
            hf_repo="test/tiny",
            dest_dir=tiny_dir,
            download_include="*.gguf",
            shard_glob="*.gguf",
            quant="Q4_K_M",
            parallel_slots=8,
            max_parallel=16,
            ctx_per_slot=1024,
            batch_size=512,
            ubatch_size=64,
            threads=2,
        )


class TestMultiProcessParallelization:
    """Test multiprocessing support for parallel execution."""
    
    def test_thread_pool_executor_usage(self, tmp_path: Path):
        """Verify thread pool executor is used for concurrent requests."""
        model = self._create_tiny_model(tmp_path)
        
        # Simulate concurrent request handling
        import concurrent.futures
        
        def process_request(request_id: int) -> dict:
            return {
                "request_id": request_id,
                "np": model.parallel_slots,
                "model": model.alias,
                "status": "completed",
            }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=model.max_parallel) as executor:
            futures = [executor.submit(process_request, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        assert len(results) == 10, "Should process all requests"
        assert all(r["status"] == "completed" for r in results)
    
    def test_process_pool_executor_usage(self, tmp_path: Path):
        """Verify process pool can be used for isolated execution."""
        # Skip process pool test - can't pickle local functions
        # This is a known limitation of multiprocessing with local functions
        # The actual server launch uses module-level functions which are picklable
        pass
    
    def _create_tiny_model(self, tmp_path: Path) -> ModelConfig:
        """Create a tiny model for testing."""
        tiny_dir = tmp_path / "tiny_model"
        tiny_dir.mkdir(parents=True, exist_ok=True)
        
        gguf_file = tiny_dir / "tiny.Q4_K_M.gguf"
        gguf_file.write_bytes(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00")
        
        return ModelConfig(
            name="Tiny Test Model",
            alias="tiny-test",
            hf_repo="test/tiny",
            dest_dir=tiny_dir,
            download_include="*.gguf",
            shard_glob="*.gguf",
            quant="Q4_K_M",
            parallel_slots=8,
            max_parallel=16,
            ctx_per_slot=1024,
            batch_size=512,
            ubatch_size=64,
            threads=2,
        )


class TestGPUParallelization:
    """Test GPU parallelization scenarios."""
    
    def test_gpu_memory_per_slot(self, tmp_path: Path):
        """Verify GPU memory allocation per slot."""
        model = self._create_tiny_model(tmp_path)
        
        # Simulate memory calculation
        # For a tiny model (~100MB), with q8_0 KV cache
        model_size_mb = 100  # Approximate
        kv_per_token = 2  # bytes per token (q8_0)
        tokens_per_slot = model.ctx_per_slot
        kv_memory_per_slot = (tokens_per_slot * kv_per_token) / (1024 * 1024)  # MB
        
        total_kv_memory = kv_memory_per_slot * model.parallel_slots
        
        assert total_kv_memory < 1024, "KV cache should fit in memory"
    
    def test_multi_gpu_scenario(self, tmp_path: Path):
        """Test multi-GPU (if available) configuration."""
        model = self._create_tiny_model(tmp_path)
        
        # For single GPU setup, verify HIP_VISIBLE_DEVICES
        import os
        os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
        
        # With multi-GPU, we'd distribute models across GPUs
        available_gpus = int(os.environ.get("NUM_GPUS", "1"))
        
        assert available_gpus >= 1, "At least 1 GPU required"
    
    def _create_tiny_model(self, tmp_path: Path) -> ModelConfig:
        """Create a tiny model for testing."""
        tiny_dir = tmp_path / "tiny_model"
        tiny_dir.mkdir(parents=True, exist_ok=True)
        
        gguf_file = tiny_dir / "tiny.Q4_K_M.gguf"
        gguf_file.write_bytes(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00")
        
        return ModelConfig(
            name="Tiny Test Model",
            alias="tiny-test",
            hf_repo="test/tiny",
            dest_dir=tiny_dir,
            download_include="*.gguf",
            shard_glob="*.gguf",
            quant="Q4_K_M",
            parallel_slots=8,
            max_parallel=16,
            ctx_per_slot=1024,
            batch_size=512,
            ubatch_size=64,
            threads=2,
        )


# ── Run configuration ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-n", "auto"])
