from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DraftModel, ModelConfig, SpecConfig


@pytest.fixture
def dummy_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelConfig:
    """Create a downloaded model fixture backed by a dummy GGUF file."""
    monkeypatch.delenv("API_KEY", raising=False)
    model_path = tmp_path / "dummy-model.Q4_K_M.gguf"
    model_path.write_text("dummy")
    return ModelConfig(
        name="Dummy Model",
        alias="dummy-model",
        hf_repo="owner/dummy-model",
        dest_dir=tmp_path,
        download_include="*.gguf",
        shard_glob="*.gguf",
        quant="Q4_K_M",
        parallel_slots=2,
        max_parallel=4,
        ctx_per_slot=2048,
    )


@pytest.fixture
def dummy_draft(tmp_path: Path) -> DraftModel:
    draft_path = tmp_path / "draft.gguf"
    draft_path.write_text("draft")
    return DraftModel(
        hf_repo="owner/draft-model",
        filename="draft.gguf",
        dest_dir=tmp_path,
    )


@pytest.fixture
def undownloaded_model(tmp_path: Path) -> ModelConfig:
    return ModelConfig(
        name="Undownloaded Model",
        alias="undownloaded-model",
        hf_repo="owner/undownloaded",
        dest_dir=tmp_path,
        download_include="*.gguf",
        shard_glob="*.gguf",
        quant="Q4_K_M",
        parallel_slots=1,
        max_parallel=2,
        ctx_per_slot=1024,
        spec=SpecConfig(strategy="ngram"),
    )
