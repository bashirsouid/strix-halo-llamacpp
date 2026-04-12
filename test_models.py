from __future__ import annotations

import os
from pathlib import Path

import pytest

from models import MODELS, DraftModel, ModelConfig, SpecConfig, get_model


class TestModelLookup:
    def test_get_model_supports_exact_name_alias_and_substring(self):
        exact = get_model("Qwen3 Coder Next (Q6_K)")
        alias = get_model("qwen3-coder-next-q6")
        substring = get_model("next (q6")

        assert exact.alias == "qwen3-coder-next-q6"
        assert alias.name == exact.name
        assert substring.alias == exact.alias

    def test_get_model_is_case_insensitive(self):
        assert get_model("QWEN3-CODER-NEXT-Q6").alias == "qwen3-coder-next-q6"

    def test_get_model_supports_minimax_budget_aliases(self):
        assert get_model("minimax-m2.7-udiq3xxs").quant == "UD-IQ3_XXS"
        assert get_model("minimax-m2.7-udq2xl").quant == "UD-Q2_K_XL"

    def test_get_model_reports_ambiguous_and_missing_inputs(self):
        with pytest.raises(ValueError, match="Ambiguous model name"):
            get_model("nemotron")

        with pytest.raises(ValueError, match="Ambiguous model name"):
            get_model("minimax-m2.7")

        with pytest.raises(ValueError, match="Unknown model"):
            get_model("definitely-not-a-real-model")


class TestSpecConfig:
    def test_no_strategy_returns_empty_args(self):
        assert SpecConfig().server_args() == []

    def test_ngram_without_draft_uses_ngram_limits(self):
        cfg = SpecConfig(strategy="ngram", ngram_type="ngram-map-k", ngram_size_n=32)

        assert cfg.server_args() == [
            "--spec-type",
            "ngram-map-k",
            "--spec-ngram-size-n",
            "32",
            "--draft-max",
            "64",
            "--draft-min",
            "48",
        ]

    def test_draft_strategy_uses_existing_draft_model(self, dummy_draft: DraftModel):
        cfg = SpecConfig(strategy="draft", draft=dummy_draft, draft_max=6, draft_min=3)

        assert cfg.server_args() == [
            "--model-draft",
            str(dummy_draft.path),
            "--draft-max",
            "6",
            "--draft-min",
            "3",
        ]

    def test_draft_plus_ngram_avoids_ngram_fallback_when_draft_exists(self, dummy_draft: DraftModel):
        cfg = SpecConfig(strategy="draft+ngram", draft=dummy_draft)
        args = cfg.server_args()

        assert args.count("--draft-max") == 1
        assert args.count("--draft-min") == 1
        assert "--spec-type" in args


class TestModelServerArgs:
    def test_model_server_args_apply_overrides_and_optional_flags(
        self, dummy_model: ModelConfig
    ):
        dummy_model.reasoning = True
        dummy_model.reasoning_format = "traverse"
        dummy_model.reasoning_budget = 0
        dummy_model.cache_ram = True
        dummy_model.kv_unified = True
        dummy_model.clear_idle = 60
        dummy_model.cpu_moe = 2
        dummy_model.n_cpu_moe = 128
        dummy_model.temperature = 0.7
        dummy_model.top_p = 0.95
        dummy_model.top_k = 40
        dummy_model.min_p = 0.0
        dummy_model.repeat_penalty = 1.05
        dummy_model.chat_template_kwargs = {
            "enable_thinking": True,
            "reasoning_effort": "high",
        }
        dummy_model.extra_args = ["--mirostat", "2"]

        args = dummy_model.server_args(parallel_override=4, ctx_override=16384)

        assert args[args.index("-m") + 1].endswith("dummy-model.Q4_K_M.gguf")
        assert args[args.index("--parallel") + 1] == "4"
        assert args[args.index("--ctx-size") + 1] == "16384"
        assert args[args.index("--temp") + 1] == "0.7"
        assert args[args.index("--top-p") + 1] == "0.95"
        assert args[args.index("--top-k") + 1] == "40"
        assert args[args.index("--min-p") + 1] == "0.0"
        assert args[args.index("--repeat-penalty") + 1] == "1.05"
        assert args[args.index("--chat-template-kwargs") + 1] == (
            '{"enable_thinking":true,"reasoning_effort":"high"}'
        )
        assert "--reasoning" in args
        assert args[args.index("--reasoning-format") + 1] == "traverse"
        assert args[args.index("--reasoning-budget") + 1] == "0"
        assert "--cache-ram" in args
        assert "--kv-unified" in args
        assert args[args.index("--clear-idle") + 1] == "60"
        assert args[args.index("--cpu-moe") + 1] == "2"
        assert args[args.index("--n-cpu-moe") + 1] == "128"
        assert args[-2:] == ["--mirostat", "2"]

    def test_model_server_args_include_mmproj_when_present(
        self, dummy_model: ModelConfig, tmp_path: Path
    ):
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_text("projector")
        dummy_model.mmproj = "mmproj.gguf"

        args = dummy_model.server_args()

        assert args[args.index("--mmproj") + 1] == str(mmproj)

    def test_model_server_args_sets_skip_mmproj_check_when_missing(
        self, dummy_model: ModelConfig, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("LLAMA_SKIP_MMPROJ_CHECK", raising=False)
        dummy_model.mmproj = "missing-projector.gguf"

        args = dummy_model.server_args()

        assert "--mmproj" not in args
        assert os.environ["LLAMA_SKIP_MMPROJ_CHECK"] == "1"

    def test_model_server_args_fail_for_missing_model(self, undownloaded_model: ModelConfig):
        with pytest.raises(FileNotFoundError, match="Model not downloaded"):
            undownloaded_model.server_args()

    def test_model_api_key_defaults_from_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("API_KEY", "secret-key")
        (tmp_path / "model.gguf").write_text("dummy")

        model = ModelConfig(
            name="API Key Model",
            alias="api-key-model",
            hf_repo="owner/repo",
            dest_dir=tmp_path,
            download_include="*.gguf",
            shard_glob="*.gguf",
        )

        assert model.api_key == "secret-key"
        assert model.server_args()[model.server_args().index("--api-key") + 1] == "secret-key"


class TestCatalogSanity:
    def test_catalog_has_unique_aliases_and_valid_parallel_settings(self):
        aliases = [model.alias for model in MODELS]
        assert aliases
        assert len(aliases) == len(set(aliases))

        for model in MODELS:
            assert model.parallel_slots >= 1, model.alias
            assert model.max_parallel >= model.parallel_slots, model.alias
            assert model.ctx_per_slot > 0, model.alias
            assert "/" in model.hf_repo, model.alias
