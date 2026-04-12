"""Model catalog for Strix Halo llama.cpp launcher.

Each model is a ModelConfig dataclass.
Add new models by appending to MODELS.
The server reads this list at startup — nothing else to change.

Parallelization and context strategy
  parallel_slots   number of concurrent request slots (--parallel / -np).
                   Higher values increase *aggregate* throughput when multiple
                   requests arrive simultaneously, at the cost of per-request
                   latency and memory.  For single-user interactive use, 1 is
                   usually fastest.  For API / multi-agent workloads, 2-6 on
                   small MoE models can significantly boost total tok/s.

  ctx_per_slot     context window *per slot* (in tokens).  The total context
                   passed to llama-server is ctx_per_slot × parallel_slots.
                   This keeps each request's usable context constant regardless
                   of how many slots are active.

  max_parallel     upper bound for the bench-parallel sweep.  The sweep will
                   test --np 1…max_parallel and find the throughput peak.

Use `python server.py bench-parallel [MODEL]` to find the optimal slot count
for your exact hardware config, then update parallel_slots to match. The
Aider benchmark defaults to this value unless you override it with `--threads`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MODELS_DIR = Path("/mnt/data/models")


# ── dataclasses ──

@dataclass
class DraftModel:
    """Small model used for classic speculative decoding."""
    hf_repo: str
    filename: str
    dest_dir: Path

    @property
    def path(self) -> Path:
        return self.dest_dir / self.filename


@dataclass
class SpecConfig:
    """Speculation strategy for a model.

    strategy:
        "draft"         use a separate small draft model
        "ngram"         draftless n-gram self-speculation  (best for MoE on UMA)
        "draft+ngram"   combine both (ngram has priority when it matches)
        "ngram-cache"   ngram with cache (new)
        "ngram-simple"  simple ngram mode (new)
        "ngram-map-k"   ngram with map-k (new)
        "ngram-map-k4v" ngram with map-k4v (new)
        "ngram-mod"     modified ngram (default, most stable)
        None            no speculation
    """
    strategy: str | None = None

    # Draft-model settings (used when strategy contains "draft")
    draft: DraftModel | None = None
    draft_max: int = 8
    draft_min: int = 2

    # N-gram settings (used when strategy contains "ngram")
    ngram_type: str = "ngram-mod"
    ngram_size_n: int = 24
    ngram_draft_max: int = 64
    ngram_draft_min: int = 48

    def server_args(self) -> list[str]:
        """Return the llama-server CLI flags for this speculation config."""
        args: list[str] = []
        if self.strategy is None:
            return args

        if "draft" in self.strategy and self.draft and self.draft.path.exists():
            args += [
                "--model-draft", str(self.draft.path),
                "--draft-max",   str(self.draft_max),
                "--draft-min",   str(self.draft_min),
            ]

        if "ngram" in self.strategy:
            args += [
                "--spec-type",        self.ngram_type,
                "--spec-ngram-size-n", str(self.ngram_size_n),
            ]
            if "draft" not in self.strategy or not (self.draft and self.draft.path.exists()):
                args += [
                    "--draft-max", str(self.ngram_draft_max),
                    "--draft-min", str(self.ngram_draft_min),
                ]

        return args


@dataclass
class ModelConfig:
    name: str                   # human-readable name (include quant for clarity)
    alias: str                  # OpenAI-compatible model alias
    hf_repo: str                # Hugging Face repo  (owner/repo)
    dest_dir: Path              # local directory for the GGUF files
    download_include: str       # glob pattern passed to `hf download --include`
    shard_glob: str             # glob to find shard-1 (the file passed to -m)
    quant: str = ""             # quantization level (e.g. "Q4_K_M", "Q6_K")

    #  Parallelization  
    parallel_slots: int = 1     # --parallel / -np (concurrent request slots, also default aider-bench threads)
    max_parallel: int = 8       # upper bound for bench-parallel sweep
    ctx_per_slot: int = 32768   # context window per slot (total = this × slots)

    #  Inference tuning  
    batch_size: int = 4096      # -b   (logical max batch)
    ubatch_size: int = 256      # -ub  (physical max ubatch)
    threads: int = 4            # -t   (CPU threads)
    cache_type_k: str = "q8_0"  # --cache-type-k  (q8_0 saves ~50% vs f16)
    cache_type_v: str = "q8_0"  # --cache-type-v

    #  Sampling defaults
    temperature: float | None = None     # --temp
    top_p: float | None = None           # --top-p
    top_k: int | None = None             # --top-k
    min_p: float | None = None           # --min-p
    repeat_penalty: float | None = None  # --repeat-penalty
    presence_penalty: float | None = None  # --presence-penalty
    frequency_penalty: float | None = None  # --frequency-penalty

    #  Feature flags  
    reasoning_format: str | None = None  # --reasoning-format (none, deepseek, deepseek-legacy, auto)
    reasoning_budget: int | None = None  # --reasoning-budget (tokens)
    reasoning: bool = False              # --reasoning (enable reasoning mode)
    cache_prompt: bool = True            # --cache-prompt / --no-cache-prompt
    cache_reuse: int = 256               # --cache-reuse (min reusable chunk)
    cache_ram: int | bool | None = None  # --cache-ram MiB; True keeps legacy 8 GiB behavior
    slot_save_path: str | None = None    # --slot-save-path (enables /slots save/restore)
    kv_unified: bool = False             # --kv-unified
    clear_idle: int = 0                  # --clear-idle (seconds)
    cpu_moe: int = 0                     # --cpu-moe (layers)
    n_cpu_moe: int = 0                   # --n-cpu-moe (n-batch)

    #  Speculation  
    spec: SpecConfig = field(default_factory=SpecConfig)

    #  Extras  
    chat_template_file: str | None = None
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    mmproj: str | None = None  # Path or filename for multimodal projector (GGUF)
    extra_args: list[str] = field(default_factory=list)  # One-off / experimental llama.cpp flags
    notes: str = ""
    hidden: bool = False
    api_key: str | None = None  # Global API key override (use .env file)

    def __post_init__(self):
        """Load API key from environment if set and not manually configured."""
        import os
        if self.api_key is None:
            self.api_key = os.environ.get("API_KEY")

    @property
    def ctx_size(self) -> int:
        """Total context passed to llama-server (ctx_per_slot × parallel_slots)."""
        return self.ctx_per_slot * self.parallel_slots

    @property
    def model_path(self) -> Path | None:
        """Return path to the first shard, or None if not downloaded."""
        matches = sorted(self.dest_dir.glob(self.shard_glob))
        return matches[0] if matches else None

    @property
    def is_downloaded(self) -> bool:
        return self.model_path is not None

    def server_args(self, parallel_override: int | None = None,
                    ctx_override: int | None = None) -> list[str]:
        """Build the full llama-server argument list for this model.

        Args:
            parallel_override: If set, use this instead of self.parallel_slots.
                               Used by bench-parallel to sweep different values.
            ctx_override:      If set, use as total context instead of calculating.
        """
        model_path = self.model_path
        if model_path is None:
            raise FileNotFoundError(f"Model not downloaded: {self.name}")

        np = parallel_override if parallel_override is not None else self.parallel_slots
        if ctx_override is not None:
            total_ctx = ctx_override
        else:
            total_ctx = self.ctx_per_slot * np

        args = [
            "-m",             str(model_path),
            "--host",         "0.0.0.0",
            "--port",         "8000",
            "--ctx-size",     str(total_ctx),
            "-ngl",           "999",
            "--flash-attn",   "on",
            "--parallel",     str(np),
            "-a",             self.alias,
            "--cache-prompt" if self.cache_prompt else "--no-cache-prompt",
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
            "-b",             str(self.batch_size),
            "-ub",            str(self.ubatch_size),
            "-t",             str(self.threads),
            "--jinja",
        ]

        if self.temperature is not None:
            args += ["--temp", str(self.temperature)]
        if self.top_p is not None:
            args += ["--top-p", str(self.top_p)]
        if self.top_k is not None:
            args += ["--top-k", str(self.top_k)]
        if self.min_p is not None:
            args += ["--min-p", str(self.min_p)]
        if self.repeat_penalty is not None:
            args += ["--repeat-penalty", str(self.repeat_penalty)]
        if self.presence_penalty is not None:
            args += ["--presence-penalty", str(self.presence_penalty)]
        if self.frequency_penalty is not None:
            args += ["--frequency-penalty", str(self.frequency_penalty)]

        if self.chat_template_file:
            args += ["--chat-template-file", self.chat_template_file]
        if self.chat_template_kwargs:
            args += [
                "--chat-template-kwargs",
                json.dumps(self.chat_template_kwargs, separators=(",", ":"), sort_keys=True),
            ]

        if self.mmproj:
            mmproj_path = self.dest_dir / self.mmproj if "/" not in self.mmproj else Path(self.mmproj)
            if mmproj_path.exists():
                args += ["--mmproj", str(mmproj_path)]
            else:
                import os
                os.environ.setdefault("LLAMA_SKIP_MMPROJ_CHECK", "1")

        if self.api_key:
            args += ["--api-key", self.api_key]

        if self.reasoning_format:
            args += ["--reasoning-format", self.reasoning_format]
        if self.reasoning_budget is not None:
            args += ["--reasoning-budget", str(self.reasoning_budget)]
        if self.reasoning:
            args += ["--reasoning"]
        if self.cache_prompt and self.cache_reuse > 0:
            args += ["--cache-reuse", str(self.cache_reuse)]
        if self.cache_ram not in (None, False, 0):
            cache_ram = 8192 if self.cache_ram is True else int(self.cache_ram)
            args += ["--cache-ram", str(cache_ram)]
        if self.slot_save_path:
            args += ["--slots", "--slot-save-path", self.slot_save_path]
        if self.kv_unified:
            args += ["--kv-unified"]
        if self.clear_idle > 0:
            args += ["--clear-idle", str(self.clear_idle)]
        if self.cpu_moe > 0:
            args += ["--cpu-moe", str(self.cpu_moe)]
        if self.n_cpu_moe > 0:
            args += ["--n-cpu-moe", str(self.n_cpu_moe)]

        args += self.spec.server_args()
        args += self.extra_args
        return args


# ── Model catalog ──
#
# Defaults tuned for:
#   AMD Ryzen AI Max 395 ·  96 GB system RAM ·  90 GB mapped to GPU
#   Vulkan (RADV) backend
#
# Run `python server.py bench-parallel MODEL` to find the optimal
# parallel_slots for your exact config, then update the value here.
#
# Memory budget reasoning (90 GB GPU):
#   Model weights + KV cache + overhead must fit in ~88 GB (leave ~2 GB).
#   q8_0 KV cache ≈ 1 byte per element.  Per-token KV size varies by arch
#   but for MoE models with small active params, KV is very manageable.


MODELS: list[ModelConfig] = [

    # ── Qwen3 Coder Next ──
    # ~62 GB model weight at Q6_K → ~26 GB left for KV + overhead
    ModelConfig(
        name="Qwen3 Coder Next (Q6_K)",
        alias="qwen3-coder-next-q6",
        hf_repo="unsloth/Qwen3-Coder-Next-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Qwen3-Coder-Next-GGUF/Q6_K",
        download_include="Q6_K/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="Q6_K",
        parallel_slots=1,
        max_parallel=3,
        ctx_per_slot=262144,
        ubatch_size=512,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        min_p=0.01,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: coding agents, tool calling, agentic workflows. "
            "MoE 80B (3B active). #1 on SWE-rebench. ~62 GB at Q6_K. "
            "Non-thinking — fast direct responses, no <think> blocks. "
            "Qwen official: temp=1.0, top_p=0.95, top_k=40. "
            "Bug: -ub must be 512 on Strix Halo Vulkan (issue #18725). "
            "Bug: Do NOT use Q6_K_XL — broken architecture detection."
        ),
    ),

    # ── Qwen3 Coder Next ──
    # ~62 GB model weight at UD-Q6_K_XL → ~26 GB left for KV + overhead
    ModelConfig(
        name="Qwen3 Coder Next (UD-Q6_K_XL)",
        alias="qwen3-coder-next-udq6xl",
        hf_repo="unsloth/Qwen3-Coder-Next-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Qwen3-Coder-Next-GGUF/UD-Q6_K_XL/",
        download_include="*UD-Q6_K_XL*",
        shard_glob="*-00001-of-*.gguf",
        quant="UD-Q6_K_XL",
        parallel_slots=1,
        max_parallel=3,
        ctx_per_slot=262144,
        ubatch_size=512,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        min_p=0.01,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: coding agents, tool calling, agentic workflows. "
            "MoE 80B (3B active). #1 on SWE-rebench. ~62 GB at UD-Q6_K_XL. "
            "Non-thinking — fast direct responses, no  Witt blocks. "
            "Qwen official: temp=1.0, top_p=0.95, top_k=40. "
            "Bug: -ub must be 512 on Strix Halo Vulkan (issue #18725). "
            "Bug: Do NOT use Q6_K_XL — broken architecture detection."
        ),
    ),

    # ── Qwen3 Coder Next ──
    # ~40 GB model weight at Q4_K_XL → ~48 GB left for KV + overhead
    ModelConfig(
        name="Qwen3 Coder Next (Q4_K_XL)",
        alias="qwen3-coder-next-q4",
        hf_repo="unsloth/Qwen3-Coder-Next-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Qwen3-Coder-Next-GGUF",
        download_include="*UD-Q4_K_XL*",
        shard_glob="*UD-Q4_K_XL*.gguf",
        quant="UD-Q4_K_XL",
        parallel_slots=1,
        max_parallel=3,
        ctx_per_slot=262144,
        ubatch_size=512,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        min_p=0.01,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: coding agents, tool calling, agentic workflows. "
            "MoE 80B (3B active). #1 on SWE-rebench. ~40 GB at UD-Q4_K_XL. "
            "Non-thinking — fast direct responses, no <think> blocks. "
            "Qwen official: temp=1.0, top_p=0.95, top_k=40. "
            "Bug: -ub must be 512 on Strix Halo Vulkan (issue #18725). "
            "Bug: Do NOT use Q6_K_XL — broken architecture detection."
        ),
    ),

    # ── MiniMax M2.7 ──
    # ~80 GB at UD-IQ3_XXS → ~4–8 GB left for KV + overhead
    ModelConfig(
        name="MiniMax M2.7 (UD-IQ3_XXS)",
        alias="minimax-m2.7-udiq3xxs",
        hf_repo="unsloth/MiniMax-M2.7-GGUF",
        dest_dir=MODELS_DIR / "unsloth/MiniMax-M2.7-GGUF/UD-IQ3_XXS",
        download_include="UD-IQ3_XXS/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="UD-IQ3_XXS",
        parallel_slots=1,
        max_parallel=1,
        ctx_per_slot=16384,
        ubatch_size=512,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: strongest MiniMax M2.7 quality that still fits a Ryzen AI Max 395 / 96 GB shared-memory budget. "
            "MoE 229B total with 256 experts and 8 active per token. "
            "UD dynamic 3-bit quant (~80.1 GB) is the highest MiniMax-M2.7 variant here that still leaves room for a practical KV cache. "
            "Context is capped to 16K by default because MiniMax KV cache is large at this scale. "
            "MiniMax official: temp=1.0, top_p=0.95, top_k=40. "
            "Requires a recent llama.cpp build with MiniMax M2 architecture support; re-pull images if your local containers predate that."
        ),
    ),

    # ── MiniMax M2.7 ──
    # ~75 GB at UD-Q2_K_XL → ~8–12 GB left for KV + overhead
    ModelConfig(
        name="MiniMax M2.7 (UD-Q2_K_XL)",
        alias="minimax-m2.7-udq2xl",
        hf_repo="unsloth/MiniMax-M2.7-GGUF",
        dest_dir=MODELS_DIR / "unsloth/MiniMax-M2.7-GGUF/UD-Q2_K_XL",
        download_include="UD-Q2_K_XL/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="UD-Q2_K_XL",
        parallel_slots=1,
        max_parallel=2,
        ctx_per_slot=32768,
        ubatch_size=512,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: fitting MiniMax M2.7 more comfortably inside an ~80–90 GB mapped-memory budget. "
            "MoE 229B total with 256 experts and 8 active per token. "
            "UD dynamic Q2_K_XL (~75.3 GB) is the safest Strix Halo pick; expect a clearer speed edge over the 80 GB 3-bit entry but with a larger quality drop. "
            "32K context is a safer default on a ~90 GB mapped-memory budget. "
            "MiniMax official: temp=1.0, top_p=0.95, top_k=40. "
            "Use this alias when you want the best chance of a clean local benchmark run against qwen3-coder-next-q6."
        ),
    ),

    # ── Kimi-Dev-72B (Q6_K) ──
    # ~59 GB at Q6_K + ~10 GB KV at 65K ctx = ~69 GB total → comfortable on 96 GB
    # 131K native context possible (total ≈81 GB) if system memory is otherwise clear
    ModelConfig(
        name="Kimi-Dev-72B (Q6_K)",
        alias="kimi-dev-72b-q6",
        hf_repo="bartowski/moonshotai_Kimi-Dev-72B-GGUF",
        dest_dir=MODELS_DIR / "bartowski/moonshotai_Kimi-Dev-72B-GGUF/moonshotai_Kimi-Dev-72B-Q6_K",
        download_include="moonshotai_Kimi-Dev-72B-Q6_K/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="Q6_K",
        parallel_slots=1,
        max_parallel=2,
        ctx_per_slot=65536,
        temperature=0.3,  # Conservative for code; RL-trained on test-passing (moonshotai)
        top_p=0.95,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: hard coding tasks, real-world SWE-bench-style patching, large codebase comprehension. "
            "Dense 72B (no MoE) — 60.4% SWE-bench Verified. "
            "RL-trained exclusively on test-passing reward in Docker; no instruction tuning. "
            "~59 GB at Q6_K; KV cache ≈10 GB at 65K ctx (q8_0, GQA-8). "
            "To use 131K native context: set ctx_per_slot=131072 (total ≈81 GB — safe if system is otherwise idle). "
            "No thinking/reasoning mode. Dense = slow: expect ~4–6 tok/s vs MoE peers. "
            "MIT license. Use when quality matters more than speed."
        ),
    ),

    # ── GLM 4.7 Flash ──
    # ~30 GB at Q8 → ~58 GB for KV + overhead — plenty of room
    ModelConfig(
        name="GLM 4.7 Flash (Q8_K_XL)",
        alias="glm-4.7-flash-q8",
        hf_repo="unsloth/GLM-4.7-Flash-GGUF",
        dest_dir=MODELS_DIR / "unsloth/GLM-4.7-Flash-GGUF",
        download_include="GLM-4.7-Flash-Q8_K_XL.gguf",
        shard_glob="*Q8_K_XL*.gguf",
        quant="Q8_K_XL",
        parallel_slots=1,
        max_parallel=8,
        ctx_per_slot=32768,
        temperature=0.7,
        top_p=1.0,
        min_p=0.01,
        repeat_penalty=1.1,  # Mild prevention; GLM 4.7 Flash needs this (notes indicated issue)
        spec=SpecConfig(strategy="ngram"),
        chat_template_kwargs={"enable_thinking": True},
        reasoning_format="auto",
        notes=(
            "Best for: code + chat at high speed, interleaved thinking. "
            "Thinking is forced on via chat_template_kwargs.enable_thinking. "
            "Parallelism is only effective on ROCM; latest testing is slower than sequential on RADV."
            "MoE 30B (3B active). Best 30B model on SWE-Bench + GPQA."
            "~30 GB at Q8 — near-lossless, fits easily. 200K context."
            "Bug: needs --repeat-penalty 1.1 and --min-p 0.01."
        ),
    ),

    # ── Qwen3.5 35B ──
    # ~48 GB at Q8_K_XL → ~40 GB for KV + overhead
    ModelConfig(
        name="Qwen3.5 35B (Q8_K_XL)",
        alias="qwen3.5-35b-q8",
        hf_repo="unsloth/Qwen3.5-35B-A3B-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Qwen3.5-35B-A3B-GGUF",
        download_include="*UD-Q8_K_XL*",
        shard_glob="*UD-Q8_K_XL*.gguf",
        quant="UD-Q8_K_XL",
        parallel_slots=3,
        max_parallel=6,
        ctx_per_slot=32768,
        temperature=0.6,  # Qwen official: temp=0.6 for coding tasks (thinking mode)
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,  # Qwen official: 0.0 for coding tasks
        repeat_penalty=1.0,
        spec=SpecConfig(strategy="ngram"),
        chat_template_kwargs={"enable_thinking": True},
        reasoning_format="auto",
        notes=(
            "Best for: reasoning, summarization, general-purpose. "
            "MoE 35B (3B active). Thinking mode forced on via chat_template_kwargs.enable_thinking. "
            "~48 GB at Q8_K_XL — high quality. Multimodal. "
            "Qwen official: temp=0.6 (coding), top_p=0.95, top_k=20, presence_penalty=0.0."
        ),
    ),

    # ── Gemma 4 26B A4B MoE ──
    ModelConfig(
        name="Gemma 4 26B A4B (UD-IQ4_XS)",
        alias="gemma4-26b-a4b-udiq4",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        dest_dir=MODELS_DIR / "unsloth/gemma-4-26B-A4B-it-GGUF",
        download_include="gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
        shard_glob="*UD-IQ4_XS*.gguf",
        quant="UD-IQ4_XS",
        parallel_slots=1,
        max_parallel=6,
        ctx_per_slot=262144,
        ubatch_size=256,
        mmproj="mmproj-BF16.gguf",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        spec=SpecConfig(strategy="ngram"),
        chat_template_kwargs={"enable_thinking": True},
        reasoning_format="auto",
        notes=(
            "Best for: code, tool calling, reasoning, vision (with mmproj). "
            "MoE 25.2B (3.8B active). 128 experts, 8 active + 1 shared. "
            "UD dynamic quant — smaller than the prior Q8_0 entry while keeping the same model family. "
            "256K native context. Native function calling baked into training. "
            "Thinking mode is forced on via chat_template_kwargs.enable_thinking. "
            "Google official: temp=1.0, top_p=0.95, top_k=64. "
            "Vision: requires mmproj-BF16.gguf projector file. "
            "Apache 2.0. Day-1 note: tool-call templates still maturing in llama.cpp."
        ),
    ),

    # ── Gemma 4 31B Dense ──
    # ~32.6 GB at Q8_0 → ~57 GB for KV + overhead
    ModelConfig(
        name="Gemma 4 31B Dense (Q8_0)",
        alias="gemma4-31b-q8",
        hf_repo="unsloth/gemma-4-31B-it-GGUF",
        dest_dir=MODELS_DIR / "unsloth/gemma-4-31B-it-GGUF",
        download_include="*Q8_0*",
        shard_glob="*Q8_0*.gguf",
        quant="Q8_0",
        parallel_slots=1,
        max_parallel=2,
        ctx_per_slot=32768,
        ubatch_size=256,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        chat_template_kwargs={"enable_thinking": True},
        reasoning_format="auto",
        notes=(
            "Best for: maximum code quality, hard reasoning, vision. "
            "Dense 30.7B — all params active every token. SLOW (~6-8 tok/s). "
            "~32.6 GB at Q8_0. Highest benchmarks: LCB 80.0%, CF ELO 2150, "
            "GPQA 84.3%, AIME 89.2%. Use for hard single-shot problems where "
            "you can wait — too slow for interactive agentic loops. "
            "Thinking mode is forced on via chat_template_kwargs.enable_thinking. "
            "Google official: temp=1.0, top_p=0.95, top_k=64. "
            "Vision: download mmproj-BF16.gguf separately, use --mmproj flag. "
            "Apache 2.0. No speculation — dense models don't benefit much from "
            "ngram on UMA, and draft models add memory pressure."
        ),
    ),

    # ── Qwen3 Coder 30B A3B ──
    ModelConfig(
        name="Qwen3 Coder 30B A3B (UD-Q4_K_XL)",
        alias="qwen3-coder-30b-udq4",
        hf_repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        download_include="*UD-Q4_K_XL*",
        shard_glob="*UD-Q4_K_XL*.gguf",
        quant="UD-Q4_K_XL",
        parallel_slots=1,
        max_parallel=6,
        ctx_per_slot=262144,
        ubatch_size=512,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repeat_penalty=1.05,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: coding agents, tool calling — lighter alternative to Coder Next. "
            "MoE 30.5B (3.3B active). 128 experts, 8 activated. "
            "UD dynamic quant for better quality-per-GB than uniform low-bit quants. "
            "256K native context. Non-thinking only (no <think> blocks). "
            "Qwen official: temp=0.7, top_p=0.8, top_k=20, rep_pen=1.05. "
            "Bug: -ub must be 512 on Strix Halo Vulkan (same qwen3_moe arch as Coder Next). "
            "Apache 2.0. Agentic coding: supports Qwen Code, CLINE, function call format."
        ),
    ),

    # ── Mistral Small 4 Tool/Code ──
    # ~50–60 GB at Q4 → ~28–38 GB for KV + overhead
    ModelConfig(
        name="Mistral Small 4 (Q4_K_M) Tool/Code",
        alias="mistral-small-4-q4-tool",
        hf_repo="unsloth/Mistral-Small-4-119B-2603-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Mistral-Small-4-119B-2603-GGUF/UD-Q4_K_M",
        download_include="UD-Q4_K_M/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="UD-Q4_K_M",
        parallel_slots=1,
        max_parallel=4,
        ctx_per_slot=32768,
        temperature=0.1,  # Mistral official: temp=0.1 for tool/code (reasoning_effort=none)
        top_k=50,
        chat_template_kwargs={"reasoning_effort": "none"},
        spec=SpecConfig(
            strategy="draft+ngram",
            draft=DraftModel(
                hf_repo="bartowski/alamios_Mistral-Small-3.1-DRAFT-0.5B-GGUF",
                filename="alamios_Mistral-Small-3.1-DRAFT-0.5B-Q4_K_M.gguf",
                dest_dir=MODELS_DIR / "bartowski/alamios_Mistral-Small-3.1-DRAFT-0.5B-GGUF",
            ),
            draft_max=8,
            draft_min=2,
        ),
        notes=(
            "Best for: tool calling, code generation, fast chat. "
            "MoE 119B (6.5B active). Temperature=0.1 for deterministic outputs. "
            "Reasoning is pinned to reasoning_effort=none for the tool/code alias. "
            "Draft model + ngram speculation for speed. "
            "Mistral official: temp=0.1 (tool/code), 0.7 (reasoning). "
            "Unsloth recommends --jinja template."
        ),
    ),

    # ── Mistral Small 4 Reasoning ──
    # ~50–60 GB at Q4 → ~28–38 GB for KV + overhead
    ModelConfig(
        name="Mistral Small 4 (Q4_K_M) Reasoning",
        alias="mistral-small-4-q4-reasoning",
        hf_repo="unsloth/Mistral-Small-4-119B-2603-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Mistral-Small-4-119B-2603-GGUF/UD-Q4_K_M",
        download_include="UD-Q4_K_M/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="UD-Q4_K_M",
        parallel_slots=1,
        max_parallel=2,
        ctx_per_slot=131072,
        temperature=0.7,  # Mistral official: temp=0.7 for reasoning (reasoning_effort=high)
        top_p=0.95,
        top_k=20,
        chat_template_kwargs={"reasoning_effort": "high"},
        spec=SpecConfig(
            strategy="ngram",
        ),
        notes=(
            "Best for: long-context reasoning, complex problem solving. "
            "MoE 119B (6.5B active). Temperature=0.7 for creative reasoning. "
            "Reasoning is forced on via chat_template_kwargs.reasoning_effort=high. "
            "N-gram speculation (draft model not recommended for reasoning). "
            "256K context window (limited to 128K here). "
            "Mistral official: temp=0.7 (reasoning)."
        ),
    ),

    # ── Nemotron 3 Super ──
    # ~50–60 GB at Q4 → ~28–38 GB for KV + overhead
    ModelConfig(
        name="Nemotron 3 Super (Q4_K_M)",
        alias="nemotron-super-q4",
        hf_repo="unsloth/Nemotron-3-Super-120B-A12B-GGUF",
        dest_dir=MODELS_DIR / "nvidia/nemotron-3-super/UD-Q4_K_M",
        download_include="UD-Q4_K_M/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="UD-Q4_K_M",
        parallel_slots=1,
        max_parallel=3,
        ctx_per_slot=65536,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=0.0,  # NVIDIA default (no penalty)
        frequency_penalty=0.0,  # NVIDIA default (no penalty)
        spec=SpecConfig(strategy="ngram"),
        chat_template_kwargs={"enable_thinking": True},
        reasoning_format="auto",
        notes=(
            "Best for: long-context reasoning, multi-agent workflows. "
            "Hybrid Mamba2-Transformer MoE 120B (12B active). "
            "Natively supports 1M context (limited here by memory). "
            "NVIDIA official: temp=1.0, top_p=0.95. "
            "Thinking mode is forced on via chat_template_kwargs.enable_thinking. "
            "Use --reasoning-budget and --reasoning-format controls for advanced usage."
        ),
    ),

        # ── Nemotron Nano Q4 ──
    ModelConfig(
        name="Nemotron Nano (UD-Q4_K_XL)",
        alias="nemotron-nano-udq4",
        hf_repo="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        dest_dir=MODELS_DIR / "nvidia/nemotron-nano",
        download_include="*UD-Q4_K_XL*",
        shard_glob="*UD-Q4_K_XL*.gguf",
        quant="UD-Q4_K_XL",
        parallel_slots=8,
        max_parallel=12,
        ctx_per_slot=1048576,
        temperature=0.6,
        top_p=0.95,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        spec=SpecConfig(strategy="ngram"),
        chat_template_kwargs={"enable_thinking": True},
        reasoning_format="auto",
        notes=(
            "Best for: speed, lightweight tasks, tool calling, quick iteration. "
            "MoE 30B (3B active). "
            "UD dynamic quant for better quality-per-GB than the prior uniform Q4 entry. "
            "Thinking mode is forced on via chat_template_kwargs.enable_thinking. "
            "Good for drafting, quick Q&A, low-latency tool calls. "
            "NVIDIA: temp=0.6 for lightweight tasks."
        ),
    ),

    # ── Nemotron Nano Q8 ──
    # ~12 GB at Q8 → ~76 GB for KV + overhead
    ModelConfig(
        name="Nemotron Nano (UD-Q8_K_XL)",
        alias="nemotron-nano-q8",
        hf_repo="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        dest_dir=MODELS_DIR / "nvidia/nemotron-nano",
        download_include="*UD-Q8_K_XL*.gguf",
        shard_glob="*UD-Q8_K_XL*.gguf",
        quant="UD-Q8_K_XL",
        parallel_slots=8,
        max_parallel=10,
        ctx_per_slot=1048576,
        temperature=0.6,
        top_p=0.95,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        spec=SpecConfig(strategy="ngram"),
        chat_template_kwargs={"enable_thinking": True},
        reasoning_format="auto",
        notes=(
            "Best for: speed/quality balance, lightweight tasks, tool calling. "
            "MoE 30B (3B active). ~45+ tok/s at Q8. "
            "Thinking mode is forced on via chat_template_kwargs.enable_thinking. "
            "Good for drafting, quick Q&A, low-latency tool calls. "
            "NVIDIA: temp=0.6 for lightweight tasks."
        ),
    ),

    # ── Qwen3.5 122B A10B (Architect) ──
    ModelConfig(
        name="Qwen3.5 122B A10B (UD-Q4_K_XL)",
        alias="qwen3.5-122b-udq4",
        hf_repo="unsloth/Qwen3.5-122B-A10B-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL",
        download_include="*UD-Q4_K_XL*",
        shard_glob="*UD-Q4_K_XL*.gguf",
        quant="UD-Q4_K_XL",
        parallel_slots=1,
        max_parallel=2,
        ctx_per_slot=131072, 
        ubatch_size=512,
        temperature=0.6,  # Qwen official: temp=0.6 for coding tasks (thinking mode)
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,  # Qwen official: 0.0 for coding tasks
        repeat_penalty=1.0,       # Keep at 1.0 (off) to prevent degrading <think> loops
        spec=SpecConfig(strategy="ngram"),
        chat_template_kwargs={"enable_thinking": True},
        reasoning_format="auto",
        notes=(
            "Best for: The 'Architect' model in multi-file agentic refactoring. "
            "MoE 122B (10B active). Fits in ~70GB, leaving ~20GB for the 128K context cache. "
            "Expect ~12-15 tok/sec on Strix Halo 395 due to 10B active parameters. "
            "Thinking mode is forced on via chat_template_kwargs.enable_thinking. "
            "Qwen official: temp=0.6 (coding), top_p=0.95, top_k=20, presence_penalty=0.0. "
            "Do not use for fast autocomplete; use strictly as a planner/architect."
        ),
    ),

    # ── DeepSeek Coder V2 Lite ──
    ModelConfig(
        name="DeepSeek Coder V2 Lite (Q8_0_L)",
        alias="deepseek-coder-v2-lite",
        hf_repo="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        dest_dir=MODELS_DIR / "deepseek/DeepSeek-Coder-V2-Lite-Instruct",
        download_include="*Q8_0_L.gguf",
        shard_glob="*Q8_0_L.gguf",
        quant="Q8_0_L",
        parallel_slots=1,
        max_parallel=4,
        ctx_per_slot=65536,       # Model supports 128K, but 64K is safer for VRAM at Q8
        ubatch_size=512,
        temperature=0.0,          # DeepSeek official: temp=0.0 for pure coding tasks
        top_p=1.0,                # DeepSeek official: top_p=1.0 when temp is 0.0
        repeat_penalty=1.0,
        spec=SpecConfig(strategy="ngram"),
        # Note: DeepSeek-V2 requires the deepseek2/deepseek-chat chat template
        chat_template_kwargs={},
        notes=(
            "Best for: Fast agentic coding, autocomplete, and massive context ingestion. "
            "MoE 16B (2.4B active). Uses Multi-Head Latent Attention (MLA). "
            "Will run extremely fast on Strix Halo due to only 2.4B active parameters. "
            "Native 128K context. DeepSeek official: temp=0.0, top_p=1.0 for coding. "
        ),
    ),

    # ── Hidden smoke-test model ──
    ModelConfig(
        name="SmolLM2 135M Instruct (Q4_K_M) [test]",
        alias="smollm2-135m-test-q4",
        hf_repo="bartowski/SmolLM2-135M-Instruct-GGUF",
        dest_dir=MODELS_DIR / "bartowski/SmolLM2-135M-Instruct-GGUF",
        download_include="SmolLM2-135M-Instruct-Q4_K_M.gguf",
        shard_glob="*Q4_K_M*.gguf",
        quant="Q4_K_M",
        parallel_slots=1,
        max_parallel=2,
        ctx_per_slot=4096,
        batch_size=256,
        ubatch_size=64,
        threads=2,
        hidden=True,
        notes=(
            "Hidden smoke-test model for ./test.sh. "
            "Kept out of the interactive picker and --all flows, but still "
            "addressable by alias from the CLI."
        ),
    ),

]


# ── helpers ──

def get_model(name_or_alias: str) -> ModelConfig:
    """Look up a model by name, alias, or unambiguous substring."""
    key = name_or_alias.lower().strip()
    for m in MODELS:
        if key in (m.alias.lower(), m.name.lower()):
            return m
    # substring match
    matches = [m for m in MODELS if key in m.alias.lower() or key in m.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(m.alias for m in matches)
        raise ValueError(f"Ambiguous model name '{name_or_alias}' — matches: {names}")
    raise ValueError(
        f"Unknown model '{name_or_alias}'. Available: "
        + ", ".join(m.alias for m in MODELS)
    )
