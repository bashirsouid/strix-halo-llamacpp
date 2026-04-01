"""
Model catalog for Strix Halo llama.cpp launcher.

Each model is a ModelConfig dataclass.  Add new models by appending to MODELS.
The server reads this list at startup — nothing else to change.

Parallelization and context strategy
─────────────────────────────────────
  parallel_slots  — number of concurrent request slots (--parallel / -np).
                    Higher values increase *aggregate* throughput when multiple
                    requests arrive simultaneously, at the cost of per-request
                    latency and memory.  For single-user interactive use, 1 is
                    usually fastest.  For API / multi-agent workloads, 2-6 on
                    small MoE models can significantly boost total tok/s.

  ctx_per_slot    — context window *per slot* (in tokens).  The total context
                    passed to llama-server is ctx_per_slot × parallel_slots.
                    This keeps each request's usable context constant regardless
                    of how many slots are active.

  max_parallel    — upper bound for the bench-parallel sweep.  The sweep will
                    test --np 1 … max_parallel and find the throughput peak.

Use `python server.py bench-parallel [MODEL]` to find the optimal slot count
for your exact hardware config, then update parallel_slots to match.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

MODELS_DIR = Path("/mnt/data/models")


# ── dataclasses ──────────────────────────────────────────────────────────────

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
        "draft"        – use a separate small draft model
        "ngram"        – draftless n-gram self-speculation  (best for MoE on UMA)
        "draft+ngram"  – combine both (ngram has priority when it matches)
        None           – no speculation
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

    # ── Parallelization ──────────────────────────────────────────────────────
    parallel_slots: int = 1     # --parallel / -np (concurrent request slots)
    max_parallel: int = 8       # upper bound for bench-parallel sweep
    ctx_per_slot: int = 32768   # context window per slot (total = this × slots)

    # ── Inference tuning ─────────────────────────────────────────────────────
    batch_size: int = 4096      # -b   (logical max batch)
    ubatch_size: int = 256      # -ub  (physical max ubatch)
    threads: int = 4            # -t   (CPU threads)
    cache_type_k: str = "q8_0"  # --cache-type-k  (q8_0 saves ~50% vs f16)
    cache_type_v: str = "q8_0"  # --cache-type-v

    # ── Speculation ──────────────────────────────────────────────────────────
    spec: SpecConfig = field(default_factory=SpecConfig)

    # ── Extras ───────────────────────────────────────────────────────────────
    chat_template_file: str | None = None
    extra_args: list[str] = field(default_factory=list)
    notes: str = ""

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
            "--no-mmap",
            "--flash-attn",   "on",
            "--parallel",     str(np),
            "-a",             self.alias,
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
            "-b",             str(self.batch_size),
            "-ub",            str(self.ubatch_size),
            "-t",             str(self.threads),
            "--jinja",
        ]

        if self.chat_template_file:
            args += ["--chat-template-file", self.chat_template_file]

        args += self.spec.server_args()
        args += self.extra_args
        return args


# ── Model catalog ────────────────────────────────────────────────────────────
#
# Defaults tuned for:
#   AMD Ryzen AI Max 395  ·  96 GB system RAM  ·  90 GB mapped to GPU
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

    # ── Qwen3 Coder Next  ────────────────────────────────────────────────────
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
        ctx_per_slot=32768,
        ubatch_size=512,
        spec=SpecConfig(strategy="ngram"),
        extra_args=["--temp", "1.0", "--top-p", "0.95", "--top-k", "40", "--min-p", "0.01"],
        notes=(
            "Best for: coding agents, tool calling, agentic workflows.  "
            "MoE 80B (3B active).  #1 on SWE-rebench.  ~62 GB at Q6_K.  "
            "Non-thinking — fast direct responses, no <think> blocks.  "
            "Bug: -ub must be 512 on Strix Halo Vulkan (issue #18725).  "
            "Bug: Do NOT use Q6_K_XL — broken architecture detection."
        ),
    ),

    # ── GLM 4.7 Flash  ───────────────────────────────────────────────────────
    # ~30 GB at Q8 → ~58 GB for KV + overhead → plenty of room
    ModelConfig(
        name="GLM 4.7 Flash (Q8_K_XL)",
        alias="glm-4.7-flash-q8",
        hf_repo="unsloth/GLM-4.7-Flash-GGUF",
        dest_dir=MODELS_DIR / "unsloth/GLM-4.7-Flash-GGUF",
        download_include="GLM-4.7-Flash-Q8_K_XL.gguf",
        shard_glob="*Q8_K_XL*.gguf",
        quant="Q8_K_XL",
        parallel_slots=1,
        max_parallel=4,
        ctx_per_slot=32768,
        spec=SpecConfig(strategy="ngram"),
        extra_args=["--repeat-penalty", "1.0", "--min-p", "0.01"],
        notes=(
            "Best for: code + chat at high speed, interleaved thinking.  "
            "MoE 30B (3B active).  Best 30B model on SWE-Bench + GPQA.  "
            "~30 GB at Q8 — near-lossless, fits easily.  200K context.  "
            "Bug: needs --repeat-penalty 1.0 and --min-p 0.01."
        ),
    ),

    # ── Qwen3.5 35B  ─────────────────────────────────────────────────────────
    # ~48 GB at Q8_K_XL → ~40 GB for KV + overhead
    ModelConfig(
        name="Qwen3.5 35B (Q8_K_XL)",
        alias="qwen3.5-35b-q8",
        hf_repo="unsloth/Qwen3.5-35B-A3B-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Qwen3.5-35B-A3B-GGUF",
        download_include="*UD-Q8_K_XL*",
        shard_glob="*UD-Q8_K_XL*.gguf",
        quant="UD-Q8_K_XL",
        parallel_slots=6,
        max_parallel=10,
        ctx_per_slot=32768,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: reasoning, summarization, general-purpose.  "
            "MoE 35B (3B active).  Thinking mode on by default.  "
            "~48 GB at Q8_K_XL — high quality.  Multimodal."
        ),
    ),

    # ── Mistral Small 4  ─────────────────────────────────────────────────────
    # ~50–60 GB at Q4 → ~28–38 GB for KV + overhead
    ModelConfig(
        name="Mistral Small 4 (Q4_K_M)",
        alias="mistral-small-4-q4",
        hf_repo="unsloth/Mistral-Small-4-119B-2603-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Mistral-Small-4-119B-2603-GGUF/UD-Q4_K_M",
        download_include="UD-Q4_K_M/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="UD-Q4_K_M",
        parallel_slots=1,
        max_parallel=4,
        ctx_per_slot=32768,
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
            "Best for: all-round chat + code, tool calling.  "
            "MoE 119B (6.5B active).  Largest model that fits well at Q4.  "
            "Only model where draft-model speculation helps on UMA."
        ),
    ),

    # ── Nemotron 3 Super  ────────────────────────────────────────────────────
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
        ctx_per_slot=16384,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: long-context reasoning, multi-agent workflows.  "
            "Hybrid Mamba2-Transformer MoE 120B (12B active).  "
            "Natively supports 1M context (limited here by memory)."
        ),
    ),

    # ── Nemotron Nano Q4  ────────────────────────────────────────────────────
    # ~6 GB at Q4 → ~82 GB for KV + overhead → max parallelism
    ModelConfig(
        name="Nemotron Nano (Q4_K_M)",
        alias="nemotron-nano-q4",
        hf_repo="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        dest_dir=MODELS_DIR / "nvidia/nemotron-nano",
        download_include="*Q4_K_M*.gguf",
        shard_glob="*Q4_K_M*.gguf",
        quant="Q4_K_M",
        parallel_slots=8, # 116 tok/s combined
        max_parallel=12,
        ctx_per_slot=65536,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: speed, lightweight tasks, quick iteration.  "
            "MoE 30B (3B active).  Fastest model in catalog (~60+ tok/s).  "
            "Good for drafting, quick Q&A, low-latency tool calls."
        ),
    ),

    # ── Nemotron Nano Q8  ────────────────────────────────────────────────────
    # ~12 GB at Q8 → ~76 GB for KV + overhead
    ModelConfig(
        name="Nemotron Nano (UD-Q8_K_XL)",
        alias="nemotron-nano-q8",
        hf_repo="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        dest_dir=MODELS_DIR / "nvidia/nemotron-nano",
        download_include="*UD-Q8_K_XL*.gguf",
        shard_glob="*UD-Q8_K_XL*.gguf",
        quant="UD-Q8_K_XL",
        parallel_slots=8, # 83 tok/s combined
        max_parallel=10,
        ctx_per_slot=49152,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: speed/quality balance, lightweight tasks.  "
            "MoE 30B (3B active).  ~45+ tok/s at Q8.  "
            "Good for drafting, quick Q&A, low-latency tool calls."
        ),
    ),
]


# ── helpers ──────────────────────────────────────────────────────────────────

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
        f"Unknown model '{name_or_alias}'.  Available: "
        + ", ".join(m.alias for m in MODELS)
    )
