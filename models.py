"""
Model catalog for Strix Halo llama.cpp launcher.

Each model is a ModelConfig dataclass.  Add new models by appending to MODELS.
The server reads this list at startup — nothing else to change.
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
            # When ngram is the *only* strategy, use its wider draft window.
            # When combined with a draft model the shared flags are already set above.
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
    quant: str = ""             # quantization level (e.g. "Q4_K_M", "Q6_K", "Q8_K_XL")
    ctx_size: int = 32768
    batch_size: int = 4096      # -b   (logical max batch)
    ubatch_size: int = 256      # -ub  (physical max ubatch)
    threads: int = 4            # -t   (CPU threads)
    spec: SpecConfig = field(default_factory=SpecConfig)
    chat_template_file: str | None = None
    extra_args: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def model_path(self) -> Path | None:
        """Return path to the first shard, or None if not downloaded."""
        matches = sorted(self.dest_dir.glob(self.shard_glob))
        return matches[0] if matches else None

    @property
    def is_downloaded(self) -> bool:
        return self.model_path is not None

    def server_args(self) -> list[str]:
        """Build the full llama-server argument list for this model."""
        model_path = self.model_path
        if model_path is None:
            raise FileNotFoundError(f"Model not downloaded: {self.name}")

        args = [
            "-m",             str(model_path),
            "--host",         "0.0.0.0",
            "--port",         "8000",
            "--ctx-size",     str(self.ctx_size),
            "-ngl",           "999",
            "--no-mmap",
            "--flash-attn",   "on",
            "--parallel",     "1",
            "-a",             self.alias,
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
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

MODELS: list[ModelConfig] = [

    # ── Qwen3 Coder Next  ────────────────────────────────────────────────────
    # BEST FOR: Coding agents, tool calling, agentic workflows
    ModelConfig(
        name="Qwen3 Coder Next (Q6_K)",
        alias="qwen3-coder-next-q6",
        hf_repo="unsloth/Qwen3-Coder-Next-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Qwen3-Coder-Next-GGUF/Q6_K",
        download_include="Q6_K/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="Q6_K",
        ctx_size=32768,
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
    # BEST FOR: Code + chat quality at high speed, interleaved thinking
    ModelConfig(
        name="GLM 4.7 Flash (Q8_K_XL)",
        alias="glm-4.7-flash-q8",
        hf_repo="unsloth/GLM-4.7-Flash-GGUF",
        dest_dir=MODELS_DIR / "unsloth/GLM-4.7-Flash-GGUF",
        download_include="GLM-4.7-Flash-Q8_K_XL.gguf",
        shard_glob="*Q8_K_XL*.gguf",
        quant="Q8_K_XL",
        ctx_size=32768,
        spec=SpecConfig(strategy="ngram"),
        extra_args=[
            "--repeat-penalty", "1.0",
            "--min-p", "0.01",
        ],
        notes=(
            "Best for: code + chat at high speed, interleaved thinking.  "
            "MoE 30B (3B active).  Best 30B model on SWE-Bench + GPQA.  "
            "~30 GB at Q8 — near-lossless, fits easily.  200K context.  "
            "Bug: needs --repeat-penalty 1.0 and --min-p 0.01 (set automatically)."
        ),
    ),

    # ── Qwen3.5 35B  ─────────────────────────────────────────────────────────
    # BEST FOR: Reasoning, summarization, general-purpose with thinking mode
    ModelConfig(
        name="Qwen3.5 35B (Q8_K_XL)",
        alias="qwen3.5-35b-q8",
        hf_repo="unsloth/Qwen3.5-35B-A3B-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Qwen3.5-35B-A3B-GGUF",
        download_include="*UD-Q8_K_XL*",
        shard_glob="*UD-Q8_K_XL*.gguf",
        quant="UD-Q8_K_XL",
        ctx_size=98304, # 32768*3=98304
        extra_args=[
            "-np", "3",
        ],
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: reasoning, summarization, general-purpose. Parallelization enabled (3)"
            "MoE 35B (3B active).  Thinking mode on by default (/no_think to disable).  "
            "~48 GB at Q8_K_XL — high quality.  Multimodal (text + vision)."
        ),
    ),

    # ── Mistral Small 4  ─────────────────────────────────────────────────────
    # BEST FOR: All-round chat + code, tool calling, largest MoE that fits well
    ModelConfig(
        name="Mistral Small 4 (Q4_K_M)",
        alias="mistral-small-4-q4",
        hf_repo="unsloth/Mistral-Small-4-119B-2603-GGUF",
        dest_dir=MODELS_DIR / "unsloth/Mistral-Small-4-119B-2603-GGUF/UD-Q4_K_M",
        download_include="UD-Q4_K_M/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="UD-Q4_K_M",
        ctx_size=32768,
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
    # BEST FOR: Long-context reasoning, multi-agent, 1M context potential
    ModelConfig(
        name="Nemotron 3 Super (Q4_K_M)",
        alias="nemotron-super-q4",
        hf_repo="unsloth/Nemotron-3-Super-120B-A12B-GGUF",
        dest_dir=MODELS_DIR / "nvidia/nemotron-3-super/UD-Q4_K_M",
        download_include="UD-Q4_K_M/*.gguf",
        shard_glob="*-00001-of-*.gguf",
        quant="UD-Q4_K_M",
        ctx_size=16384,
        spec=SpecConfig(strategy="ngram"),
        notes=(
            "Best for: long-context reasoning, multi-agent workflows.  "
            "Hybrid Mamba2-Transformer MoE 120B (12B active).  "
            "Has built-in MTP heads (not yet supported in llama.cpp).  "
            "Natively supports 1M context (limited here by memory)."
        ),
    ),

    # ── Nemotron Nano  ───────────────────────────────────────────────────────
    # BEST FOR: Speed, lightweight tasks, quick iteration. Supports parallelization.
    ModelConfig(
        name="Nemotron Nano (Q4_K_M)",
        alias="nemotron-nano-q4",
        hf_repo="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        dest_dir=MODELS_DIR / "nvidia/nemotron-nano",
        download_include="*Q4_K_M*.gguf",
        shard_glob="*Q4_K_M*.gguf",
        quant="Q4_K_M",
        ctx_size=393216, # 65536*6=393216
        extra_args=[
            "-np", "6",
        ],
        notes=(
            "Best for: speed, lightweight tasks, quick iteration. Parallelization enabled (6)."
            "MoE 30B (3B active).  Fastest model in the catalog (~60+ tok/s)."
            "Good for drafting, quick Q&A, and low-latency tool calls."
        ),
    ),

    # ── Nemotron Nano  ───────────────────────────────────────────────────────
    # BEST FOR: Speed/qualty balance, lightweight tasks, quick iteration. No parallelization.
    ModelConfig(
        name="Nemotron Nano (UD-Q8_K_XL)",
        alias="nemotron-nano-q8",
        hf_repo="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        dest_dir=MODELS_DIR / "nvidia/nemotron-nano",
        download_include="*UD-Q8_K_XL*.gguf",
        shard_glob="*UD-Q8_K_XL*.gguf",
        quant="UD-Q8_K_XL",
        ctx_size=196608, # 65536*3=196608
        extra_args=[
            "-np", "3",
        ],
        notes=(
            "Best for: speed, lightweight tasks, quick iteration. Parallelization enabled (3)."
            "MoE 30B (3B active).  Fastest model in the catalog (~45+ tok/s)."
            "Good for drafting, quick Q&A, and low-latency tool calls."
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
