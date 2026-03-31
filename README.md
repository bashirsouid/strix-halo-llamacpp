# strix-halo-llamacpp

Local llama.cpp launcher for **AMD Strix Halo** (Ryzen AI Max / gfx1151),
optimized for maximum tokens-per-second on the integrated GPU.

Builds llama.cpp from source with Vulkan cooperative-matrix shaders,
downloads models from Hugging Face, and serves them as an OpenAI-compatible
API with per-model performance tuning.

## Hardware target

| Item       | Value                                 |
|------------|---------------------------------------|
| SoC        | AMD Ryzen AI Max 395 (Strix Halo)     |
| GPU        | Radeon 8060S (gfx1151)                |
| UMA pool   | ~90 GB mapped to GPU                  |
| Bandwidth  | ~215 GB/s shared CPU+GPU              |
| GPU driver | Mesa RADV (Vulkan) — recommended      |

## Quick start

```bash
# 1. Install build dependencies
# Fedora:
sudo dnf install cmake ninja-build gcc-c++ vulkan-headers vulkan-loader-devel glslang shaderc spirv-tools git

# Ubuntu/Debian:
sudo apt install cmake ninja-build build-essential libvulkan-dev glslang-tools glslc spirv-tools git

# 2. Install Python dependencies
pip install huggingface_hub hf_transfer

# 3. Build llama.cpp from source (one-time, ~2 min)
python server.py build

# 4. Serve a model (shows interactive picker, or specify by name)
python server.py serve                    # interactive picker
python server.py serve qwen3-coder       # by name (substring match)

# 5. Stop the server
python server.py stop
```

## Recommended models

All models are MoE — they only activate a fraction of their parameters per
token, which is ideal for the Strix Halo's 215 GB/s bandwidth wall.  Ordered
by what they're best at:

| Model | Quant | Size | Best for | tok/s (est.) |
|-------|-------|------|----------|:------------:|
| Qwen3 Coder Next | Q6_K | ~62 GB | Coding agents, tool calling | ~40 |
| GLM 4.7 Flash | Q8_K_XL | ~30 GB | Code + chat, interleaved thinking | ~55 |
| Qwen3.5 35B | Q8_K_XL | ~48 GB | Reasoning, summarization, vision | ~50 |
| Mistral Small 4 | Q4_K_M | ~57 GB | All-round chat + code | ~15 |
| Nemotron 3 Super | Q4_K_M | ~63 GB | Long-context reasoning, multi-agent | ~20 |
| Nemotron Nano | Q4_K_M | ~17 GB | Speed, quick iteration, drafting | ~60 |

**If you only download one:** Qwen3 Coder Next for coding, GLM 4.7 Flash for
everything else.  Both are fast (3B active params) and fit comfortably in
your 90 GB GPU allocation.

## Commands

| Command                    | Description                            |
|----------------------------|----------------------------------------|
| `python server.py build`   | Clone + build llama.cpp with Vulkan    |
| `python server.py list`    | Show all models + download status      |
| `python server.py serve MODEL` | Download (if needed) + launch      |
| `python server.py stop`    | Stop the running server                |
| `python server.py bench`   | Benchmark the currently running server |
| `python server.py bench MODEL` | Start model, benchmark, stop       |
| `python server.py bench-all`  | Benchmark all downloaded models     |
| `python server.py download MODEL` | Download without serving         |

### Serve options

```bash
python server.py serve qwen3-coder-next-q6 \
    --port 8000 \
    --ctx 32768 \
    --threads 4 \
    --backend radv \        # or amdvlk
    --no-spec \             # disable speculation
    --extra -- --verbose    # extra flags to llama-server
```

## Performance tuning notes

### Why build from source?

Native Vulkan builds with cooperative-matrix kernels (`VK_KHR_cooperative_matrix`)
consistently outperform packaged runtimes.  Community benchmarks show ~45–50 tok/s
on 120B MoE models with a local build vs ~30 tok/s from Docker/packaged binaries.
The difference comes from shader paths, reduced IPC overhead, and tuned flags.

### Speculation strategies

Different models benefit from different speculation approaches on UMA:

| Model              | Active params | Strategy        | Why                                |
|--------------------|:------------:|-----------------|------------------------------------|
| Qwen3 Coder Next   | 3B           | ngram only      | Tiny active params, fast already   |
| GLM 4.7 Flash      | 3B           | ngram only      | Same — n-gram helps on code output |
| Qwen3.5 35B        | 3B           | ngram only      | Fast, n-gram helps on reasoning    |
| Mistral Small 4    | 6.5B         | draft + ngram   | Small enough for draft to help     |
| Nemotron 3 Super   | 12B          | ngram only      | Has MTP heads (not yet supported)  |
| Nemotron Nano      | 3B           | none            | Already ~60 tok/s, not worth it    |

**N-gram self-speculation** (`--spec-type ngram-mod`) drafts from patterns already
in the context window.  It uses zero extra memory and zero extra bandwidth — ideal
for UMA systems like Strix Halo.  Particularly effective for code generation and
any output with repetitive structure.

**Draft-model speculation** uses a separate small model.  On UMA, the draft model
competes for the same memory bandwidth as the target.  Only worthwhile when the
target's active parameters are very small (e.g., Mistral Small 4's 6.5B active).

### Key flags

All models launch with these tuned defaults:

- `--no-mmap` — prevents lazy loading that causes Vulkan slowdowns
- `--flash-attn` — essential for long-context performance
- `--cache-type-k q8_0 --cache-type-v q8_0` — saves memory for larger context
- `-b 4096 -ub 256` — batch/ubatch sizes tuned for Strix Halo
- `-t 4` — CPU threads (Vulkan does the heavy compute on GPU)
- `-ngl 999` — offload all layers to GPU

### Backend choice: RADV vs AMDVLK

Both backends are Vulkan — the difference is the GPU driver. RADV is the
open-source Mesa driver; AMDVLK is AMD's official one. You switch between
them with `--backend radv` or `--backend amdvlk`.

**RADV** (the default) is the safer all-round choice. It's good at both
prompt processing and token generation, and it handles long contexts well.

**AMDVLK** generates tokens 5–16% faster once it gets going. The tradeoff
is twofold:

1. **Slower prompt processing.** Before the model starts generating, it has
   to ingest your entire prompt. AMDVLK is significantly slower at this
   step (~358 tok/s vs ~500+ tok/s on RADV in one benchmark). So if you
   paste in a long document or use a big system prompt, you'll wait longer
   for the first token to appear. Once generation starts, AMDVLK is faster.

2. **2 GiB single-buffer limit.** AMDVLK cannot allocate a single
   contiguous block of GPU memory larger than 2 GB. This is a driver
   limitation. In practice your sharded MoE models are fine because
   individual layers are under 2 GB. But a single-file dense 70B model
   or a very large KV cache allocation could fail to load on AMDVLK
   while RADV handles it without issue.

**Rule of thumb:** For interactive chat with short prompts where you want
the fastest streaming response, try AMDVLK. For anything with long prompts
(RAG, document analysis, large system prompts), stick with RADV.

### Thread count

`-t 4` is the default. Token generation is GPU-bound so thread count barely matters
for tg.  Prompt processing can benefit from more threads on the CPU side.  If you
see CPU usage during generation (check `htop`), something else is wrong — likely
mmap is on or not all layers are offloaded.  The `-t 1` approach works too; the
difference is small when all layers are on GPU.

### Nemotron 3 Super and MTP

Nemotron 3 Super has built-in Multi-Token Prediction (MTP) heads that enable
self-speculative decoding without a draft model.  However, llama.cpp does not yet
support MTP inference — those layers are currently skipped when loading the model.
There are active PRs (e.g., #20700 for Qwen3.5 MTP) but nothing merged for
Nemotron yet.  Until MTP lands, n-gram speculation is the best we can do.  Once
MTP is supported, Nemotron Super should see significant speedups for free.

## Benchmarking

Three ways to benchmark:

```bash
# 1. Test whatever server is currently running
python server.py bench

# 2. Start a specific model, benchmark it, stop it
python server.py bench qwen3-coder-next

# 3. Benchmark ALL downloaded models — generates a comparison report
python server.py bench-all
python server.py bench-all --backend amdvlk   # compare backends
```

`bench-all` starts each downloaded model one by one, runs three prompts
(short/medium/long), records average tok/s, stops the server, and moves to
the next. At the end it prints a ranked summary and appends the results to
`bench_results.jsonl` so you can track regressions across llama.cpp updates.

## Project layout

```
strix-halo-llamacpp/
├── server.py      # CLI entry point — build, serve, stop, bench, list
├── models.py      # Model catalog — add new models here
├── README.md
└── llama.cpp/     # (created by `build`) cloned source + build artifacts
```

## Adding a new model

Edit `models.py` and append a `ModelConfig` to the `MODELS` list:

```python
ModelConfig(
    name="My New Model",
    alias="my-model",
    hf_repo="user/repo-GGUF",
    dest_dir=MODELS_DIR / "user/repo-GGUF/Q4_K_M",
    download_include="Q4_K_M/*.gguf",
    shard_glob="*-00001-of-*.gguf",   # or "*.gguf" for single file
    ctx_size=32768,
    spec=SpecConfig(strategy="ngram"),  # or None for small models
),
```

## API access

The server is OpenAI-compatible on port 8000:

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-next-q6",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

Works with Open WebUI, Continue, Cursor, or any OpenAI-compatible client.

## Migrating from the old bash scripts

The old `load_*.sh` scripts, `config.env`, `lib.sh`, and `docker-compose*.yml`
files are no longer needed.  The Python rewrite:

- Eliminates ~500 lines of duplicated bash across 10+ loader scripts
- Builds llama.cpp natively (faster than Docker containers)
- Defines all model config as data (not imperative scripts)
- Replaces draft-model speculation with n-gram for models where it doesn't help
- Tunes batch sizes and threads based on community benchmarks
- Curated model list focused on best-in-class for each use case

## Updating llama.cpp

```bash
python server.py build             # pulls latest + rebuilds
python server.py build --rebuild   # clean rebuild from scratch
```

## Future work

Things to revisit as llama.cpp and the ecosystem evolve:

**Multi-Token Prediction (MTP) for Nemotron 3 Super and GLM 4.7.**
Nemotron 3 Super and GLM 4.7 ship with built-in MTP
heads that enable self-speculative decoding without a separate draft model.
llama.cpp currently skips these layers when loading the model.  There is
active development (PR #20700 adds MTP for Qwen3.5, PR #18886 defines an
MTP API) but nothing merged for Nemotron or GLM architectures yet.  Once
MTP support lands, these models should see significant speedups — MTP uses
less than 1% additional memory compared to the bandwidth cost of a
separate draft model.  When this ships, update the spec config for these
models from `strategy="ngram"` to whatever the MTP flag ends up being, and
re-run `bench-all` to measure the improvement.

**Qwen3-Coder-Next Vulkan ubatch regression (issue #18725).**
On Strix Halo with Vulkan/RADV, Qwen3-Coder-Next (and Qwen3-Next models
generally) suffer a prompt-processing performance regression when ubatch
size exceeds 512 — performance roughly halves.  This appears to be
specific to this model architecture on this GPU.  We currently work around
it with `-ub 512` in the model config.  Monitor the upstream issue; when
it's fixed, the model can be switched back to the standard `-ub 256` that
other models use (or higher), which should improve PP throughput.

**NPU (XDNA 2) for prompt processing.**
The Strix Halo's 50 TOPS NPU is detected by Linux (kernel 6.14+) and
ROCm is adding support, but llama.cpp has no XDNA backend.  The only
current NPU path is Lemonade Server SDK on Windows, which is limited to
~8B models with 2K-3K context — not useful for our workloads.  If a
llama.cpp XDNA backend or a hybrid NPU+iGPU inference path emerges for
Linux, it could meaningfully reduce time-to-first-token on long prompts
by offloading prompt processing to the NPU while the iGPU handles
generation.

**AMDVLK discontinuation.**
As of September 2025, AMD has discontinued AMDVLK in favor of focusing on
Mesa RADV.  The driver still works and can still be faster for token
generation, but expect no further updates.  RADV is improving rapidly and
the gap is narrowing.  Eventually the `--backend amdvlk` option may
become irrelevant.

**Eagle-3 speculative decoding.**
Eagle-3 is the current state-of-the-art speculative decoding algorithm
(2-2.5x speedup in other frameworks).  llama.cpp does not support it yet
but there is active discussion (#15902) and a PR in progress (#18039).
Eagle-3 checkpoints are available on Hugging Face for several model
families.  When support lands, this could be a significant speedup for
larger dense models and MoE models with higher active param counts.

## Troubleshooting

**Build fails with missing Vulkan headers** — Install the Vulkan development
packages listed in Quick Start above.

**`vulkaninfo` shows no devices** — Make sure your user is in the `video` and
`render` groups: `sudo usermod -aG video,render $USER` then log out/in.

**Server starts but inference is slow** — Check that `--no-mmap` is set (it is by
default).  Memory mapping causes severe slowdowns with Vulkan on large models.

**Out of memory** — The UMA pool is shared between CPU and GPU.  Close other
applications or try a smaller quant / lower context size.
