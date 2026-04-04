# strix-halo-llamacpp

Local LLM launcher and benchmark suite for the **AMD Strix Halo** (Ryzen AI Max / gfx1151) integrated GPU.
Builds llama.cpp from source, downloads GGUF models from Hugging Face, and serves them as an **OpenAI-compatible REST API** with hardware-tuned defaults.

Supports **Vulkan (RADV / AMDVLK)** and **ROCm (HIP)** backends side by side — build both and switch at serve time.

## Test Suite

This project includes a comprehensive test suite to validate configuration, model arguments, environment variables, and parallelization settings.

### Quick Test (Dry-Run Mode)
```bash
# Validate configuration without affecting running server
python server.py test --sequential --dry-run

# Test with different parallel slots
python server.py test --np 4 --sequential

# Test with different backend
python server.py test --backend rocm --sequential
```

### Python pytest Tests
```bash
# Run all tests
pytest tests/ -v

# Parallel execution (uses multiple CPU cores)
pytest tests/ -n auto

# Specific test files
pytest tests/test_server.py -v
pytest tests/test_parallelization.py -v
```

For detailed test documentation, see [TESTING.md](TESTING.md).

---

## Hardware target

| Item | Value |
|---|---|
| SoC | AMD Ryzen AI Max 395 (Strix Halo) |
| GPU | Radeon 8060S (gfx1151, RDNA 3.5) |
| RAM | 96 GB LPDDR5X (~90 GB mapped to GPU via GTT) |
| UMA VRAM | 512 MB (display only — all inference runs in GTT) |
| Backends | Vulkan (RADV/AMDVLK) · ROCm (HIP + rocWMMA) |

---

## Quick start

```bash
git clone https://github.com/bashirsouid/strix-halo-llamacpp
cd strix-halo-llamacpp
./start.sh                              # Interactive: pick backend and model
./start.sh --backend rocm               # Backend specified, pick model
./start.sh nemotron-nano-q4             # Model specified, pick backend
./start.sh --backend radv nemotron-nano-q4  # specific model + backend
```

Or step by step:

```bash
python server.py build                    # build llama.cpp with Vulkan
python server.py download nemotron-nano-q4  # download a model
python server.py serve nemotron-nano-q4   # serve it
```

### Configuration

Create a `.env` file (ignored by git) with your settings:

```bash
# API key for server authentication (required for production use)
API_KEY=your-api-key-here

# Optional: LLaMA.cpp version pin (default: HEAD)
# LLAMACPP_REF=b8656
```

Generate an API key with:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Backends — Vulkan vs ROCm

Both backends can coexist in separate build directories (`build-vulkan/` and `build-rocm/`).

| Backend | Flag (build) | Flag (serve) | Best for |
|---|---|---|---|
| Vulkan RADV | `--backend vulkan` | `--backend radv` | **Generation speed** — fast GEMV decode, stable |
| Vulkan AMDVLK | `--backend vulkan` | `--backend amdvlk` | Prompt processing on small models (2 GB buffer limit) |
| ROCm HIP | `--backend rocm` | `--backend rocm` | **Prefill speed** — uses WMMA matrix units via rocWMMA |

Vulkan's GEMM shaders on RDNA 3.5 cannot access the hardware matrix accelerator units, so prefill (prompt processing) is significantly slower than generation.  ROCm's rocWMMA path can use them, giving much better prefill throughput.

```bash
# Build both backends
python server.py build --backend vulkan
python server.py build --backend rocm

# Compare them
python server.py bench nemotron-nano-q4 --backend radv
python server.py bench nemotron-nano-q4 --backend rocm
```

### ROCm build

AMD's ROCm packages have dependency issues on **Debian Trixie** (they require Ubuntu's libstdc++-11-dev).  The build system handles this automatically by extracting pre-built binaries from a container image:

```bash
# Option 1: Container extraction (recommended for Debian Trixie)
# The kyuz0 toolbox images ship with llama.cpp pre-built for gfx1151.
# Just needs podman or docker — binaries are copied out and run natively.
sudo apt install podman
python server.py build --backend rocm    # extracts binary from container

# Option 2: Native hipcc (Fedora, or Ubuntu with ROCm repo)
sudo dnf install rocm-dev hipcc          # Fedora
python server.py build --backend rocm    # builds from source with native hipcc
```

The container approach uses the [kyuz0 Strix Halo toolbox](https://github.com/kyuz0/amd-strix-halo-toolboxes) image (`rocm-7.2`) which is rebuilt on every llama.cpp commit.  The `build` step just pulls the image.  At serve time, `llama-server` runs inside the container with your GPU devices and model directory mounted — the API is still on `localhost:8000` and everything else (benchmarks, TUI, model management) runs natively.  Run `python server.py build --backend rocm --rebuild` to pull the latest image.

---

## Model catalog

Models are defined in `models.py` with per-model defaults tuned for 90 GB GPU memory.

| Model | Params | Active | Quant | Weight ~GB | np | ctx/slot |
|---|---|---|---|---|---|---|
| Qwen3 Coder 30B A3B | 30.5B MoE | 3.3B | Q8_0 | ~32.5 | 1 | 256K |
| Qwen3 Coder Next | 80B MoE | 3B | Q6_K | ~62 | 1 | 262K |
| GLM 4.7 Flash | 30B MoE | 3B | Q8_K_XL | ~30 | 1 | 32K |
| Qwen3.5 35B Throughput | 35B MoE | 3B | Q8_K_XL | ~48 | 3 | 32K |
| Qwen3.5 35B LongCtx | 35B MoE | 3B | Q8_K_XL | ~48 | 1 | 256K |
| Mistral Small 4 Tool/Code | 119B MoE | 6.5B | Q4_K_M | ~55 | 1 | 32K |
| Mistral Small 4 Reasoning | 119B MoE | 6.5B | Q4_K_M | ~55 | 1 | 128K |
| Nemotron 3 Super Reasoning | 120B MoE | 12B | Q4_K_M | ~55 | 1 | 64K |
| Nemotron Nano Q4 Router | 30B MoE | 3B | Q4_K_M | ~6 | 8 | 32K |
| Nemotron Nano Q4 LongCtx | 30B MoE | 3B | Q4_K_M | ~6 | 1 | 256K |
| Nemotron Nano Q8 Router | 30B MoE | 3B | Q8_K_XL | ~12 | 8 | 32K |
| Nemotron Nano Q8 LongCtx | 30B MoE | 3B | Q8_K_XL | ~12 | 1 | 256K |

### Parallelization

Parallelization (`--parallel` / `-np`) is a first-class setting per model.  Each model has:

- **`parallel_slots`** — number of concurrent request slots (used by default)
- **`ctx_per_slot`** — context window per slot (total context = ctx_per_slot × parallel_slots)
- **`max_parallel`** — upper bound for the `bench-parallel` sweep

The `ctx_per_slot` approach keeps each request's usable context constant regardless of how many slots are active.  The model's training context limit applies to each slot independently — total context going above the model's max is fine.

### Speculative decoding

N-gram speculation is enabled by default on most models.  Mistral Small 4 additionally uses a draft model for draft+ngram speculation.

---

## CLI reference

```
python server.py build    [--backend vulkan|radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly] [--rebuild]
python server.py list
python server.py serve    [MODEL] [--backend radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly] [--np N] [--ctx-per-slot N] ...
python server.py stop
python server.py bench    [MODEL] [--backend radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly]
python server.py bench-all          [--backend radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly]
python server.py bench-parallel [MODEL] [--backend radv|amdvlk|rocm|rocm6|rocm7|rocm7-nightly] [--max-np N]
python server.py download [MODEL]
```

### `serve` options

| Flag | Description |
|---|---|
| `--backend radv\|amdvlk\|rocm\|rocm6\|rocm7\|rocm7-nightly` | Backend to use (default: radv) |
| `--np N` | Override parallel slots |
| `--ctx N` | Override total context size |
| `--ctx-per-slot N` | Override context per slot (total = this × np) |
| `--threads N` / `-t N` | Override CPU thread count |
| `--no-spec` | Disable speculative decoding |
| `--verbose` / `-v` | Show full llama-server output |
| `--extra ...` | Extra args passed to llama-server |

---

## Benchmarking

### Single model benchmark

Tests both **generation** (short prompt → long output) and **prefill** (long prompt → short output) in a single run, reporting both speeds plus a combined score.

```bash
python server.py bench nemotron-nano-q4 --backend radv
```

Output:
```
  ── Generation (short prompt → 512 tok output) ──────────────
  [gen-1]   512 tok in    8.5s  →   60.2 tok/s  (prompt: 12 tok)
  [gen-2]   512 tok in    8.6s  →   59.5 tok/s  (prompt: 18 tok)
  [gen-3]   512 tok in    8.9s  →   57.5 tok/s  (prompt: 42 tok)

  ── Prefill (long prompt → 16 tok output) ───────────────────
  [pp-1 ]    526 prompt tok in   52.6s  →   10.0 pp tok/s  (output: 16 tok)
  [pp-2 ]   1007 prompt tok in  100.7s  →   10.0 pp tok/s  (output: 16 tok)

  ═══════════════════════════════════════════════
  Generation:    59.1 tok/s  (decode throughput)
  Prefill:       10.0 tok/s  (prompt processing)
  Combined:      34.5 tok/s  (average of both)
  ═══════════════════════════════════════════════
```

### Benchmark all models

Runs the full generation + prefill benchmark for every downloaded model:

```bash
./benchmark-run-all.sh --backend radv
./benchmark-run-all.sh --backend rocm    # compare backends
```

### Parallel sweep

Sweeps `--parallel` from 1 to `max_parallel`, measuring both single-request and concurrent throughput at each level.  Finds the optimal slot count for your hardware.

```bash
./benchmark-parallel.sh nemotron-nano-q4 --backend radv
./benchmark-parallel.sh nemotron-nano-q4 --backend rocm --max-np 10
```

### Visualizers

Open interactive HTML charts in your browser:

```bash
./benchmark-view.sh              # bench_results.jsonl → bench_report.html
./benchmark-parallel-view.sh     # bench_parallel_results.jsonl → parallel_report.html
```

---

## Shell scripts

| Script | Purpose |
|---|---|
| `start.sh` | One-command bootstrap: install deps, build, serve |
| `stop.sh` | Stop the running server |
| `load_nemotron_nano_q4.sh` | Shortcut to serve Nemotron Nano Q4 |
| `load_nemotron_nano_q8.sh` | Shortcut to serve Nemotron Nano Q8 |
| `load_glm4.7_flash_q8.sh` | Shortcut to serve GLM 4.7 Flash Q8 |
| `load_qwen3.5_35b_q8.sh` | Shortcut to serve Qwen3.5 35B Q8 |
| `benchmark-run.sh` | Run bench on running/specified model |
| `benchmark-run-all.sh` | Benchmark all downloaded models |
| `benchmark-parallel.sh` | Parallel sweep |
| `benchmark-view.sh` | Open bench results chart |
| `benchmark-parallel-view.sh` | Open parallel sweep chart |

All shell scripts pass through arguments, so `./benchmark-run-all.sh --backend rocm` works.

---

## Memory configuration (Strix Halo)

This setup is optimized for **GTT-only operation** with minimal VRAM aperture:

| Setting | Value | Why |
|---|---|---|
| BIOS UMA Frame Buffer | 512 MB | Just enough for display output |
| GTT mapped to GPU | ~90 GB | All model weights + KV cache live here |
| Kernel boot params | `iommu=pt amdgpu.gttsize=92160 ttm.pages_limit=23592960` | Maps 90 GB of system RAM to GPU GTT |
| `--no-mmap` | Always set | Critical — mmap causes severe slowdowns on UMA |
| `--flash-attn on` | Always set | Reduces KV cache memory, stable at large context |
| `--cache-type-k q8_0` | Default | 50% less KV memory vs f16 with minimal quality loss |

### Important: do NOT use `RADV_PERFTEST=nogttspill`

Since all inference runs in GTT (not VRAM), preventing GTT spilling would be counterproductive.  The standard RADV memory management handles GTT-only operation correctly.

---

## Environment variables

Set automatically per backend by `server.py`:

### Vulkan (RADV)
```
AMD_VULKAN_ICD=RADV
HSA_ENABLE_SDMA=0
GGML_VK_PREFER_HOST_MEMORY=0
```

### Vulkan (AMDVLK)
```
AMD_VULKAN_ICD=AMDVLK
HSA_ENABLE_SDMA=0
GGML_VK_PREFER_HOST_MEMORY=0
```

### ROCm (HIP)
```
HSA_ENABLE_SDMA=0
ROCBLAS_USE_HIPBLASLT=1
HIP_VISIBLE_DEVICES=0
```

---

## API access

Once the server is ready, the endpoint is OpenAI-compatible at port 8000:

```bash
# List loaded models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-nano-q4",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }'
```

Works as a drop-in replacement for the OpenAI API in any compatible client (Open WebUI, Continue, Cursor, Claude Code, etc.).

---

## Adding new models

Add a new `ModelConfig` to the `MODELS` list in `models.py`:

```python
ModelConfig(
    name="My Model (Q4_K_M)",
    alias="my-model-q4",
    hf_repo="owner/repo-GGUF",
    dest_dir=MODELS_DIR / "owner/repo",
    download_include="*Q4_K_M*.gguf",
    shard_glob="*Q4_K_M*.gguf",
    quant="Q4_K_M",
    parallel_slots=4,        # start conservative, tune with bench-parallel
    max_parallel=8,
    ctx_per_slot=32768,
    spec=SpecConfig(strategy="ngram"),
)
```

Then tune: `python server.py bench-parallel my-model-q4`

---

## Troubleshooting

**Server fails to start with ROCm:**
Ensure `/dev/kfd` and `/dev/dri` are accessible and your user is in the `video` and `render` groups.

**Very slow first run:**
llama.cpp performs a warm-up pass on startup.  Normal for large models — takes 30–90s.

**Out of memory:**
Close other RAM-heavy applications.  Try a lower quant or reduce `ctx_per_slot`.

**Prefill much slower than generation (Vulkan):**
This is expected — Vulkan's GEMM shaders can't use RDNA 3.5 matrix units.  Build and test with `--backend rocm` for better prefill.  See the benchmarking section.

**GLM 4.7 Flash hangs with `--parallel > 1`:**
Known issue — concurrent generation collapses to ~3 tok/s.  Keep this model at `parallel_slots=1`.
