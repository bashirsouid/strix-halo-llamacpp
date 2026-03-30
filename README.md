# strix-halo-llamacpp

Personal llama.cpp launcher for the **AMD Strix Halo** (Ryzen AI Max / gfx1151) integrated GPU.
Serves any supported GGUF model as an **OpenAI-compatible REST API** over Docker Compose, using kyuz0's pre-built Vulkan RADV toolbox containers.

Speculative decoding is opt-in per loader — currently enabled for `Mistral-Small-4-119B` using a small draft model.

***

## Hardware target

| Item | Value |
|---|---|
| SoC | AMD Ryzen AI Max (Strix Halo) |
| GPU | Radeon 8060S (gfx1151) |
| GPU driver | Mesa RADV (Vulkan) |
| VRAM mode | UMA — system RAM is GPU RAM |

***

## Prerequisites

- Linux (Fedora 42 / Ubuntu 24.04 recommended)
- Docker + Docker Compose plugin
- User in the `video` group (`sudo usermod -aG video $USER`)
- `huggingface-hub` CLI (for loaders that download models)
  ```bash
  pip install -U huggingface-hub hf-transfer
  ```

***

## Quick start

```bash
git clone https://github.com/bashirsouid/strix-halo-llamacpp
cd strix-halo-llamacpp
./start.sh          # loads DEFAULT_LOADER from config.env
```

The server is ready when `start.sh` prints **"Server is ready"** and shows the access URLs.
The container keeps running after the script exits.

```bash
./stop.sh           # stops and removes both containers
```

***

## Repository layout

```
strix-halo-llamacpp/
│
├── start.sh                    # Entry point — runs DEFAULT_LOADER
├── stop.sh                     # Stops and removes all containers
├── lib.sh                      # Shared helpers (sourced by every loader/script)
├── config.env                  # Global settings + DEFAULT_LOADER
│
├── docker-compose.yml          # Base compose — no speculative decoding
├── docker-compose.spec.yml     # Spec compose — adds --model-draft flags
│
├── load_mistral-small-4_q4.sh  # Mistral-Small-4-119B  Q4_K_M (spec enabled)
├── load_mistral-small-4_q3.sh  # Mistral-Small-4-119B  Q3_K_M
├── load_deepseek-coder-v2-lite.sh
├── load_devstral-small-2.sh
├── load_glm_4.5_air.sh
├── load_llama33_70b.sh
├── load_nemotron_nano.sh
├── load_nemotron_super.sh
├── load_qwen_chat_7b.sh
├── load_qwen_coder_7b.sh
│
├── scan_models.sh              # Discovers all GGUFs on disk
└── bench_current.py            # Quick throughput benchmark
```

***

## Configuration

### `config.env` — global knobs

```bash
# Which loader to run when you call ./start.sh
DEFAULT_LOADER=load_mistral-small-4_q4.sh

# Docker image tag (swap to switch backend)
LLAMA_IMAGE_TAG=vulkan-radv
# LLAMA_IMAGE_TAG=vulkan-amdvlk
# LLAMA_IMAGE_TAG=rocm-6.4.4-rocwmma

# Where models live on the host
MODELS_DIR=/mnt/data/models
HOST_LLAMA_CACHE=/mnt/data/llama.cpp-cache

# Server port
SERVER_PORT=8000

# Default inference tuning (loaders can override)
LLAMA_NGL=999
LLAMA_THREADS=1
```

> Per-model settings (HF repo, quant file, alias, context size) live **entirely** in each loader — not here.

***

## Loader scripts

Every `load_*.sh` is a self-contained script that:

1. Sources `lib.sh` and `config.env`
2. Calls `clear_draft_config` to reset any state from a previous run
3. Downloads the model shards if not already on disk
4. Exports `MODEL_FLAG` / `MODEL_VALUE` (and optionally draft variables)
5. Calls `launch_server` and `wait_for_server`

Run a loader directly to switch models without restarting the whole shell:

```bash
./load_nemotron_nano.sh
```

### Switching models

Edit `DEFAULT_LOADER` in `config.env`, then run `./start.sh`.  
Or call any loader directly — it will stop the running container first.

***

## Speculative decoding

Speculative decoding is configured **per loader**. Only `load_mistral-small-4_q4.sh` currently enables it.

### How it works

| File | Role |
|---|---|
| `docker-compose.yml` | Base — used by all loaders that do not set `LLAMA_COMPOSE_FILE` |
| `docker-compose.spec.yml` | Adds `--model-draft`, `--draft-max`, `--draft-min`, `--ubatch-size` |

A loader opts in by exporting these variables before calling `launch_server`:

```bash
export DRAFT_MODEL_PATH="/path/to/draft.gguf"
export DRAFT_MAX=8
export DRAFT_MIN=2
export LLAMA_COMPOSE_FILE="docker-compose.spec.yml"
```

If `DRAFT_MODEL_PATH` is not set (or the file is missing), the loader falls back to the base compose automatically via `clear_draft_config`.

### Enabling speculative decoding for a new model

1. Add a draft-model download block to the loader (follow the pattern in `load_mistral-small-4_q4.sh`).
2. Export `DRAFT_MODEL_PATH`, `DRAFT_MAX`, `DRAFT_MIN`, and `LLAMA_COMPOSE_FILE`.
3. That's it — `launch_server` picks up `LLAMA_COMPOSE_FILE` and uses the spec compose.

### Keeping a model non-speculative

Just call `clear_draft_config` at the top of the loader (as all loaders already do) and **do not** set `LLAMA_COMPOSE_FILE`. The base `docker-compose.yml` is used automatically.

***

## Docker backend images

Containers are built and published by [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) and rebuilt automatically on every llama.cpp master commit.

| `LLAMA_IMAGE_TAG` | Backend | Notes |
|---|---|---|
| `vulkan-radv` | Vulkan (Mesa RADV) | Stable default; recommended starting point |
| `vulkan-amdvlk` | Vulkan (AMDVLK) | Sometimes faster; 2 GiB single-buffer limit can block large models |
| `rocm-6.4.4-rocwmma` | ROCm 6.4.4 + ROCWMMA | Best for BF16; requires `/dev/kfd`; occasional instability |
| `rocm-6.4.4` | ROCm 6.4.4 | Stable ROCm baseline without ROCWMMA |

Pull the latest image for your tag at any time:

```bash
docker pull docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv
```

***

## API access

Once the server is ready the endpoint is OpenAI-compatible at port `8000`:

```bash
# List loaded models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-4-119b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }'
```

Works as a drop-in replacement for the OpenAI API in any compatible client (Open WebUI, Continue, Cursor, etc.).

***

## Benchmarking

`bench_current.py` runs a quick decode-throughput test against the running server:

```bash
python bench_current.py
```

It exercises several prompt types and reports median TTFT and tok/s per run.
Best suited as a regression check ("did this change make things faster?") rather than a rigorous speculative-decoding evaluation.

> For a more accurate speculative-decoding comparison, run the benchmark once with `docker-compose.spec.yml` active and once without, using longer output lengths (512+ tokens) and streaming measurement.

***

## `lib.sh` helper reference

Sourced by every script. Key exports:

| Symbol | Type | Description |
|---|---|---|
| `clear_draft_config` | function | Unsets `DRAFT_MODEL_PATH`, `DRAFT_MAX`, `DRAFT_MIN`, `LLAMA_COMPOSE_FILE` |
| `launch_server [alias]` | function | Runs `docker compose up` with the active compose file |
| `wait_for_server` | function | Polls `/v1/models`, streams logs, prints access info on success |
| `_build_search_dirs` | function | Populates `_SEARCH_DIRS` for `scan_models.sh` |
| `_lib_info/ok/warn/fail` | functions | Colour log helpers |

***

## Scanning for local models

```bash
./scan_models.sh
```

Searches all known model directories on the host (HF cache, LM Studio, Ollama, `/mnt/data/models`, etc.) and lists every `.gguf` file found.

***

## Troubleshooting

**`error: invalid argument: --speculative`**  
Remove the `--speculative` flag. In this build, `--model-draft <path>` is sufficient to enable speculative decoding — `--speculative` is not a valid argument.

**`gguf_init_from_file: failed to open GGUF file 'http://...'`**  
`--model-draft` expects a local file path, not a URL. The draft model must be downloaded to disk first; `load_mistral-small-4_q4.sh` handles this automatically.

**Container exits immediately, no GPU seen**  
Check that `/dev/dri` is accessible and the user is in the `video` group:
```bash
ls -la /dev/dri
groups $USER
```

**Very slow first run**  
llama.cpp performs a warm-up inference pass on startup. This is normal for large models and takes 30–90 s. Subsequent requests are fast.

**Out-of-memory / model fails to load**  
The Strix Halo UMA pool is shared between CPU and GPU. Close other applications consuming large amounts of RAM, or try a lower-quant variant of the model (e.g., Q3_K_M instead of Q4_K_M).