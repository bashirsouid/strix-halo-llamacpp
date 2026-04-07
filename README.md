# strix-halo-llamacpp

Container-based local LLM launcher, benchmark suite, and eval harness for the AMD Strix Halo family.

The project pulls prebuilt `llama.cpp` container images from `kyuz0/amd-strix-halo-toolboxes`, downloads GGUF models from Hugging Face, and serves them through the OpenAI-compatible `llama-server` API.

## What it does

- launches models with tuned defaults from `models.py`
- supports `radv`/`vulkan`, `amdvlk`, `rocm`, `rocm6`, `rocm7`, and `rocm7-nightly`
- benchmarks single-request and concurrent throughput
- sweeps `--parallel` values to find the best aggregate throughput
- runs EvalPlus against the local server
- includes shell entrypoints for quick start and watch mode
- ships a pytest suite that covers Python helpers and shell entrypoints

## Requirements

- Linux
- Docker available on `PATH`
- Python 3.10+
- enough disk space for your chosen GGUF models

## Quick start

Interactive mode:

```bash
./start.sh
```

Explicit model and backend:

```bash
python server.py download-images
python server.py download qwen3-coder-next-q6
python server.py serve qwen3-coder-next-q6 --backend radv
```

Auto-restart if the launcher notices the server stopped:

```bash
./watch.sh --backend radv qwen3-coder-next-q6
```

Stop the running server:

```bash
./stop.sh
# or
python server.py stop
```

## Configuration

Optional environment variables can be placed in `.env`.

Example:

```bash
API_KEY=replace-me
HF_REVISION=main
```

`server.py` loads `.env` on startup without overwriting variables that are already set in the environment.

## Core commands

List models:

```bash
python server.py list
```

Serve a model:

```bash
python server.py serve MODEL --backend radv
python server.py serve MODEL --backend rocm --np 4 --ctx-per-slot 32768
```

Download a model only:

```bash
python server.py download MODEL
```

Pull all backend images:

```bash
python server.py download-images
```

Benchmark the currently running server or a specific model:

```bash
python server.py bench
python server.py bench MODEL --backend radv
```

Sweep parallelism:

```bash
python server.py bench-parallel MODEL --backend radv --max-np 8
```

Run EvalPlus:

```bash
python server.py eval MODEL --suite humaneval --backend radv
python server.py eval-all --suite humaneval --backend rocm
```

Dry-run launcher checks without touching a live server:

```bash
python server.py test --dry-run --sequential
```

## Backends

All backends run in containers.

- `vulkan` / `radv`: `docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv`
- `amdvlk`: `docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-amdvlk`
- `rocm`: `docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-nightly`
- `rocm6`: `docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-6.4.4`
- `rocm7`: `docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2`
- `rocm7-nightly`: `docker.io/kyuz0/amd-strix-halo-toolboxes:rocm7-nightlies`

There is no native `build` step in the current launcher flow. Pull the container images, then `serve`.

## Models

Models are defined in `models.py`.

Each model can specify:

- `parallel_slots`
- `max_parallel`
- `ctx_per_slot`
- speculation settings
- cache and reasoning flags
- backend-independent launch defaults

Use `python server.py list` to inspect the current catalog.

## API

The server exposes the OpenAI-compatible API on localhost.

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-next-q6",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }'
```

## Testing

Run the full automated suite:

```bash
pytest -q
```

Run only the shell entrypoint coverage:

```bash
pytest tests/test_entrypoints.py -v
```

Run the manual live inference smoke test against an already-running local server:

```bash
STRIX_RUN_LIVE_INFERENCE=1 pytest tests/test_inference.py -m integration -v
```

Or use the helper script:

```bash
./test.sh
```

More details live in [TESTING.md](TESTING.md).
