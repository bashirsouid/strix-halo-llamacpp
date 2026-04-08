# strix-halo-llamacpp

Container-based local LLM launcher, benchmark suite, and eval harness for the AMD Strix Halo family.

The project pulls prebuilt `llama.cpp` container images from `kyuz0/amd-strix-halo-toolboxes`, downloads GGUF models from Hugging Face, and serves them through the OpenAI-compatible `llama-server` API.

## What it does

- launches models with tuned defaults from `models.py`
- supports `radv`/`vulkan`, `amdvlk`, `rocm`, `rocm6`, `rocm7`, and `rocm7-nightly`
- benchmarks single-request and concurrent throughput
- sweeps `--parallel` values to find the best aggregate throughput
- runs Aider's Dockerized code-edit benchmark against the local server
- keeps the older EvalPlus flow available as a legacy smoke test
- exposes repo-aware caching helpers for local coding tools such as OpenCode
- generates a stable per-repo architecture summary that can be reused across many requests
- can warm, save, and restore llama.cpp slot state for one repo at a time
- ships a pytest suite plus bash smoke tests that cover model helpers, shell entrypoints, wrapper scripts, and the top-level `./test.sh` runner

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

Useful cache-related serve flags:

```bash
python server.py serve MODEL --backend radv --cache-reuse 256
python server.py serve MODEL --backend radv --cache-ram 8192
python server.py serve MODEL --backend radv --disable-prompt-cache
python server.py serve MODEL --backend radv --slot-save-path ~/.cache/strix-halo-llamacpp/slots
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

Run the recommended Aider benchmark:

```bash
python server.py aider-setup
python server.py aider-bench MODEL --backend radv --profile python-quick
python server.py aider-bench MODEL --backend radv --profile python-all
python server.py aider-bench-all --backend rocm --profile python-quick
```

The default `python-quick` profile is a fixed harder 9-exercise subset chosen to keep local runs near ~30 minutes while still separating models on parsing, stateful logic, tree/graph handling, and API-style edits. `python-all` is the full 34-exercise Python set from Aider's polyglot benchmark. Every Aider run now uses a fresh benchmark directory, forces a clean rerun, fixes the exercise order with a deterministic seed, injects local model metadata so token-limit warnings stop falling back to `0`, and filters terminal output down to progress plus actionable warnings. The full raw log is saved under `results/aider/logs/`.

Visualize the Aider results:

```bash
python tools/eval_viewer.py
```

Legacy EvalPlus commands are still available, but the repo is now set up for Aider-style code editing benchmarks first:

```bash
python server.py eval MODEL --suite humaneval --backend radv
python server.py eval-all --suite humaneval --backend rocm
```

Dry-run launcher checks without touching a live server:

```bash
python server.py test --dry-run --sequential
```

Repo-aware caching helpers:

```bash
python server.py repo-init --repo ~/code/my-app --model qwen3-coder-next-q6
python server.py repo-refresh --repo ~/code/my-app
python server.py repo-proxy --repo ~/code/my-app
python server.py repo-warm --repo ~/code/my-app
python server.py repo-save --repo ~/code/my-app
python server.py repo-restore --repo ~/code/my-app
python server.py repo-status --repo ~/code/my-app
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

Useful caching fields in `ModelConfig` now include:

- `cache_prompt`
- `cache_reuse`
- `cache_ram`
- `slot_save_path`
- `cache_type_k` / `cache_type_v`

Use `python server.py list` to inspect the current catalog.

## Repo-aware caching and OpenCode workflow

This launcher now includes a stateful workflow aimed at code work on a single repository at a time.

### What gets cached

1. A stable repo summary is generated from files such as `README.md`, `AGENTS.md`, `ARCHITECTURE.md`, key manifests, and a trimmed file tree.
2. The repo summary is injected as the first system message through a tiny local proxy.
3. Requests are pinned to one llama.cpp slot by setting `id_slot`.
4. The slot can be warmed once, then saved to disk and restored later.

The design goal is to keep the reusable architecture context byte-stable so llama.cpp prompt caching can avoid recomputing the same prefix over and over.

### Files written by the repo helper

For a repo at `~/code/my-app`, the helper writes cache files under:

```text
~/.cache/strix-halo-llamacpp/repositories/<repo-slug>/repo-context.md
~/.cache/strix-halo-llamacpp/repositories/<repo-slug>/repo-context.json
```

`repo-init` also writes an `opencode.json` inside the target project so OpenCode can talk to the local proxy.

### Recommended flow

1. Start the local model server.

```bash
python server.py serve qwen3-coder-next-q6 --backend radv
```

2. Generate repo context and OpenCode config.

```bash
python server.py repo-init --repo ~/code/my-app --model qwen3-coder-next-q6
```

3. Run the repo-aware proxy in another terminal.

```bash
python server.py repo-proxy --repo ~/code/my-app
```

4. Warm the repo slot once after each server restart.

```bash
python server.py repo-warm --repo ~/code/my-app
```

5. Start OpenCode from inside the repo.

```bash
cd ~/code/my-app
opencode
```

6. Before shutting down or switching workloads, save the warmed slot.

```bash
python server.py repo-save --repo ~/code/my-app
```

7. After launching the model server again, restore it.

```bash
python server.py repo-restore --repo ~/code/my-app
```

### Notes

- `repo-proxy` listens on `127.0.0.1:8001` by default and forwards to the local `llama-server` on `127.0.0.1:8000`.
- If your upstream local server uses an API key, the proxy forwards that key upstream. The generated `opencode.json` does not need to store secrets.
- The launcher publishes the container port to `127.0.0.1` only. This keeps the slot management endpoints local to the machine.
- Use `repo-refresh` after major refactors so the cached architecture summary stays relevant.

## API

The server exposes the OpenAI-compatible API on localhost only.

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

When slot persistence is enabled, llama.cpp slot save and restore are available through the local `/slots` endpoints. This launcher intentionally binds the server to localhost on the host side so those endpoints are not exposed on the LAN by default.

## Testing

Run the default automated suite:

```bash
pytest -q
```

That default collection now includes `test_models.py`, `test_entrypoints.py`, and everything under `tests/`.

Run the subprocess-based shell and wrapper entrypoint coverage only:

```bash
pytest test_entrypoints.py -v
```

Run the lightweight bash smoke scripts only:

```bash
bash tests/test_start.sh
bash tests/test_bash_entrypoints.sh
```

Run the manual live inference smoke test against an already-running local server:

```bash
STRIX_RUN_LIVE_INFERENCE=1 pytest tests/test_inference.py -m integration -v
```

Or use the helper script:

```bash
./test.sh
```

`./test.sh` runs the bash smoke scripts, the default pytest suite, and the live inference smoke test. It always ends with a single `FINAL RESULT:` line so failures from any sub-step show up in the last line of output.

More details live in [TESTING.md](TESTING.md).
