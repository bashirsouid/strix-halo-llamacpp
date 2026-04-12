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
- auto-starts a transparent proxy on port 8001 for model switching and repo-aware caching while keeping the raw llama-server on port 8000
- can warm, save, and restore model-scoped llama.cpp slot state per repo
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

`python server.py serve ...` and `./start.sh` now do two things automatically:

- keep the raw `llama-server` on `http://127.0.0.1:8000/v1` for debugging, and
- start a transparent proxy on `http://127.0.0.1:8001/v1` for automatic model switching and repo-aware cache management.

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

By default this also starts the transparent proxy on `127.0.0.1:8001`. Use `--no-proxy` when you explicitly want only the raw server.

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
python server.py aider-bench MODEL --backend radv --profile python-quick --threads 1
python server.py aider-bench MODEL --backend radv --profile python-all --threads 3
python server.py aider-bench-all --backend rocm --profile python-quick
```

The default `python-quick` profile is a fixed harder 9-exercise subset chosen to keep local runs near ~30 minutes while still separating models on parsing, stateful logic, tree/graph handling, and API-style edits. `python-all` is the full 34-exercise Python set from Aider's polyglot benchmark. Every Aider run now uses a fresh benchmark directory, forces a clean rerun, fixes the exercise order with a deterministic seed, injects local model metadata so token-limit warnings stop falling back to `0`, and filters terminal output down to progress plus actionable warnings. The full raw log is saved under `results/aider/logs/`.

### Aider parallelism

By default, `python server.py aider-bench ...` now starts from the model's `parallel_slots` value in `models.py`, but it caps the **default eval worker count at 3**. That keeps quality runs from blindly inheriting very high throughput-tuning values that can create long tail latencies or harder-to-debug hangs. If you want another value, pass `--threads` explicitly.

Examples:

```bash
# Use the model's tuned default from models.py, capped at 3 for evals
python server.py aider-bench qwen3-coder-next-q6 --backend radv --profile python-quick

# Force a strict serial run for apples-to-apples comparisons
python server.py aider-bench qwen3-coder-next-q6 --backend radv --profile python-quick --threads 1

# Try a higher-concurrency screen run
python server.py aider-bench qwen3-coder-next-q6 --backend radv --profile python-quick --threads 3

# Let each downloaded model use its own tuned parallel_slots value, capped at 3
python server.py aider-bench-all --backend rocm7 --profile python-quick

# Force the same parallelism across every model in aider-bench-all
python server.py aider-bench-all --backend rocm7 --profile python-quick --threads 2
```

When `--threads` is provided, the launcher starts `llama-server` with matching `--parallel` so the benchmark worker count and server slot count stay aligned. Context still scales as `ctx_per_slot × threads`, just like the rest of the launcher.

### Aider verbose diagnostics

Use `--verbose` when you want to debug a noisy or suspicious run. In verbose mode the wrapper:

- streams the full raw Aider subprocess output back to the terminal,
- keeps writing the normal benchmark log under `results/aider/logs/`, and
- routes requests through a tiny local OpenAI-compatible proxy that writes a second text log with one line per request completion.

That proxy log is especially useful for finalization stalls because it shows when `/v1` requests start and finish, the active request count, and token usage from the upstream response when available.

```bash
python server.py aider-bench MODEL --backend rocm7 --profile python-quick --verbose
```

Aider runs now refresh `eval_report.html` automatically after `aider-bench` and `aider-bench-all`, so the dashboard stays current as new JSONL rows land in `results/aider/aider_results.jsonl`. The viewer keeps `python-quick` and `python-all` variants separate in the charts because the profile is part of each run's series key.

You can still regenerate or open the report manually at any time:

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

Repo-aware caching helpers and manual controls:

```bash
python server.py repo-refresh --repo ~/code/my-app
python server.py repo-save --repo ~/code/my-app --model qwen3-coder-next-udq6xl
python server.py repo-restore --repo ~/code/my-app --model qwen3-coder-next-udq6xl
python server.py repo-status --repo ~/code/my-app --model qwen3-coder-next-udq6xl

# Optional convenience helpers if you want them:
python server.py repo-init --repo ~/code/my-app
python server.py repo-proxy --backend radv
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

## Repo-aware caching, transparent proxy, and OpenCode

The normal flow is now intentionally simple:

1. Start the model once with `./start.sh` or `python server.py serve MODEL --backend ...`.
2. Point your client at `http://127.0.0.1:8001/v1`.
3. Send the repo path with `X-Repo-Path` and send the model alias in the normal OpenAI payload.

That is enough for the proxy to:

- auto-generate stable repo context the first time it sees a repository,
- switch the upstream `llama-server` whenever the requested `model` changes,
- restore the matching repo+model slot snapshot when it exists,
- warm a fresh slot when no snapshot exists yet, and
- save the active slot automatically when the proxy exits, the launcher stops, or traffic moves away from that repo/model pair.

You do **not** need to run `repo-init` or `repo-proxy` for the standard OpenCode workflow anymore. Those commands still exist as manual/debugging helpers.

### What gets cached

1. A stable repo summary is generated from files such as `README.md`, `AGENTS.md`, `ARCHITECTURE.md`, key manifests, and a trimmed file tree.
2. The repo summary is injected as the first system message through the local proxy.
3. Requests are pinned to one llama.cpp slot by setting `id_slot`.
4. Slot snapshots are scoped to **repo slug + exact model alias + slot id**.
5. The proxy saves before leaving a repo/model pair and restores before returning.

The design goal is to keep the reusable architecture context byte-stable so llama.cpp prompt caching can avoid recomputing the same prefix over and over.

### Files written by the repo helper

For a repo at `~/code/my-app`, the helper writes cache files under:

```text
~/.cache/strix-halo-llamacpp/repositories/<repo-slug>/repo-context.md
~/.cache/strix-halo-llamacpp/repositories/<repo-slug>/repo-context.json
~/.cache/strix-halo-llamacpp/slots/<repo-slug>--m_<exact-model-key>--slot0.bin
~/.cache/strix-halo-llamacpp/proxy-state.json
~/.cache/strix-halo-llamacpp/proxy-metrics.jsonl
./.proxy.log
```

### Standard usage

Start the local server and transparent proxy:

```bash
./start.sh
# or
python server.py serve qwen3-coder-next-udq6xl --backend radv
```

After startup:

- raw upstream server: `http://127.0.0.1:8000/v1`
- transparent proxy: `http://127.0.0.1:8001/v1`

The proxy is the endpoint you should give to OpenCode and other coding tools. The raw upstream server stays useful for debugging config issues or talking directly to llama.cpp.

### OpenCode setup

OpenCode supports custom `baseURL`, `headers`, `timeout`, `chunkTimeout`, project-local `opencode.json`, and environment-variable substitution. `repo-init` now writes this shape by default with `qwen3-coder-next-udq6xl` as the main and `small_model`, plus `agent.plan.model = qwen3.5-122b-udq4`, unless you override it with flags. A practical project-local config for the architect/coder split looks like this:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "strix-local": {
      "name": "Strix Halo llama.cpp",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8001/v1",
        "headers": {
          "X-Repo-Path": "{env:PWD}"
        },
        "chunkTimeout": 120000,
        "timeout": 900000
      },
      "models": {
        "qwen3-coder-next-udq6xl": {
          "name": "Qwen3 Coder Next (UD-Q6_K_XL)",
          "limit": {
            "context": 262144,
            "output": 8192
          }
        },
        "qwen3.5-122b-udq4": {
          "name": "Qwen3.5 122B A10B (UD-Q4_K_XL)",
          "limit": {
            "context": 131072,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "strix-local/qwen3-coder-next-udq6xl",
  "small_model": "strix-local/qwen3-coder-next-udq6xl",
  "agent": {
    "plan": {
      "model": "strix-local/qwen3.5-122b-udq4"
    }
  }
}
```

That header-based setup is the important bit. When OpenCode starts inside the repo, `{env:PWD}` resolves to the current working directory, and the proxy walks up to the Git root automatically before building or loading the cache.

Typical flow:

```bash
cd ~/code/my-app
opencode
```

After that, switching the selected model in OpenCode causes the proxy to save the old slot, start the requested model if needed, restore the matching slot for the current repo, and then forward the request.

### Other OpenAI-compatible tools

Preferred routing for first-time use is the repo-path header because it lets the proxy auto-initialize a brand-new repo without any setup:

```text
http://127.0.0.1:8001/v1   +   X-Repo-Path: /absolute/path/to/repo
```

The older cached-repo slug routing still works after a repo has been seen once:

```text
http://127.0.0.1:8001/r/<repo-slug>/v1
http://127.0.0.1:8001/v1   +   X-Repo-Slug: <repo-slug>
```

The proxy also serves a synthetic `/v1/models` response so clients can discover the advertised aliases even when the upstream `llama-server` is currently running some other model.

### Automatic save and restore behavior

Automatic slot persistence is handled in four places:

- the proxy saves the active slot before switching away to another repo/model pair,
- the proxy saves the active slot when repo-scoped traffic gives way to unscoped traffic,
- the proxy saves the active slot on clean shutdown, and
- `python server.py stop` and `./stop.sh` stop the managed proxy before stopping the running container.

Hard kills and power loss still cannot be made crash-safe, so `repo-save` remains available as a manual checkpoint if you want one.

### Metrics and logs

The proxy prints one human-readable line per request and also appends JSONL metrics to `~/.cache/strix-halo-llamacpp/proxy-metrics.jsonl`. The request lines now make the cache hit rate explicit in two ways:

- `prompt_cache=` shows how much of the **prompt** came from the KV cache.
- `call_cache=` shows how much of the **entire request** (prompt + generated tokens) came from cache.

That makes it easier to decide whether prompt caching is saving enough work to justify keeping it on for a given workflow. For streamed chat completions, the proxy automatically asks llama.cpp to include final usage stats so those cache fields still show up in the log.

Switch/save events are logged separately, for example:

```text
[switch] repo=my-app-a1b2c3d4 from=qwen3.5-122b-udq4 save=ok(...) to=qwen3-coder-next-udq6xl cold_start=13.8s restore=ok(...)
[proxy] repo=my-app-a1b2c3d4 status=200 path=/v1/chat/completions time=3.42s model=qwen3-coder-next-udq6xl slot=0 prompt=4102 prompt_cache=3800/4102(92.6%) prompt_eval=302/4102(7.4%) call_cache=3800/4514(84.2%) pp=894tok/s gen=412 gen_tps=44.7
```

The detached proxy process started by `serve`/`start.sh` also writes its stdout/stderr to `./.proxy.log`.

### Notes and caveats

- The proxy assumes slot snapshots are only safe to reuse for the **exact same model alias**. Cross-family KV reuse is intentionally not enabled by default.
- The standard launcher flow now treats `repo-init` and `repo-proxy` as optional helpers, not required setup steps.
- If your upstream local server uses an API key, the proxy forwards the correct key upstream after model switches.
- The launcher publishes the container port to `127.0.0.1` only. This keeps the slot management endpoints local to the machine.
- Use `repo-refresh` after major refactors so the cached architecture summary stays relevant.

## API

The raw upstream server and the transparent proxy both expose OpenAI-compatible APIs on localhost only. In day-to-day coding-tool use, prefer the proxy on port `8001`.

Raw upstream llama.cpp:

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

Transparent proxy with automatic repo routing:

```bash
curl http://127.0.0.1:8001/v1/models

curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Repo-Path: $PWD" \
  -d '{
    "model": "qwen3-coder-next-q6",
    "messages": [{"role": "user", "content": "Summarize the project architecture."}],
    "max_tokens": 256
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
