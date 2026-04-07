# Testing

The test suite is split into three layers:

- default pytest coverage that is safe for local development and CI
- lightweight bash smoke scripts that keep shell-level checks honest
- an optional live inference smoke test that talks to a real local server

## Run everything that is safe in CI

```bash
pytest -q
```

`pytest.ini` collects these targets by default:

- `test_models.py` for model lookup and argument-generation coverage
- `test_entrypoints.py` for subprocess-based coverage of `start.sh`, `watch.sh`, `stop.sh`, `source-me.sh`, `benchmark-run.sh`, `evaluate.sh`, and `test.sh`
- `tests/` for `server.py`, parallelization helpers, and the opt-in live inference test wrapper

## Focused runs

Model helpers only:

```bash
pytest test_models.py -v
```

Shell and wrapper entrypoints only:

```bash
pytest test_entrypoints.py -v
```

Server helpers only:

```bash
pytest tests/test_server.py -v
```

Parallel benchmark helpers only:

```bash
pytest tests/test_parallelization.py -v
```

Live inference wrapper only:

```bash
pytest tests/test_inference.py -v
```

Legacy bash smoke scripts only:

```bash
bash tests/test_start.sh
bash tests/test_bash_entrypoints.sh
```

## Dry-run launcher validation

`server.py` also exposes a dry-run command that checks model resolution, container selection, and argument generation without starting a new server:

```bash
python server.py test --dry-run --sequential
python server.py test --dry-run --backend rocm --np 4
```

## Optional live smoke test

If you already have a local server running and want to verify a real completion request, enable the live integration test explicitly:

```bash
STRIX_RUN_LIVE_INFERENCE=1 pytest tests/test_inference.py -m integration -v
```

Without that environment variable, the live test is skipped by default so `pytest` stays safe for local development and CI.

## Helper script

`./test.sh` is the top-level end-to-end helper:

```bash
./test.sh
```

What it does:

- optionally installs `requirements-test.txt`
- runs `bash tests/test_start.sh`
- runs `bash tests/test_bash_entrypoints.sh`
- runs the default pytest collection (`test_models.py`, `test_entrypoints.py`, and `tests/`)
- reuses an already-running local server on `STRIX_TEST_PORT` when possible, otherwise starts a temporary server
- runs the live inference smoke test with `tests/test_inference.py`

`./test.sh` keeps going across independent phases, records failures, and always prints a single trailing `FINAL RESULT:` line. That last line includes any failed sub-step, so failures from the bash smoke scripts, pytest suite, server startup, or live inference smoke test bubble all the way up to the final report.

Useful environment variables for `./test.sh`:

```bash
STRIX_TEST_PORT=8000
STRIX_TEST_BACKEND=radv
STRIX_TEST_MODEL_ALIAS=smollm2-135m-test-q4
STRIX_TEST_TIMEOUT=30
```

## Notes for contributors

- keep default tests hermetic and offline unless the test is explicitly marked as live integration
- prefer subprocess-based tests in `test_entrypoints.py` for shell wrappers so real argument forwarding stays covered
- keep the lightweight bash smoke scripts small and focused on shell-only behavior
- when adding a new CLI wrapper or test phase, update both the default pytest collection and `./test.sh` so the final summary remains authoritative
