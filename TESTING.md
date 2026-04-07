# Testing

The test suite is split into fast automated coverage and an optional live smoke test.

## Run everything that is safe in CI

```bash
pytest -q
```

That command covers:

- model lookup and argument generation in `models.py`
- environment loading and launcher helpers in `server.py`
- benchmark/eval helper behavior with mocks
- shell entrypoints such as `start.sh`, `watch.sh`, `stop.sh`, and `source-me.sh`

## Focused runs

Python helpers only:

```bash
pytest tests/test_models.py tests/test_server.py -v
```

Shell entrypoints only:

```bash
pytest tests/test_entrypoints.py -v
```

Parallel benchmark helpers only:

```bash
pytest tests/test_parallelization.py -v
```

Inference helper unit tests only:

```bash
pytest tests/test_inference.py -v
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

`test.sh` installs the test requirements and runs the default pytest command:

```bash
./test.sh
```

## Notes for contributors

- keep default tests hermetic and offline
- mock Docker, HTTP, and subprocess boundaries unless the test is explicitly marked as live integration
- prefer exercising shell entrypoints through subprocess-based tests so argument forwarding stays covered
- when adding a new backend or CLI flag, update both Python tests and shell entrypoint tests
