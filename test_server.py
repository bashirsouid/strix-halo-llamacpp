from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import server
from models import DraftModel, ModelConfig, SpecConfig


class DummyResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = json.dumps(payload).encode()
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyPopen:
    def __init__(self):
        self.stdout = []

    def wait(self) -> int:
        return 0


class DummyThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True


class TestEnvironmentAndHelpers:
    def test_load_env_file_reads_values_without_overwriting_existing_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / ".env").write_text("API_KEY=from-file\nHF_REVISION=abc123\n")
        monkeypatch.setattr(server, "PROJECT_DIR", tmp_path)
        monkeypatch.setenv("API_KEY", "existing-key")
        monkeypatch.delenv("HF_REVISION", raising=False)

        server.load_env_file()

        assert server.os.environ["API_KEY"] == "existing-key"
        assert server.os.environ["HF_REVISION"] == "abc123"

    def test_container_helpers_cover_every_backend(self):
        for backend in server.VALID_BACKENDS:
            assert server._container_image(backend).startswith(server.CONTAINER_REGISTRY)
            assert server._container_name(backend).startswith("strix-llama-")
            assert server._is_container_backend(backend) is True

        assert server._is_rocm("rocm7") is True
        assert server._is_rocm("radv") is False
        assert server._local_url(8123, "/health") == "http://127.0.0.1:8123/health"

    def test_resolve_backend_handles_valid_invalid_and_interactive_inputs(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        assert server.resolve_backend("radv") == "radv"

        with pytest.raises(SystemExit):
            server.resolve_backend("not-a-backend")

        monkeypatch.setattr(server, "pick_backend", lambda prompt_text="": "rocm7")
        assert server.resolve_backend(None) == "rocm7"


class TestDownloadLogic:
    def test_hf_download_prefers_cli_then_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls: list[tuple[str, str, str, str, str | None]] = []

        def fake_cli(cli: str, repo: str, pattern: str, local_dir: str, revision=None):
            calls.append(("cli", cli, repo, pattern, revision))
            raise subprocess.CalledProcessError(returncode=1, cmd=[cli])

        def fake_python(repo: str, pattern: str, local_dir: str, revision=None):
            calls.append(("python", repo, pattern, local_dir, revision))

        monkeypatch.setattr(server, "_find_hf_cli", lambda: "hf")
        monkeypatch.setattr(server, "_hf_download_cli", fake_cli)
        monkeypatch.setattr(server, "_hf_download_python", fake_python)

        server._hf_download("owner/repo", "*.gguf", "/tmp/models", revision="main")

        assert calls == [
            ("cli", "hf", "owner/repo", "*.gguf", "main"),
            ("python", "owner/repo", "*.gguf", "/tmp/models", "main"),
        ]

    def test_download_model_uses_parent_dir_for_nested_patterns_and_downloads_draft(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("HF_REVISION", raising=False)
        model_dir = tmp_path / "Q4_K_M"
        draft_dir = tmp_path / "draft-dir"
        cfg = ModelConfig(
            name="Nested Download Model",
            alias="nested-download-model",
            hf_repo="owner/repo",
            dest_dir=model_dir,
            download_include="Q4_K_M/*.gguf",
            shard_glob="*.gguf",
            spec=SpecConfig(
                strategy="draft",
                draft=DraftModel(
                    hf_repo="owner/draft",
                    filename="draft.gguf",
                    dest_dir=draft_dir,
                ),
            ),
        )

        calls: list[tuple[str, str, str]] = []

        def fake_download(repo: str, pattern: str, local_dir: str, revision=None):
            calls.append((repo, pattern, local_dir))
            if repo == cfg.hf_repo:
                download_root = Path(local_dir)
                (download_root / "Q4_K_M").mkdir(parents=True, exist_ok=True)
                (download_root / "Q4_K_M" / "model.gguf").write_text("model")
            else:
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                (Path(local_dir) / "draft.gguf").write_text("draft")

        monkeypatch.setattr(server, "_hf_download", fake_download)

        server.download_model(cfg)

        assert calls[0] == (cfg.hf_repo, cfg.download_include, str(cfg.dest_dir.parent))
        assert calls[1] == (cfg.spec.draft.hf_repo, cfg.spec.draft.filename, str(cfg.spec.draft.dest_dir))
        assert cfg.is_downloaded is True
        assert cfg.spec.draft.path.exists() is True


class TestServerLifecycle:
    def test_stop_server_stops_known_containers_and_cleans_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pid_file = tmp_path / ".server.pid"
        state_file = tmp_path / ".server.json"
        pid_file.write_text("123\n")
        state_file.write_text("{}")

        monkeypatch.setattr(server, "PID_FILE", pid_file)
        monkeypatch.setattr(server, "STATE_FILE", state_file)
        monkeypatch.setattr(server, "CONTAINER_NAMES", {"radv": "one", "rocm": "two"})
        monkeypatch.setattr(server, "_find_container_runtime", lambda: "docker")
        monkeypatch.setattr(server.time, "sleep", lambda seconds: None)

        commands: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs):
            commands.append(cmd)
            if cmd[:2] == ["docker", "inspect"]:
                return SimpleNamespace(returncode=0)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(server.subprocess, "run", fake_run)

        server.stop_server()

        assert not pid_file.exists()
        assert not state_file.exists()
        assert ["docker", "stop", "-t", "5", "one"] in commands
        assert ["docker", "rm", "-f", "two"] in commands

    def test_wait_for_server_uses_localhost_and_reports_success(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls: list[str] = []

        def fake_urlopen(url: str, timeout: int = 0):
            calls.append(url)
            return DummyResponse({}, status=200)

        monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(server.time, "time", Mock(side_effect=[0, 0]))
        monkeypatch.setattr(server.time, "sleep", lambda seconds: None)

        assert server.wait_for_server(port=9000, timeout=5, verbose=True) is True
        assert calls == ["http://127.0.0.1:9000/health"]

    def test_wait_for_server_times_out_cleanly(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            server.urllib.request,
            "urlopen",
            Mock(side_effect=server.urllib.error.URLError("refused")),
        )
        monkeypatch.setattr(server.time, "time", Mock(side_effect=[0, 0, 1]))
        monkeypatch.setattr(server.time, "sleep", lambda seconds: None)

        assert server.wait_for_server(port=9000, timeout=1, verbose=False) is False

    def test_launch_server_writes_state_and_runs_container(
        self, dummy_model: ModelConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        state_file = tmp_path / ".server.json"
        pid_file = tmp_path / ".server.pid"
        monkeypatch.setattr(server, "STATE_FILE", state_file)
        monkeypatch.setattr(server, "PID_FILE", pid_file)
        monkeypatch.setattr(server, "_find_container_runtime", lambda: "docker")
        monkeypatch.setattr(server, "download_model", lambda cfg: None)
        monkeypatch.setattr(server, "stop_server", lambda: None)
        monkeypatch.setattr(server, "wait_for_server", lambda port, verbose=False: True)
        monkeypatch.setattr(server.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(server.threading if hasattr(server, 'threading') else __import__('threading'), 'Thread', DummyThread)

        dummy_model.temperature = 0.7
        dummy_model.chat_template_kwargs = {"enable_thinking": True}

        run_commands: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs):
            run_commands.append(cmd)
            if cmd[:2] == ["docker", "inspect"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            if cmd[:2] == ["docker", "run"]:
                return SimpleNamespace(returncode=0, stdout="abcdef1234567890\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(server.subprocess, "run", fake_run)
        monkeypatch.setattr(server.subprocess, "Popen", lambda *args, **kwargs: DummyPopen())

        server.launch_server(
            dummy_model,
            port=8081,
            backend="rocm",
            extra_args=["--mirostat", "2"],
            parallel_override=4,
            ctx_override=16384,
        )

        state = json.loads(state_file.read_text())
        assert state == {
            "model": "dummy-model",
            "backend": "rocm",
            "port": 8081,
            "parallel": 4,
            "container": "strix-llama-rocm",
        }

        docker_run = next(cmd for cmd in run_commands if cmd[:2] == ["docker", "run"])
        assert ["-p", "8081:8081"] == docker_run[docker_run.index("-p") : docker_run.index("-p") + 2]
        assert "--mmap" in docker_run
        assert "--no-direct-io" in docker_run
        assert docker_run[docker_run.index("--temp") + 1] == "0.7"
        assert docker_run[docker_run.index("--chat-template-kwargs") + 1] == '{"enable_thinking":true}'
        assert "--mirostat" in docker_run
        assert "2" in docker_run


class TestBenchmarkAndEvalHelpers:
    def test_bench_one_uses_localhost_and_reports_token_rates(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        open_calls: list[str] = []

        def fake_urlopen(req, timeout: int = 0):
            open_calls.append(req.full_url)
            return DummyResponse({"usage": {"prompt_tokens": 11, "completion_tokens": 22}})

        monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(server.time, "perf_counter", Mock(side_effect=[0.0, 2.0]))

        result = server._bench_one(port=8010, prompt="hello", max_tokens=22, label="smoke")

        assert result["ok"] is True
        assert result["prompt_tok"] == 11
        assert result["comp_tok"] == 22
        assert result["tok_s"] == pytest.approx(11.0)
        assert open_calls == ["http://127.0.0.1:8010/v1/chat/completions"]

    def test_fire_one_request_uses_localhost(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        open_calls: list[str] = []

        def fake_urlopen(req, timeout: int = 0):
            open_calls.append(req.full_url)
            return DummyResponse({"usage": {"prompt_tokens": 3, "completion_tokens": 9}})

        monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(server.time, "perf_counter", Mock(side_effect=[1.0, 4.0]))

        result = server._fire_one_request(port=8011, prompt="hello", max_tokens=9)

        assert result["ok"] is True
        assert result["tok_s"] == pytest.approx(3.0)
        assert open_calls == ["http://127.0.0.1:8011/v1/chat/completions"]

    def test_run_evalplus_uses_localhost_base_url_and_parses_single_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(server, "PROJECT_DIR", tmp_path)
        monkeypatch.setattr(server, "EVAL_RAW_DIR", tmp_path / "eval_raw")
        monkeypatch.setattr(server, "EVAL_RESULTS_FILE", tmp_path / "eval_results.jsonl")
        monkeypatch.setattr(
            server.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="pass@1: 0.5\n"),
        )
        monkeypatch.setattr(server.time, "perf_counter", Mock(side_effect=[0.0, 1.5]))

        result = server.run_evalplus(
            port=8123,
            suite="humaneval",
            model_alias="qwen3-coder-next-q6",
            backend="radv",
        )

        assert result["ok"] is True
        assert result["pass_at_1_base"] == 0.5
        assert result["pass_at_1_plus"] is None
        raw_logs = list((tmp_path / "eval_raw").glob("*.log"))
        assert raw_logs, "expected EvalPlus raw log to be written"
        assert "pass@1: 0.5" in raw_logs[0].read_text()
        assert server.EVAL_RESULTS_FILE.exists()
