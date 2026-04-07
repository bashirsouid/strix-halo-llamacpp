from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import server
from models import ModelConfig
from repo_cache import (
    REPO_CONTEXT_MARKER,
    build_repo_context,
    inject_repo_context,
    make_warm_payload,
    refresh_repo_context,
    repo_paths,
    write_opencode_config,
)


def test_build_repo_context_detects_stack_and_commands(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Demo Repo\n\nA small test repository.\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1.0'\n\n[tool.ruff]\nline-length = 100\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hello')\n")

    context, metadata = build_repo_context(tmp_path)

    assert "Cached repo context: " + tmp_path.name in context
    assert "Python" in context
    assert "`pytest -q`" in context
    assert "## File tree (trimmed)" in context
    assert metadata["repo_root"] == str(tmp_path)
    assert "README.md" in metadata["source_files"]


def test_write_opencode_config_creates_openai_compatible_provider(tmp_path: Path):
    config_path = write_opencode_config(
        tmp_path,
        model_alias="qwen3-coder-next-q6",
        model_name="Qwen3 Coder Next (Q6_K)",
        context_limit=32768,
        proxy_port=8124,
        provider_id="strix-local",
        provider_name="Strix Halo llama.cpp",
        output_limit=4096,
    )

    config = json.loads(config_path.read_text())

    assert config["$schema"] == "https://opencode.ai/config.json"
    assert config["model"] == "strix-local/qwen3-coder-next-q6"
    provider = config["provider"]["strix-local"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8124/v1"
    model = provider["models"]["qwen3-coder-next-q6"]
    assert model["limit"]["context"] == 32768
    assert model["limit"]["output"] == 4096


def test_inject_repo_context_sets_cache_prompt_and_slot_id():
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Explain this repo."}],
    }

    injected = inject_repo_context(payload, "# repo summary", slot_id=3)

    assert injected["cache_prompt"] is True
    assert injected["id_slot"] == 3
    assert injected["messages"][0]["role"] == "system"
    assert REPO_CONTEXT_MARKER in injected["messages"][0]["content"]
    assert injected["messages"][1]["content"] == "Explain this repo."


def test_make_warm_payload_is_stable():
    payload = make_warm_payload("# repo summary", model_alias="demo-model", slot_id=7)

    assert payload["model"] == "demo-model"
    assert payload["cache_prompt"] is True
    assert payload["id_slot"] == 7
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 1
    assert payload["messages"][0]["role"] == "system"
    assert REPO_CONTEXT_MARKER in payload["messages"][0]["content"]


def test_model_server_args_include_prompt_cache_and_slot_flags(dummy_model: ModelConfig, tmp_path: Path):
    dummy_model.cache_reuse = 384
    dummy_model.cache_ram = True
    dummy_model.slot_save_path = str(tmp_path / "slots")

    args = dummy_model.server_args()

    assert "--cache-prompt" in args
    assert args[args.index("--cache-reuse") + 1] == "384"
    assert args[args.index("--cache-ram") + 1] == "8192"
    assert "--slots" in args
    assert args[args.index("--slot-save-path") + 1] == str(tmp_path / "slots")


def test_launch_server_binds_localhost_and_mounts_slot_cache(tmp_path: Path, dummy_model: ModelConfig, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []
    state_file = tmp_path / ".server.json"
    pid_file = tmp_path / ".server.pid"
    slot_root = tmp_path / "slot-cache"
    slot_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(server, "download_model", lambda cfg: None)
    monkeypatch.setattr(server, "stop_server", lambda: None)
    monkeypatch.setattr(server, "wait_for_server", lambda port, verbose=False: True)
    monkeypatch.setattr(server, "_find_container_runtime", lambda: "docker")
    monkeypatch.setattr(server, "_container_image", lambda backend: "image:test")
    monkeypatch.setattr(server, "_container_name", lambda backend: "strix-test")
    monkeypatch.setattr(server, "STATE_FILE", state_file)
    monkeypatch.setattr(server, "PID_FILE", pid_file)
    monkeypatch.setattr(server, "SLOT_CACHE_ROOT", slot_root)
    monkeypatch.setattr(server, "ensure_cache_dirs", lambda: None)

    def fake_run(cmd, capture_output=False, text=False, check=False, timeout=None):
        calls.append([str(part) for part in cmd])
        if len(cmd) >= 2 and cmd[0] == "docker" and cmd[1] == "inspect":
            return Mock(returncode=1, stdout="", stderr="")
        if len(cmd) >= 3 and cmd[0] == "docker" and cmd[1] == "run":
            return Mock(returncode=0, stdout="abcdef1234567890", stderr="")
        return Mock(returncode=0, stdout="", stderr="")

    class DummyPopen:
        def __init__(self, *args, **kwargs):
            self.stdout = []

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.subprocess, "Popen", DummyPopen)

    server.launch_server(dummy_model, port=8123, backend="radv")

    docker_run = next(cmd for cmd in calls if len(cmd) >= 3 and cmd[0] == "docker" and cmd[1] == "run")
    assert "-p" in docker_run
    assert docker_run[docker_run.index("-p") + 1] == "127.0.0.1:8123:8123"
    assert "-v" in docker_run
    mounts = [docker_run[i + 1] for i, item in enumerate(docker_run) if item == "-v"]
    assert f"{slot_root}:{slot_root}" in mounts

    state = json.loads(state_file.read_text())
    assert state["slot_save_path"] == str(slot_root)
    assert state["cache_reuse"] == dummy_model.cache_reuse


def test_refresh_repo_context_writes_cache_files(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Repo\n")

    paths = refresh_repo_context(tmp_path)

    assert paths.context_file.exists()
    assert paths.metadata_file.exists()
    assert repo_paths(tmp_path).slug == paths.slug
