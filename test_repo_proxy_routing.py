import json
from pathlib import Path

from repo_cache import (
    PublishedModel,
    REPO_CONTEXT_MARKER,
    REPO_PATH_HEADER,
    RepoProxyController,
    _update_sse_metrics_buffer,
    inject_repo_context,
    provider_payload,
    refresh_repo_context,
    repo_paths,
    repo_system_prompt,
    slot_filename_for,
    write_opencode_config,
)


def test_slot_filename_for_scopes_by_model_and_slot(tmp_path: Path) -> None:
    repo_dir = tmp_path / "demo-repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("# Demo\n", encoding="utf-8")

    first = slot_filename_for(repo_dir, "qwen3-coder-next-udq6xl", slot_id=0)
    second = slot_filename_for(repo_dir, "qwen3.5-122b-udq4", slot_id=0)
    third = slot_filename_for(repo_dir, "qwen3-coder-next-udq6xl", slot_id=1)

    assert first != second
    assert first != third
    assert first.endswith("--slot0.bin")
    assert third.endswith("--slot1.bin")
    assert repo_paths(repo_dir).slug in first


def test_provider_payload_uses_repo_path_header_for_project_local_clients(tmp_path: Path) -> None:
    repo_dir = tmp_path / "demo-repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("# Demo\n", encoding="utf-8")

    payload = provider_payload(
        provider_id="strix-local",
        provider_name="Strix Halo llama.cpp",
        proxy_port=8001,
        repo_dir=repo_dir,
        repo_path_header_value="{env:PWD}",
        models=[
            PublishedModel(
                alias="qwen3-coder-next-udq6xl",
                name="Coder",
                context_limit=262144,
                output_limit=8192,
            ),
            PublishedModel(
                alias="qwen3.5-122b-udq4",
                name="Architect",
                context_limit=131072,
                output_limit=8192,
            ),
        ],
    )

    provider = payload["strix-local"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8001/v1"
    assert provider["options"]["headers"][REPO_PATH_HEADER] == "{env:PWD}"
    assert set(provider["models"]) == {"qwen3-coder-next-udq6xl", "qwen3.5-122b-udq4"}


def test_controller_resolves_repo_route_and_synthetic_models(tmp_path: Path) -> None:
    repo_dir = tmp_path / "demo-repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
    refresh_repo_context(repo_dir)
    slug = repo_paths(repo_dir).slug

    controller = RepoProxyController(
        upstream_port=8000,
        default_repo_dir=repo_dir,
        available_models=[
            {
                "alias": "qwen3-coder-next-udq6xl",
                "name": "Coder",
                "context_limit": 262144,
                "output_limit": 8192,
            }
        ],
    )

    route = controller.resolve_route(f"/r/{slug}/v1/models", {})
    assert route.repo_slug == slug
    assert route.upstream_path == "/v1/models"
    assert route.synthetic_response is not None
    assert route.synthetic_response["data"][0]["id"] == "qwen3-coder-next-udq6xl"

    chat_route = controller.resolve_route(f"/r/{slug}/v1/chat/completions", {})
    assert chat_route.repo_dir == repo_dir.resolve()
    assert chat_route.inject_context is True


def test_controller_auto_initializes_repo_from_repo_path_header(tmp_path: Path) -> None:
    repo_root = tmp_path / "demo-repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    subdir = repo_root / "src"
    subdir.mkdir()

    controller = RepoProxyController(upstream_port=8000)
    route = controller.resolve_route(
        "/v1/chat/completions",
        {REPO_PATH_HEADER: str(subdir)},
    )

    assert route.repo_dir == repo_root.resolve()
    assert route.repo_slug == repo_paths(repo_root).slug
    assert route.inject_context is True
    assert repo_paths(repo_root).context_file.exists()


def test_update_sse_metrics_buffer_tracks_latest_json_payload() -> None:
    chunks = [
        b'data: {"model":"first","timings":{"prompt_n":4}}\n\n',
        b'data: {"model":"second","timings":{"cache_n":12}}\n\n',
        b'data: [DONE]\n\n',
    ]

    buffer = ""
    latest_payload = None
    for chunk in chunks:
        buffer, latest_payload = _update_sse_metrics_buffer(
            buffer,
            chunk,
            latest_payload=latest_payload,
        )

    assert buffer == ""
    assert latest_payload == {"model": "second", "timings": {"cache_n": 12}}


def test_inject_repo_context_collapses_late_system_messages() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "late system"},
            {"role": "assistant", "content": "world"},
        ]
    }

    injected = inject_repo_context(payload, "repo details go here")

    messages = injected["messages"]
    assert messages[0]["role"] == "system"
    assert REPO_CONTEXT_MARKER in messages[0]["content"]
    assert "late system" in messages[0]["content"]
    assert [message["role"] for message in messages[1:]] == ["user", "assistant"]


def test_inject_repo_context_does_not_duplicate_existing_repo_context() -> None:
    existing = repo_system_prompt("existing repo details")
    payload = {
        "messages": [
            {"role": "system", "content": existing},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "tooling rules"},
        ]
    }

    injected = inject_repo_context(payload, "new repo details")
    system_message = injected["messages"][0]["content"]

    assert system_message.count(REPO_CONTEXT_MARKER) == 1
    assert "existing repo details" in system_message
    assert "tooling rules" in system_message


def test_write_opencode_config_sets_plan_agent_and_small_model(tmp_path: Path) -> None:
    repo_dir = tmp_path / "demo-repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("# Demo\n", encoding="utf-8")

    config_path = write_opencode_config(
        repo_dir,
        models=[
            PublishedModel(
                alias="qwen3-coder-next-udq6xl",
                name="Coder",
                context_limit=262144,
                output_limit=8192,
            ),
            PublishedModel(
                alias="qwen3.5-122b-udq4",
                name="Architect",
                context_limit=131072,
                output_limit=8192,
            ),
        ],
        default_model="qwen3-coder-next-udq6xl",
        small_model="qwen3-coder-next-udq6xl",
        plan_model="qwen3.5-122b-udq4",
        proxy_port=8001,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["model"] == "strix-local/qwen3-coder-next-udq6xl"
    assert config["small_model"] == "strix-local/qwen3-coder-next-udq6xl"
    assert config["agent"]["plan"]["model"] == "strix-local/qwen3.5-122b-udq4"
