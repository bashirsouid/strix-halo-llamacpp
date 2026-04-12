from pathlib import Path

from repo_cache import (
    PublishedModel,
    RepoProxyController,
    provider_payload,
    refresh_repo_context,
    repo_paths,
    slot_filename_for,
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


def test_provider_payload_uses_repo_scoped_base_url_and_multiple_models(tmp_path: Path) -> None:
    repo_dir = tmp_path / "demo-repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("# Demo\n", encoding="utf-8")

    payload = provider_payload(
        provider_id="strix-local",
        provider_name="Strix Halo llama.cpp",
        proxy_port=8001,
        repo_dir=repo_dir,
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

    slug = repo_paths(repo_dir).slug
    provider = payload["strix-local"]
    assert provider["options"]["baseURL"] == f"http://127.0.0.1:8001/r/{slug}/v1"
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
