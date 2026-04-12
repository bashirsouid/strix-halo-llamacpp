from __future__ import annotations

import copy
import hashlib
import http.server
import json
import os
import re
import signal
import socketserver
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CACHE_ROOT = Path.home() / ".cache" / "strix-halo-llamacpp"
REPO_CACHE_ROOT = CACHE_ROOT / "repositories"
SLOT_CACHE_ROOT = CACHE_ROOT / "slots"
PROXY_STATE_FILE = CACHE_ROOT / "proxy-state.json"
PROXY_METRICS_FILE = CACHE_ROOT / "proxy-metrics.jsonl"
DEFAULT_PROXY_PORT = 8001
DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_SLOT_ID = 0
REPO_ROUTE_PREFIX = "/r"
REPO_PATH_HEADER = "X-Repo-Path"
REPO_SLUG_HEADER = "X-Repo-Slug"
REPO_CONTEXT_MARKER = "<!-- strix-halo-repo-context -->"
MAX_DOC_CHARS = 8000
MAX_TOTAL_CONTEXT_CHARS = 24000
MAX_TREE_DEPTH = 4
MAX_TREE_ENTRIES = 250

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".turbo",
    ".cache",
    ".direnv",
    "vendor",
    "tmp",
    "temp",
}

DOC_CANDIDATES = (
    "AGENTS.md",
    "CLAUDE.md",
    "ARCHITECTURE.md",
    "README.md",
    "README.rst",
    "CONTRIBUTING.md",
    "TESTING.md",
)

MANIFEST_CANDIDATES = (
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Dockerfile",
    "requirements.txt",
)

TEXT_EXTENSIONS = {
    ".md",
    ".rst",
    ".txt",
    ".py",
    ".sh",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".env",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
}


@dataclass(frozen=True)
class RepoPaths:
    repo_dir: Path
    slug: str
    cache_dir: Path
    context_file: Path
    metadata_file: Path
    slot_filename: str
    opencode_file: Path


@dataclass(frozen=True)
class PublishedModel:
    alias: str
    name: str
    context_limit: int
    output_limit: int = 8192


@dataclass(frozen=True)
class ProxyRoute:
    repo_slug: str | None
    repo_dir: Path | None
    upstream_path: str
    inject_context: bool = False
    synthetic_response: dict[str, Any] | None = None


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def ensure_cache_dirs() -> None:
    REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    SLOT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _discover_repo_root(candidate: Path) -> Path:
    current = candidate
    for probe in (current, *current.parents):
        if (probe / ".git").exists():
            return probe
    return candidate


def normalize_repo_dir(repo_dir: str | Path | None) -> Path:
    candidate = Path(repo_dir or ".").expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Repository path does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {candidate}")
    return _discover_repo_root(candidate)


def repo_slug(repo_dir: str | Path) -> str:
    repo_path = normalize_repo_dir(repo_dir)
    safe_name = re.sub(r"[^a-z0-9._-]+", "-", repo_path.name.lower()).strip("-") or "repo"
    digest = hashlib.sha1(str(repo_path).encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}-{digest}"


def repo_paths(repo_dir: str | Path) -> RepoPaths:
    ensure_cache_dirs()
    root = normalize_repo_dir(repo_dir)
    slug = repo_slug(root)
    cache_dir = REPO_CACHE_ROOT / slug
    return RepoPaths(
        repo_dir=root,
        slug=slug,
        cache_dir=cache_dir,
        context_file=cache_dir / "repo-context.md",
        metadata_file=cache_dir / "repo-context.json",
        slot_filename=f"{slug}-slot0.bin",
        opencode_file=root / "opencode.json",
    )


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")
    return safe or "value"


def exact_model_key(model_alias: str | None) -> str:
    raw = (model_alias or "default").strip() or "default"
    safe = _safe_component(raw)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}"


def slot_filename_for(
    repo_dir: str | Path,
    model_alias: str | None,
    *,
    slot_id: int = DEFAULT_SLOT_ID,
    exact_key: str | None = None,
) -> str:
    paths = repo_paths(repo_dir)
    model_key = exact_key or exact_model_key(model_alias)
    return f"{paths.slug}--m_{model_key}--slot{slot_id}.bin"


def repo_proxy_base_url(
    repo_dir: str | Path,
    *,
    proxy_host: str = DEFAULT_PROXY_HOST,
    proxy_port: int = DEFAULT_PROXY_PORT,
) -> str:
    slug = repo_paths(repo_dir).slug
    return f"http://{proxy_host}:{proxy_port}{REPO_ROUTE_PREFIX}/{slug}/v1"


def discover_cached_repos() -> dict[str, Path]:
    ensure_cache_dirs()
    discovered: dict[str, Path] = {}
    for cache_dir in sorted(REPO_CACHE_ROOT.iterdir() if REPO_CACHE_ROOT.exists() else [], key=lambda item: item.name):
        if not cache_dir.is_dir():
            continue
        metadata_file = cache_dir / "repo-context.json"
        if not metadata_file.exists():
            continue
        try:
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        repo_root = data.get("repo_root") if isinstance(data, dict) else None
        if not isinstance(repo_root, str) or not repo_root.strip():
            continue
        repo_path = Path(repo_root).expanduser()
        if repo_path.exists() and repo_path.is_dir():
            discovered[cache_dir.name] = repo_path.resolve()
    return discovered


def _normalize_published_models(
    *,
    models: list[PublishedModel | dict[str, Any]] | None = None,
    model_alias: str | None = None,
    model_name: str | None = None,
    context_limit: int | None = None,
    output_limit: int = 8192,
) -> list[PublishedModel]:
    normalized: list[PublishedModel] = []
    if models:
        for entry in models:
            if isinstance(entry, PublishedModel):
                normalized.append(entry)
                continue
            if not isinstance(entry, dict):
                raise TypeError(f"Unsupported model entry type: {type(entry)!r}")
            alias = entry.get("alias") or entry.get("model_alias")
            name = entry.get("name") or entry.get("model_name")
            ctx = entry.get("context_limit")
            out = entry.get("output_limit", output_limit)
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(f"Model entry is missing a valid alias: {entry!r}")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Model entry is missing a valid name: {entry!r}")
            if ctx is None:
                raise ValueError(f"Model entry is missing context_limit: {entry!r}")
            normalized.append(
                PublishedModel(
                    alias=alias.strip(),
                    name=name.strip(),
                    context_limit=int(ctx),
                    output_limit=int(out),
                )
            )

    if not normalized:
        if not model_alias or not model_name or context_limit is None:
            raise ValueError("Need either models=... or model_alias/model_name/context_limit")
        normalized.append(
            PublishedModel(
                alias=model_alias,
                name=model_name,
                context_limit=int(context_limit),
                output_limit=int(output_limit),
            )
        )

    deduped: list[PublishedModel] = []
    seen: set[str] = set()
    for item in normalized:
        if item.alias in seen:
            continue
        deduped.append(item)
        seen.add(item.alias)
    return deduped


def _collapse_blank_lines(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output: list[str] = []
    blank = False
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            if not blank:
                output.append("")
            blank = True
            continue
        output.append(stripped)
        blank = False
    return "\n".join(output).strip()


def _read_text(path: Path, limit_chars: int = MAX_DOC_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    text = _collapse_blank_lines(text)
    if len(text) <= limit_chars:
        return text
    return text[:limit_chars].rstrip() + "\n… [truncated]"


def _walk_tree(root: Path, depth: int = MAX_TREE_DEPTH, max_entries: int = MAX_TREE_ENTRIES) -> tuple[list[str], bool]:
    lines: list[str] = []
    truncated = False

    def should_skip(path: Path) -> bool:
        name = path.name
        if name in SKIP_DIRS:
            return True
        if path.is_file() and path.suffix.lower() not in TEXT_EXTENSIONS and path.stat().st_size > 128_000:
            return True
        return False

    def visit(path: Path, prefix: str = "", level: int = 0) -> None:
        nonlocal truncated
        if truncated or level > depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        except Exception:
            return
        visible = [entry for entry in entries if not should_skip(entry)]
        total = len(visible)
        for index, entry in enumerate(visible):
            if len(lines) >= max_entries:
                truncated = True
                return
            connector = "└──" if index == total - 1 else "├──"
            marker = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector} {entry.name}{marker}")
            if entry.is_dir():
                extension = "    " if index == total - 1 else "│   "
                visit(entry, prefix + extension, level + 1)
                if truncated:
                    return

    visit(root)
    return lines, truncated


def _load_package_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_stack(repo_dir: Path) -> list[str]:
    stack: list[str] = []

    package_json = repo_dir / "package.json"
    if package_json.exists():
        package_data = _load_package_json(package_json)
        deps = {
            **package_data.get("dependencies", {}),
            **package_data.get("devDependencies", {}),
        }
        label = "Node.js"
        if "typescript" in deps:
            label += " / TypeScript"
        if "next" in deps:
            label += " / Next.js"
        elif "react" in deps:
            label += " / React"
        stack.append(label)

    pyproject = repo_dir / "pyproject.toml"
    if pyproject.exists() or (repo_dir / "requirements.txt").exists():
        pyproject_data = _load_toml(pyproject) if pyproject.exists() else {}
        tool = pyproject_data.get("tool", {})
        label = "Python"
        if "poetry" in tool:
            label += " / Poetry"
        elif "hatch" in tool:
            label += " / Hatch"
        elif "uv" in tool:
            label += " / uv"
        stack.append(label)

    if (repo_dir / "Cargo.toml").exists():
        stack.append("Rust / Cargo")
    if (repo_dir / "go.mod").exists():
        stack.append("Go")
    if (repo_dir / "Dockerfile").exists() or (repo_dir / "docker-compose.yml").exists() or (repo_dir / "docker-compose.yaml").exists():
        stack.append("Docker")
    if (repo_dir / "server.py").exists() and (repo_dir / "models.py").exists():
        stack.append("llama.cpp launcher")

    return stack or ["Unclassified repository"]


def infer_commands(repo_dir: Path) -> list[str]:
    commands: list[str] = []

    package_json = repo_dir / "package.json"
    if package_json.exists():
        package_data = _load_package_json(package_json)
        scripts = package_data.get("scripts", {}) if isinstance(package_data.get("scripts"), dict) else {}
        for key in ("dev", "build", "lint", "test", "typecheck"):
            if key in scripts:
                commands.append(f"npm run {key}")

    pyproject = repo_dir / "pyproject.toml"
    if pyproject.exists():
        pyproject_data = _load_toml(pyproject)
        tool = pyproject_data.get("tool", {})
        if (repo_dir / "tests").exists() or (repo_dir / "pytest.ini").exists():
            commands.append("pytest -q")
        if "ruff" in tool:
            commands.append("ruff check .")
        if "mypy" in tool:
            commands.append("mypy .")

    if (repo_dir / "requirements.txt").exists() and "pytest -q" not in commands:
        if (repo_dir / "tests").exists() or (repo_dir / "pytest.ini").exists():
            commands.append("pytest -q")

    if (repo_dir / "Cargo.toml").exists():
        commands.extend(["cargo test", "cargo fmt --check"])

    if (repo_dir / "go.mod").exists():
        commands.append("go test ./...")

    if (repo_dir / "Makefile").exists():
        makefile_text = _read_text(repo_dir / "Makefile", limit_chars=4000)
        for target in ("test", "lint", "build"):
            pattern = re.compile(rf"^{re.escape(target)}\s*:", re.MULTILINE)
            if pattern.search(makefile_text):
                commands.append(f"make {target}")

    unique: list[str] = []
    seen: set[str] = set()
    for command in commands:
        if command not in seen:
            unique.append(command)
            seen.add(command)
    return unique


def _top_level_entries(repo_dir: Path, limit: int = 20) -> list[str]:
    try:
        entries = sorted(repo_dir.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
    except Exception:
        return []
    result: list[str] = []
    for entry in entries:
        if entry.name in SKIP_DIRS:
            continue
        suffix = "/" if entry.is_dir() else ""
        result.append(f"- `{entry.name}{suffix}`")
        if len(result) >= limit:
            break
    return result


def _collect_doc_snippets(repo_dir: Path) -> list[tuple[str, str]]:
    snippets: list[tuple[str, str]] = []
    for name in DOC_CANDIDATES + MANIFEST_CANDIDATES:
        path = repo_dir / name
        if not path.exists() or not path.is_file():
            continue
        text = _read_text(path)
        if text:
            snippets.append((name, text))
    return snippets


def build_repo_context(repo_dir: str | Path) -> tuple[str, dict[str, Any]]:
    repo_path = normalize_repo_dir(repo_dir)
    stack = detect_stack(repo_path)
    commands = infer_commands(repo_path)
    tree_lines, tree_truncated = _walk_tree(repo_path)
    snippets = _collect_doc_snippets(repo_path)
    top_level = _top_level_entries(repo_path)

    sections: list[str] = []
    sections.append(f"# Cached repo context: {repo_path.name}")
    sections.append(
        "This file is intentionally stable so llama.cpp prompt caching can reuse it across many coding requests. "
        "It omits volatile Git state and recent commit noise on purpose. Use it as orientation, then verify exact details in the source before editing."
    )
    sections.append("## Detected stack\n\n" + "\n".join(f"- {item}" for item in stack))
    if commands:
        sections.append("## Likely build / test commands\n\n" + "\n".join(f"- `{cmd}`" for cmd in commands))
    if top_level:
        sections.append("## Top-level layout\n\n" + "\n".join(top_level))
    tree_block = "\n".join(tree_lines)
    if tree_truncated:
        tree_block += "\n… [tree truncated]"
    sections.append("## File tree (trimmed)\n\n```text\n" + tree_block + "\n```")

    for name, text in snippets:
        sections.append(f"## {name}\n\n{text}")

    context = "\n\n".join(sections).strip() + "\n"
    if len(context) > MAX_TOTAL_CONTEXT_CHARS:
        context = context[:MAX_TOTAL_CONTEXT_CHARS].rstrip() + "\n\n… [repo context truncated]\n"

    metadata = {
        "repo_root": str(repo_path),
        "stack": stack,
        "commands": commands,
        "source_files": [name for name, _ in snippets],
        "tree_truncated": tree_truncated,
    }
    return context, metadata


def refresh_repo_context(repo_dir: str | Path) -> RepoPaths:
    paths = repo_paths(repo_dir)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    context, metadata = build_repo_context(paths.repo_dir)
    paths.context_file.write_text(context, encoding="utf-8")
    paths.metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def load_repo_context(repo_dir: str | Path, refresh_if_missing: bool = True) -> str:
    paths = repo_paths(repo_dir)
    if refresh_if_missing and not paths.context_file.exists():
        refresh_repo_context(paths.repo_dir)
    return paths.context_file.read_text(encoding="utf-8")


def ensure_gitignore_entry(repo_dir: str | Path, pattern: str) -> Path:
    repo_path = normalize_repo_dir(repo_dir)
    gitignore = repo_path / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    lines = [line.rstrip() for line in existing.splitlines()]
    if pattern not in lines:
        updated = existing
        if updated and not updated.endswith("\n"):
            updated += "\n"
        updated += f"{pattern}\n"
        gitignore.write_text(updated, encoding="utf-8")
    return gitignore


def provider_payload(
    *,
    provider_id: str,
    provider_name: str,
    proxy_port: int,
    models: list[PublishedModel | dict[str, Any]] | None = None,
    repo_dir: str | Path | None = None,
    repo_path_header_value: str | None = None,
    model_alias: str | None = None,
    model_name: str | None = None,
    context_limit: int | None = None,
    output_limit: int = 8192,
    api_key: str | None = None,
    timeout_ms: int = 900_000,
    chunk_timeout_ms: int = 120_000,
) -> dict[str, Any]:
    published_models = _normalize_published_models(
        models=models,
        model_alias=model_alias,
        model_name=model_name,
        context_limit=context_limit,
        output_limit=output_limit,
    )

    base_url = f"http://127.0.0.1:{proxy_port}/v1"
    if repo_dir is not None and not repo_path_header_value:
        base_url = repo_proxy_base_url(repo_dir, proxy_port=proxy_port)

    options: dict[str, Any] = {
        "baseURL": base_url,
        "timeout": timeout_ms,
        "chunkTimeout": chunk_timeout_ms,
    }
    if repo_path_header_value:
        options["headers"] = {REPO_PATH_HEADER: repo_path_header_value}
    if api_key:
        options["apiKey"] = api_key

    models_payload: dict[str, Any] = {}
    for item in published_models:
        models_payload[item.alias] = {
            "name": item.name,
            "limit": {
                "context": item.context_limit,
                "output": item.output_limit,
            },
        }

    return {
        provider_id: {
            "npm": "@ai-sdk/openai-compatible",
            "name": provider_name,
            "options": options,
            "models": models_payload,
        }
    }



def write_opencode_config(
    repo_dir: str | Path,
    *,
    models: list[PublishedModel | dict[str, Any]] | None = None,
    default_model: str | None = None,
    small_model: str | None = None,
    plan_model: str | None = None,
    model_alias: str | None = None,
    model_name: str | None = None,
    context_limit: int | None = None,
    proxy_port: int = DEFAULT_PROXY_PORT,
    provider_id: str = "strix-local",
    provider_name: str = "Strix Halo llama.cpp",
    output_limit: int = 8192,
    api_key: str | None = None,
    timeout_ms: int = 900_000,
    chunk_timeout_ms: int = 120_000,
    repo_path_header_value: str | None = "{env:PWD}",
) -> Path:
    paths = repo_paths(repo_dir)
    config_path = paths.opencode_file
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Existing opencode.json is not valid JSON: {config_path}") from exc
    else:
        config = {}

    if not isinstance(config, dict):
        raise ValueError(f"Existing opencode.json must contain a JSON object: {config_path}")

    published_models = _normalize_published_models(
        models=models,
        model_alias=model_alias,
        model_name=model_name,
        context_limit=context_limit,
        output_limit=output_limit,
    )
    default_alias = default_model or published_models[0].alias

    config.setdefault("$schema", "https://opencode.ai/config.json")
    provider = config.setdefault("provider", {})
    if not isinstance(provider, dict):
        raise ValueError(f"The provider field in {config_path} must be an object")

    provider.update(
        provider_payload(
            provider_id=provider_id,
            provider_name=provider_name,
            proxy_port=proxy_port,
            repo_dir=paths.repo_dir,
            repo_path_header_value=repo_path_header_value,
            models=published_models,
            api_key=api_key,
            timeout_ms=timeout_ms,
            chunk_timeout_ms=chunk_timeout_ms,
        )
    )
    config["model"] = f"{provider_id}/{default_alias}"
    if small_model:
        config["small_model"] = f"{provider_id}/{small_model}"

    if plan_model:
        agent = config.setdefault("agent", {})
        if not isinstance(agent, dict):
            raise ValueError(f"The agent field in {config_path} must be an object")
        plan_agent = agent.setdefault("plan", {})
        if not isinstance(plan_agent, dict):
            raise ValueError(f"The agent.plan field in {config_path} must be an object")
        plan_agent["model"] = f"{provider_id}/{plan_model}"

    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def repo_system_prompt(context_text: str) -> str:
    return f"{REPO_CONTEXT_MARKER}\n{context_text.strip()}"


def _content_contains_marker(content: Any) -> bool:
    if isinstance(content, str):
        return REPO_CONTEXT_MARKER in content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and REPO_CONTEXT_MARKER in text:
                    return True
            elif isinstance(item, str) and REPO_CONTEXT_MARKER in item:
                return True
    return False


def _merge_system_contents(contents: list[Any]) -> Any:
    normalized: list[str | list[Any]] = []
    for content in contents:
        if content is None:
            continue
        if isinstance(content, str):
            stripped = content.strip()
            if stripped:
                normalized.append(stripped)
            continue
        if isinstance(content, list):
            if content:
                normalized.append(copy.deepcopy(content))
            continue
        rendered = str(content).strip()
        if rendered:
            normalized.append(rendered)

    if not normalized:
        return ""
    if not any(isinstance(item, list) for item in normalized):
        return "\n\n".join(item for item in normalized if isinstance(item, str))

    merged: list[Any] = []
    for item in normalized:
        if merged:
            merged.append({"type": "text", "text": "\n\n"})
        if isinstance(item, str):
            merged.append({"type": "text", "text": item})
        else:
            merged.extend(item)
    return merged


def payload_has_repo_context(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "system":
            continue
        if _content_contains_marker(message.get("content")):
            return True
    return False


def collapse_system_messages(
    messages: list[dict[str, Any]],
    *,
    repo_context_text: str | None = None,
) -> list[dict[str, Any]]:
    system_contents: list[Any] = []
    remaining_messages: list[dict[str, Any]] = []
    has_repo_context = False

    for message in messages:
        if not isinstance(message, dict):
            remaining_messages.append(message)
            continue
        if message.get("role") == "system":
            content = message.get("content")
            system_contents.append(content)
            has_repo_context = has_repo_context or _content_contains_marker(content)
            continue
        remaining_messages.append(copy.deepcopy(message))

    if repo_context_text and not has_repo_context:
        system_contents.insert(0, repo_system_prompt(repo_context_text))

    if not system_contents:
        return remaining_messages

    merged_system = {
        "role": "system",
        "content": _merge_system_contents(system_contents),
    }
    return [merged_system] + remaining_messages


def inject_repo_context(payload: dict[str, Any], context_text: str, slot_id: int = DEFAULT_SLOT_ID) -> dict[str, Any]:
    cloned = copy.deepcopy(payload)
    messages = cloned.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected an OpenAI-compatible payload with a 'messages' array")
    cloned["messages"] = collapse_system_messages(messages, repo_context_text=context_text)
    cloned.setdefault("cache_prompt", True)
    cloned.setdefault("id_slot", slot_id)
    return cloned


def make_warm_payload(context_text: str, model_alias: str | None = None, slot_id: int = DEFAULT_SLOT_ID) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": repo_system_prompt(context_text)},
            {"role": "user", "content": "Acknowledge with OK."},
        ],
        "cache_prompt": True,
        "id_slot": slot_id,
        "stream": False,
        "temperature": 0,
        "max_tokens": 1,
    }
    if model_alias:
        payload["model"] = model_alias
    return payload


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_completion_metrics(payload: dict[str, Any] | None) -> dict[str, float | int | None]:
    if not isinstance(payload, dict):
        return {}

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        timings = {}

    cache_tokens = _as_int(timings.get("cache_n"))
    prompt_eval_tokens = _as_int(timings.get("prompt_n"))
    completion_tokens = (
        _as_int(usage.get("completion_tokens"))
        or _as_int(usage.get("output_tokens"))
        or _as_int(timings.get("predicted_n"))
    )
    prompt_tokens = (
        _as_int(usage.get("prompt_tokens"))
        or _as_int(usage.get("input_tokens"))
        or ((cache_tokens or 0) + (prompt_eval_tokens or 0) or None)
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_tokens": cache_tokens,
        "prompt_eval_tokens": prompt_eval_tokens,
        "prompt_ms": _as_float(timings.get("prompt_ms")),
        "prompt_tps": _as_float(timings.get("prompt_per_second")),
        "completion_ms": _as_float(timings.get("predicted_ms")),
        "completion_tps": _as_float(timings.get("predicted_per_second")),
    }


def _extract_error_message(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return None


def format_proxy_metrics_line(
    *,
    path: str,
    status: int,
    elapsed_sec: float,
    request_payload: dict[str, Any] | None = None,
    response_payload: dict[str, Any] | None = None,
    error: str | None = None,
    repo_slug: str | None = None,
) -> str:
    metrics = extract_completion_metrics(response_payload)
    model = None
    if isinstance(response_payload, dict):
        model = response_payload.get("model")
    if model is None and isinstance(request_payload, dict):
        model = request_payload.get("model")
    slot_id = request_payload.get("id_slot") if isinstance(request_payload, dict) else None

    parts = ["[proxy]"]
    if isinstance(repo_slug, str) and repo_slug:
        parts.append(f"repo={repo_slug}")
    parts.extend([f"status={status}", f"path={path}", f"time={elapsed_sec:.2f}s"])
    if isinstance(model, str) and model:
        parts.append(f"model={model}")
    if isinstance(slot_id, int):
        parts.append(f"slot={slot_id}")

    prompt_eval_tokens = metrics.get("prompt_eval_tokens")
    cache_tokens = metrics.get("cache_tokens")
    ctx_tokens = None
    if isinstance(prompt_eval_tokens, int) or isinstance(cache_tokens, int):
        ctx_tokens = (prompt_eval_tokens or 0) + (cache_tokens or 0)
        if ctx_tokens > 0:
            parts.append(f"ctx={ctx_tokens}")
    if isinstance(cache_tokens, int):
        if ctx_tokens and ctx_tokens > 0:
            cache_pct = 100.0 * cache_tokens / ctx_tokens
            parts.append(f"cache={cache_tokens}/{ctx_tokens}({cache_pct:.1f}%)")
        else:
            parts.append(f"cache={cache_tokens}")

    prompt_tps = metrics.get("prompt_tps")
    if isinstance(prompt_tps, float):
        parts.append(f"pp={prompt_tps:.0f}tok/s")

    completion_tokens = metrics.get("completion_tokens")
    if isinstance(completion_tokens, int):
        parts.append(f"gen={completion_tokens}")

    completion_tps = metrics.get("completion_tps")
    if isinstance(completion_tps, float):
        parts.append(f"gen_tps={completion_tps:.1f}")
    elif isinstance(completion_tokens, int) and elapsed_sec > 0:
        parts.append(f"gen_tps={completion_tokens / elapsed_sec:.1f}")

    if isinstance(error, str) and error.strip():
        clean_error = " ".join(error.strip().split())
        if len(clean_error) > 160:
            clean_error = clean_error[:157].rstrip() + "..."
        parts.append(f"error={clean_error}")

    return " ".join(parts)


def _update_sse_metrics_buffer(
    buffer: str,
    chunk: bytes,
    *,
    latest_payload: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    text = chunk.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    buffer += text
    while "\n\n" in buffer:
        event, buffer = buffer.split("\n\n", 1)
        if not event.strip():
            continue
        data_lines: list[str] = []
        for line in event.split("\n"):
            if not line.startswith("data:"):
                continue
            data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            latest_payload = payload
    return buffer, latest_payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)



def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_cache_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")



def _slot_action_summary(result: dict[str, Any] | None, *, action: str) -> str:
    if not isinstance(result, dict):
        return f"{action}=ok"
    fields: list[str] = []
    for key in ("n_saved", "n_written", "n_restored", "filename"):
        value = result.get(key)
        if value is not None:
            fields.append(f"{key}={value}")
    for key in ("save_ms", "restore_ms", "ms"):
        value = result.get(key)
        if isinstance(value, (int, float)):
            fields.append(f"{key}={value:.0f}ms")
            break
    return f"{action}=ok" + (f"({', '.join(fields)})" if fields else "")



class RepoProxyController:
    def __init__(
        self,
        *,
        upstream_port: int,
        slot_id: int = DEFAULT_SLOT_ID,
        default_repo_dir: str | Path | None = None,
        available_models: list[PublishedModel | dict[str, Any]] | None = None,
        default_model: str | None = None,
        switch_model_callback: Callable[[str], None] | None = None,
        save_slot_callback: Callable[..., dict[str, Any]] | None = None,
        restore_slot_callback: Callable[..., dict[str, Any]] | None = None,
        warm_slot_callback: Callable[..., dict[str, Any]] | None = None,
        current_model_callback: Callable[[], str | None] | None = None,
        state_file: Path = PROXY_STATE_FILE,
        metrics_file: Path = PROXY_METRICS_FILE,
    ) -> None:
        self.upstream_port = upstream_port
        self.slot_id = slot_id
        self.default_repo_dir = normalize_repo_dir(default_repo_dir) if default_repo_dir else None
        self.default_repo_slug = repo_slug(self.default_repo_dir) if self.default_repo_dir else None
        self.available_models = {
            item.alias: item
            for item in _normalize_published_models(models=available_models)
        } if available_models else {}
        self.default_model = default_model
        self.switch_model_callback = switch_model_callback
        self.save_slot_callback = save_slot_callback
        self.restore_slot_callback = restore_slot_callback
        self.warm_slot_callback = warm_slot_callback
        self.current_model_callback = current_model_callback
        self.state_file = state_file
        self.metrics_file = metrics_file
        self.lock = threading.RLock()
        self.active_repo_slug: str | None = None
        self.active_repo_dir: Path | None = None
        self.active_model: str | None = None
        self.active_slot_filename: str | None = None
        self.active_dirty = False
        self._repo_map: dict[str, Path] = {}
        self._refresh_repo_map()
        self._load_state()

    def _refresh_repo_map(self) -> None:
        self._repo_map = discover_cached_repos()
        if self.default_repo_dir is not None:
            self._repo_map[self.default_repo_slug or repo_slug(self.default_repo_dir)] = self.default_repo_dir

    def _remember_repo_dir(self, repo_dir: str | Path, *, refresh_if_missing: bool = True) -> tuple[str, Path]:
        root = normalize_repo_dir(repo_dir)
        slug = repo_slug(root)
        self._repo_map[slug] = root
        if refresh_if_missing and not repo_paths(root).context_file.exists():
            refresh_repo_context(root)
        return slug, root

    def _current_model(self) -> str | None:
        return self.current_model_callback() if self.current_model_callback else self.active_model

    def _save_active_if_needed(self) -> tuple[dict[str, Any] | None, str | None]:
        current_model = self._current_model()
        if not (
            self.active_dirty
            and self.active_repo_dir is not None
            and self.active_model
            and self.active_slot_filename
            and self.save_slot_callback is not None
            and current_model
        ):
            return None, None
        try:
            result = self.save_slot_callback(
                self.active_repo_dir,
                model_alias=self.active_model,
                slot_id=self.slot_id,
                filename=self.active_slot_filename,
            )
            self.active_dirty = False
            self._write_state()
            return result, None
        except Exception as exc:  # pragma: no cover - defensive logging path
            return None, str(exc)

    def _switch_model_if_needed(self, model_alias: str) -> tuple[bool, float | None]:
        current_model = self._current_model()
        if current_model == model_alias:
            return False, None
        if self.switch_model_callback is None:
            raise RuntimeError(f"Model switch requested for {model_alias!r}, but no switch callback is configured")
        started = time.perf_counter()
        self.switch_model_callback(model_alias)
        return True, time.perf_counter() - started

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        repo_root = data.get("repo_root")
        self.active_repo_slug = data.get("repo_slug") if isinstance(data.get("repo_slug"), str) else None
        if isinstance(repo_root, str) and repo_root:
            repo_path = Path(repo_root).expanduser()
            if repo_path.exists() and repo_path.is_dir():
                self.active_repo_dir = repo_path.resolve()
        self.active_model = data.get("model_alias") if isinstance(data.get("model_alias"), str) else None
        self.active_slot_filename = data.get("slot_filename") if isinstance(data.get("slot_filename"), str) else None
        self.active_dirty = bool(data.get("dirty", False))

    def _write_state(self) -> None:
        ensure_cache_dirs()
        payload = {
            "repo_slug": self.active_repo_slug,
            "repo_root": str(self.active_repo_dir) if self.active_repo_dir else None,
            "model_alias": self.active_model,
            "slot_id": self.slot_id,
            "slot_filename": self.active_slot_filename,
            "dirty": self.active_dirty,
            "upstream_port": self.upstream_port,
            "updated_at": int(time.time()),
        }
        self.state_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def models_payload(self) -> dict[str, Any]:
        aliases = list(self.available_models)
        if not aliases:
            current = self.current_model_callback() if self.current_model_callback else None
            if current:
                aliases = [current]
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "strix-local",
                }
                for alias in aliases
            ],
        }

    def resolve_route(self, raw_path: str, headers: http.client.HTTPMessage | dict[str, Any]) -> ProxyRoute:
        self._refresh_repo_map()
        parsed = urllib.parse.urlsplit(raw_path)
        upstream_path = parsed.path or "/"
        repo_slug_value: str | None = None
        repo_dir: Path | None = None

        query_items = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        forwarded_query: list[tuple[str, str]] = []
        repo_path_value: str | None = None
        for key, value in query_items:
            if key in {"repo", "repo_path"} and value.strip() and repo_path_value is None:
                repo_path_value = value.strip()
                continue
            forwarded_query.append((key, value))

        header_repo_path = headers.get(REPO_PATH_HEADER) if hasattr(headers, "get") else None
        if isinstance(header_repo_path, str) and header_repo_path.strip():
            repo_path_value = header_repo_path.strip()

        if repo_path_value:
            repo_slug_value, repo_dir = self._remember_repo_dir(repo_path_value)

        if repo_dir is None and upstream_path.startswith(f"{REPO_ROUTE_PREFIX}/"):
            parts = upstream_path.split("/")
            if len(parts) >= 4 and parts[1] == REPO_ROUTE_PREFIX.strip("/"):
                repo_slug_value = parts[2]
                remainder = "/" + "/".join(parts[3:])
                upstream_path = remainder if remainder != "//" else "/"

        if repo_dir is None and repo_slug_value is None:
            header_value = headers.get(REPO_SLUG_HEADER) if hasattr(headers, "get") else None
            if isinstance(header_value, str) and header_value.strip():
                repo_slug_value = header_value.strip()

        if repo_dir is None and repo_slug_value is None:
            if self.default_repo_slug:
                repo_slug_value = self.default_repo_slug
            elif len(self._repo_map) == 1:
                repo_slug_value = next(iter(self._repo_map))

        if repo_dir is None and repo_slug_value:
            repo_dir = self._repo_map.get(repo_slug_value)
            if repo_dir is not None:
                repo_slug_value, repo_dir = self._remember_repo_dir(repo_dir, refresh_if_missing=False)

        if forwarded_query:
            upstream_path = f"{upstream_path}?{urllib.parse.urlencode(forwarded_query)}"

        if repo_dir is not None and not repo_paths(repo_dir).context_file.exists():
            refresh_repo_context(repo_dir)

        synthetic_response = None
        inject_context = False
        plain_path = upstream_path.split("?", 1)[0]
        if plain_path == "/v1/models":
            synthetic_response = self.models_payload()
        elif plain_path == "/v1/chat/completions" and repo_dir is not None:
            inject_context = True
        return ProxyRoute(
            repo_slug=repo_slug_value,
            repo_dir=repo_dir,
            upstream_path=upstream_path,
            inject_context=inject_context,
            synthetic_response=synthetic_response,
        )

    def resolve_requested_model(self, payload: dict[str, Any] | None) -> str | None:
        model = payload.get("model") if isinstance(payload, dict) else None
        if isinstance(model, str) and model.strip():
            return model.strip()
        if self.default_model:
            return self.default_model
        if self.active_model:
            return self.active_model
        if self.current_model_callback:
            return self.current_model_callback()
        return None

    def ensure_target(self, repo_slug: str, repo_dir: Path, model_alias: str) -> dict[str, Any]:
        previous_model = self.active_model
        current_model = self._current_model()
        target_slot_filename = slot_filename_for(repo_dir, model_alias, slot_id=self.slot_id)
        no_change = (
            self.active_repo_slug == repo_slug
            and self.active_model == model_alias
            and self.active_slot_filename == target_slot_filename
            and current_model == model_alias
        )
        if no_change:
            return {
                "repo_slug": repo_slug,
                "model_alias": model_alias,
                "slot_filename": target_slot_filename,
                "switched": False,
                "restored": False,
                "warmed": False,
            }

        save_result, save_error = self._save_active_if_needed()
        switched, cold_start_sec = self._switch_model_if_needed(model_alias)

        restore_result = None
        restore_error = None
        restored = False
        if self.restore_slot_callback is not None:
            try:
                restore_result = self.restore_slot_callback(
                    repo_dir,
                    model_alias=model_alias,
                    slot_id=self.slot_id,
                    filename=target_slot_filename,
                )
                restored = True
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    restore_error = str(exc)
            except Exception as exc:  # pragma: no cover - defensive logging path
                restore_error = str(exc)

        warm_result = None
        warmed = False
        if not restored and self.warm_slot_callback is not None:
            warm_result = self.warm_slot_callback(repo_dir, model_alias=model_alias, slot_id=self.slot_id)
            warmed = True

        self.active_repo_slug = repo_slug
        self.active_repo_dir = repo_dir
        self.active_model = model_alias
        self.active_slot_filename = target_slot_filename
        self.active_dirty = bool(warmed)
        self._write_state()

        message = ["[switch]", f"repo={repo_slug}"]
        if previous_model:
            message.append(f"from={previous_model}")
        if save_result is not None or save_error:
            if save_error:
                message.append(f"save=error({save_error})")
            else:
                message.append(_slot_action_summary(save_result, action="save"))
        message.append(f"to={model_alias}")
        if switched and cold_start_sec is not None:
            message.append(f"cold_start={cold_start_sec:.1f}s")
        if restored:
            message.append(_slot_action_summary(restore_result, action="restore"))
        elif warm_result is not None:
            message.append("restore=miss")
            message.append("warm=ok")
        elif restore_error:
            message.append(f"restore=error({restore_error})")
        print(" ".join(message), flush=True)
        _append_jsonl(
            self.metrics_file,
            {
                "event": "switch",
                "repo_slug": repo_slug,
                "model_alias": model_alias,
                "slot_filename": target_slot_filename,
                "save": _jsonable(save_result),
                "save_error": save_error,
                "restore": _jsonable(restore_result),
                "restore_error": restore_error,
                "warm": _jsonable(warm_result),
                "switched": switched,
                "cold_start_sec": cold_start_sec,
                "timestamp": time.time(),
            },
        )

        return {
            "repo_slug": repo_slug,
            "model_alias": model_alias,
            "slot_filename": target_slot_filename,
            "switched": switched,
            "restored": restored,
            "warmed": warmed,
            "save_result": save_result,
            "restore_result": restore_result,
            "warm_result": warm_result,
        }

    def prepare_unscoped(self, model_alias: str | None = None) -> dict[str, Any]:
        previous_repo = self.active_repo_slug
        previous_model = self.active_model
        save_result, save_error = self._save_active_if_needed()
        switched = False
        cold_start_sec = None
        if model_alias:
            switched, cold_start_sec = self._switch_model_if_needed(model_alias)

        current_model = self._current_model()
        self.active_repo_slug = None
        self.active_repo_dir = None
        self.active_model = model_alias or current_model
        self.active_slot_filename = None
        self.active_dirty = False
        self._write_state()

        if previous_repo or save_result is not None or save_error or switched:
            message = ["[switch]", "repo=-"]
            if previous_repo:
                message.append(f"from_repo={previous_repo}")
            if previous_model:
                message.append(f"from={previous_model}")
            if save_result is not None or save_error:
                if save_error:
                    message.append(f"save=error({save_error})")
                else:
                    message.append(_slot_action_summary(save_result, action="save"))
            if model_alias:
                message.append(f"to={model_alias}")
            if switched and cold_start_sec is not None:
                message.append(f"cold_start={cold_start_sec:.1f}s")
            print(" ".join(message), flush=True)
            _append_jsonl(
                self.metrics_file,
                {
                    "event": "switch",
                    "repo_slug": None,
                    "model_alias": model_alias or current_model,
                    "slot_filename": None,
                    "save": _jsonable(save_result),
                    "save_error": save_error,
                    "restore": None,
                    "restore_error": None,
                    "warm": None,
                    "switched": switched,
                    "cold_start_sec": cold_start_sec,
                    "timestamp": time.time(),
                },
            )

        return {
            "repo_slug": None,
            "model_alias": model_alias or current_model,
            "slot_filename": None,
            "switched": switched,
            "save_result": save_result,
        }

    def mark_active_dirty(self) -> None:
        if self.active_repo_dir is None or not self.active_slot_filename:
            return
        self.active_dirty = True
        self._write_state()

    def flush_active(self, *, reason: str = "shutdown") -> dict[str, Any] | None:
        current_model = self.current_model_callback() if self.current_model_callback else self.active_model
        if not (
            self.active_dirty
            and self.active_repo_dir is not None
            and self.active_model
            and self.active_slot_filename
            and self.save_slot_callback is not None
            and current_model
        ):
            return None
        result = self.save_slot_callback(
            self.active_repo_dir,
            model_alias=self.active_model,
            slot_id=self.slot_id,
            filename=self.active_slot_filename,
        )
        self.active_dirty = False
        self._write_state()
        print(
            " ".join(
                [
                    "[save]",
                    f"repo={self.active_repo_slug}",
                    f"model={self.active_model}",
                    _slot_action_summary(result, action="save"),
                    f"reason={reason}",
                ]
            ),
            flush=True,
        )
        _append_jsonl(
            self.metrics_file,
            {
                "event": "save",
                "repo_slug": self.active_repo_slug,
                "model_alias": self.active_model,
                "slot_filename": self.active_slot_filename,
                "reason": reason,
                "result": _jsonable(result),
                "timestamp": time.time(),
            },
        )
        return result

    def log_request(
        self,
        *,
        path: str,
        status: int,
        elapsed_sec: float,
        request_payload: dict[str, Any] | None = None,
        response_payload: dict[str, Any] | None = None,
        error: str | None = None,
        repo_slug: str | None = None,
    ) -> None:
        line = format_proxy_metrics_line(
            path=path,
            status=status,
            elapsed_sec=elapsed_sec,
            request_payload=request_payload,
            response_payload=response_payload,
            error=error,
            repo_slug=repo_slug,
        )
        print(line, flush=True)
        _append_jsonl(
            self.metrics_file,
            {
                "event": "request",
                "repo_slug": repo_slug,
                "path": path,
                "status": status,
                "elapsed_sec": elapsed_sec,
                "request": _jsonable(request_payload),
                "response": _jsonable(response_payload),
                "error": error,
                "metrics": _jsonable(extract_completion_metrics(response_payload)),
                "timestamp": time.time(),
            },
        )


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def _metrics_enabled(self) -> bool:
        return bool(getattr(self.server, "metrics_enabled", True))

    def _log_proxy_summary(
        self,
        *,
        status: int,
        started_at: float,
        request_payload: dict[str, Any] | None = None,
        response_payload: dict[str, Any] | None = None,
        error: str | None = None,
        repo_slug: str | None = None,
        upstream_path: str | None = None,
    ) -> None:
        if not self._metrics_enabled():
            return
        elapsed_sec = time.perf_counter() - started_at
        controller: RepoProxyController | None = getattr(self.server, "controller", None)
        if controller is not None:
            controller.log_request(
                path=upstream_path or self.path,
                status=status,
                elapsed_sec=elapsed_sec,
                request_payload=request_payload,
                response_payload=response_payload,
                error=error,
                repo_slug=repo_slug,
            )
            return
        print(
            format_proxy_metrics_line(
                path=upstream_path or self.path,
                status=status,
                elapsed_sec=elapsed_sec,
                request_payload=request_payload,
                response_payload=response_payload,
                error=error,
                repo_slug=repo_slug,
            ),
            flush=True,
        )

    def _forward(self) -> None:
        upstream_base: str = getattr(self.server, "upstream_base")
        upstream_headers: dict[str, str] = getattr(self.server, "upstream_headers")
        upstream_headers_callback: Callable[[], dict[str, str]] | None = getattr(self.server, "upstream_headers_callback", None)
        slot_id: int = getattr(self.server, "slot_id")
        controller: RepoProxyController | None = getattr(self.server, "controller", None)
        started_at = time.perf_counter()

        body = b""
        request_payload: dict[str, Any] | None = None
        if self.command in {"POST", "PUT", "PATCH"}:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                content_length = int(raw_length)
            except ValueError:
                content_length = 0
            body = self.rfile.read(content_length) if content_length > 0 else b""
            if body:
                try:
                    parsed_payload = json.loads(body.decode("utf-8"))
                except Exception:
                    parsed_payload = None
                if isinstance(parsed_payload, dict):
                    request_payload = parsed_payload

        with (controller.lock if controller is not None else threading.RLock()):
            try:
                route = controller.resolve_route(self.path, self.headers) if controller is not None else ProxyRoute(
                    repo_slug=None,
                    repo_dir=None,
                    upstream_path=self.path,
                )
            except Exception as exc:
                self.send_error(400, f"Invalid repo routing metadata: {exc}")
                self._log_proxy_summary(
                    status=400,
                    started_at=started_at,
                    request_payload=request_payload,
                    error=str(exc),
                    upstream_path=self.path,
                )
                return

            if route.synthetic_response is not None:
                data = json.dumps(route.synthetic_response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self._log_proxy_summary(
                    status=200,
                    started_at=started_at,
                    request_payload=request_payload,
                    response_payload=route.synthetic_response,
                    repo_slug=route.repo_slug,
                    upstream_path=route.upstream_path,
                )
                return

            if controller is not None and self.command in {"POST", "PUT", "PATCH"}:
                requested_model = controller.resolve_requested_model(request_payload)
                try:
                    if route.repo_dir is not None and requested_model:
                        controller.ensure_target(route.repo_slug or repo_slug(route.repo_dir), route.repo_dir, requested_model)
                    elif route.repo_dir is None:
                        controller.prepare_unscoped(model_alias=requested_model)
                except Exception as exc:
                    self.send_error(502, f"Failed to prepare target model: {exc}")
                    self._log_proxy_summary(
                        status=502,
                        started_at=started_at,
                        request_payload=request_payload,
                        error=str(exc),
                        repo_slug=route.repo_slug,
                        upstream_path=route.upstream_path,
                    )
                    return

            if route.inject_context:
                try:
                    payload = request_payload or {}
                    context_text = load_repo_context(route.repo_dir, refresh_if_missing=True)
                    payload = inject_repo_context(payload, context_text, slot_id=slot_id)
                    request_payload = payload
                    body = json.dumps(payload).encode("utf-8")
                except Exception as exc:
                    self.send_error(400, f"Invalid chat payload: {exc}")
                    self._log_proxy_summary(
                        status=400,
                        started_at=started_at,
                        request_payload=request_payload,
                        error=str(exc),
                        repo_slug=route.repo_slug,
                        upstream_path=route.upstream_path,
                    )
                    return

            url = upstream_base + route.upstream_path
            request_headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "content-length", "connection"}
            }
            request_headers.update(upstream_headers)
            if callable(upstream_headers_callback):
                try:
                    request_headers.update(upstream_headers_callback())
                except Exception:
                    pass

            req = urllib.request.Request(url, data=body or None, headers=request_headers, method=self.command)
            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    self._relay_response(
                        response,
                        started_at=started_at,
                        request_payload=request_payload,
                        controller=controller,
                        route=route,
                    )
            except urllib.error.HTTPError as exc:
                self._relay_error(
                    exc,
                    started_at=started_at,
                    request_payload=request_payload,
                    controller=controller,
                    route=route,
                )
            except urllib.error.URLError as exc:
                self.send_error(502, f"Upstream request failed: {exc}")
                self._log_proxy_summary(
                    status=502,
                    started_at=started_at,
                    request_payload=request_payload,
                    error=str(exc),
                    repo_slug=route.repo_slug,
                    upstream_path=route.upstream_path,
                )

    def _relay_response(
        self,
        response: Any,
        *,
        started_at: float,
        request_payload: dict[str, Any] | None = None,
        controller: RepoProxyController | None = None,
        route: ProxyRoute | None = None,
    ) -> None:
        content_type = response.headers.get("Content-Type", "application/octet-stream")
        stream = "text/event-stream" in content_type
        self.send_response(response.status)
        for key, value in response.headers.items():
            if key.lower() in {"transfer-encoding", "content-length", "connection"}:
                continue
            self.send_header(key, value)
        self.send_header("Connection", "close")

        if stream:
            self.end_headers()
            reader = getattr(response, "read1", response.read)
            sse_buffer = ""
            latest_payload: dict[str, Any] | None = None
            while True:
                chunk = reader(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                sse_buffer, latest_payload = _update_sse_metrics_buffer(
                    sse_buffer,
                    chunk,
                    latest_payload=latest_payload,
                )
            if controller is not None and route is not None and response.status < 400 and request_payload is not None:
                controller.mark_active_dirty()
            self._log_proxy_summary(
                status=response.status,
                started_at=started_at,
                request_payload=request_payload,
                response_payload=latest_payload,
                repo_slug=route.repo_slug if route else None,
                upstream_path=route.upstream_path if route else None,
            )
            return

        data = response.read()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

        response_payload = None
        if data:
            try:
                parsed = json.loads(data.decode("utf-8"))
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                response_payload = parsed
        if controller is not None and route is not None and response.status < 400 and request_payload is not None:
            controller.mark_active_dirty()
        self._log_proxy_summary(
            status=response.status,
            started_at=started_at,
            request_payload=request_payload,
            response_payload=response_payload,
            repo_slug=route.repo_slug if route else None,
            upstream_path=route.upstream_path if route else None,
        )

    def _relay_error(
        self,
        error: urllib.error.HTTPError,
        *,
        started_at: float,
        request_payload: dict[str, Any] | None = None,
        controller: RepoProxyController | None = None,
        route: ProxyRoute | None = None,
    ) -> None:
        data = error.read()
        self.send_response(error.code)
        for key, value in error.headers.items():
            if key.lower() in {"transfer-encoding", "content-length", "connection"}:
                continue
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

        response_payload = None
        if data:
            try:
                parsed = json.loads(data.decode("utf-8"))
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                response_payload = parsed
        self._log_proxy_summary(
            status=error.code,
            started_at=started_at,
            request_payload=request_payload,
            response_payload=response_payload,
            error=_extract_error_message(response_payload) or str(error),
            repo_slug=route.repo_slug if route else None,
            upstream_path=route.upstream_path if route else None,
        )


def start_repo_proxy(
    repo_dir: str | Path | None = None,
    *,
    listen_host: str = DEFAULT_PROXY_HOST,
    listen_port: int = DEFAULT_PROXY_PORT,
    upstream_host: str = "127.0.0.1",
    upstream_port: int = 8000,
    slot_id: int = DEFAULT_SLOT_ID,
    upstream_headers: dict[str, str] | None = None,
    upstream_headers_callback: Callable[[], dict[str, str]] | None = None,
    verbose: bool = False,
    metrics: bool = True,
    available_models: list[PublishedModel | dict[str, Any]] | None = None,
    default_model: str | None = None,
    switch_model_callback: Callable[[str], None] | None = None,
    save_slot_callback: Callable[..., dict[str, Any]] | None = None,
    restore_slot_callback: Callable[..., dict[str, Any]] | None = None,
    warm_slot_callback: Callable[..., dict[str, Any]] | None = None,
    current_model_callback: Callable[[], str | None] | None = None,
) -> None:
    default_repo = normalize_repo_dir(repo_dir) if repo_dir is not None else None
    if default_repo is not None and not repo_paths(default_repo).context_file.exists():
        refresh_repo_context(default_repo)

    server = _ThreadingHTTPServer((listen_host, listen_port), _ProxyHandler)
    server.upstream_base = f"http://{upstream_host}:{upstream_port}"
    server.upstream_headers = dict(upstream_headers or {})
    server.upstream_headers_callback = upstream_headers_callback
    server.slot_id = slot_id
    server.verbose = verbose
    server.metrics_enabled = metrics
    server.controller = RepoProxyController(
        upstream_port=upstream_port,
        slot_id=slot_id,
        default_repo_dir=default_repo,
        available_models=available_models,
        default_model=default_model,
        switch_model_callback=switch_model_callback,
        save_slot_callback=save_slot_callback,
        restore_slot_callback=restore_slot_callback,
        warm_slot_callback=warm_slot_callback,
        current_model_callback=current_model_callback,
    )

    previous_handlers: dict[int, Any] = {}

    def _shutdown_handler(signum: int, _frame: Any) -> None:
        try:
            server.shutdown()
        except Exception:
            pass

    for signum in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if signum is None:
            continue
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _shutdown_handler)
        except Exception:
            continue

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            controller = getattr(server, "controller", None)
            if controller is not None:
                controller.flush_active(reason="proxy-exit")
        except Exception:
            pass
        server.server_close()
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except Exception:
                continue
