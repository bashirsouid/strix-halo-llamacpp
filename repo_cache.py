from __future__ import annotations

import copy
import hashlib
import http.server
import json
import os
import re
import socketserver
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_ROOT = Path.home() / ".cache" / "strix-halo-llamacpp"
REPO_CACHE_ROOT = CACHE_ROOT / "repositories"
SLOT_CACHE_ROOT = CACHE_ROOT / "slots"
DEFAULT_PROXY_PORT = 8001
DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_SLOT_ID = 0
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


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def ensure_cache_dirs() -> None:
    REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    SLOT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def normalize_repo_dir(repo_dir: str | Path | None) -> Path:
    candidate = Path(repo_dir or ".").expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Repository path does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {candidate}")
    return candidate


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
    model_alias: str,
    model_name: str,
    context_limit: int,
    output_limit: int = 8192,
    api_key: str | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "baseURL": f"http://127.0.0.1:{proxy_port}/v1",
    }
    if api_key:
        options["apiKey"] = api_key

    return {
        provider_id: {
            "npm": "@ai-sdk/openai-compatible",
            "name": provider_name,
            "options": options,
            "models": {
                model_alias: {
                    "name": model_name,
                    "limit": {
                        "context": context_limit,
                        "output": output_limit,
                    },
                }
            },
        }
    }


def write_opencode_config(
    repo_dir: str | Path,
    *,
    model_alias: str,
    model_name: str,
    context_limit: int,
    proxy_port: int = DEFAULT_PROXY_PORT,
    provider_id: str = "strix-local",
    provider_name: str = "Strix Halo llama.cpp",
    output_limit: int = 8192,
    api_key: str | None = None,
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

    config.setdefault("$schema", "https://opencode.ai/config.json")
    provider = config.setdefault("provider", {})
    if not isinstance(provider, dict):
        raise ValueError(f"The provider field in {config_path} must be an object")

    provider.update(
        provider_payload(
            provider_id=provider_id,
            provider_name=provider_name,
            proxy_port=proxy_port,
            model_alias=model_alias,
            model_name=model_name,
            context_limit=context_limit,
            output_limit=output_limit,
            api_key=api_key,
        )
    )
    config["model"] = f"{provider_id}/{model_alias}"

    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def repo_system_prompt(context_text: str) -> str:
    return f"{REPO_CONTEXT_MARKER}\n{context_text.strip()}"


def payload_has_repo_context(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and REPO_CONTEXT_MARKER in content:
            return True
    return False


def inject_repo_context(payload: dict[str, Any], context_text: str, slot_id: int = DEFAULT_SLOT_ID) -> dict[str, Any]:
    cloned = copy.deepcopy(payload)
    messages = cloned.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected an OpenAI-compatible payload with a 'messages' array")
    if not payload_has_repo_context(messages):
        messages = [{"role": "system", "content": repo_system_prompt(context_text)}] + messages
    cloned["messages"] = messages
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
) -> str:
    metrics = extract_completion_metrics(response_payload)
    model = None
    if isinstance(response_payload, dict):
        model = response_payload.get("model")
    if model is None and isinstance(request_payload, dict):
        model = request_payload.get("model")
    slot_id = request_payload.get("id_slot") if isinstance(request_payload, dict) else None

    parts = ["[proxy]", f"status={status}", f"path={path}", f"time={elapsed_sec:.2f}s"]
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
    text = chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
    buffer += text
    while "\n\n" in buffer:
        event, buffer = buffer.split("\n\n", 1)
        for line in event.split("\n"):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                latest_payload = payload
    return buffer, latest_payload



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
    ) -> None:
        if not self._metrics_enabled():
            return
        elapsed_sec = time.perf_counter() - started_at
        print(
            format_proxy_metrics_line(
                path=self.path,
                status=status,
                elapsed_sec=elapsed_sec,
                request_payload=request_payload,
                response_payload=response_payload,
                error=error,
            ),
            flush=True,
        )

    def _forward(self) -> None:
        upstream_base: str = getattr(self.server, "upstream_base")
        upstream_headers: dict[str, str] = getattr(self.server, "upstream_headers")
        repo_context_file: Path = getattr(self.server, "repo_context_file")
        slot_id: int = getattr(self.server, "slot_id")
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

        if self.command == "POST" and self.path == "/v1/chat/completions":
            try:
                payload = request_payload or {}
                context_text = repo_context_file.read_text(encoding="utf-8")
                payload = inject_repo_context(payload, context_text, slot_id=slot_id)
                request_payload = payload
                body = json.dumps(payload).encode("utf-8")
            except Exception as exc:
                self.send_error(400, f"Invalid chat payload: {exc}")
                self._log_proxy_summary(status=400, started_at=started_at, request_payload=request_payload, error=str(exc))
                return

        url = upstream_base + self.path
        request_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        request_headers.update(upstream_headers)

        req = urllib.request.Request(url, data=body or None, headers=request_headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                self._relay_response(response, started_at=started_at, request_payload=request_payload)
        except urllib.error.HTTPError as exc:
            self._relay_error(exc, started_at=started_at, request_payload=request_payload)
        except urllib.error.URLError as exc:
            self.send_error(502, f"Upstream request failed: {exc}")
            self._log_proxy_summary(status=502, started_at=started_at, request_payload=request_payload, error=str(exc))

    def _relay_response(
        self,
        response: Any,
        *,
        started_at: float,
        request_payload: dict[str, Any] | None = None,
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
            self._log_proxy_summary(
                status=response.status,
                started_at=started_at,
                request_payload=request_payload,
                response_payload=latest_payload,
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
        self._log_proxy_summary(
            status=response.status,
            started_at=started_at,
            request_payload=request_payload,
            response_payload=response_payload,
        )

    def _relay_error(
        self,
        error: urllib.error.HTTPError,
        *,
        started_at: float,
        request_payload: dict[str, Any] | None = None,
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
        )


def start_repo_proxy(
    repo_dir: str | Path,
    *,
    listen_host: str = DEFAULT_PROXY_HOST,
    listen_port: int = DEFAULT_PROXY_PORT,
    upstream_host: str = "127.0.0.1",
    upstream_port: int = 8000,
    slot_id: int = DEFAULT_SLOT_ID,
    upstream_headers: dict[str, str] | None = None,
    verbose: bool = False,
    metrics: bool = True,
) -> None:
    paths = repo_paths(repo_dir)
    if not paths.context_file.exists():
        refresh_repo_context(paths.repo_dir)

    server = _ThreadingHTTPServer((listen_host, listen_port), _ProxyHandler)
    server.upstream_base = f"http://{upstream_host}:{upstream_port}"
    server.upstream_headers = dict(upstream_headers or {})
    server.repo_context_file = paths.context_file
    server.slot_id = slot_id
    server.verbose = verbose
    server.metrics_enabled = metrics

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
