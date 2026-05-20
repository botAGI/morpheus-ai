"""
Morpheus API Server
"""
from difflib import SequenceMatcher
import json
import os
import re
import shlex
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlsplit, urlunsplit

from pydantic import BaseModel
import toml

try:
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse
except ModuleNotFoundError:
    def Body(default=None, **_kwargs):
        return default

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, *_args, **_kwargs):
            pass

        def add_middleware(self, *_args, **_kwargs):
            pass

        def get(self, *_args, **_kwargs):
            return lambda func: func

        def post(self, *_args, **_kwargs):
            return lambda func: func

    class PlainTextResponse:
        def __init__(
            self,
            content: str,
            media_type: str = "text/plain",
            headers: dict | None = None,
        ):
            self.content = content
            self.media_type = media_type
            self.headers = headers or {}

    class Request:
        base_url: str

    class CORSMiddleware:
        pass

from morpheus.core.config import MorpheusConfig
from morpheus.core.check import check_text
from morpheus.core.compiler import compile_project
from morpheus.core.wake import generate_wake_md
from morpheus.core.provenance import (
    compute_sha256_file,
    compute_sha256_bytes,
    build_receipt,
    evidence_jsonl_bytes,
    latest_receipt_file,
    new_receipt_id,
    receipt_file_name,
)
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.integrations.manifest import integration_manifest

WILDCARD_HOSTS = {"0.0.0.0", "::", ""}

app = FastAPI(
    title="Morpheus API",
    description="Agent State Compiler API",
    version="0.2.0b1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompileRequest(BaseModel):
    project_root: Optional[str] = None

class InitRequest(BaseModel):
    project_root: Optional[str] = None

class AgentBootstrapRequest(BaseModel):
    project_root: Optional[str] = None

class AgentPrepareRequest(BaseModel):
    project_root: Optional[str] = None

class ProjectConfigRequest(BaseModel):
    project_root: Optional[str] = None
    watch_dirs: Optional[list[str]] = None

class ModelSmokeRequest(BaseModel):
    base_model: Optional[str] = None
    prompt: Optional[str] = None

class VerifyRequest(BaseModel):
    project_root: Optional[str] = None

class CompileResponse(BaseModel):
    receipt_id: str
    claim_count: dict
    source_count: int
    wake_md: str

class VerifyResponse(BaseModel):
    valid: bool
    errors: list[str]
    receipt_id: str

class InitResponse(BaseModel):
    initialized: bool
    project_root: str
    created: bool

class AgentBootstrapResponse(BaseModel):
    project_root: str
    path: str
    created: bool
    updated: bool
    content: str
    agent_connect_url: str

class ModelSmokeResponse(BaseModel):
    ok: bool
    base_model: str
    prompt: str
    answer: str
    error: Optional[str] = None


MORPHEUS_AGENT_BEGIN = "<!-- MORPHEUS:BEGIN -->"
MORPHEUS_AGENT_END = "<!-- MORPHEUS:END -->"
DEFAULT_MODEL_SMOKE_MODEL = "qwen2.5:0.5b"
DEFAULT_MODEL_SMOKE_PROMPT = (
    "Reply with one short sentence confirming Morpheus model smoke test is working."
)
MCP_PROTOCOL_VERSION = "2025-11-25"
TRUTH_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")
TRUTH_MARKER_RE = re.compile(r"^(TODO|DECISION|FIXME|NOTE|HACK|XXX):\s*", re.IGNORECASE)


def latest_receipt_or_http_error(receipts_dir: Path) -> Path | None:
    """Return the receipt chain tail or fail with a client-visible API error."""
    try:
        return latest_receipt_file(receipts_dir)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Receipt chain invalid: {exc}") from exc


def load_json_object_or_http_error(path: Path, label: str) -> dict:
    """Load a JSON object or fail with a client-visible API error."""
    try:
        reject_symlink_paths([path], label)
        data = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"{label} invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"{label} invalid: expected JSON object")
    return data


def load_jsonl_rows_or_value_error(path: Path, label: str) -> list[dict]:
    """Load local JSONL rows for MCP truth tools."""
    try:
        reject_symlink_paths([path], label)
        reject_symlink_components(path, label)
        lines = path.read_text().splitlines()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} invalid: {exc}") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number} invalid JSON: {exc.msg}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _list_count(value) -> int:
    return len(value) if isinstance(value, list) else 0


def _is_real_directory(path: Path) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    try:
        reject_symlink_components(path, "Directory path")
    except ValueError:
        return False
    return True


def _has_symlink_component(path: Path) -> bool:
    return any(component.is_symlink() for component in (path, *path.parents))


def project_status_payload(root: Path) -> dict:
    """Return UI/API project status without binding it to a specific route."""
    morpheus_dir = root / ".morpheus"

    if not _is_real_directory(root):
        return {"initialized": False}

    if morpheus_dir.exists() and not _is_real_directory(morpheus_dir):
        return {"initialized": False}

    state_path = morpheus_dir / "state.json"
    if not state_path.exists():
        if _is_real_directory(morpheus_dir):
            return {
                "initialized": True,
                "compiled": False,
                "sources": 0,
                "claims": 0,
                "evidence": 0,
                "compiled_at": None,
            }
        return {"initialized": False}

    state = load_json_object_or_http_error(state_path, "State file")
    return {
        "initialized": True,
        "compiled": True,
        "sources": _list_count(state.get("sources")),
        "claims": _list_count(state.get("claims")),
        "evidence": _list_count(state.get("evidence")),
        "compiled_at": state.get("compiled_at")
    }


def normalize_agent_state(status_payload: dict) -> dict:
    """Give agents a stable state shape even before project initialization."""
    return {
        "initialized": bool(status_payload.get("initialized")),
        "compiled": bool(status_payload.get("compiled")),
        "sources": int(status_payload.get("sources") or 0),
        "claims": int(status_payload.get("claims") or 0),
        "evidence": int(status_payload.get("evidence") or 0),
        "compiled_at": status_payload.get("compiled_at"),
    }


def normalized_watch_dirs(project_root: Path, watch_dirs: list[str] | None) -> list[str]:
    """Normalize UI/API watch path input to project-relative paths."""
    raw_dirs = [
        str(watch_dir).strip()
        for watch_dir in (watch_dirs or ["."])
        if str(watch_dir).strip()
    ]
    if not raw_dirs:
        raw_dirs = ["."]

    root = project_root.resolve()
    normalized = []
    for raw_dir in raw_dirs:
        candidate = Path(raw_dir)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        reject_symlink_components(candidate, "Watch path")
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Watch path must stay inside project root: {raw_dir}") from exc
        stored = relative.as_posix()
        if stored in ("", "."):
            stored = "."
        if stored not in normalized:
            normalized.append(stored)
    return normalized


def watch_path_info(project_root: Path, watch_dir: str) -> dict:
    try:
        normalized = normalized_watch_dirs(project_root, [watch_dir])[0]
        path = project_root if normalized == "." else project_root / normalized
        exists = path.exists()
        if path.is_dir():
            kind = "directory"
        elif path.is_file():
            kind = "file"
        elif exists:
            kind = "other"
        else:
            kind = "missing"
        return {
            "path": normalized,
            "absolute_path": str(path),
            "exists": exists,
            "kind": kind,
            "valid": True,
            "detail": kind,
        }
    except ValueError as exc:
        return {
            "path": str(watch_dir),
            "absolute_path": str(Path(watch_dir)),
            "exists": False,
            "kind": "invalid",
            "valid": False,
            "detail": str(exc),
        }


def project_config_payload(project_root: Path) -> dict:
    if not _is_real_directory(project_root):
        raise HTTPException(status_code=400, detail="Project root must be an existing real directory")

    morpheus_dir = project_root / ".morpheus"
    initialized = _is_real_directory(morpheus_dir)
    try:
        config = MorpheusConfig(project_root=project_root).load()
        watch_dirs = normalized_watch_dirs(project_root, config.watch_dirs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "service": "morpheus",
        "version": "0.2.0b1",
        "project_root": str(project_root),
        "initialized": initialized,
        "config_path": str(morpheus_dir / "morpheus.toml"),
        "watch_dirs": watch_dirs,
        "watch_paths": [watch_path_info(project_root, watch_dir) for watch_dir in watch_dirs],
    }


def write_project_config(project_root: Path, watch_dirs: list[str] | None) -> dict:
    if not _is_real_directory(project_root):
        raise HTTPException(status_code=400, detail="Project root must be an existing real directory")

    try:
        normalized_dirs = normalized_watch_dirs(project_root, watch_dirs)
        MorpheusConfig(project_root=project_root).init_default()
        config_path = project_root / ".morpheus" / "morpheus.toml"
        reject_symlink_paths([config_path], "Config path")
        reject_symlink_components(config_path, "Config path")
        if config_path.exists() and not config_path.is_file():
            raise ValueError(f"Config path is not a file: {config_path}")
        data = toml.loads(config_path.read_text()) if config_path.exists() else {}
        data["watch_dirs"] = normalized_dirs
        config_path.write_text(toml.dumps(data))
    except (OSError, toml.TomlDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return project_config_payload(project_root)


def api_base_url(request: Request) -> str:
    """Return the externally visible API base from the incoming request."""
    return str(request.base_url).rstrip("/")


def embedded_agent_api_base_url(request: Request) -> str:
    """Return the stable API base to persist inside AGENTS.md."""
    configured = getattr(request, "embedded_agent_api_base", None)
    if configured:
        return str(configured).rstrip("/")

    parsed = urlsplit(api_base_url(request))
    port = parsed.port or (443 if parsed.scheme == "https" else 8000)
    return urlunsplit((parsed.scheme, f"127.0.0.1:{port}", "", "", ""))


def endpoint_url(api_base: str, path: str, project_root: Path | None = None) -> str:
    query = urlencode({"project_root": str(project_root)}) if project_root else ""
    return f"{api_base}{path}{'?' + query if query else ''}"


def ui_url_for_request(request: Request) -> str:
    """Return the UI URL that matches the active API request and CLI UI port."""
    configured_url = os.environ.get("MORPHEUS_UI_URL")
    if configured_url:
        return configured_url.rstrip("/")

    parsed = urlsplit(api_base_url(request))
    configured_host = os.environ.get("MORPHEUS_UI_HOST")
    host = configured_host if configured_host not in (None, *WILDCARD_HOSTS) else parsed.hostname
    if not host:
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    port = os.environ.get("MORPHEUS_UI_PORT", "5173")
    return urlunsplit((parsed.scheme or "http", f"{host}:{port}", "/ui/index.html", "", ""))


def quickstart_payload(request: Request, project_root: Path) -> dict:
    api_base = api_base_url(request)
    ui_url = ui_url_for_request(request)
    native_manifest = endpoint_url(api_base, "/agent/connect", project_root)
    a2a_card = f"{api_base}/.well-known/agent-card.json"
    mcp_url = f"{api_base}/mcp"
    return {
        "service": "morpheus",
        "version": "0.2.0b1",
        "project_root": str(project_root),
        "ui_url": ui_url,
        "commands": {
            "clone": "git clone https://github.com/botAGI/morpheus-ai && cd morpheus-ai",
            "install": [
                "uvx --from morpheus-wake morpheus wake .",
                "pipx run --spec morpheus-wake morpheus wake .",
            ],
            "development_install": [
                "python3 -m venv .venv",
                "source .venv/bin/activate",
                "python -m pip install -e .",
            ],
            "run_local": "morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173",
            "run_lan": "morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173",
            "prepare_agent": "morpheus prepare-agent",
            "agent_manifest": "morpheus agent-connect --json",
            "verify": "morpheus verify --all",
            "model_smoke": f"morpheus model-smoke --base-model {DEFAULT_MODEL_SMOKE_MODEL}",
        },
        "connect": {
            "native": native_manifest,
            "a2a_agent_card": a2a_card,
            "mcp": mcp_url,
            "ui": ui_url,
        },
        "human_path": [
            {
                "id": "install",
                "label": "Run once",
                "detail": "Use uvx or pipx with the morpheus-wake package.",
            },
            {
                "id": "run",
                "label": "Run Morpheus",
                "detail": "Start backend and UI with morpheus serve --ui.",
            },
            {
                "id": "prepare",
                "label": "Prepare Agent",
                "detail": "Click Prepare Agent or run morpheus prepare-agent.",
            },
        ],
        "agent_path": [
            {
                "id": "discover",
                "label": "Discover",
                "detail": "Fetch the native manifest, A2A Agent Card, or MCP endpoint.",
            },
            {
                "id": "read",
                "label": "Read State",
                "detail": "Read WAKE.md before changing the project.",
            },
            {
                "id": "verify",
                "label": "Verify",
                "detail": "Compile and verify after meaningful changes.",
            },
        ],
        "copy_paste_agent_prompt": "\n".join([
            "Connect to Morpheus before changing this project.",
            f"Native manifest: {native_manifest}",
            f"A2A Agent Card: {a2a_card}",
            f"MCP endpoint: {mcp_url}",
            "Run `morpheus prepare-agent` if next_action.id is prepare_agent.",
            "Read WAKE.md before edits, then compile and verify after meaningful changes.",
        ]),
    }


def a2a_agent_card_payload(request: Request) -> dict:
    api_base = api_base_url(request)
    return {
        "name": "Morpheus AI",
        "description": (
            "Agent State Compiler with verifiable provenance, WAKE.md handoff, "
            "integration status, and local model smoke testing."
        ),
        "supportedInterfaces": [
            {
                "url": f"{api_base}/agent/connect",
                "protocolBinding": "https://morpheus.ai/protocols/agent-connect/v1",
                "protocolVersion": "0.2.0b1",
            },
            {
                "url": f"{api_base}/mcp",
                "protocolBinding": "MCP",
                "protocolVersion": MCP_PROTOCOL_VERSION,
            },
        ],
        "provider": {
            "organization": "Morpheus AI",
            "url": api_base,
        },
        "version": "0.2.0b1",
        "documentationUrl": f"{api_base}/.well-known/morpheus.json",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
            "extendedAgentCard": False,
        },
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json", "text/markdown", "text/plain"],
        "skills": [
            {
                "id": "compile-project-state",
                "name": "Compile Project State",
                "description": "Compile watched sources into WAKE.md and signed receipts.",
                "tags": ["state", "provenance", "handoff"],
                "examples": ["Compile this project before an agent starts work."],
                "inputModes": ["application/json"],
                "outputModes": ["application/json", "text/markdown"],
            },
            {
                "id": "build-agent-handoff",
                "name": "Build Agent Handoff",
                "description": "Return a copyable handoff bundle for another autonomous agent.",
                "tags": ["handoff", "agents", "wake"],
                "examples": ["Build a handoff for the next coding agent."],
                "inputModes": ["application/json"],
                "outputModes": ["application/json", "text/markdown"],
            },
            {
                "id": "inspect-integrations",
                "name": "Inspect Integrations",
                "description": "Report GitHub, Gmail, Calendar, Slack, and Linear setup state.",
                "tags": ["integrations", "context", "mcp"],
                "examples": ["Which external context sources are configured?"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
            },
            {
                "id": "smoke-test-local-model",
                "name": "Smoke Test Local Model",
                "description": "Run a direct Ollama smoke test through Morpheus.",
                "tags": ["ollama", "model", "health"],
                "examples": ["Check qwen2.5:0.5b through the local API."],
                "inputModes": ["application/json", "text/plain"],
                "outputModes": ["application/json", "text/plain"],
            },
        ],
    }


def prepare_agent_request(api_base: str, project_root: Path) -> dict:
    return {
        "method": "POST",
        "url": f"{api_base}/agent/prepare",
        "json": {"project_root": str(project_root)},
    }


def handoff_request(api_base: str, project_root: Path) -> dict:
    return {
        "method": "GET",
        "url": endpoint_url(api_base, "/agent/handoff", project_root),
    }


def prepare_agent_action(api_base: str, project_root: Path) -> dict:
    return {
        "id": "prepare_agent",
        "label": "Prepare Agent",
        "detail": "Initialize, compile, bootstrap AGENTS.md, verify, and produce handoff.",
        "command": "morpheus prepare-agent",
        "request": prepare_agent_request(api_base, project_root),
    }


def handoff_action(api_base: str, project_root: Path) -> dict:
    return {
        "id": "handoff",
        "label": "Build Handoff",
        "detail": "Project is ready. Build the copyable handoff bundle.",
        "command": "morpheus handoff",
        "request": handoff_request(api_base, project_root),
    }


def set_project_root_action() -> dict:
    return {
        "id": "set_project_root",
        "label": "Set Project Root",
        "detail": "Choose an existing safe project directory.",
        "command": None,
        "request": None,
    }


def diagnostics_next_action(api_base: str, project_root: Path, checks: list[dict]) -> dict:
    project_root_check = next(
        (check for check in checks if check["id"] == "project_root"),
        None,
    )
    if project_root_check and not project_root_check["ok"]:
        return set_project_root_action()

    ready_checks = {
        "project_root",
        "initialized",
        "compiled",
        "wake",
        "agent_bootstrap",
    }
    checks_by_id = {check["id"]: check for check in checks}
    if all(checks_by_id.get(check_id, {}).get("ok") for check_id in ready_checks):
        return handoff_action(api_base, project_root)
    return prepare_agent_action(api_base, project_root)


def agent_connect_payload(request: Request, project_root: Path) -> dict:
    """Build a self-contained connection manifest for autonomous agents."""
    api_base = api_base_url(request)
    project_root_text = str(project_root)
    json_body = {"project_root": project_root_text}
    status_url = endpoint_url(api_base, "/status", project_root)
    wake_url = endpoint_url(api_base, "/wake", project_root)

    endpoints = {
        "status": {
            "method": "GET",
            "url": status_url,
        },
        "initialize": {
            "method": "POST",
            "url": f"{api_base}/init",
            "json": json_body,
        },
        "compile": {
            "method": "POST",
            "url": f"{api_base}/compile",
            "json": json_body,
        },
        "config": {
            "method": "GET",
            "url": endpoint_url(api_base, "/config", project_root),
        },
        "integrations": {
            "method": "GET",
            "url": f"{api_base}/integrations",
        },
        "model_smoke": {
            "method": "POST",
            "url": f"{api_base}/models/smoke",
            "json": {
                "base_model": DEFAULT_MODEL_SMOKE_MODEL,
                "prompt": DEFAULT_MODEL_SMOKE_PROMPT,
            },
        },
        "prepare_agent": prepare_agent_request(api_base, project_root),
        "wake": {
            "method": "GET",
            "url": wake_url,
        },
        "verify": {
            "method": "POST",
            "url": endpoint_url(api_base, "/verify", project_root),
        },
        "handoff": handoff_request(api_base, project_root),
    }

    connect_url = endpoint_url(api_base, "/agent/connect", project_root)
    state = normalize_agent_state(project_status_payload(project_root))
    bootstrap_ok = agent_bootstrap_diagnostic(request, project_root)["ok"]
    if not _is_real_directory(project_root):
        next_action = set_project_root_action()
    elif state["initialized"] and state["compiled"] and bootstrap_ok:
        next_action = handoff_action(api_base, project_root)
    else:
        next_action = prepare_agent_action(api_base, project_root)
    return {
        "service": "morpheus",
        "version": "0.2.0b1",
        "api_base": api_base,
        "project_root": project_root_text,
        "state": state,
        "next_action": next_action,
        "sequence": [
            {
                "id": "status",
                "goal": "Check whether Morpheus already has compiled state.",
                "request": endpoints["status"],
            },
            {
                "id": "prepare_agent",
                "goal": "Run the one-step prepare flow when next_action.id is prepare_agent.",
                "request": endpoints["prepare_agent"],
            },
            {
                "id": "read_wake",
                "goal": "Load WAKE.md before making project changes.",
                "request": endpoints["wake"],
            },
            {
                "id": "inspect_integrations",
                "goal": "Inspect optional external context sources such as GitHub, Slack, and Linear.",
                "request": endpoints["integrations"],
            },
            {
                "id": "verify",
                "goal": "Verify receipt integrity after compilation.",
                "request": endpoints["verify"],
            },
        ],
        "endpoints": endpoints,
        "integrations": integration_manifest(),
        "cli": {
            "agent_connect": "morpheus agent-connect --json",
            "diagnostics": "morpheus diagnostics --json",
            "initialize": "morpheus init",
            "compile": "morpheus compile",
            "read_wake": "morpheus wake",
            "verify": "morpheus verify --all",
            "model_smoke": f"morpheus model-smoke --base-model {DEFAULT_MODEL_SMOKE_MODEL}",
            "serve": "morpheus serve --host 127.0.0.1 --port 8000",
            "serve_ui": "morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173",
        },
        "curl": {
            "connect": f"curl -s {shlex.quote(connect_url)}",
            "initialize": (
                f"curl -s -X POST {shlex.quote(f'{api_base}/init')} "
                "-H 'Content-Type: application/json' "
                f"-d {shlex.quote(json.dumps(json_body))}"
            ),
            "compile": (
                f"curl -s -X POST {shlex.quote(f'{api_base}/compile')} "
                "-H 'Content-Type: application/json' "
                f"-d {shlex.quote(json.dumps(json_body))}"
            ),
            "wake": f"curl -s {shlex.quote(wake_url)}",
            "verify": f"curl -s -X POST {shlex.quote(endpoints['verify']['url'])}",
            "config": f"curl -s {shlex.quote(endpoints['config']['url'])}",
            "integrations": f"curl -s {shlex.quote(endpoints['integrations']['url'])}",
            "model_smoke": (
                f"curl -s -X POST {shlex.quote(endpoints['model_smoke']['url'])} "
                "-H 'Content-Type: application/json' "
                f"-d {shlex.quote(json.dumps(endpoints['model_smoke']['json']))}"
            ),
        },
        "agent_prompt": (
            "Fetch the connect manifest before working on this project. "
            f"Use {connect_url}, run `morpheus prepare-agent` when next_action.id is prepare_agent, "
            "follow sequence in order, read WAKE.md before edits, "
            "and run compile plus verify after meaningful changes."
        ),
    }


def mcp_tool_definitions() -> list[dict]:
    project_root_schema = {
        "type": "object",
        "properties": {
            "project_root": {
                "type": "string",
                "description": "Optional absolute project root. Defaults to the server working directory.",
            }
        },
        "additionalProperties": False,
    }
    check_text_schema = {
        "type": "object",
        "properties": {
            "project_root": project_root_schema["properties"]["project_root"],
            "text": {
                "type": "string",
                "description": "Agent-written project claim text to verify locally.",
            },
            "fail_on_unknown": {
                "type": "boolean",
                "description": "Treat unknown claims as failures in the returned payload.",
                "default": False,
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }
    evidence_schema = {
        "type": "object",
        "properties": {
            "project_root": project_root_schema["properties"]["project_root"],
            "claim": {
                "type": "string",
                "description": "Claim text to match against active source-backed state.",
            },
            "claim_id": {
                "type": "string",
                "description": "Exact Morpheus claim id such as clm_0001.",
            },
        },
        "additionalProperties": False,
    }
    return [
        {
            "name": "morpheus_status",
            "title": "Morpheus Project Status",
            "description": "Return initialization, compilation, source, claim, and evidence counts.",
            "inputSchema": project_root_schema,
        },
        {
            "name": "morpheus_diagnostics",
            "title": "Morpheus Diagnostics",
            "description": "Return readiness checks and the recommended next action for an agent.",
            "inputSchema": project_root_schema,
        },
        {
            "name": "morpheus_integrations",
            "title": "Morpheus Integrations",
            "description": "Return machine-readable integration setup status.",
            "inputSchema": {"type": "object", "additionalProperties": False},
        },
        {
            "name": "morpheus_model_smoke",
            "title": "Morpheus Model Smoke",
            "description": "Run a local Ollama smoke test through Morpheus.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "base_model": {
                        "type": "string",
                        "description": "Ollama model name.",
                        "default": DEFAULT_MODEL_SMOKE_MODEL,
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt to send to the model.",
                        "default": DEFAULT_MODEL_SMOKE_PROMPT,
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "morpheus_check_text",
            "title": "Morpheus Check Text",
            "description": "Verify agent text against local source-backed project state.",
            "inputSchema": check_text_schema,
        },
        {
            "name": "morpheus_get_active_state",
            "title": "Morpheus Active State",
            "description": "Return active source-backed claims and evidence spans.",
            "inputSchema": project_root_schema,
        },
        {
            "name": "morpheus_get_evidence_for_claim",
            "title": "Morpheus Evidence For Claim",
            "description": "Return source evidence spans for a claim id or claim text.",
            "inputSchema": evidence_schema,
        },
        {
            "name": "morpheus_get_wake",
            "title": "Morpheus WAKE.md",
            "description": "Return the compiled local WAKE.md state artifact.",
            "inputSchema": project_root_schema,
        },
    ]


def mcp_success(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def mcp_error(request_id, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def mcp_tool_result(payload: dict) -> dict:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, indent=2, default=str),
            }
        ],
        "isError": False,
    }


def mcp_project_root(arguments: dict) -> Path:
    value = arguments.get("project_root") if isinstance(arguments, dict) else None
    return Path(value) if value else Path.cwd()


def mcp_truth_paths(project_root: Path) -> dict[str, Path]:
    if not _is_real_directory(project_root):
        raise ValueError("Project root is missing or unsafe")
    morpheus_dir = project_root / ".morpheus"
    if not _is_real_directory(morpheus_dir):
        raise ValueError("Morpheus state not found. Run: morpheus wake .")
    return {
        "morpheus_dir": morpheus_dir,
        "state": morpheus_dir / "state.json",
        "evidence": morpheus_dir / "evidence.jsonl",
        "wake": morpheus_dir / "WAKE.md",
    }


def mcp_load_truth_state(project_root: Path) -> tuple[dict, list[dict]]:
    paths = mcp_truth_paths(project_root)
    try:
        state = load_json_object_or_http_error(paths["state"], "state.json")
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc
    evidence_rows = load_jsonl_rows_or_value_error(paths["evidence"], "evidence.jsonl")
    return state, evidence_rows


def mcp_claim_text(claim: dict) -> str:
    return TRUTH_MARKER_RE.sub("", str(claim.get("excerpt") or "")).strip()


def mcp_evidence_span(evidence: dict | None) -> dict | None:
    if not evidence:
        return None
    line_start = int(evidence.get("line_start") or 1)
    return {
        "path": str(evidence.get("path") or "evidence.jsonl"),
        "line_start": line_start,
        "line_end": int(evidence.get("line_end") or line_start),
        "excerpt": str(evidence.get("excerpt") or ""),
        "claim_id": str(evidence.get("claim_id") or ""),
        "source_sha256": evidence.get("source_sha256"),
        "excerpt_sha256": evidence.get("excerpt_sha256"),
    }


def mcp_active_claims(project_root: Path) -> list[dict]:
    state, evidence_rows = mcp_load_truth_state(project_root)
    evidence_by_claim = {
        str(row.get("claim_id")): row
        for row in evidence_rows
        if isinstance(row, dict) and row.get("claim_id")
    }
    active_claims = []
    for claim in state.get("claims", []):
        if not isinstance(claim, dict):
            continue
        if claim.get("category") == "outdated" or claim.get("status") in {"outdated", "superseded"}:
            continue
        if claim.get("status", "active") != "active":
            continue
        claim_id = str(claim.get("id") or "")
        evidence = mcp_evidence_span(evidence_by_claim.get(claim_id))
        active_claims.append({
            "id": claim_id,
            "claim_id": claim_id,
            "text": mcp_claim_text(claim),
            "status": claim.get("status", "active"),
            "category": claim.get("category"),
            "source_id": claim.get("source_id"),
            "line_start": claim.get("line_start"),
            "line_end": claim.get("line_end"),
            "evidence": evidence,
        })
    return active_claims


def mcp_normalize_claim(text: str) -> str:
    text = TRUTH_MARKER_RE.sub("", text).casefold()
    return " ".join(TRUTH_WORD_RE.findall(text))


def mcp_claim_score(query: str, candidate: str) -> float:
    query_norm = mcp_normalize_claim(query)
    candidate_norm = mcp_normalize_claim(candidate)
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm in candidate_norm or candidate_norm in query_norm:
        return 1.0
    query_tokens = set(query_norm.split())
    candidate_tokens = set(candidate_norm.split())
    token_score = 0.0
    if query_tokens and candidate_tokens:
        token_score = len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)
    return max(SequenceMatcher(None, query_norm, candidate_norm).ratio(), token_score)


def mcp_check_text_payload(arguments: dict) -> dict:
    text = str(arguments.get("text") or "").strip()
    if not text:
        raise ValueError("morpheus_check_text requires non-empty text")
    return check_text(
        text,
        project_root=mcp_project_root(arguments),
        fail_on_unknown=bool(arguments.get("fail_on_unknown", False)),
    )


def mcp_active_state_payload(project_root: Path) -> dict:
    state, _evidence_rows = mcp_load_truth_state(project_root)
    claims = mcp_active_claims(project_root)
    return {
        "project_root": str(project_root),
        "receipt_id": state.get("receipt_id"),
        "compiled_at": state.get("compiled_at"),
        "claims_count": len(claims),
        "active_claims": claims,
    }


def mcp_evidence_for_claim_payload(arguments: dict) -> dict:
    project_root = mcp_project_root(arguments)
    claim_id = str(arguments.get("claim_id") or "").strip()
    claim_text = str(arguments.get("claim") or "").strip()
    if not claim_id and not claim_text:
        raise ValueError("morpheus_get_evidence_for_claim requires claim or claim_id")
    matches = []
    for claim in mcp_active_claims(project_root):
        if claim_id and claim["id"] == claim_id:
            matches.append(claim)
            continue
        if claim_text and mcp_claim_score(claim_text, claim["text"]) >= 0.72:
            matches.append(claim)
    return {
        "project_root": str(project_root),
        "query": {"claim": claim_text or None, "claim_id": claim_id or None},
        "match_count": len(matches),
        "matches": matches,
    }


def mcp_wake_payload(project_root: Path) -> dict:
    paths = mcp_truth_paths(project_root)
    wake_path = paths["wake"]
    try:
        reject_symlink_paths([wake_path], "WAKE.md")
        reject_symlink_components(wake_path, "WAKE.md")
        wake_md = wake_path.read_text()
    except (OSError, ValueError) as exc:
        raise ValueError(f"WAKE.md unreadable: {exc}") from exc
    return {
        "project_root": str(project_root),
        "path": wake_path.relative_to(project_root).as_posix(),
        "wake_md": wake_md,
    }


def mcp_call_tool(request: Request, name: str, arguments: dict) -> dict:
    if name == "morpheus_status":
        return mcp_tool_result(project_status_payload(mcp_project_root(arguments)))
    if name == "morpheus_diagnostics":
        return mcp_tool_result(diagnostics_payload(request, mcp_project_root(arguments)))
    if name == "morpheus_integrations":
        return mcp_tool_result(integration_manifest())
    if name == "morpheus_model_smoke":
        payload = model_smoke_payload(ModelSmokeRequest(**(arguments or {}))).model_dump()
        return mcp_tool_result(payload)
    if name == "morpheus_check_text":
        return mcp_tool_result(mcp_check_text_payload(arguments))
    if name == "morpheus_get_active_state":
        return mcp_tool_result(mcp_active_state_payload(mcp_project_root(arguments)))
    if name == "morpheus_get_evidence_for_claim":
        return mcp_tool_result(mcp_evidence_for_claim_payload(arguments))
    if name == "morpheus_get_wake":
        return mcp_tool_result(mcp_wake_payload(mcp_project_root(arguments)))
    raise ValueError(f"Unknown MCP tool: {name}")


def mcp_origin_allowed(request: Request) -> bool:
    origin = request.headers.get("origin") if hasattr(request, "headers") else None
    if not origin:
        return True
    origin_host = urlsplit(origin).hostname
    request_host = urlsplit(api_base_url(request)).hostname
    return origin_host in {"127.0.0.1", "localhost", request_host}


def mcp_payload(request: Request, payload: dict) -> dict | None:
    request_id = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        return mcp_error(request_id, -32600, "Invalid JSON-RPC request")

    method = payload.get("method")
    raw_params = payload.get("params")
    if raw_params is None:
        params = {}
    elif isinstance(raw_params, dict):
        params = raw_params
    else:
        return mcp_error(request_id, -32602, "MCP params must be a JSON object")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        protocol_version = params.get("protocolVersion") or MCP_PROTOCOL_VERSION
        return mcp_success(
            request_id,
            {
                "protocolVersion": protocol_version
                if protocol_version == MCP_PROTOCOL_VERSION
                else MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "morpheus",
                    "title": "Morpheus AI",
                    "version": "0.2.0b1",
                    "description": "Agent State Compiler with verifiable provenance.",
                },
                "instructions": (
                    "Use Morpheus tools to inspect project state, build handoffs, "
                    "check integrations, and smoke-test local models."
                ),
            },
        )
    if method == "tools/list":
        return mcp_success(request_id, {"tools": mcp_tool_definitions()})
    if method == "tools/call":
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            return mcp_error(request_id, -32602, "tools/call requires a tool name")
        try:
            result = mcp_call_tool(
                request,
                params["name"],
                params.get("arguments") if isinstance(params.get("arguments"), dict) else {},
            )
        except ValueError as exc:
            return mcp_error(request_id, -32602, str(exc))
        return mcp_success(request_id, result)
    return mcp_error(request_id, -32601, "Method not found")


def diagnostic_check(check_id: str, label: str, ok: bool, detail: str) -> dict:
    return {
        "id": check_id,
        "label": label,
        "ok": ok,
        "detail": detail,
    }


def morpheus_agent_section_current(existing: str) -> bool:
    required = [
        MORPHEUS_AGENT_BEGIN,
        MORPHEUS_AGENT_END,
        "morpheus prepare-agent",
        "morpheus handoff",
        "morpheus agent-connect --json",
        "morpheus diagnostics --json",
        "morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173",
        "Use `0.0.0.0` only for explicit user-approved trusted LAN testing",
        "Run compile and verify after meaningful changes",
    ]
    return all(snippet in existing for snippet in required)


def agent_bootstrap_diagnostic(request: Request, project_root: Path) -> dict:
    agents_path = project_root / "AGENTS.md"
    try:
        reject_symlink_paths([agents_path], "AGENTS.md")
        if not agents_path.exists():
            return diagnostic_check(
                "agent_bootstrap",
                "AGENTS.md bootstrap",
                False,
                "Run Bootstrap AGENTS.md",
            )
        if not agents_path.is_file():
            return diagnostic_check(
                "agent_bootstrap",
                "AGENTS.md bootstrap",
                False,
                "AGENTS.md path is not a file",
            )
        existing = agents_path.read_text()
    except (OSError, ValueError) as exc:
        return diagnostic_check(
            "agent_bootstrap",
            "AGENTS.md bootstrap",
            False,
            str(exc),
        )

    return diagnostic_check(
        "agent_bootstrap",
        "AGENTS.md bootstrap",
        morpheus_agent_section_current(existing),
        "AGENTS.md is current"
        if morpheus_agent_section_current(existing)
        else "Refresh AGENTS.md bootstrap",
    )


def diagnostics_payload(request: Request, project_root: Path) -> dict:
    api_base = api_base_url(request)
    status_payload = normalize_agent_state(project_status_payload(project_root))
    morpheus_dir = project_root / ".morpheus"
    wake_path = morpheus_dir / "WAKE.md"
    project_root_ok = _is_real_directory(project_root)
    wake_ok = bool(
        status_payload["compiled"]
        and wake_path.exists()
        and wake_path.is_file()
        and not wake_path.is_symlink()
    )

    checks = [
        diagnostic_check("backend", "Backend API", True, api_base),
        diagnostic_check(
            "project_root",
            "Project root",
            project_root_ok,
            str(project_root) if project_root_ok else "Path is missing or unsafe",
        ),
        diagnostic_check(
            "initialized",
            "Morpheus initialized",
            status_payload["initialized"],
            ".morpheus is ready" if status_payload["initialized"] else "Run Initialize",
        ),
        diagnostic_check(
            "compiled",
            "WAKE compiled",
            status_payload["compiled"],
            "WAKE.md is current" if status_payload["compiled"] else "Run Compile",
        ),
        diagnostic_check(
            "wake",
            "WAKE readable",
            wake_ok,
            str(wake_path) if wake_ok else "WAKE.md not available yet",
        ),
        agent_bootstrap_diagnostic(request, project_root),
    ]

    return {
        "service": "morpheus",
        "version": "0.2.0b1",
        "api_base": api_base,
        "project_root": str(project_root),
        "cwd": str(Path.cwd()),
        "state": status_payload,
        "checks": checks,
        "next_action": diagnostics_next_action(api_base, project_root, checks),
        "agent_connect_url": endpoint_url(api_base, "/agent/connect", project_root),
        "commands": {
            "agent_connect": "morpheus agent-connect --json",
            "diagnostics": "morpheus diagnostics --json",
            "serve": "morpheus serve --host 127.0.0.1 --port 8000",
            "initialize": "morpheus init",
            "compile": "morpheus compile",
            "read_wake": "morpheus wake",
            "verify": "morpheus verify --all",
            "serve_ui": "morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173",
        },
    }


def project_wake_or_none(project_root: Path) -> str | None:
    wake_path = project_root / ".morpheus" / "WAKE.md"
    if not wake_path.exists():
        return None
    try:
        reject_symlink_paths([wake_path], "WAKE.md")
        reject_symlink_components(wake_path, "WAKE.md")
        return wake_path.read_text()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"WAKE.md unreadable: {exc}") from exc


def agent_handoff_markdown(payload: dict) -> str:
    diagnostics_checks = payload["diagnostics"].get("checks", [])
    readiness_lines = [
        f"- {check['label']}: {'OK' if check['ok'] else 'Needs action'} - {check['detail']}"
        for check in diagnostics_checks
    ]
    command_lines = [
        f"- `{command}`"
        for command in payload["commands"].values()
    ]
    wake_md = payload.get("wake_md")
    wake_section = wake_md if wake_md else "WAKE.md is not compiled yet. Run `morpheus compile` first."

    return "\n".join([
        "# Morpheus Agent Handoff",
        "",
        f"Project: `{payload['project_root']}`",
        f"API: `{payload['api_base']}`",
        "",
        "## Agent Sequence",
        "",
        "1. Read this handoff.",
        "2. Run diagnostics and bootstrap AGENTS.md when needed.",
        "3. Compile, then read WAKE.md before edits.",
        "4. Make the requested change.",
        "5. Compile again and run `morpheus verify --all`.",
        "",
        "## Readiness",
        "",
        *readiness_lines,
        "",
        "## Commands",
        "",
        *command_lines,
        "",
        "## Agent Prompt",
        "",
        payload["manifest"]["agent_prompt"],
        "",
        "## AGENTS.md Preview",
        "",
        "````markdown",
        payload["agent_bootstrap_preview"]["content"].rstrip(),
        "````",
        "",
        "## WAKE.md",
        "",
        "````markdown",
        wake_section.rstrip(),
        "````",
        "",
    ])


def agent_handoff_payload(request: Request, project_root: Path) -> dict:
    api_base = api_base_url(request)
    commands = {
        "prepare_agent": "morpheus prepare-agent",
        "prepare_agent_json": "morpheus prepare-agent --json",
        "handoff": "morpheus handoff",
        "handoff_json": "morpheus handoff --json",
        "agent_connect": "morpheus agent-connect --json",
        "diagnostics": "morpheus diagnostics --json",
        "bootstrap_preview": "morpheus bootstrap-agent --dry-run",
        "bootstrap_write": "morpheus bootstrap-agent",
        "compile": "morpheus compile",
        "read_wake": "morpheus wake",
        "verify": "morpheus verify --all",
        "serve_ui": "morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173",
    }
    payload = {
        "service": "morpheus",
        "version": "0.2.0b1",
        "api_base": api_base,
        "project_root": str(project_root),
        "manifest": agent_connect_payload(request, project_root),
        "diagnostics": diagnostics_payload(request, project_root),
        "agent_bootstrap_preview": preview_agent_bootstrap(request, project_root).model_dump(),
        "wake_md": project_wake_or_none(project_root),
        "commands": commands,
    }
    payload["markdown"] = agent_handoff_markdown(payload)
    return payload


def prepare_step(step_id: str, label: str, ok: bool, detail: str) -> dict:
    return {
        "id": step_id,
        "label": label,
        "ok": ok,
        "detail": detail,
    }


def agent_prepare_payload(request: Request, project_root: Path) -> dict:
    project_root_text = str(project_root)
    steps = []

    initialized = init_project(InitRequest(project_root=project_root_text))
    steps.append(prepare_step(
        "initialize",
        "Initialize",
        True,
        "Created .morpheus" if initialized.created else ".morpheus already exists",
    ))

    compiled = compile(CompileRequest(project_root=project_root_text))
    steps.append(prepare_step(
        "compile",
        "Compile",
        True,
        f"{compiled.source_count} sources, receipt {compiled.receipt_id}",
    ))

    bootstrapped = write_agent_bootstrap(request, project_root)
    if bootstrapped.created:
        bootstrap_detail = "Created AGENTS.md"
    elif bootstrapped.updated:
        bootstrap_detail = "Updated AGENTS.md"
    else:
        bootstrap_detail = "AGENTS.md already current"
    steps.append(prepare_step(
        "bootstrap_agent",
        "Bootstrap AGENTS.md",
        True,
        bootstrap_detail,
    ))

    verified = verify(VerifyRequest(project_root=project_root_text))
    steps.append(prepare_step(
        "verify",
        "Verify",
        verified.valid,
        f"Receipt {verified.receipt_id}" if verified.valid else "; ".join(verified.errors),
    ))

    handoff = agent_handoff_payload(request, project_root)
    steps.append(prepare_step(
        "handoff",
        "Handoff",
        True,
        "Handoff bundle ready",
    ))

    return {
        "service": "morpheus",
        "version": "0.2.0b1",
        "api_base": api_base_url(request),
        "project_root": project_root_text,
        "steps": steps,
        "initialized": initialized.model_dump(),
        "compiled": compiled.model_dump(),
        "bootstrapped": bootstrapped.model_dump(),
        "verified": verified.model_dump(),
        "handoff": handoff,
    }


def morpheus_agent_section(request: Request, project_root: Path) -> str:
    return "\n".join([
        MORPHEUS_AGENT_BEGIN,
        "## Morpheus Bootstrap",
        "",
        "Fetch the Morpheus manifest before making changes:",
        "",
        "- Connect manifest: start the API/UI, then fetch "
        "`/agent/connect?project_root=<PROJECT_ROOT>`.",
        "- One-command prepare: `morpheus prepare-agent`.",
        "- Local handoff bundle: `morpheus handoff`.",
        "- Local CLI manifest: `morpheus agent-connect --json`.",
        "- Local diagnostics: `morpheus diagnostics --json`.",
        "- Read `WAKE.md` before edits.",
        "- Run compile and verify after meaningful changes.",
        "- If the API/UI are unavailable, start them with "
        "`morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173`.",
        "- Use `0.0.0.0` only for explicit user-approved trusted LAN testing or "
        "authenticated proxy environments.",
        "",
        "Agent sequence:",
        "",
        "1. Fetch `/agent/connect` for this project root.",
        "2. Initialize only when `state.initialized` is false.",
        "3. Compile, then read WAKE.md.",
        "4. Make the requested project change.",
        "5. Compile again and run `morpheus verify --all`.",
        MORPHEUS_AGENT_END,
    ])


def merge_morpheus_agent_section(existing: str, section: str) -> str:
    start = existing.find(MORPHEUS_AGENT_BEGIN)
    end = existing.find(MORPHEUS_AGENT_END, start)
    if start != -1 and end != -1:
        end += len(MORPHEUS_AGENT_END)
        prefix = existing[:start].rstrip()
        suffix = existing[end:].lstrip()
        parts = [part for part in [prefix, section, suffix] if part]
        return "\n\n".join(parts).rstrip() + "\n"

    if existing.strip():
        return existing.rstrip() + "\n\n" + section + "\n"
    return "# AGENTS.md\n\n" + section + "\n"


def agent_bootstrap_response(
    request: Request,
    project_root: Path,
    *,
    write: bool,
) -> AgentBootstrapResponse:
    if not _is_real_directory(project_root):
        raise HTTPException(
            status_code=400,
            detail="Project root must be an existing real directory",
        )

    agents_path = project_root / "AGENTS.md"
    try:
        reject_symlink_paths([agents_path], "AGENTS.md")
        if agents_path.exists() and not agents_path.is_file():
            raise ValueError("AGENTS.md path is not a file")
        created = not agents_path.exists()
        existing = agents_path.read_text() if agents_path.exists() else ""
        content = merge_morpheus_agent_section(
            existing,
            morpheus_agent_section(request, project_root),
        )
        updated = content != existing
        if write and updated:
            agents_path.write_text(content)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AgentBootstrapResponse(
        project_root=str(project_root),
        path=str(agents_path),
        created=created,
        updated=updated,
        content=content,
        agent_connect_url=endpoint_url(api_base_url(request), "/agent/connect", project_root),
    )


def preview_agent_bootstrap(request: Request, project_root: Path) -> AgentBootstrapResponse:
    return agent_bootstrap_response(request, project_root, write=False)


def write_agent_bootstrap(request: Request, project_root: Path) -> AgentBootstrapResponse:
    return agent_bootstrap_response(request, project_root, write=True)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0b1"}


@app.get("/.well-known/morpheus.json")
def well_known_morpheus(request: Request):
    """Discovery document for tools and agents looking for Morpheus."""
    api_base = api_base_url(request)
    return {
        "service": "morpheus",
        "version": "0.2.0b1",
        "description": "Agent State Compiler with verifiable provenance",
        "connect_url": f"{api_base}/agent/connect",
        "handoff_url": f"{api_base}/agent/handoff",
        "handoff_markdown_url": f"{api_base}/agent/handoff.md",
        "agent_card_url": f"{api_base}/.well-known/agent-card.json",
        "mcp_url": f"{api_base}/mcp",
        "quickstart_url": f"{api_base}/quickstart",
        "docs": {
            "human_quickstart": "README.md",
            "state_file": ".morpheus/WAKE.md",
        },
    }


@app.get("/quickstart")
def quickstart(request: Request, project_root: Optional[str] = None):
    """Return one-screen install, run, and connect instructions."""
    root = Path(project_root) if project_root else Path.cwd()
    return quickstart_payload(request, root)


@app.get("/.well-known/agent-card.json")
def a2a_agent_card(request: Request):
    """Return an A2A-compatible Agent Card for Morpheus discovery."""
    return PlainTextResponse(
        json.dumps(a2a_agent_card_payload(request)),
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.post("/mcp")
def mcp_endpoint(request: Request, payload: dict = Body(default=None)):
    """Minimal MCP Streamable HTTP endpoint exposing read-only Morpheus tools."""
    if not mcp_origin_allowed(request):
        raise HTTPException(status_code=403, detail="Origin not allowed")
    response = mcp_payload(request, payload)
    if response is None:
        return PlainTextResponse("", media_type="application/json", headers={})
    return response


@app.get("/agent/connect")
def agent_connect(request: Request, project_root: Optional[str] = None):
    """Return a machine-readable connection manifest for autonomous agents."""
    root = Path(project_root) if project_root else Path.cwd()
    return agent_connect_payload(request, root)


@app.get("/agent/handoff")
def agent_handoff(request: Request, project_root: Optional[str] = None):
    """Return a complete agent handoff bundle for the selected project."""
    root = Path(project_root) if project_root else Path.cwd()
    return agent_handoff_payload(request, root)


@app.get("/agent/handoff.md", response_class=PlainTextResponse)
def agent_handoff_markdown_route(request: Request, project_root: Optional[str] = None):
    """Return the complete agent handoff as plain markdown."""
    root = Path(project_root) if project_root else Path.cwd()
    return PlainTextResponse(
        agent_handoff_payload(request, root)["markdown"],
        media_type="text/markdown",
    )


@app.post("/agent/prepare")
def agent_prepare(request: Request, prepare_request: AgentPrepareRequest):
    """Initialize, compile, bootstrap AGENTS.md, verify, and return handoff."""
    root = (
        Path(prepare_request.project_root)
        if prepare_request.project_root
        else Path.cwd()
    )
    return agent_prepare_payload(request, root)


@app.get("/diagnostics")
def diagnostics(request: Request, project_root: Optional[str] = None):
    """Return backend and project readiness diagnostics for the UI."""
    root = Path(project_root) if project_root else Path.cwd()
    return diagnostics_payload(request, root)


@app.get("/integrations")
def integrations():
    """Return integration setup status for humans, UI, and agents."""
    return integration_manifest()


def model_smoke_payload(smoke_request: ModelSmokeRequest) -> ModelSmokeResponse:
    from morpheus.training.eval import query_model

    base_model = (smoke_request.base_model or DEFAULT_MODEL_SMOKE_MODEL).strip()
    prompt = (smoke_request.prompt or DEFAULT_MODEL_SMOKE_PROMPT).strip()
    if not base_model:
        base_model = DEFAULT_MODEL_SMOKE_MODEL
    if not prompt:
        prompt = DEFAULT_MODEL_SMOKE_PROMPT

    answer = query_model(prompt, base_model=base_model)
    if answer.startswith("Error:"):
        return ModelSmokeResponse(
            ok=False,
            base_model=base_model,
            prompt=prompt,
            answer="",
            error=answer,
        )
    return ModelSmokeResponse(
        ok=True,
        base_model=base_model,
        prompt=prompt,
        answer=answer,
        error=None,
    )


@app.post("/models/smoke", response_model=ModelSmokeResponse)
def model_smoke(smoke_request: ModelSmokeRequest):
    """Run a direct local model smoke test through Ollama."""
    return model_smoke_payload(smoke_request)


@app.get("/config")
def get_project_config(project_root: Optional[str] = None):
    """Return Morpheus project context source configuration."""
    root = Path(project_root) if project_root else Path.cwd()
    return project_config_payload(root)


@app.post("/config")
def update_project_config(config_request: ProjectConfigRequest):
    """Save Morpheus project context source configuration."""
    root = (
        Path(config_request.project_root)
        if config_request.project_root
        else Path.cwd()
    )
    return write_project_config(root, config_request.watch_dirs)


@app.post("/agent/bootstrap", response_model=AgentBootstrapResponse)
def agent_bootstrap(request: Request, bootstrap_request: AgentBootstrapRequest):
    """Create or refresh the Morpheus-managed AGENTS.md section."""
    root = (
        Path(bootstrap_request.project_root)
        if bootstrap_request.project_root
        else Path.cwd()
    )
    return write_agent_bootstrap(request, root)


@app.post("/agent/bootstrap/preview", response_model=AgentBootstrapResponse)
def agent_bootstrap_preview(request: Request, bootstrap_request: AgentBootstrapRequest):
    """Preview the Morpheus-managed AGENTS.md content without writing it."""
    root = (
        Path(bootstrap_request.project_root)
        if bootstrap_request.project_root
        else Path.cwd()
    )
    return preview_agent_bootstrap(request, root)


@app.post("/init", response_model=InitResponse)
def init_project(request: InitRequest):
    """Initialize Morpheus project state for the desktop UI."""
    project_root = Path(request.project_root) if request.project_root else Path.cwd()
    morpheus_dir = project_root / ".morpheus"
    created = not morpheus_dir.exists()

    try:
        MorpheusConfig(project_root=project_root).init_default()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return InitResponse(
        initialized=True,
        project_root=str(project_root),
        created=created,
    )


@app.get("/wake")
def get_project_wake(project_root: Optional[str] = None):
    """Get WAKE.md for an explicit project root."""
    root = Path(project_root) if project_root else Path.cwd()
    morpheus_dir = root / ".morpheus"

    if not _is_real_directory(root) or not _is_real_directory(morpheus_dir):
        raise HTTPException(status_code=400, detail="Not initialized")

    wake_path = morpheus_dir / "WAKE.md"
    try:
        reject_symlink_paths([wake_path], "WAKE.md")
        reject_symlink_components(wake_path, "WAKE.md")
        if not wake_path.exists():
            raise HTTPException(status_code=404, detail="WAKE.md not found")
        wake_md = wake_path.read_text()
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"WAKE.md unreadable: {exc}") from exc

    return {"project_root": str(root), "wake_md": wake_md}

@app.post("/compile", response_model=CompileResponse)
def compile(request: CompileRequest):
    """Compile project state"""
    project_root = Path(request.project_root) if request.project_root else Path.cwd()
    morpheus_dir = project_root / ".morpheus"
    
    if not _is_real_directory(project_root):
        raise HTTPException(status_code=400, detail="Not initialized. Run 'morpheus init'")

    if not _is_real_directory(morpheus_dir):
        raise HTTPException(status_code=400, detail="Not initialized. Run 'morpheus init'")
    
    # Compile
    try:
        state = compile_project(project_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    # Get previous receipt
    receipts_dir = morpheus_dir / "receipts"
    prev_hash = None
    if receipts_dir.exists():
        latest = latest_receipt_or_http_error(receipts_dir)
        if latest:
            try:
                prev_hash = compute_sha256_file(latest)
            except OSError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Receipt chain invalid: {latest.name}: unreadable receipt ({exc})",
                ) from exc
    
    # Build sources
    sources_data = [{
        "id": s.id,
        "path": s.path,
        "sha256": s.sha256,
        "size_bytes": s.size_bytes,
        "line_count": s.line_count
    } for s in state.sources]
    
    # Generate final WAKE.md before signing so the receipt hashes the artifact on disk.
    receipt_id = new_receipt_id()
    state.receipt_id = receipt_id
    state_dump = state.model_dump()
    state_json = json.dumps(state_dump, indent=2, default=str)
    state_json_sha = compute_sha256_bytes(state_json.encode())
    evidence_jsonl = evidence_jsonl_bytes(state_dump.get("evidence", []))
    evidence_jsonl_sha = compute_sha256_bytes(evidence_jsonl)

    wake_md = generate_wake_md(state, receipt_id)
    wake_sha = compute_sha256_bytes(wake_md.encode())
    
    private_key_path = morpheus_dir / "keys" / "local.key"
    try:
        receipt = build_receipt(
            state_dump,
            wake_sha,
            sources_data,
            private_key_path,
            prev_hash,
            receipt_id=receipt_id,
            state_json_sha=state_json_sha,
            evidence_jsonl_sha=evidence_jsonl_sha,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Signing failed: {exc}") from exc
    
    wake_path = morpheus_dir / "WAKE.md"
    state_path = morpheus_dir / "state.json"
    evidence_path = morpheus_dir / "evidence.jsonl"
    receipt_path = receipts_dir / receipt_file_name(receipt["receipt_id"])
    audit_log = receipts_dir / "audit.log"

    try:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_paths(
            [wake_path, state_path, evidence_path, receipt_path, audit_log],
            "Output path",
        )

        # Write WAKE with real receipt
        wake_path.write_text(wake_md)

        # Save state
        state_path.write_text(state_json)

        # Save evidence
        evidence_path.write_bytes(evidence_jsonl)
        
        # Save receipt
        receipt_path.write_text(json.dumps(receipt, indent=2, default=str))

        with audit_log.open("a") as f:
            f.write(f"{receipt['issued_at']} {receipt['receipt_id']}\n")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Output write failed: {exc}") from exc
    
    return CompileResponse(
        receipt_id=receipt["receipt_id"],
        claim_count=receipt["claim_count"],
        source_count=len(state.sources),
        wake_md=wake_md
    )

@app.get("/wake/{project}")
def get_wake(project: str):
    """Get WAKE.md for a project"""
    project_path = Path(project)
    if (
        project in ("", ".", "..")
        or project_path.is_absolute()
        or project_path.name != project
    ):
        raise HTTPException(status_code=400, detail="Invalid project name")

    # Look for project in common locations
    possible_paths = [
        Path.home() / ".morpheus" / project / "WAKE.md",
        Path.cwd() / project / "WAKE.md",
        Path(project) / "WAKE.md",
    ]
    
    for p in possible_paths:
        if _has_symlink_component(p):
            continue
        if p.exists():
            try:
                wake_md = p.read_text()
            except OSError as exc:
                raise HTTPException(status_code=400, detail=f"WAKE.md unreadable: {exc}") from exc
            return {"project": project, "wake_md": wake_md}
    
    raise HTTPException(status_code=404, detail="WAKE.md not found")

@app.post("/verify")
def verify(
    request: VerifyRequest | None = Body(default=None),
    project_root: Optional[str] = None,
):
    """Verify receipt chain"""
    from morpheus.core.verify import verify_receipt_chain
    
    root_value = project_root or (request.project_root if request else None)
    root = Path(root_value) if root_value else Path.cwd()
    morpheus_dir = root / ".morpheus"
    
    if not _is_real_directory(root):
        raise HTTPException(status_code=400, detail="Not initialized")

    if not _is_real_directory(morpheus_dir):
        raise HTTPException(status_code=400, detail="Not initialized")
    
    valid, errors = verify_receipt_chain(morpheus_dir)
    
    receipts_dir = morpheus_dir / "receipts"
    latest_path = None
    if receipts_dir.exists() and valid:
        latest_path = latest_receipt_or_http_error(receipts_dir)
    receipt_id = "none"
    if latest_path:
        receipt_id = load_json_object_or_http_error(
            latest_path,
            "Latest receipt",
        ).get("receipt_id", latest_path.stem)
    
    return VerifyResponse(
        valid=valid,
        errors=errors,
        receipt_id=receipt_id,
    )

@app.get("/status")
def status(project_root: Optional[str] = None):
    """Get project status"""
    root = Path(project_root) if project_root else Path.cwd()
    return project_status_payload(root)
