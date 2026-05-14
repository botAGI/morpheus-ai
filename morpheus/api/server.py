"""
Morpheus API Server
"""
from urllib.parse import urlencode
import json
import shlex
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from morpheus.core.config import MorpheusConfig
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

app = FastAPI(
    title="Morpheus API",
    description="Agent State Compiler API",
    version="0.1.0"
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


def api_base_url(request: Request) -> str:
    """Return the externally visible API base from the incoming request."""
    return str(request.base_url).rstrip("/")


def endpoint_url(api_base: str, path: str, project_root: Path | None = None) -> str:
    query = urlencode({"project_root": str(project_root)}) if project_root else ""
    return f"{api_base}{path}{'?' + query if query else ''}"


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
        "wake": {
            "method": "GET",
            "url": wake_url,
        },
        "verify": {
            "method": "POST",
            "url": endpoint_url(api_base, "/verify", project_root),
        },
    }

    connect_url = endpoint_url(api_base, "/agent/connect", project_root)
    return {
        "service": "morpheus",
        "version": "0.1.0",
        "api_base": api_base,
        "project_root": project_root_text,
        "state": normalize_agent_state(project_status_payload(project_root)),
        "sequence": [
            {
                "id": "status",
                "goal": "Check whether Morpheus already has compiled state.",
                "request": endpoints["status"],
            },
            {
                "id": "initialize_if_needed",
                "goal": "Create .morpheus when state.initialized is false.",
                "request": endpoints["initialize"],
            },
            {
                "id": "compile",
                "goal": "Compile project sources into WAKE.md and a signed receipt.",
                "request": endpoints["compile"],
            },
            {
                "id": "read_wake",
                "goal": "Load WAKE.md before making project changes.",
                "request": endpoints["wake"],
            },
            {
                "id": "verify",
                "goal": "Verify receipt integrity after compilation.",
                "request": endpoints["verify"],
            },
        ],
        "endpoints": endpoints,
        "cli": {
            "initialize": "morpheus init",
            "compile": "morpheus compile",
            "read_wake": "morpheus wake",
            "verify": "morpheus verify --all",
            "serve": "morpheus serve --host 0.0.0.0 --port 8000",
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
        },
        "agent_prompt": (
            "Fetch the connect manifest before working on this project. "
            f"Use {connect_url}, follow sequence in order, read WAKE.md before edits, "
            "and run compile plus verify after meaningful changes."
        ),
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/.well-known/morpheus.json")
def well_known_morpheus(request: Request):
    """Discovery document for tools and agents looking for Morpheus."""
    api_base = api_base_url(request)
    return {
        "service": "morpheus",
        "version": "0.1.0",
        "description": "Agent State Compiler with verifiable provenance",
        "connect_url": f"{api_base}/agent/connect",
        "docs": {
            "human_quickstart": "README.md",
            "state_file": ".morpheus/WAKE.md",
        },
    }


@app.get("/agent/connect")
def agent_connect(request: Request, project_root: Optional[str] = None):
    """Return a machine-readable connection manifest for autonomous agents."""
    root = Path(project_root) if project_root else Path.cwd()
    return agent_connect_payload(request, root)


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
