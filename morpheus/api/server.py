"""
Morpheus API Server
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

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

app = FastAPI(
    title="Morpheus API",
    description="Agent State Compiler API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompileRequest(BaseModel):
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


def latest_receipt_or_http_error(receipts_dir: Path) -> Path | None:
    """Return the receipt chain tail or fail with a client-visible API error."""
    try:
        return latest_receipt_file(receipts_dir)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Receipt chain invalid: {exc}") from exc


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

@app.post("/compile", response_model=CompileResponse)
def compile(request: CompileRequest):
    """Compile project state"""
    project_root = Path(request.project_root) if request.project_root else Path.cwd()
    morpheus_dir = project_root / ".morpheus"
    
    if not morpheus_dir.exists():
        raise HTTPException(status_code=400, detail="Not initialized. Run 'morpheus init'")
    
    # Compile
    state = compile_project(project_root)
    
    # Get previous receipt
    receipts_dir = morpheus_dir / "receipts"
    prev_hash = None
    if receipts_dir.exists():
        latest = latest_receipt_or_http_error(receipts_dir)
        if latest:
            prev_hash = compute_sha256_file(latest)
    
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
    
    # Write WAKE with real receipt
    (morpheus_dir / "WAKE.md").write_text(wake_md)

    # Save state
    state_path = morpheus_dir / "state.json"
    state_path.write_text(state_json)

    # Save evidence
    evidence_path = morpheus_dir / "evidence.jsonl"
    evidence_path.write_bytes(evidence_jsonl)
    
    # Save receipt
    receipt_path = receipts_dir / receipt_file_name(receipt["receipt_id"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))

    audit_log = receipts_dir / "audit.log"
    with audit_log.open("a") as f:
        f.write(f"{receipt['issued_at']} {receipt['receipt_id']}\n")
    
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
        if p.exists():
            return {"project": project, "wake_md": p.read_text()}
    
    raise HTTPException(status_code=404, detail="WAKE.md not found")

@app.post("/verify")
def verify(project_root: Optional[str] = None):
    """Verify receipt chain"""
    from morpheus.core.verify import verify_receipt_chain
    
    root = Path(project_root) if project_root else Path.cwd()
    morpheus_dir = root / ".morpheus"
    
    if not morpheus_dir.exists():
        raise HTTPException(status_code=400, detail="Not initialized")
    
    valid, errors = verify_receipt_chain(morpheus_dir)
    
    receipts_dir = morpheus_dir / "receipts"
    latest_path = None
    if receipts_dir.exists() and valid:
        latest_path = latest_receipt_file(receipts_dir)
    receipt_id = "none"
    if latest_path:
        receipt_id = json.loads(latest_path.read_text()).get("receipt_id", latest_path.stem)
    
    return VerifyResponse(
        valid=valid,
        errors=errors,
        receipt_id=receipt_id,
    )

@app.get("/status")
def status(project_root: Optional[str] = None):
    """Get project status"""
    root = Path(project_root) if project_root else Path.cwd()
    morpheus_dir = root / ".morpheus"
    
    state_path = morpheus_dir / "state.json"
    if not state_path.exists():
        return {"initialized": False}
    
    state = json.loads(state_path.read_text())
    return {
        "initialized": True,
        "sources": len(state.get("sources", [])),
        "claims": len(state.get("claims", [])),
        "evidence": len(state.get("evidence", [])),
        "compiled_at": state.get("compiled_at")
    }
