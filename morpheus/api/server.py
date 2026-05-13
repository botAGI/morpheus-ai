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
from morpheus.core.provenance import compute_sha256_file, build_receipt, receipt_file_name

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
        existing = sorted(receipts_dir.glob("receipt_*.json"))
        if existing:
            prev_hash = compute_sha256_file(existing[-1])
    
    # Build sources
    sources_data = [{
        "id": s.id,
        "path": s.path,
        "sha256": s.sha256,
        "size_bytes": s.size_bytes,
        "line_count": s.line_count
    } for s in state.sources]
    
    # Generate WAKE.md
    wake_md = generate_wake_md(state, "pending")
    
    # Sign
    temp_path = morpheus_dir / "temp_wake.md"
    temp_path.write_text(wake_md)
    wake_sha = compute_sha256_file(temp_path)
    temp_path.unlink()
    
    private_key_path = morpheus_dir / "keys" / "local.key"
    receipt = build_receipt(
        state.model_dump(),
        wake_sha,
        sources_data,
        private_key_path,
        prev_hash
    )
    
    # Update WAKE with real receipt
    wake_md = wake_md.replace("pending", receipt["receipt_id"])
    (morpheus_dir / "WAKE.md").write_text(wake_md)
    
    # Save receipt
    receipt_path = receipts_dir / receipt_file_name(receipt["receipt_id"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    
    return CompileResponse(
        receipt_id=receipt["receipt_id"],
        claim_count=receipt["claim_count"],
        source_count=len(state.sources),
        wake_md=wake_md
    )

@app.get("/wake/{project}")
def get_wake(project: str):
    """Get WAKE.md for a project"""
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
    latest = sorted(receipts_dir.glob("receipt_*.json"))[-1].name if receipts_dir.exists() else None
    
    return VerifyResponse(
        valid=valid,
        errors=errors,
        receipt_id=latest or "none"
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
