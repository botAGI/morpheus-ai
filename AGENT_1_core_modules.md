# Agent 1 — Core Python Modules

## Your Task

Implement the following files in `/Users/testbot/.openclaw/workspace/morpheus-ai/morpheus/core/`:

### 1. `__init__.py`
Export all public classes/functions.

### 2. `models.py`
Pydantic models:

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class Source(BaseModel):
    id: str
    path: str
    kind: str  # "markdown", "json", "email", "file"
    sha256: str
    size_bytes: int
    line_count: int
    modified_at: datetime

class Claim(BaseModel):
    id: str
    source_id: str
    line_start: int
    line_end: int
    excerpt: str
    status: str = "active"  # "active", "superseded", "unverified"
    category: str = "fact"  # "decision", "task", "preference", "fact"
    inference: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Evidence(BaseModel):
    id: str
    claim_id: str
    source_id: str
    path: str
    line_start: int
    line_end: int
    excerpt: str
    source_sha256: str
    excerpt_sha256: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ProjectState(BaseModel):
    project_name: str
    root_sha: str
    compiled_at: datetime
    sources: list[Source] = []
    claims: list[Claim] = []
    evidence: list[Evidence] = []
    open_questions: list[str] = []

class Receipt(BaseModel):
    schema: str = "morpheus-receipt/1"
    receipt_id: str
    project: dict
    wake_md_sha256: str
    state_json_sha256: str
    evidence_jsonl_sha256: str
    sources: list[dict]
    claim_count: dict
    tool: dict
    issued_at: datetime
    previous_receipt_sha256: Optional[str] = None
    signature: dict
```

### 3. `config.py`
Handle `.morpheus/config.toml`:

```python
from pathlib import Path
import toml

class MorpheusConfig:
    DEFAULT_DIR = Path.home() / ".morpheus"
    
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.morpheus_dir = self.project_root / ".morpheus"
        self.config_path = self.morpheus_dir / "config.toml"
    
    def load(self) -> dict:
        if self.config_path.exists():
            return toml.load(self.config_path)
        return {}
    
    def save(self, config: dict):
        self.morpheus_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            toml.dump(config, f)
    
    def init_default(self):
        default = {
            "project": {"name": self.project_root.name},
            "integrations": {},
            "retention": {"receipts_days": 365}
        }
        self.save(default)
        # Generate keypair
        self.generate_keys()
    
    def generate_keys(self):
        from .provenance import generate_keypair
        keys_dir = self.morpheus_dir / "keys"
        keys_dir.mkdir(parents=True, exist_ok=True)
        generate_keypair(keys_dir / "local.key", keys_dir / "local.pub")
```

### 4. `provenance.py`
Receipt chain + ed25519 signing:

```python
import hashlib
import base64
import json
from pathlib import Path
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def generate_keypair(private_key_path: Path, public_key_path: Path):
    private_key = ed25519.Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()
    )
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo
    )
    private_key_path.write_bytes(private_bytes)
    public_key_path.write_bytes(public_bytes)
    private_key_path.chmod(0o600)
    public_key_path.chmod(0o644)
    return private_key, private_key.public_key()

def sign_data(data: dict, private_key_path: Path) -> str:
    private_bytes = private_key_path.read_bytes()
    private_key = serialization.load_pem_private_key(private_bytes, password=None)
    data_bytes = json.dumps(data, sort_keys=True, default=str).encode()
    signature = private_key.sign(data_bytes)
    return base64.b64encode(signature).decode()

def verify_signature(data: dict, signature_b64: str, public_key_path: Path) -> bool:
    public_bytes = public_key_path.read_bytes()
    public_key = serialization.load_pem_public_key(public_bytes)
    data_bytes = json.dumps(data, sort_keys=True, default=str).encode()
    signature = base64.b64decode(signature_b64)
    try:
        public_key.verify(signature, data_bytes)
        return True
    except:
        return False

def compute_sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compute_sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def build_receipt(state: dict, wake_md_sha256: str, sources: list, 
                  private_key_path: Path, previous_receipt_sha256: str | None = None) -> dict:
    from .models import Receipt
    import uuid
    
    claim_count = {"active": 0, "superseded": 0, "unverified": 0}
    for c in state.get("claims", []):
        claim_count[c.get("status", "active")] += 1
    
    evidence_jsonl = "\n".join(json.dumps(e, sort_keys=True, default=str) for e in state.get("evidence", []))
    
    receipt_data = {
        "receipt_id": f"rcpt_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}",
        "project": state.get("project", {}),
        "wake_md_sha256": wake_md_sha256,
        "state_json_sha256": compute_sha256_str(json.dumps(state, sort_keys=True, default=str)),
        "evidence_jsonl_sha256": compute_sha256_str(evidence_jsonl),
        "sources": sources,
        "claim_count": claim_count,
        "tool": {"name": "morpheus", "version": "0.1.0"},
        "issued_at": datetime.utcnow().isoformat() + "Z",
        "previous_receipt_sha256": previous_receipt_sha256,
    }
    
    sig_b64 = sign_data(receipt_data, private_key_path)
    
    receipt_data["signature"] = {
        "algo": "ed25519",
        "key_id": "morpheus-local-key-001",
        "signature_b64": sig_b64
    }
    
    return receipt_data
```

### 5. `compiler.py`
Source compilation logic:

```python
import hashlib
import json
from pathlib import Path
from datetime import datetime
from .models import Source, Claim, Evidence, ProjectState

SUPPORTED_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".py", ".js", ".ts"}

def compute_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def extract_sources(project_root: Path) -> list[Source]:
    sources = []
    for path in project_root.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            if ".morpheus" in path.parts:
                continue
            content = path.read_bytes()
            lines = content.decode("utf-8", errors="replace").splitlines()
            sources.append(Source(
                id=f"src_{len(sources) + 1:04d}",
                path=str(path.relative_to(project_root)),
                kind=path.suffix.lstrip("."),
                sha256=compute_file_hash(path),
                size_bytes=len(content),
                line_count=len(lines),
                modified_at=datetime.fromtimestamp(path.stat().st_mtime)
            ))
    return sources

def extract_claims_from_source(source: Source, project_root: Path) -> list[Claim]:
    """Simple claim extraction - looks for TODO, FIXME, DECISION markers"""
    path = project_root / source.path
    lines = path.read_text("utf-8", errors="replace").splitlines()
    claims = []
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line.startswith("TODO:") or line.startswith("FIXME:") or line.startswith("[DECISION]"):
            claims.append(Claim(
                id=f"clm_{len(claims) + 1:04d}",
                source_id=source.id,
                line_start=i,
                line_end=i,
                excerpt=line,
                category="task" if "TODO" in line else "decision" if "DECISION" in line else "fact",
                created_at=datetime.utcnow()
            ))
    return claims

def compile_project(project_root: Path) -> ProjectState:
    sources = extract_sources(project_root)
    all_claims = []
    
    for source in sources:
        claims = extract_claims_from_source(source, project_root)
        all_claims.extend(claims)
    
    evidence = []
    for claim in all_claims:
        source = next(s for s in sources if s.id == claim.source_id)
        evidence.append(Evidence(
            id=f"ev_{len(evidence) + 1:04d}",
            claim_id=claim.id,
            source_id=claim.source_id,
            path=source.path,
            line_start=claim.line_start,
            line_end=claim.line_end,
            excerpt=claim.excerpt,
            source_sha256=source.sha256,
            excerpt_sha256=hashlib.sha256(claim.excerpt.encode()).hexdigest(),
            timestamp=datetime.utcnow()
        ))
    
    return ProjectState(
        project_name=project_root.name,
        root_sha=compute_file_hash(project_root / "SPEC.md") if (project_root / "SPEC.md").exists() else compute_file_hash(project_root),
        compiled_at=datetime.utcnow(),
        sources=sources,
        claims=all_claims,
        evidence=evidence,
        open_questions=[]
    )
```

### 6. `wake.py`
WAKE.md generator:

```python
from pathlib import Path
from datetime import datetime
from .models import ProjectState

def generate_wake_md(state: ProjectState, receipt_id: str) -> str:
    lines = [
        "# WAKE.md — Project State",
        "",
        f"**Compiled:** {state.compiled_at.isoformat()}Z",
        f"**Receipt:** {receipt_id}",
        f"**Morpheus:** v0.1.0",
        "",
        "---",
        "",
        "## Current State",
        "",
    ]
    
    # Group claims by category
    by_category = {}
    for claim in state.claims:
        cat = claim.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(claim)
    
    for cat, claims in by_category.items():
        lines.append(f"### {cat.capitalize()}s")
        for c in claims:
            lines.append(f"- {c.excerpt} ({c.source_id}:{c.line_start})")
        lines.append("")
    
    lines.extend([
        "## Questions",
        "",
    ])
    for q in state.open_questions:
        lines.append(f"- {q}")
    
    lines.extend([
        "",
        "## Evidence Summary",
        f"- {len(state.claims)} claims from {len(state.sources)} sources",
        f"- Last compiled: {state.compiled_at.isoformat()}Z",
        "",
        "---",
        "*Generated by Morpheus. Verify with `morpheus verify --provenance`.*",
    ])
    
    return "\n".join(lines)
```

### 7. `verify.py`
Verification logic:

```python
import json
from pathlib import Path
from .provenance import verify_signature, compute_sha256_file

def verify_receipt_chain(morpheus_dir: Path) -> tuple[bool, list[str]]:
    receipts_dir = morpheus_dir / "receipts"
    if not receipts_dir.exists():
        return False, ["No receipts directory"]
    
    receipt_files = sorted(receipts_dir.glob("receipt_*.json"))
    if not receipt_files:
        return False, ["No receipts found"]
    
    public_key_path = receipts_dir / "keys" / "local.pub"
    if not public_key_path.exists():
        return False, ["No public key found"]
    
    errors = []
    prev_hash = None
    
    for i, rf in enumerate(receipt_files):
        receipt = json.loads(rf.read_text())
        
        # Verify chain
        if i == 0:
            if receipt.get("previous_receipt_sha256") is not None:
                errors.append(f"First receipt has previous_receipt_sha256")
        else:
            if receipt.get("previous_receipt_sha256") != prev_hash:
                errors.append(f"Receipt {i} chain broken")
        
        # Verify signature
        sig_data = {k: v for k, v in receipt.items() if k != "signature"}
        if not verify_signature(sig_data, receipt["signature"]["signature_b64"], public_key_path):
            errors.append(f"Receipt {i} signature invalid")
        
        prev_hash = json.dumps(receipt, sort_keys=True, default=str)
    
    return len(errors) == 0, errors
```

## Instructions

1. Create all files in `/Users/testbot/.openclaw/workspace/morpheus-ai/morpheus/core/`
2. Use exact imports as shown above
3. Add docstrings to each module
4. Ensure imports are correct
5. Create `__init__.py` that exports: `Source, Claim, Evidence, ProjectState, Receipt, MorpheusConfig, compile_project, generate_wake_md, verify_receipt_chain`
