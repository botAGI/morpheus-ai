# Morpheus AI — Product Specification

## 1. Overview

**Morpheus AI** is an open-source personal AI agent with verifiable provenance and daily memory consolidation via LoRA fine-tuning.

**Core differentiation:**
- Input-side provenance (signed receipts, evidence chain)
- Daily training = weights-as-memory (vs retrieval-only competitors)
- EU AI Act compliant by design

**Positioning:** "Agent State Compiler with verifiable provenance" — stops AI agents from starting from scratch.

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│                    MORPHEUS AI                        │
├──────────────────────────────────────────────────────┤
│  UI Layer (Tauri + WebView)                          │
│  - Chat interface                                    │
│  - Voice (TTS/Whisper)                               │
│  - Optional: animated mascot                         │
├──────────────────────────────────────────────────────┤
│  Integration Layer                                   │
│  - Gmail, Calendar (Google API)                      │
│  - GitHub, Filesystem                                │
│  - Generic MCP gateway                               │
├──────────────────────────────────────────────────────┤
│  Core Engine                                         │
│  - State Compiler → WAKE.md                         │
│  - Provenance (receipt chain, ed25519 signing)      │
│  - Evidence extraction from sources                  │
│  - Verification CLI                                 │
├──────────────────────────────────────────────────────┤
│  Memory Layer (Phase 3)                              │
│  - QLoRA daily fine-tuning                           │
│  - Weights-as-memory (differentiator)              │
└──────────────────────────────────────────────────────┘
```

## 3. Directory Structure

```
morpheus-ai/
├── SPEC.md                    # This file
├── README.md
├── pyproject.toml             # Python package config
├── morpheus/                  # Python package
│   ├── __init__.py
│   ├── cli.py                # CLI: morpheus <command>
│   ├── core/                 # Core engine
│   │   ├── __init__.py
│   │   ├── models.py         # Pydantic models (Source, Claim, Receipt, etc)
│   │   ├── config.py         # morpheus.toml reading/writing
│   │   ├── compiler.py       # Compile sources → state.json
│   │   ├── provenance.py     # Receipt chain, ed25519 signing
│   │   ├── wake.py           # WAKE.md generator
│   │   └── verify.py         # morpheus verify --all
│   ├── integrations/         # External data sources
│   │   ├── __init__.py
│   │   ├── gmail.py          # Gmail API → emails as evidence
│   │   ├── calendar.py       # Google Calendar → events as evidence
│   │   ├── github.py         # GitHub API → PRs, issues, commits
│   │   └── filesystem.py     # Local files → sources
│   └── api/                  # FastAPI server (optional, for cloud)
│       ├── __init__.py
│       ├── server.py         # FastAPI app
│       └── routes/
│           ├── __init__.py
│           ├── compile.py     # POST /compile
│           ├── verify.py      # POST /verify
│           └── wake.py        # GET /wake/{project}
├── ui/                       # Tauri desktop app
│   ├── src/
│   │   ├── main.rs           # Tauri entry
│   │   ├── lib.rs            # App logic
│   │   └── styles.css
│   ├── index.html
│   ├── Cargo.toml
│   └── tauri.conf.json
├── assets/                   # UI assets (mascot, icons)
└── tests/
    ├── __init__.py
    ├── test_compiler.py
    ├── test_provenance.py
    └── test_verify.py
```

## 4. Core Data Models

### Source
```python
class Source(BaseModel):
    id: str                    # "src_001"
    path: str                  # "README.md"
    kind: str                  # "markdown", "json", "email", etc
    sha256: str                # Full file hash at capture
    size_bytes: int
    line_count: int
    modified_at: datetime
```

### Claim
```python
class Claim(BaseModel):
    id: str                    # "clm_001"
    source_id: str              # Reference to Source.id
    line_start: int
    line_end: int
    excerpt: str               # Exact text
    status: str                # "active", "superseded", "unverified"
    category: str              # "decision", "task", "preference", "fact"
    inference: bool             # True if derived by model
    created_at: datetime
```

### Evidence
```python
class Evidence(BaseModel):
    id: str                    # "ev_001"
    claim_id: str
    source_id: str
    path: str
    line_start: int
    line_end: int
    excerpt: str
    source_sha256: str         # Hash of source file at capture
    excerpt_sha256: str        # Hash of excerpt bytes
    timestamp: datetime
```

### Receipt
```python
class Receipt(BaseModel):
    schema: str                # "morpheus-receipt/1"
    receipt_id: str            # "rcpt_20260512T154233Z_a1b2c3"
    project: dict              # {"name": "...", "root_sha": "..."}
    wake_md_sha256: str
    state_json_sha256: str
    evidence_jsonl_sha256: str
    sources: list[dict]        # [{id, path, sha256, size_bytes, line_count}]
    claim_count: dict          # {"active": 23, "superseded": 7, "unverified": 2}
    tool: dict                 # {"name": "morpheus", "version": "0.1.0"}
    issued_at: datetime
    previous_receipt_sha256: str | None
    signature: dict            # {"algo": "ed25519", "key_id": "...", "signature_b64": "..."}
```

## 5. CLI Commands

```bash
morpheus init                    # Initialize .morpheus/ in current dir
morpheus compile                 # Compile sources → state.json + WAKE.md + receipt
morpheus verify                  # Verify receipt chain
morpheus verify --all           # Full provenance verification
morpheus status                  # Show current state summary
morpheus integrate <service>    # Connect Gmail/Calendar/GitHub
morpheus wake                    # Print WAKE.md to stdout
morpheus train                   # Phase 3: QLoRA fine-tune (future)
```

## 6. Integration Specs

### Gmail
- OAuth2 authentication
- Read emails from INBOX (last 30 days by default)
- Extract: subject, sender, date, snippet, thread_id
- Evidence: email body lines can back claims

### Google Calendar
- OAuth2 authentication
- Read events from primary calendar
- Extract: title, description, attendees, timestamps
- Evidence: event details can back claims

### GitHub
- Personal Access Token authentication
- Read: issues, PRs, commits, discussions
- Extract: title, body, state, author, dates
- Evidence: PR/issue content can back claims

### Filesystem
- Watch指定的 directories
- Track changes via git or inotify/fsevents
- Evidence: file content can back claims

## 7. WAKE.md Format

```markdown
# WAKE.md — Project State

**Compiled:** 2026-05-13T09:17:00Z  
**Receipt:** rcpt_20260513T091700Z_a1b2c3  
**Morpheus:** v0.1.0

---

## Current State

### Active Decisions
- ...

### Open Tasks
- ...

### Recent Changes
- ...

## Questions

## Evidence Summary
- 23 claims from 5 sources
- Last verified: 2026-05-13T09:17:00Z

---

*Generated by Morpheus. Verify with `morpheus verify --all`.*
```

## 8. Implementation Notes

### Python
- `typer` for CLI (like Typer/Styles)
- `pydantic` for models
- `cryptography` for ed25519 signing
- `httpx` for API calls
- `python-dotenv` for env vars

### Tauri
- Rust backend
- WebView frontend (vanilla JS + CSS for now)
- No heavy frameworks (avoid React/Vue complexity)

### Signing
- `cryptography.hazmat.primitives.asymmetric.ed25519`
- Generate keypair on `morpheus init`
- Store private key at `.morpheus/keys/local.key` (mode 0600)

## 9. Phase 3: Daily Training (Future)

```text
Daily cycle:
  Evening: sessions.jsonl → consolidation dataset
  Night: QLoRA fine-tune → adapter
  Morning: base model + adapter → "woken" agent
```

Not in v0.1. Spec'd for future.

## 10. Acceptance Criteria (v0.1)

- [x] `morpheus init` creates .morpheus/ with morpheus.toml
- [x] `morpheus compile` produces WAKE.md + receipt
- [x] Receipt is ed25519 signed
- [x] `morpheus verify` validates chain
- [ ] Integration with Gmail/Calendar works (OAuth flow) — Phase 2
- [x] Tauri app skeleton builds and runs
- [x] Enhanced UI with mascot animations, voice indicators
- [x] Automated test suite passes for core, CLI, API, and training modules
- [x] CLI improved with panels, tables, verbose mode
- [x] Phase 3 training pipeline (consolidate → train → eval)
