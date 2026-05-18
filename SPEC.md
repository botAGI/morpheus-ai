# Morpheus Product Specification

## 1. Product Frame

Morpheus is a source-grounded truth layer with an experimental learning core.

It verifies coding-agent claims against project sources, compiles current state
into `WAKE.md`, and can run a local learning lab that turns accepted
source-backed claims into experimental adapter training data.

Morpheus is not a personal AI agent. Morpheus is not a generic memory layer.
Morpheus is not a real-world truth oracle. It verifies source-grounded project
truth. Local adapter learning is experimental and remains behind check, review,
eval, and rollback gates.

**Public pitch:**

```text
Stop coding agents from hallucinating about your repo.
First verify. Then learn.
```

**Core loop:**

```text
sources -> WAKE.md -> check -> accepted dataset -> local adapter lab -> eval
```

## 2. Core Differentiation

- **GitHub-native state artifact**: `WAKE.md` sits next to `README.md` and
  `AGENTS.md`.
- **Source-grounded claim verification**: `morpheus check` classifies agent text
  as `verified`, `stale`, `incorrect`, or `unknown` against local project state.
- **Current project truth, not old fragments**: Morpheus compiles what is
  supported now, instead of only retrieving past notes.
- **Verifiable provenance**: state is backed by `state.json`, `evidence.jsonl`,
  SHA-256 hashes, and signed ed25519 receipts.
- **Agent handoff**: CLI, HTTP, A2A-style discovery, and MCP expose the same
  state to coding agents.
- **Local-first operation**: private projects and vaults can keep generated
  state under `.morpheus/`.
- **Review-gated semantic state**: inferred claims must be labeled and reviewed
  before they become active state.
- **Experimental learning lab**: accepted source-backed claims can become a local
  dataset for adapter experiments. Adapter output is not the source of truth;
  source spans and `morpheus check` remain the gate.

## 3. Non-Goals

- Morpheus does not autonomously run as the agent.
- Morpheus does not claim legal compliance as a product guarantee.
- Morpheus does not use raw private vaults as model-training input.
- Morpheus does not activate adapters without eval and rollback support.

The safe framing is:

```text
Designed for provenance, local-first operation, source attribution,
and user-controlled state export.
```

## 4. Architecture

```text
┌──────────────────────────────────────────────────────┐
│                    MORPHEUS                          │
├──────────────────────────────────────────────────────┤
│  Public Repo Primitives                              │
│  - README.md: human explanation                      │
│  - AGENTS.md: agent instructions                     │
│  - WAKE.md: current project state                    │
├──────────────────────────────────────────────────────┤
│  Core Engine                                         │
│  - Deterministic compiler                            │
│  - WAKE.md generator                                 │
│  - Local claim checker                               │
│  - Evidence extraction from sources                  │
│  - Receipt chain and ed25519 signing                 │
│  - Verification CLI                                  │
│  - Reviewed dataset compiler                         │
│  - Autonomous learning lab                           │
├──────────────────────────────────────────────────────┤
│  Agent Interfaces                                    │
│  - CLI handoff                                       │
│  - HTTP /agent/connect                               │
│  - A2A-style Agent Card                              │
│  - MCP Streamable HTTP endpoint                      │
├──────────────────────────────────────────────────────┤
│  Optional Surfaces                                   │
│  - Browser UI launchpad                              │
│  - Local integration cache readers                   │
│  - Experimental consolidation and adapter helpers    │
└──────────────────────────────────────────────────────┘
```

## 5. Repository Structure

```text
morpheus-ai/
├── README.md                  # Human-facing pitch and quick start
├── README.ru.md               # Russian README
├── AGENTS.md                  # Agent instructions
├── WAKE.md                    # Public showcase of compiled project state
├── SPEC.md                    # Product and architecture specification
├── pyproject.toml             # Python package config
├── morpheus/
│   ├── cli.py                 # Typer CLI
│   ├── core/                  # Compiler, models, wake generation, provenance
│   ├── integrations/          # Filesystem and local integration caches
│   ├── api/                   # FastAPI, handoff, MCP, A2A-style discovery
│   └── training/              # Experimental adapter helpers
├── ui/                        # Static browser UI and Tauri shell
├── docs/                      # Testing, launch notes, product framing
└── tests/                     # pytest suite
```

## 6. WAKE.md Format

```markdown
# WAKE.md - Project State

**Compiled:** 2026-05-17T17:16:59Z
**Receipt:** rcpt_20260517T171659Z_da4bf751
**Morpheus:** v0.1.1

---

## Current State

### Active Decisions
- DECISION: The public primitive is WAKE.md. *(src:src_001:12)*

### Open Tasks
- TODO: Expand richer stale-claim detection. *(src:src_002:8)*

## Source References
- src_001: `SPEC.md`
- src_002: `docs/ROADMAP.md`

## Evidence Summary
- 2 active claims from 2 sources
- Compiled: 2026-05-17T17:16:59Z

---

*Generated by Morpheus. Verify with `morpheus verify --all`.*
```

## 7. Compiler Modes

### v0.1 Deterministic Compiler

The current compiler extracts explicit source markers:

```text
TODO: DECISION: FIXME: NOTE: HACK: XXX:
```

This is intentionally simple. It gives reproducible receipts and predictable
evidence, but it will produce weak `WAKE.md` files for projects that never mark
decisions or tasks explicitly.

### Semantic Compiler Alpha

`morpheus compile --semantic` should read high-signal sources such as README,
SPEC, AGENTS, CHANGELOG, issues, and notes. It should create candidate claims
with one of these labels:

- `source_backed`: directly supported by cited source text.
- `inferred`: derived by model reasoning from source-backed material.
- `needs_review`: useful but not safe to activate automatically.

Only reviewed claims should become active state.

The default semantic alpha provider is local/offline heuristic extraction. It is
safe for deterministic dogfood and never calls cloud providers unless the user
explicitly configures a provider such as Ollama or a future cloud backend.

## 8. CLI Commands

PyPI distribution name:

```toml
name = "morpheus-wake"
```

The installed executable remains `morpheus`.

```bash
morpheus wake .                 # Init if needed, compile, verify, write root WAKE.md
morpheus wake . --private       # Keep WAKE.md under .morpheus/
morpheus check                  # Check agent text from stdin against local state
morpheus check --input FILE     # Check agent text from a file
morpheus stale .                # Report stale launch-positioning claims
morpheus learn lab .            # Run autonomous local learning experiment
morpheus learn dataset .        # Build dataset from accepted candidates
morpheus learn train . --dry-run
morpheus learn eval .
morpheus init                   # Initialize .morpheus/
morpheus compile                # Compile sources into state artifacts
morpheus verify --all           # Verify receipt chain and artifacts
morpheus wake                   # Print .morpheus/WAKE.md
morpheus status                 # Show state summary
morpheus prepare-agent          # Prepare AGENTS.md + handoff
morpheus handoff                # Print copyable agent handoff
morpheus agent-connect --json   # Print machine-readable agent manifest
morpheus diagnostics --json     # Print readiness checks
morpheus serve --ui             # Run backend and browser UI
```

## 9. Data Artifacts

### Source

```python
class Source(BaseModel):
    id: str
    path: str
    kind: str
    sha256: str
    size_bytes: int
    line_count: int
    modified_at: datetime
```

### Claim

```python
class Claim(BaseModel):
    id: str
    source_id: str
    line_start: int
    line_end: int
    excerpt: str
    status: str
    category: str
    inference: bool
    created_at: datetime
```

### Evidence

```python
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
    timestamp: datetime
```

### Receipt

```python
class Receipt(BaseModel):
    schema: str
    receipt_id: str
    project: dict
    wake_md_sha256: str
    state_json_sha256: str
    evidence_jsonl_sha256: str
    sources: list[dict]
    claim_count: dict
    tool: dict
    issued_at: datetime
    previous_receipt_sha256: str | None
    signature: dict
```

### Learning Gate

Training examples may be created only from claims that satisfy all of these:

- `status == accepted`
- `label == source_backed`
- source path exists and is not ignored
- source SHA/span still match
- evidence excerpt matches the source span
- no secret-like content
- not pending, rejected, `needs_review`, or inferred-only

Outdated claims can become correction/negative examples. They must not become
positive project facts.

Activation requires eval. Production use requires rollback.

## 10. Integrations

Integrations contribute evidence. They do not change the product identity.

- `github`: issues, pull requests, commits, and cached metadata.
- `gmail`: local Gmail cache and OAuth-oriented token path.
- `calendar`: local Calendar cache and OAuth-oriented token path.
- `slack`: local Slack export cache plus optional token file.
- `linear`: local Linear issue cache plus optional token file.
- `filesystem`: project files, docs, specs, notes, and vaults.

Local tokens and caches should stay outside the repository by default.

## 11. Security Model

- Default bind address should be `127.0.0.1`.
- `.morpheus/`, receipts, caches, generated datasets, model outputs, and token
  files must stay out of git unless explicitly exported.
- Symlink escapes must be rejected for project roots, watched paths, output
  paths, and sensitive state files.
- MCP and A2A-style endpoints are automation surfaces and should be exposed only
  in trusted environments.

## 12. Launch Acceptance Criteria

- [x] First README screen frames Morpheus as `WAKE.md for AI agents`.
- [x] Root `WAKE.md` is committed as a public showcase.
- [x] `morpheus wake .` provides a one-command demo path.
- [x] `morpheus stale .` detects known stale positioning claims.
- [x] SPEC avoids agent, training, and legal-overclaim positioning.
- [x] LoRA/training is documented as experimental.
- [x] Receipts are ed25519 signed and verifiable.
- [x] `morpheus verify --all` validates receipt chain and artifacts.
- [x] UI exposes setup, diagnostics, integrations, FAQ, MCP probe, and handoff.
- [x] Public hygiene tests reject local assistant artifacts.
- [x] Semantic compiler mode is implemented and review-gated in v0.2 alpha.
- [x] Before/after visual demo or GIF is published.
- [x] First GitHub release is cut.
