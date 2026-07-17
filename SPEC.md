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
│  - MCP truth tools over Streamable HTTP              │
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

**Compiled:** 2026-05-20T18:14:59Z
**Receipt:** rcpt_20260520T181459Z_c6e9028d
**Morpheus:** v0.2.0b1

---

## Current State

### Active Decisions
- DECISION: Morpheus verifies source-grounded project truth before learning. *(src:src_001:12)*

### Open Tasks
- TODO: Harden beta exit gates before stable v0.2.0. *(src:src_002:8)*

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

`morpheus compile --semantic --review` reads high-signal sources such as README,
SPEC, AGENTS, CHANGELOG, issues, and notes. It creates candidate claims with one
of these labels:

- `source_backed`: directly supported by cited source text.
- `inferred`: derived by model reasoning from source-backed material.
- `needs_review`: useful but not safe to activate automatically.

Only reviewed claims should become active state.

Semantic provider modes are explicit. `MORPHEUS_SEMANTIC_PROVIDER=local` is the
default offline heuristic provider for deterministic dogfood,
`MORPHEUS_SEMANTIC_PROVIDER=null` is a no-op review run, and
`MORPHEUS_SEMANTIC_PROVIDER=ollama` is an explicit local model opt-in. No cloud
provider may be called unless the user explicitly configures one.

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
morpheus learn quality .        # Show trainability, routing, and dataset blockers
morpheus learn benchmark . --dry-run
morpheus learn team-loop . --input feedback.jsonl --json
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
morpheus review accept-proposed # Accept rescored ACCEPT_SAFE candidates only
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

### Team Feedback Event

The local team-learning input is newline-delimited JSON:

```python
class TeamFeedbackEvent(BaseModel):
    source_type: Literal["pr_comment", "rejected_agent_claim", "human_correction"]
    external_id: str
    claim: str
    correction: str | None = None
    author: str | None = None
    url: str | None = None
```

Each validated event is stored as an immutable one-line local evidence artifact
and projected into a `pending` `outdated_claim` candidate. The content hash is
the stable candidate identity, so exact replay is idempotent and changed input
creates a new auditable version. Acceptance may route the original claim to a
negative/correction example; rejection or unresolved review state excludes it.
The team loop never builds a dataset, trains, evaluates, or activates an adapter.
The API returns `422` for malformed JSON or an invalid outer request shape and
`400` for a parsed feedback event rejected by the local ingestion policy.

Activation requires eval. Production use requires rollback.
Deterministic fake/dry-run eval artifacts are diagnostic only and cannot satisfy
the activation gate. A force flag must never bypass a failed eval gate.

## 10. Integrations

Integrations contribute evidence. They do not change the product identity.

- `github`: issues, pull requests, commits, and cached metadata.
- `gmail`: local Gmail cache and OAuth-oriented token path.
- `calendar`: local Calendar cache and OAuth-oriented token path.
- `slack`: local Slack export cache plus optional token file.
- `linear`: local Linear issue cache plus optional token file.
- `filesystem`: project files, docs, specs, notes, and vaults.

Local tokens and caches should stay outside the repository by default.

## 11. Product Roadmap

Morpheus should not evolve into another generic review bot. The product core is
a verified classification-to-training pipeline:

- **v0.3 Semantic classifier**: classify source-backed claims as architecture,
  implementation, product identity, security, command facts, integration facts,
  stale claims, team conventions, open tasks, or temporary facts.
- **v0.4 Dataset quality dashboard**: expose trainable, retrievable, stale,
  unsafe, needs-review, negative, and eval-only state.
- **v0.5 Adapter memory benchmark**: evaluate product identity, commands,
  architecture, safety rules, team conventions, stale correction, and
  unsupported-claim refusal separately.
- **v0.6 Agent memory routing**: route facts to prompt context, retrieval,
  adapter training, eval-only, negative examples, stale archive, or human review.
- **v0.7 Team learning loop**: convert PR comments, rejected agent claims, human
  corrections, accepted review candidates, check results, and stale-claim
  corrections into reviewed continual learning candidates.

The public claim is not "fine-tune an AI model on your codebase." The claim is:
Morpheus builds a verified learning layer for agents, classifies project
knowledge, proves what is source-backed, and distills stable truth into local
model memory experiments.

## 12. Security Model

- Default bind address should be `127.0.0.1`.
- `.morpheus/`, receipts, caches, generated datasets, model outputs, and token
  files must stay out of git unless explicitly exported.
- Symlink escapes must be rejected for project roots, watched paths, output
  paths, and sensitive state files.
- MCP and A2A-style endpoints are automation surfaces and should be exposed only
  in trusted environments.
- MCP truth tools are local and read-only by default:
  `morpheus_check_text`, `morpheus_get_active_state`,
  `morpheus_get_evidence_for_claim`, and `morpheus_get_wake`.

## 13. Beta Release Acceptance Criteria

- [x] First README screen frames Morpheus as a source-grounded truth layer:
  "First verify. Then learn."
- [x] Root `WAKE.md` is committed as a public showcase.
- [x] `morpheus wake .` provides a one-command demo path.
- [x] `morpheus check` verifies local agent text from file or stdin and returns
  deterministic exit codes.
- [x] `morpheus stale .` detects known stale positioning claims.
- [x] SPEC avoids agent, training, and legal-overclaim positioning.
- [x] LoRA/training is documented as experimental and downstream of source spans,
  review, check, and eval.
- [x] Receipts are ed25519 signed and verifiable.
- [x] `morpheus verify --all` validates receipt chain and artifacts.
- [x] UI exposes setup, diagnostics, integrations, FAQ, MCP probe, and handoff.
- [x] Public hygiene tests reject local assistant artifacts.
- [x] Semantic compiler mode is implemented and review-gated in v0.2 alpha.
- [x] Autonomous learning lab can build a real dogfood dataset from strict
  source-backed candidates.
- [x] Repeat-2 live MLX dogfood lab reaches `ML_CORE_PASS` without activating
  an adapter.
- [x] Before/after visual demo or GIF is published.
- [x] First GitHub release is cut.

Prerelease builds must still run package build checks, CI, and PyPI trusted
publishing verification before any tag is pushed.
