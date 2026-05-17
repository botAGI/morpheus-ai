# WAKE.md - Morpheus Project State

This repository intentionally commits `WAKE.md` as a public example.
Private projects can keep this file inside `.morpheus/`.

## Current State

Morpheus is the Agent State Compiler.

It generates `WAKE.md` so AI agents can continue from current project state
instead of starting cold every session.

## Active Decisions

- The public primitive is `WAKE.md`.
- Morpheus is positioned as `WAKE.md for AI agents`.
- `README.md` explains the project to humans.
- `AGENTS.md` tells agents how to work.
- `WAKE.md` tells agents where the project is now.
- Provenance receipts are a core differentiator.
- LoRA/training is experimental and not the core launch path.
- Private projects can keep generated state under `.morpheus/`.

## Outdated Claims

- "Morpheus is mainly a personal AI agent." Outdated.
- "Daily LoRA is the main differentiator." Outdated.
- "Memory compiler" is too weak as public positioning.
- Broad legal compliance promises should be replaced with provenance,
  local-first operation, source attribution, and user-controlled export.

## How Agents Should Work Here

1. Read this file before editing.
2. Treat it as current project state, not as a full source of truth.
3. Prefer source-backed claims over inferred claims.
4. Run `morpheus compile` and `morpheus verify --all` after meaningful changes.
5. If public positioning changes, update `README.md`, `README.ru.md`, `SPEC.md`,
   and this file together.

## Next Product Work

- Add review-gated semantic compilation.
- Add richer stale-claim detection.
- Add a visual before/after demo.
- Cut the first public release.

## Verification

Latest verified local gate for this state:

```bash
ruff check .
pytest tests/ -q
morpheus compile
morpheus verify --all
```

Generated private receipts remain in `.morpheus/` and are intentionally not
committed.
