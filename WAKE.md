# WAKE.md - Morpheus Project State

This repository intentionally commits `WAKE.md` as a public showcase state file.
Private projects can keep generated `WAKE.md` inside `.morpheus/`.

## Current State

Morpheus is becoming a source-grounded truth layer with an autonomous learning
lab.

It compiles `WAKE.md`, checks coding-agent claims against local source-backed
state, and can run a local experiment that turns strictly accepted claims into a
training dataset and adapter smoke test.

## Active Decisions

- Public framing is now: "First verify. Then learn."
- `morpheus check` is the differentiator for stale and incorrect project claims.
- `morpheus learn lab` is the autonomous experiment lane for source-backed
  dataset and adapter testing.
- The public primitive remains `WAKE.md`, but `WAKE.md` alone is no longer the
  full product story.
- `README.md` explains the project to humans.
- `AGENTS.md` tells agents how to work.
- `WAKE.md` tells agents where the project is now.
- Source spans, check results, and receipts are the truth gate.
- The README files include a visual terminal demo.
- Local adapter learning is experimental until eval passes.
- Adapter output is not the source of truth.
- Private projects can keep generated state under `.morpheus/`.

## Outdated Claims

- "Morpheus is mainly a personal AI agent." Outdated.
- "Daily LoRA is the main differentiator." Outdated.
- "Memory compiler" is too weak as public positioning.
- "WAKE.md alone is the product." Outdated.
- "LoRA trains on raw markdown." Outdated.
- "Receipts are the main value." Outdated.
- "Adapter output is source of truth." Outdated.
- Broad legal compliance promises should be replaced with provenance,
  local-first operation, source attribution, and user-controlled export.

## How Agents Should Work Here

1. Read this file before editing.
2. Treat it as current project state, not as a full source of truth.
3. Prefer source-backed claims over inferred claims.
4. Run `morpheus compile` and `morpheus verify --all` after meaningful changes.
5. If public positioning changes, update `README.md`, `README.ru.md`, `SPEC.md`,
   and this file together.

## Source References

- `README.md` - public framing, install story, and quick start.
- `README.ru.md` - Russian public framing and quick start.
- `SPEC.md` - product frame, architecture, non-goals, and release criteria.
- `AGENTS.md` - agent bootstrap behavior.
- `docs/WHY_WAKE.md` - category rationale for `WAKE.md`.
- `docs/RELEASE.md` - release process and PyPI Trusted Publishing setup.
- `CHANGELOG.md` - current launch delta.
- `docs/TESTING.md` - local quality gate and release checks.

## Next Product Work

1. Run autonomous lab on fixture benchmark and dogfood candidates.
2. Increase strict accepted source-backed candidates without human bypass.
3. Improve eval quality for base vs adapter comparison.
4. Keep CLI/API split and broader integrations behind core truth-layer work.

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
