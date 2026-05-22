# WAKE.md - Morpheus Project State

This repository intentionally commits `WAKE.md` as a public showcase state file.
Private projects can keep generated `WAKE.md` inside `.morpheus/`.

## Current State

Morpheus is a published `v0.2.0b1` beta and is moving toward a verified
classification-to-training pipeline.

It compiles `WAKE.md`, checks coding-agent claims against local source-backed
state, and can run a local experiment that turns strictly accepted claims into a
training dataset and adapter smoke test.

The latest live dogfood MLX stability gate on main reached repeat-2
`ML_CORE_PASS`: 69 strict source-backed candidates, 290 training examples,
148/148 base-vs-adapter eval coverage in both runs, adapter pass rate 0.9932
versus base pass rate 0.7973, zero critical failures, zero regressions, and no
automatic adapter activation.

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
- `docs/ROADMAP.md` - staged product direction from classifier to team learning.
- `SPEC.md` - product frame, architecture, non-goals, and release criteria.
- `AGENTS.md` - agent bootstrap behavior.
- `docs/WHY_WAKE.md` - category rationale for `WAKE.md`.
- `docs/RELEASE.md` - release process and PyPI Trusted Publishing setup.
- `CHANGELOG.md` - current launch delta.
- `docs/TESTING.md` - local quality gate and release checks.

## Next Product Work

1. v0.3 semantic classifier as product core: classify architecture,
   implementation, product, security, command, integration, stale, convention,
   task, and temporary facts.
2. v0.4 dataset quality dashboard: show trainable, retrievable, stale, unsafe,
   needs-review, negative, and eval-only claims.
3. v0.5 adapter memory benchmark: report category-level base-vs-adapter deltas
   and critical regressions.
4. v0.6 agent memory routing: decide whether a fact belongs in prompt,
   retrieval, adapter training, eval, negative examples, stale archive, or human
   review.
5. v0.7 team learning loop: turn PR comments, rejected agent claims, human
   corrections, accepted candidates, and check results into reviewed continual
   learning data.

## Verification

Latest verified local gate for this state:

```bash
ruff check .
pytest tests/ -q
morpheus wake . --private
morpheus verify --all
morpheus check --input tests/fixtures/check_stale_input.txt --local
morpheus check --input tests/fixtures/check_correct_input.txt --local
morpheus learn status
```

Generated private receipts remain in `.morpheus/` and are intentionally not
committed.
