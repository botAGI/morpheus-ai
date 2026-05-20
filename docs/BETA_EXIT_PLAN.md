# Morpheus Beta Exit Plan

## Goal

Move Morpheus from alpha to beta without weakening the core rule:

```text
First verify. Then learn.
```

Beta means the source-grounded truth layer is reliable enough for external
coding-agent users, and the learning lab can repeatedly prove whether adapter
memory helped or hurt without automatic activation.

## Beta Must-Haves

- Local `morpheus check` supports file input, stdin, JSON, summary output, and
  deterministic exit codes.
- `morpheus wake . --private` and `morpheus verify --all` stay green after docs
  and code changes.
- MCP truth tools stay local and read-only by default.
- Semantic/review/check never promotes inferred, rejected, pending,
  `needs_review`, stale, secret-like, or ignored claims into training data.
- `morpheus learn status` shows the effective trainable dataset, not only the
  latest standalone dataset directory.
- `morpheus learn train . --dry-run` can plan from the latest trainable lab
  dataset when no standalone reviewed dataset exists.
- Legacy root training commands are visibly deprecated and cannot execute
  raw-dataset training without an explicit unsafe confirmation flag.
- Repeated live dogfood MLX lab can run with full eval coverage and produce a
  clear `ML_CORE_PASS`, `ML_CORE_PARTIAL`, `ML_CORE_FAIL`, or
  `ML_CORE_DATASET_BLOCKED` verdict.
- No adapter is activated automatically.
- Package build and `twine check dist/*` pass on macOS with the project venv.

## Current Status

- Truth-layer CLI: beta candidate.
- WAKE compile and receipts: beta candidate.
- MCP truth tools: beta candidate after the local live smoke on 2026-05-20.
- Semantic review: alpha, review-gated.
- Learning dataset/status/train dry-run: beta candidate after effective dataset
  status support.
- Live MLX dogfood lab: beta candidate as an experiment lane, not adapter
  production activation. Repeat-2 stability passed on 2026-05-20 with full eval
  coverage, zero critical failures, zero regressions, and no production blockers.
- Adapter activation/rollback: implemented, but beta release should keep
  activation conservative and explicitly gated.

## Remaining Beta Work

1. Push the release candidate to `main` and confirm CI green.
2. Tag `v0.2.0b1` only after explicit release approval.
3. Watch the release workflow and verify PyPI/uvx only after the tag is pushed.
4. Create the GitHub prerelease only after PyPI publish is verified.

## Non-Negotiables

- No raw markdown training.
- No cloud calls by default.
- No hidden auto-accept of ambiguous candidates.
- No adapter activation without eval.
- No production claim based only on fixture data.
- No tag, publish, or release without explicit instruction.
