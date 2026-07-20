# Morpheus Training Legacy Notes

The old root-level `morpheus consolidate`, `morpheus train`, and `morpheus eval`
commands are retained for compatibility and dry-run script generation.

They are not the beta learning path.

The safe path is:

```bash
morpheus wake . --private
morpheus verify --all
morpheus learn lab . --dogfood --no-train
morpheus learn status
morpheus learn train . --dry-run
morpheus learn eval .
```

## Current Rule

```text
No accepted source span -> no training example.
No eval pass -> no adapter activation.
No rollback means no production activation.
```

Do not train on raw Markdown, raw private vaults, raw chat logs, pending review
candidates, rejected candidates, `needs_review` candidates, inferred-only
candidates, ignored files, or secret-like content.

## Legacy Command Safety

`morpheus train` now defaults to dry-run script generation. Executing the legacy
raw-dataset training path requires:

```bash
morpheus train --execute --yes-i-know-this-is-legacy-raw-training
```

Use this only for old local experiments. New learning work should use
`morpheus learn dataset`, `morpheus learn train`, `morpheus learn eval`, and
`morpheus learn lab`.

## Daily Script

`scripts/daily_training.sh` is now a safe daily lab gate. It refreshes private
WAKE state, verifies receipts, runs `morpheus learn lab . --dogfood --no-train`,
and prints learning status. It does not train or activate adapters.
