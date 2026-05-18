# Morpheus Autonomous Lab

The autonomous lab is a local experiment lane for the Morpheus learning core.
It exists to test the ML hypothesis without requiring a human to review every
candidate during development.

Human review remains the best path for production state. Autonomous lab mode is
stricter and narrower: it accepts only exact, source-backed, atomic claims from
high-signal sources, or it falls back to a generated benchmark fixture.

The lab never trains on raw Markdown. Raw files are read only to verify source
spans. Dataset examples are derived from accepted candidates.

The lab never activates adapters. It can return:

- `ML_CORE_PASS`
- `ML_CORE_PARTIAL`
- `ML_CORE_FAIL`
- `ML_CORE_DATASET_BLOCKED`

The current command is:

```bash
morpheus learn lab . --no-train
morpheus learn lab . --backend mlx --max-iters 50
```

Safety rules:

- Pending, rejected, `needs_review`, inferred-only, ignored, and secret-like
  candidates are excluded.
- Outdated claims can become correction examples, not positive facts.
- Cloud providers are not called by default.
- Adapter output is not treated as project truth.

