# Morpheus Learning Core

Morpheus does not choose between truth layer and weights.

The truth layer is the data-quality gate. Weights are the long-term product.

The pipeline is:

```text
semantic compile -> review/check -> accepted source-backed state
-> dataset compiler -> local adapter experiment -> eval -> activate/rollback
```

The core rule is:

```text
No accepted source span -> no training example.
No eval pass -> no adapter activation.
No rollback -> no production use.
```

Training input may include only accepted, source-backed candidates whose source
spans still match the current project files. The compiler must exclude secrets,
ignored files, raw private notes, rejected candidates, pending candidates,
`needs_review` candidates, inferred-only candidates, and stale claims.

`morpheus learn lab` is the autonomous test lane. It proves the loop can produce
a source-backed dataset and optional local adapter smoke test without promoting
the adapter.

`morpheus learn dataset`, `morpheus learn train`, `morpheus learn eval`,
`morpheus learn activate`, and `morpheus learn rollback` are the production lane
building blocks.

The current deterministic fake evaluator is a diagnostic benchmark building
block, not production evidence. Its category reports are always marked
activation-ineligible, and `morpheus learn activate --force` cannot bypass a
failed or diagnostic eval gate.

## Roadmap Alignment

The learning core should become the center of the product, but not as raw
fine-tuning.

The next milestones are:

- v0.3 semantic classifier: classify source-backed project knowledge before it
  enters check, retrieval, eval, or training.
- v0.4 dataset quality dashboard: expose trainable, retrievable, stale, unsafe,
  needs-review, negative, and eval-only state.
- v0.5 adapter memory benchmark: measure category-level base-vs-adapter deltas.
- v0.6 agent memory routing: choose prompt, retrieval, adapter training, eval,
  negative example, stale archive, or human review per claim.
- v0.7 team learning loop: turn corrections and review outcomes into continual
  learning candidates.

See `docs/ROADMAP.md` for the public roadmap.
