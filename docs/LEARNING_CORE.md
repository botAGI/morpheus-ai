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

