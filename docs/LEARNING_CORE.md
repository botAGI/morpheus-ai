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
`needs_review` candidates, inferred-only candidates, and stale claims as
positive facts. Accepted outdated candidates may produce only source-bound
negative/correction examples. Built-in truth-gate scenarios without a reviewed
candidate remain eval-only. Every instruction, ShareGPT, and MLX split row must
carry the exact candidate ID, canonical source path, line span, evidence digest,
and `adapter_training` or `negative_example` route; semantic validation rejects
missing or inconsistent training provenance.

The dataset manifest binds a canonical snapshot of review state, the routing
policy, all source/context hashes, and authority-derived train/eval artifacts.
Consumers revalidate the binding before training, evaluation, benchmark
comparison, or activation. Review revocation and artifact changes therefore
close the gate before new output is written. Lab snapshots and active-state
receipts declare separate source scopes; v1 unbound manifests are non-executable.
Active-state scope verifies the signed receipt-chain tail. Adapter evaluation
and activation must match the dataset ID and binding recorded by training.
Dataset and eval registries publish complete entries atomically from hidden,
private staging directories. `morpheus learn train` produces a preview only and
rejects execution. The executable local MLX lab uses sealed snapshots and an
execution guard that copies validated bytes into anonymous descriptors before
its final live-authority check. The guard holds the state/review lease through
the complete backend run and validates again afterward; its authenticated MLX
loader decodes the exact inherited split descriptors and hashes instead of
reopening the mutable dataset view or original paths. Output stays bound to its
held directory descriptor.
Active-state writers, capture, activation, and rollback share a cross-process
authority lock. Activation and rollback also share the full eval gate, bind base
eval, adapter eval, and current dataset authority, and use a durable recovery
journal before committing the active pointer last.

Preview adapter manifests explicitly remain `training_status=planned` with
`weight_artifact=null`. Activation and rollback require a registered trained
adapter with one non-empty regular, non-symlink `.safetensors` file; its exact
path, SHA-256, and size are revalidated and included in authority and receipts.

`morpheus learn lab` is the autonomous test lane. It proves the loop can produce
a source-backed dataset and optional local adapter smoke test without promoting
the adapter.

`morpheus learn dataset`, preview-only `morpheus learn train`, diagnostic
`morpheus learn eval`, `morpheus learn activate`, and `morpheus learn rollback`
are explicit learning-lane building blocks. Only the local MLX lab currently
has a guarded execution path.

The current deterministic fake evaluator is a diagnostic benchmark building
block, not production evidence. Its category reports are always marked
activation-ineligible and do not receive an activation receipt. Eligible evals
must carry a local Ed25519 receipt over evaluator/provider identity, the exact
dataset and eval-seed items, and config/results hashes. Relabeling diagnostic
JSON therefore remains ineligible, and `morpheus learn activate --force` cannot
bypass a failed or diagnostic eval gate.

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
