# Morpheus Roadmap

## Product Direction

Morpheus is not trying to become another review bot.

The product direction is:

```text
First verify. Then learn.
```

Morpheus builds a verified learning layer for coding agents. It classifies
project knowledge, proves what is source-backed, excludes unsafe or stale
claims, and distills stable project truth into local model memory experiments.

The truth layer is the data-quality gate. The learning layer is the long-term
product.

## Current Release

`v0.2.0b1` proves the first loop:

```text
project sources -> source-backed state -> morpheus check
-> strict accepted candidates -> dataset -> local adapter lab -> eval report
```

It can compile state, check agent claims, expose MCP truth tools, generate
source-backed datasets, run local MLX adapter experiments, and report whether
learning helped or hurt. Adapter activation remains explicit and eval-gated.

## v0.3: Semantic Classifier As Product Core

Goal: turn source-backed extraction into a useful project-knowledge classifier.

Claim classes:

- architecture facts
- implementation facts
- product identity facts
- security and safety rules
- command and CLI facts
- integration facts
- stale claims
- team conventions
- open tasks
- temporary facts

Acceptance criteria:

- every candidate has a class, source span, confidence, and trainability status,
- classifier output is deterministic enough for review diffs,
- stale and temporary facts are never promoted as stable training facts,
- `morpheus check` can report class-specific results.

## v0.4: Dataset Quality Dashboard

Goal: show what Morpheus believes is trainable, retrievable, stale, unsafe, or
review-blocked.

Dashboard surfaces:

- trainable claims
- retrievable state
- stale claims
- unsafe or secret-like candidates
- needs-review candidates
- negative examples
- eval-only examples
- source coverage and missing evidence

Acceptance criteria:

- CLI and UI expose the same quality categories,
- dashboard can explain why a claim is not trainable,
- dataset manifests include category counts and top blockers.

## v0.5: Adapter Memory Benchmark

Goal: prove adapter memory by category, not only with a single pass-rate number.

Benchmark categories:

- product identity
- commands and CLI behavior
- architecture
- safety rules
- team conventions
- stale claim correction
- unsupported claim refusal

Acceptance criteria:

- base vs adapter reports category-level deltas,
- regressions are tracked per category,
- no adapter can be activation-ready with critical stale/safety regressions.

## v0.6: Agent Memory Routing

Goal: route each fact to the right memory channel.

Routes:

- prompt context
- retrieval/RAG
- LoRA or QLoRA adapter training
- eval-only
- negative example
- stale archive
- human review

Acceptance criteria:

- classifier chooses a route for every accepted or rejected claim,
- routing decisions are auditable,
- no raw Markdown or unreviewed candidate can enter adapter training.

## v0.7: Team Learning Loop

Goal: turn team corrections into continual project memory.

Inputs:

- PR comments
- rejected agent claims
- human corrections
- accepted review candidates
- check results
- stale claim corrections

Acceptance criteria:

- corrections become pending candidates, not silent training data,
- accepted corrections can become negative or correction examples,
- rejected or unresolved corrections never enter training,
- the loop can run repeatedly without activating adapters automatically.

## Non-Goals

Morpheus should not be sold as:

```text
We fine-tune an AI model on your codebase.
```

That framing is generic and unsafe. It skips the important questions: source
grounding, secrets, stale knowledge, overfitting, eval, rollback, and changing
repositories.

The stronger framing is:

```text
Morpheus builds a verified learning layer for your agents.
It classifies project knowledge, proves what is source-backed,
and distills stable truth into local model memory.
```

## Invariants

- No accepted source span -> no training example.
- No eval pass -> no adapter activation.
- No rollback -> no production use.
- No cloud calls by default.
- No raw Markdown training.
- No adapter output as source of truth.
