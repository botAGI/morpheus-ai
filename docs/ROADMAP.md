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

## v0.5: Adapter Memory Benchmark — Complete In Current Code

Goal: prove adapter memory by category, not only with a single pass-rate number.

Benchmark categories:

- product identity
- commands and CLI behavior
- architecture
- safety rules
- team conventions
- stale claim correction
- unsupported claim refusal

Verified acceptance criteria:

- [x] base vs adapter reports pass-rate and hallucination-rate deltas per
  category,
- [x] all regressions are tracked per category and critical regressions are a
  separate subset,
- [x] no adapter can be activation-ready with critical stale/safety/refusal
  regressions,
- [x] activation and rollback-to-adapter use the same live adapter-bound
  readiness/eval gate; force cannot bypass it, while rollback-to-none remains
  the fail-safe,
- [x] activation authority binds a registered trained weight artifact by exact
  path, size, and SHA-256; preview-only manifests remain ineligible.

Canonical schema: `morpheus-benchmark-categories/1`.

Canonical coverage IDs are exactly `product_identity`,
`commands_and_cli_behavior`, `architecture`, `safety_rules`,
`team_conventions`, `stale_claim_correction`, and
`unsupported_claim_refusal`. Diagnostic `project_recall` does not satisfy
coverage. Security/safety and convention/team-convention coverage are
independent requirements.

The dataset manifest and both eval sides must bind the current category schema
and exact dataset authority. A legacy or mismatched manifest, eval, or category
schema requires rebuilding the dataset and rerunning base and adapter evals.
Editing old artifacts cannot create activation authority.

## v0.6: Agent Memory Routing — Complete in Current Code

Goal: route each fact to the right memory channel.

Routes:

- prompt context
- retrieval/RAG
- LoRA or QLoRA adapter training
- eval-only
- negative example
- stale archive
- human review

Verified foundation:

- [x] normal review acceptance and rejection recompute a route,
- [x] routing decisions expose policy version, source span, route, and reason,
- [x] dataset validation excludes raw Markdown, rejected, pending, inferred,
  secret-like, and route-inconsistent candidates from adapter training.

Verified acceptance hardening:

- [x] every persisted lifecycle transition, including lab auto-accept and source
  invalidation, recomputes and stores the canonical route,
- [x] signed compiled active-state input is either defined and enforced as
  explicit review authority or excluded by the same no-unreviewed-input rule.

## v0.7: Team Learning Loop — Local Core Complete, Orchestration Remaining

Goal: turn team corrections into continual project memory.

Inputs:

- PR comments
- rejected agent claims
- human corrections
- accepted review candidates
- check results
- stale claim corrections

Verified local-core acceptance criteria:

- [x] corrections become pending candidates, not silent training data,
- [x] accepted corrections can become negative or correction examples,
- [x] rejected or unresolved corrections never enter training,
- [x] the loop can run repeatedly without activating adapters automatically.

Remaining orchestration acceptance:

- [ ] one idempotent input path covers all six documented sources. Direct team
  feedback currently accepts PR comments, rejected agent claims, and human
  corrections; accepted review candidates, check corrections, and stale
  corrections currently arrive through or are counted from separate review and
  check flows.

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
- No exact source-authority/artifact match -> no dataset execution.
- No eval pass -> no adapter activation.
- No rollback -> no production use.
- No cloud calls by default.
- No raw Markdown training.
- No adapter output as source of truth.
