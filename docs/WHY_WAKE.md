# Why WAKE.md

Every agent starts cold.

Humans solve that by creating repo-native context files:

- `README.md` explains the project to humans.
- `AGENTS.md` tells agents how to work.
- `WAKE.md` tells agents where the project is now.

## The Gap

Most agent memory tools store fragments. That helps retrieval, but it does not
answer the more important handoff question:

```text
What is true now, what changed, and what should the next agent do first?
```

Morpheus treats that as a compile problem.

## The Loop

```text
sources -> compile -> WAKE.md -> signed receipt -> handoff -> verify
```

The result is a small file that an agent can read before editing, plus machine
artifacts that a human or another agent can verify.

## Why It Belongs In Git Repos

GitHub developers already understand repo primitives. A file in the root of the
repo is inspectable, linkable, diffable, reviewable, and easy for agents to find.

That is the product bet:

```text
README.md is for humans.
AGENTS.md is for instructions.
WAKE.md is for continuity.
```

## Public And Private Modes

Public repositories can commit a curated `WAKE.md` as a showcase.

Private projects can keep generated state inside `.morpheus/`:

```bash
morpheus wake . --private
```

The same primitive works in both cases.

## Current Beta Pipeline

`v0.2.0b2` documents the current source-grounded loop: compile state, check agent
claims, build a strict learning dataset, and run a local adapter lab without
automatic activation.

The current beta implements the verified classification-to-training pipeline:

- **v0.3 — complete in the current code**: classify project knowledge by kind
  and safety.
- **v0.4 — complete in the current code**: decide what is trainable,
  retrievable, stale, unsafe, or eval-only.
- **v0.5 — complete in the current code**: benchmark adapter memory by category.
- **v0.6 — complete in the current code**: route each accepted fact to prompt,
  retrieval, training, eval, or review.
- **v0.7 — complete in the current code**: turn team corrections into reviewed
  continual-learning candidates without automatic activation.

These completion labels describe the current beta implementation and repository
tests, not stable maturity or broad proof across real repositories.
No milestone after v0.7 is currently defined.

The invariant stays the same: agents can continue, and humans can verify.
