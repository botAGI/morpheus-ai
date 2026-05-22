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

## What Comes Next

`v0.2.0b1` proves the first source-grounded loop: compile state, check agent
claims, build a strict learning dataset, and run a local adapter lab without
automatic activation.

The next step is not another review bot. It is a verified
classification-to-training pipeline:

- classify project knowledge by kind and safety,
- decide what is trainable, retrievable, stale, unsafe, or eval-only,
- benchmark adapter memory by category,
- route each accepted fact to prompt, retrieval, training, eval, or review,
- turn team corrections into continual learning candidates.

The invariant stays the same: agents can continue, and humans can verify.
