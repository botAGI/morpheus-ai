# Morpheus

> Stop coding agents from hallucinating about your repo.
>
> First verify. Then learn.

Morpheus checks what agents say against source-backed project state. Then it can
run an autonomous learning lab to test whether stable project truth can be
distilled into local model weights.

`README.md` tells humans what this is.
`AGENTS.md` tells agents how to work.
`WAKE.md` tells agents where we are.

[Русская версия](https://github.com/botAGI/morpheus-ai/blob/main/README.ru.md)

> Status: beta release. Latest GitHub release and beta package: v0.2.0b1. The
> deterministic compiler, local claim checker, receipts, CLI, API, UI
> launchpad, MCP truth tools, A2A-style discovery, cache-backed integrations,
> and autonomous learning lab are usable. Local adapter learning is
> experimental until eval passes; source spans remain the source of truth.
> Pin `morpheus-wake==0.2.0b1` for v0.2 features; unpinned PyPI tools may still
> choose the latest stable v0.1.1 instead of this beta.
>
> Latest live dogfood stability gate on main: repeat-2 `ML_CORE_PASS` with 69
> strict source-backed candidates, 290 training examples, full base-vs-adapter
> eval coverage, zero critical failures, and no adapter activation. See
> [`docs/reports/ML_CORE_LIVE_REPORT.md`](docs/reports/ML_CORE_LIVE_REPORT.md).

![Morpheus terminal demo](https://raw.githubusercontent.com/botAGI/morpheus-ai/main/demo/morpheus-demo.gif)

## Why

Every AI agent starts cold.

You paste context. You repeat decisions. The agent suggests old ideas. It claims
features exist that do not exist.

Morpheus compiles project state, checks agent text against source-backed
evidence, and can build an experimental local learning dataset only from
accepted claims.

```text
sources -> WAKE.md -> morpheus check -> reviewed dataset -> local adapter lab
```

## The Primitive

Morpheus is a source-grounded truth layer with an experimental learning core.

It generates `WAKE.md` - a project state file that tells agents where the
project is now. `morpheus check` verifies claims against local state, source
spans, manifests, and evidence. `morpheus learn lab` runs an autonomous local
experiment to test whether verified project truth can become useful adapter
memory.

This repository intentionally commits
[WAKE.md](https://github.com/botAGI/morpheus-ai/blob/main/WAKE.md) as a public
example.
Private projects can keep `WAKE.md` inside `.morpheus/`.

## Roadmap

Morpheus is not trying to become another review bot. The next product axis is a
verified classification-to-training pipeline:

- **v0.3**: semantic classifier for architecture, implementation, product,
  security, command, integration, stale, convention, task, and temporary facts.
- **v0.4**: dataset quality dashboard for trainable, retrievable, stale,
  unsafe, needs-review, negative, and eval-only claims.
- **v0.5**: adapter memory benchmark with category-level base-vs-adapter deltas.
- **v0.6**: agent memory routing across prompt, retrieval, adapter training,
  eval, negative examples, stale archive, and human review.
- **v0.7**: team learning loop from PR comments, rejected agent claims, human
  corrections, accepted candidates, and check results.

See [docs/ROADMAP.md](docs/ROADMAP.md). The invariant stays strict: no accepted
source span means no training example, no eval pass means no adapter activation,
and adapter output is not source of truth.

## Quick Start

Install the v0.2 beta:

```bash
uvx --from 'morpheus-wake==0.2.0b1' morpheus wake .
```

With pipx:

```bash
pipx run --spec 'morpheus-wake==0.2.0b1' morpheus wake .
```

For private workspaces:

```bash
uvx --from 'morpheus-wake==0.2.0b1' morpheus wake . --private
```

That keeps the compiled state at `.morpheus/WAKE.md`.

Three-command alpha loop:

```bash
uvx --from 'morpheus-wake==0.2.0b1' morpheus wake .
gh pr view 42 --json body -q .body | uvx --from 'morpheus-wake==0.2.0b1' morpheus check
uvx --from 'morpheus-wake==0.2.0b1' morpheus learn lab . --no-train
```

`morpheus learn lab` is experimental. It can use a strict autonomous benchmark
lane, but it never activates adapters automatically and it does not use raw
Markdown fine-tuning. On Apple Silicon with MLX installed, add `--backend mlx`
when you intentionally want to run local adapter training.

Development install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

morpheus wake .
```

## Before / After

Without Morpheus:

```text
User: What changed yesterday?
Agent: I do not have enough context.
```

With Morpheus:

```text
User: Check this agent answer before I merge it.
Agent: stale: "Morpheus is mainly a LoRA trainer."
       incorrect: "morpheus check sends text to cloud by default."
       verified: "The package name is morpheus-wake."
```

## Why Not Just Use Memory?

Memory tells an agent what happened.
Source-grounded state tells an agent what is supported now.

RAG retrieves old fragments.
Morpheus verifies current project claims before any learning experiment.

`README.md` is for humans.
`AGENTS.md` is for agent instructions.
`WAKE.md` is for agent continuity.

## Core Features

- **WAKE.md compiler**: scans watched paths and extracts marked decisions,
  tasks, notes, fixes, and evidence.
- **Local claim check**: `morpheus check` verifies agent text from a file or
  stdin against local state and returns `verified`, `stale`, `incorrect`, or
  `unknown`.
- **Autonomous learning lab**: `morpheus learn lab` builds a strict benchmark
  dataset from machine-verifiable source-backed claims, optionally runs local
  MLX LoRA smoke training, and writes a pass/partial/fail report without
  activating adapters.
- **Verifiable provenance**: writes `state.json`, `evidence.jsonl`, and signed
  ed25519 receipts with SHA-256 hashes.
- **Agent handoff**: produces copyable instructions, diagnostics, and manifest
  URLs for another coding agent.
- **Stale claim scan**: `morpheus stale .` flags launch-positioning claims that
  conflict with the current WAKE.md framing.
- **Local UI launchpad**: browser UI for setup, context sources, diagnostics,
  integrations, model smoke tests, and handoff bundles.
- **Agent interop**: native `/agent/connect`, A2A-compatible Agent Card, and
  MCP truth tools for local claim checking and evidence lookup.
- **Context sources**: compile one project, a monorepo, a workspace, or a notes
  vault by configuring watched paths.
- **Integration cache readers**: GitHub, Gmail, Calendar, Slack, and Linear can
  contribute evidence from local caches or token-backed adapters.

## Tested On Current Main

The current local gate has been run against this repository, not only fixtures:

| Capability | Tested result |
| --- | --- |
| `ruff check .` | passes |
| `pytest tests/ -q` | 678 tests pass |
| `morpheus wake . --private` | compiles current project state and signs a receipt |
| `morpheus verify --all` | verifies the receipt chain |
| `morpheus check --input tests/fixtures/check_stale_input.txt --local` | exits 1 and reports the stale claim |
| `morpheus check --input tests/fixtures/check_correct_input.txt --local` | exits 0 and verifies the claim |
| `morpheus learn lab . --dogfood --backend mlx --eval-limit 0 --repeat 2` | repeat-2 `ML_CORE_PASS` on real repo dogfood data |
| `morpheus learn train . --dry-run` | plans from the latest trainable lab dataset when standalone dataset is empty |
| local `/mcp` truth tools smoke | lists tools and verifies check/state/evidence/WAKE calls on `127.0.0.1` |

The live MLX stability run used `mlx-community/Qwen2.5-7B-Instruct-4bit`,
trained a local adapter from strict source-backed candidates, evaluated 148 base
and adapter items in each of two runs, improved pass rate from 0.7973 to
0.9932, and recorded zero regressions or critical failures. This is a lab gate,
not automatic production activation.

## Deterministic Core, Check, And Learning Beta

The deterministic compiler remains simple by design. It extracts explicit
markers:

```text
TODO: DECISION: FIXME: NOTE: HACK: XXX:
```

That makes receipts reproducible and easy to verify.

`morpheus check` is local-only by default. It does not send agent text or
project source excerpts to cloud providers.

The beta includes a review-gated semantic path:

```bash
morpheus wake . --semantic --review
morpheus review list
morpheus review accept <candidate-id>
morpheus review apply
```

Semantic extraction is review-gated. Candidates are labeled as `source_backed`
or `needs_review`, source spans are verified before apply, and accepted claims
become active only after `morpheus review apply` signs a new receipt.

The learning core sits behind that gate:

```bash
morpheus learn dataset . --from accepted
morpheus learn train . --dry-run
morpheus learn eval .
morpheus learn lab . --no-train
```

No accepted source span means no training example. No eval pass means no adapter
activation. No rollback means no production use.

## Obsidian And Personal Notes

An Obsidian vault can be used as a Morpheus context source because it is a
folder of Markdown files. The recommended path is local compilation first:
source links, evidence, receipts, and review. Do not train directly on a raw
private vault.

```bash
cd ~/Obsidian
morpheus wake . --private
```

For a workspace with several projects or vaults, set the parent folder as the
project root and configure `.morpheus/morpheus.toml`:

```toml
watch_dirs = ["project-a", "project-b", "vault"]
```

## Agent Self-Connect

Agents can discover Morpheus without reading the README:

```bash
morpheus prepare-agent
morpheus agent-connect --json
morpheus diagnostics --json
morpheus handoff
```

With the HTTP API running:

```bash
curl -s "http://127.0.0.1:8000/agent/connect?project_root=$PWD"
curl -s "http://127.0.0.1:8000/agent/handoff.md?project_root=$PWD"
curl -s http://127.0.0.1:8000/.well-known/morpheus.json
curl -s http://127.0.0.1:8000/.well-known/agent-card.json
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

The MCP endpoint exposes the local truth-layer tools `morpheus_check_text`,
`morpheus_get_active_state`, `morpheus_get_evidence_for_claim`, and
`morpheus_get_wake`. These tools read local Morpheus state and do not call cloud
providers by default.

A new agent should:

1. Read `WAKE.md`.
2. Fetch `/agent/connect` or run `morpheus agent-connect --json`.
3. Follow the returned `next_action`.
4. Run `morpheus compile` and `morpheus verify --all` after meaningful changes.

## UI Start

```bash
morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173
```

Open:

```text
http://127.0.0.1:5173/ui/index.html
```

The Start screen lets you set a project root, configure watched paths, run
diagnostics, prepare an agent, inspect integrations, probe MCP tools, and copy a
complete handoff bundle.

## Architecture

```text
morpheus/
  core/          compiler, models, receipts, verification, safe IO
  core/learning/ reviewed datasets, eval, registry, autonomous lab
  integrations/  filesystem and cache-backed external sources
  api/           FastAPI, agent connect, diagnostics, MCP, A2A card
  training/      experimental consolidation and LoRA helpers
ui/              static browser UI and Tauri shell
tests/           pytest suite for compiler, API, CLI, integrations, training
docs/            launch notes, testing notes, and product framing
```

Compile flow:

```text
morpheus compile
  -> scans configured watch_dirs
  -> extracts explicit evidence markers
  -> writes state.json and evidence.jsonl
  -> generates WAKE.md
  -> signs a receipt with ed25519
  -> links the receipt to the previous receipt hash
```

## CLI Reference

| Command | Description |
| --- | --- |
| `morpheus wake .` | Init if needed, compile, verify, and write root `WAKE.md` |
| `morpheus wake . --private` | Compile and verify, keeping `WAKE.md` in `.morpheus/` |
| `morpheus check` | Verify agent text from stdin against local project state |
| `morpheus check --input FILE` | Verify agent text from a file |
| `morpheus check --json` | Print a machine-readable check result |
| `morpheus review list` | List semantic candidates awaiting review |
| `morpheus review accept-proposed` | Accept freshly rescored `ACCEPT_SAFE` candidates without applying active state |
| `morpheus review apply` | Apply accepted candidates into active state and sign a receipt |
| `morpheus learn lab .` | Run the autonomous learning lab without activating adapters |
| `morpheus learn dataset .` | Build a dataset from accepted source-backed candidates |
| `morpheus learn quality .` | Write trainability, route, blocker, and dataset quality reports |
| `morpheus learn benchmark . --dry-run` | Write benchmark-readiness artifacts without training or activation |
| `morpheus learn status` | Show learning dataset and adapter status |
| `morpheus learn train . --dry-run` | Generate local training artifacts without training |
| `morpheus learn eval .` | Evaluate the latest dataset or planned adapter with the eval harness |
| `morpheus stale .` | Find stale launch-positioning claims |
| `morpheus init` | Initialize `.morpheus/` with config and keys |
| `morpheus compile` | Compile sources into `WAKE.md` and a signed receipt |
| `morpheus verify --all` | Verify receipt chain, signatures, and latest artifacts |
| `morpheus status` | Show source, claim, and evidence counts |
| `morpheus wake` | Print the private `.morpheus/WAKE.md` |
| `morpheus prepare-agent` | Initialize, compile, bootstrap `AGENTS.md`, verify, and produce handoff |
| `morpheus handoff` | Print a copyable markdown handoff |
| `morpheus agent-connect --json` | Print the machine-readable agent manifest |
| `morpheus diagnostics --json` | Print readiness checks and next action |
| `morpheus model-smoke` | Smoke-test a local Ollama model |
| `morpheus serve --ui` | Run FastAPI backend and browser UI |

Semantic provider modes are explicit. `MORPHEUS_SEMANTIC_PROVIDER=local` is the
default offline heuristic provider, `MORPHEUS_SEMANTIC_PROVIDER=null` is a no-op
review run, and `MORPHEUS_SEMANTIC_PROVIDER=ollama` is an explicit local model
opt-in. Cloud providers are never called by default.

## Development

```bash
make install-dev
make verify
make build
```

For the full public-repo quality gate, see [docs/TESTING.md](docs/TESTING.md).

## Security Notes

Morpheus is local-first. Keep `.morpheus/`, generated receipts, integration
caches, model outputs, and token files out of git. Bind to `127.0.0.1` unless
you are on a trusted LAN or behind an authenticated proxy.

See [SECURITY.md](SECURITY.md).

## Experimental Training

Local adapter training lives under `morpheus/training/`. It is optional,
explicit, and downstream of reviewed state. The default path is compile,
retrieve, cite evidence, and verify receipts.

## License

MIT
