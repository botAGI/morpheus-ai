# Morpheus

> Stop starting AI agents from scratch.
>
> Morpheus generates `WAKE.md` - a compiled project state file that lets any
> agent continue where you left off.

`README.md` tells humans what this is.
`AGENTS.md` tells agents how to work.
`WAKE.md` tells agents where we are.

[Русская версия](README.ru.md)

> Status: alpha. The compiler, receipts, CLI, API, UI launchpad, MCP endpoint,
> A2A-style discovery, and cache-backed integrations are usable. LoRA/training
> is experimental and not the core launch path.

## Why

Every AI agent starts cold.

You paste context. You repeat decisions. The agent suggests old ideas. It misses
what changed yesterday.

Morpheus compiles your current project state into `WAKE.md`, backed by sources,
evidence, and signed receipts.

```text
sources -> compile -> WAKE.md -> signed receipt -> agent handoff -> verify
```

## The Primitive

Morpheus is the Agent State Compiler.

It generates `WAKE.md` - a project state file that tells agents where the
project is now, backed by sources, evidence, and signed receipts.

This repository intentionally commits [WAKE.md](WAKE.md) as a public example.
Private projects can keep `WAKE.md` inside `.morpheus/`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

# One-command demo: init if needed, compile, verify, write root WAKE.md.
morpheus wake .
```

For private workspaces:

```bash
morpheus wake . --private
```

That keeps the compiled state at `.morpheus/WAKE.md`.

## Before / After

Without Morpheus:

```text
User: What changed yesterday?
Agent: I do not have enough context.
```

With Morpheus:

```text
User: Read WAKE.md. What changed yesterday?
Agent: Morpheus moved from "memory compiler" to "Agent State Compiler".
       Outdated: LoRA as the core product path.
       Current: WAKE.md with provenance receipts.
       Next action: update README, SPEC, and public repo metadata.
```

## Why Not Just Use Memory?

Memory tells an agent what happened.
State tells an agent what is true now.

RAG retrieves old fragments.
Morpheus compiles the current project state.

`README.md` is for humans.
`AGENTS.md` is for agent instructions.
`WAKE.md` is for agent continuity.

## Core Features

- **WAKE.md compiler**: scans watched paths and extracts marked decisions,
  tasks, notes, fixes, and evidence.
- **Verifiable provenance**: writes `state.json`, `evidence.jsonl`, and signed
  ed25519 receipts with SHA-256 hashes.
- **Agent handoff**: produces copyable instructions, diagnostics, and manifest
  URLs for another coding agent.
- **Stale claim scan**: `morpheus stale .` flags launch-positioning claims that
  conflict with the current WAKE.md framing.
- **Local UI launchpad**: browser UI for setup, context sources, diagnostics,
  integrations, model smoke tests, and handoff bundles.
- **Agent interop**: native `/agent/connect`, A2A-compatible Agent Card, and
  a minimal MCP Streamable HTTP endpoint.
- **Context sources**: compile one project, a monorepo, a workspace, or a notes
  vault by configuring watched paths.
- **Integration cache readers**: GitHub, Gmail, Calendar, Slack, and Linear can
  contribute evidence from local caches or token-backed adapters.

## Deterministic Now, Semantic Next

v0.1 is deterministic by design. It extracts explicit markers:

```text
TODO: DECISION: FIXME: NOTE: HACK: XXX:
```

That makes receipts reproducible and easy to verify.

The next compiler mode is `--semantic`: LLM-assisted state extraction from
README, SPEC, AGENTS, changelogs, issues, and notes. Semantic claims should be
review-gated and labeled as `source_backed`, `inferred`, or `needs_review`
before they become active project state.

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
