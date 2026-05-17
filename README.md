# Morpheus AI

**Local-first memory compiler for AI agents, with verifiable provenance.**

Morpheus turns project files, notes, decisions, tasks, and optional integration
exports into a compact handoff (`WAKE.md`) that humans and agents can trust. It
does not ask agents to start cold: it gives them current state, evidence, and a
receipt chain that proves where the state came from.

[Русская версия](README.ru.md)

> Status: alpha. The compiler, receipts, CLI, API, UI launchpad, MCP endpoint,
> and cache-backed integrations are usable. Scheduled LoRA training is
> experimental and should be treated as an advanced memory layer, not the core
> product path.

## Why Morpheus Exists

Modern agents lose context between sessions. RAG can retrieve text, but it often
cannot explain which facts are current, which source they came from, or what the
next agent should do first.

Morpheus is built around a stricter loop:

```text
sources -> compile -> WAKE.md -> signed receipt -> agent handoff -> verify
```

That makes it useful for:

- handing a project from one agent to another,
- compiling an Obsidian vault or project workspace into agent-readable memory,
- keeping decisions, tasks, and evidence tied to source files,
- exposing local state through CLI, HTTP, A2A-style discovery, and MCP tools,
- testing whether a local model and integration caches are ready.

## Core Features

- **WAKE.md compiler**: scans watched paths and extracts marked decisions,
  tasks, notes, fixes, and evidence.
- **Verifiable provenance**: writes `state.json`, `evidence.jsonl`, and signed
  ed25519 receipts with SHA-256 hashes.
- **Agent handoff**: produces copyable instructions, diagnostics, and manifest
  URLs for another coding agent.
- **Local UI launchpad**: browser UI for setup, context sources, diagnostics,
  integrations, model smoke tests, and handoff bundles.
- **Agent interop**: native `/agent/connect`, A2A-compatible Agent Card, and
  a minimal MCP Streamable HTTP endpoint.
- **Context sources**: compile one project, a monorepo, a workspace, or a notes
  vault by configuring watched paths.
- **Integration cache readers**: GitHub, Gmail, Calendar, Slack, and Linear can
  contribute evidence from local caches or token-backed adapters.
- **Experimental training pipeline**: consolidate sessions into JSONL and train
  LoRA adapters when you explicitly opt in.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quick Start

```bash
# Initialize Morpheus state in the current project.
morpheus init

# Compile watched sources into WAKE.md, state.json, evidence.jsonl, and receipt.
morpheus compile

# Verify the receipt chain and latest compiled artifacts.
morpheus verify --all

# Print the current compiled memory.
morpheus wake
```

## UI Start

Run the backend and static browser UI together:

```bash
morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173
```

Open:

```text
http://127.0.0.1:5173/ui/index.html
```

For trusted LAN testing, use:

```bash
morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173
```

The Start screen lets you set a project root, configure watched paths, run
diagnostics, prepare an agent, inspect integration status, probe MCP tools, and
copy a complete handoff bundle.

## Obsidian And Personal Notes

An Obsidian vault can be used as a Morpheus context source because it is just a
folder of Markdown files. The recommended pattern is not to fine-tune directly
on the whole vault. Compile and retrieve first, keep source links, and only
promote stable, reviewed memories into any future training dataset.

Example:

```bash
cd ~/Obsidian
morpheus init
morpheus compile
morpheus verify --all
```

For a workspace that contains several projects or vaults, set the parent folder
as the project root and configure `.morpheus/morpheus.toml`:

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

1. Fetch `/agent/connect` or run `morpheus agent-connect --json`.
2. Follow the returned `next_action`.
3. Read `WAKE.md` before editing.
4. Run `morpheus compile` and `morpheus verify --all` after meaningful changes.

## Integrations

List integration adapters:

```bash
morpheus integrate --list
morpheus integrate --list --json
```

Current adapters:

- `github`: GitHub issues, pull requests, commits, and cached metadata.
- `gmail`: local Gmail cache and OAuth-oriented token path.
- `calendar`: local Calendar cache and OAuth-oriented token path.
- `slack`: local Slack export cache plus optional token file.
- `linear`: local Linear issue cache plus optional token file.

Local tokens and caches live outside the repository by default under
`~/.morpheus/`.

## Architecture

```text
morpheus/
  core/          compiler, models, receipts, verification, safe IO
  integrations/  filesystem and cache-backed external sources
  api/           FastAPI, agent connect, diagnostics, MCP, A2A card
  training/      experimental consolidation and LoRA training helpers
ui/              static browser UI and Tauri shell
tests/           pytest suite for compiler, API, CLI, integrations, training
docs/            release and testing notes
```

Compile flow:

```text
morpheus compile
  -> scans configured watch_dirs
  -> extracts markers such as TODO:, DECISION:, FIXME:, NOTE:, HACK:, XXX:
  -> writes state.json and evidence.jsonl
  -> generates WAKE.md
  -> signs a receipt with ed25519
  -> links the receipt to the previous receipt hash
```

## CLI Reference

| Command | Description |
| --- | --- |
| `morpheus init` | Initialize `.morpheus/` with config and keys |
| `morpheus compile` | Compile sources into WAKE.md and a signed receipt |
| `morpheus verify --all` | Verify receipt chain, signatures, and latest artifacts |
| `morpheus status` | Show source, claim, and evidence counts |
| `morpheus wake` | Print WAKE.md |
| `morpheus prepare-agent` | Initialize, compile, bootstrap AGENTS.md, verify, and produce handoff |
| `morpheus handoff` | Print a copyable markdown handoff |
| `morpheus agent-connect --json` | Print the machine-readable agent manifest |
| `morpheus diagnostics --json` | Print readiness checks and next action |
| `morpheus bootstrap-agent` | Create or refresh Morpheus instructions in AGENTS.md |
| `morpheus integrate --list` | Show integration setup state |
| `morpheus consolidate` | Build a training dataset from sessions |
| `morpheus train` | Train an experimental LoRA adapter |
| `morpheus eval` | Evaluate an adapter on held-out questions |
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

## License

MIT
