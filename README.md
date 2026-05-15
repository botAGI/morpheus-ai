# Morpheus AI

**Agent State Compiler with verifiable provenance.**

Stop starting AI agents from scratch. Morpheus generates `WAKE.md` — a compiled project state with a verifiable provenance trail.

## Quick Start

```bash
# Install
pip install -e .

# Initialize project
morpheus init

# Compile state + generate receipt
morpheus compile

# Verify chain integrity
morpheus verify --all

# Show project status
morpheus status

# Inspect readiness and generate agent instructions
morpheus prepare-agent
morpheus prepare-agent --json
morpheus handoff
morpheus handoff --json
morpheus agent-connect --json
morpheus diagnostics --json
morpheus bootstrap-agent --dry-run
morpheus bootstrap-agent

# Run backend + browser UI
morpheus serve --ui --host 127.0.0.1 --port 8000
```

## UI Quick Start

Run the backend and browser UI with one command:

```bash
morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173
```

Open `http://127.0.0.1:5173/ui/index.html` on the same machine, or
`http://<machine-ip>:5173/ui/index.html` from another device on the same LAN.

The first screen is the setup checklist:

1. Set the project root.
2. Set **Context Sources**. Use `.` for the whole root, or one path per line
   for a monorepo/workspace such as `frontend`, `backend`, and `docs`.
3. Click **Prepare Agent**.
4. Copy the Agent Connect URL, Agent Prompt, or Handoff for another agent.

The Start screen also keeps recent project roots and includes **Diagnostics** for
backend, initialization, WAKE, receipt readiness, and the recommended next
action. The FAQ on the Start screen explains the main flows directly in the UI.

## Multiple Context Sources

Morpheus reads paths from `.morpheus/morpheus.toml`:

```toml
watch_dirs = ["frontend", "backend", "docs"]
```

All watched paths must stay inside the selected project root. To absorb several
projects, choose their parent workspace as the project root and list each project
folder in **Context Sources**.

## Integrations

List available integrations:

```bash
morpheus integrate --list
```

Current integration adapters:

- `github`: GitHub issues, pull requests, and commits via PAT.
- `gmail`: local Gmail cache/OAuth placeholder.
- `calendar`: local Calendar cache/OAuth placeholder.
- `slack`: local Slack message cache plus optional token file.
- `linear`: local Linear issue cache plus optional token file.

Slack and Linear cache files live in `~/.morpheus/slack_cache.json` and
`~/.morpheus/linear_cache.json`. They are useful for exports or agent-built
sync jobs before full OAuth/API sync is configured.

## Agent Self-Connect

Agents can discover Morpheus over HTTP without reading this README first:

```bash
curl -s http://127.0.0.1:8000/.well-known/morpheus.json
curl -s -X POST http://127.0.0.1:8000/agent/prepare \
  -H 'Content-Type: application/json' \
  -d "{\"project_root\":\"$PWD\"}"
curl -s "http://127.0.0.1:8000/agent/handoff.md?project_root=$PWD"
curl -s "http://127.0.0.1:8000/agent/handoff?project_root=$PWD"
curl -s "http://127.0.0.1:8000/agent/connect?project_root=$PWD"
curl -s "http://127.0.0.1:8000/diagnostics?project_root=$PWD"
curl -s "http://127.0.0.1:8000/config?project_root=$PWD"
curl -s -X POST http://127.0.0.1:8000/config \
  -H 'Content-Type: application/json' \
  -d "{\"project_root\":\"$PWD\",\"watch_dirs\":[\"frontend\",\"backend\",\"docs\"]}"
curl -s -X POST http://127.0.0.1:8000/agent/bootstrap/preview \
  -H 'Content-Type: application/json' \
  -d "{\"project_root\":\"$PWD\"}"
curl -s -X POST http://127.0.0.1:8000/agent/bootstrap \
  -H 'Content-Type: application/json' \
  -d "{\"project_root\":\"$PWD\"}"
```

`/agent/connect` returns the project state, the recommended `next_action`, ordered
request sequence, endpoint URLs, CLI equivalents, curl commands, and a
ready-to-copy agent prompt. A new agent should:

1. Fetch `/agent/connect`.
2. Run `next_action` when it is `prepare_agent`.
3. Read `WAKE.md` before making project changes.
4. Run compile and verify after meaningful changes.

`/agent/bootstrap` creates or refreshes the Morpheus-managed section in
`AGENTS.md` without overwriting existing project-specific instructions.

Agents running locally can use the CLI equivalent without starting the HTTP API:

```bash
morpheus prepare-agent
morpheus prepare-agent --json
morpheus handoff
morpheus handoff --json
morpheus agent-connect --json
morpheus diagnostics --json
morpheus bootstrap-agent --dry-run
morpheus bootstrap-agent --api-base http://127.0.0.1:8000
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `morpheus init` | Initialize .morpheus/ with keys |
| `morpheus compile` | Compile sources → WAKE.md + receipt |
| `morpheus verify` | Quick verify latest receipt |
| `morpheus verify --all` | Full chain + signature verification |
| `morpheus status` | Show sources/claims/evidence counts |
| `morpheus wake` | Print WAKE.md to stdout |
| `morpheus prepare-agent` | Initialize, compile, bootstrap AGENTS.md, verify, and print handoff |
| `morpheus prepare-agent --json` | Run the full prepare flow and print machine-readable results |
| `morpheus handoff` | Print a copyable markdown bundle for another agent |
| `morpheus handoff --json` | Print full handoff bundle as machine-readable JSON |
| `morpheus agent-connect --json` | Print full self-connect manifest for agents |
| `morpheus diagnostics --json` | Print readiness checks for agents/tools |
| `morpheus bootstrap-agent --dry-run` | Preview Morpheus instructions for AGENTS.md |
| `morpheus bootstrap-agent` | Create/update Morpheus instructions in AGENTS.md |
| `morpheus integrate --list` | Show available integrations |
| `morpheus consolidate --days 7` | Sessions → training dataset |
| `morpheus train --epochs 3` | QLoRA fine-tuning |
| `morpheus eval --test-file eval_questions.jsonl` | Evaluate adapter quality |
| `morpheus serve --port 8000` | Run FastAPI backend for the UI |
| `morpheus serve --ui --ui-port 5173` | Run backend and static browser UI together |
| `morpheus version` | Show version |

## Training Pipeline (Phase 3)

```bash
# 1. Consolidate sessions
morpheus consolidate --days 7 --min-pairs 10 --output dataset.jsonl --stats-output reports/consolidation.json

# 2. Train adapter
morpheus train --base-model qwen2.5:7b --dataset dataset.jsonl

# 3. Evaluate
morpheus eval --adapter-path morpheus_adapters/
```

## What is this?

Morpheus compiles your project sources, decisions, tasks, and agent history into a portable state (`WAKE.md`) with cryptographic receipts proving where each claim came from.

```
README.md     → tells humans what this is
AGENTS.md     → tells agents how to work here
WAKE.md       → tells agents where we are now
.morpheus/   → machine state, receipts, evidence
```

## Architecture

```
morpheus compile
  → extracts sources from project files
  → builds claims from markers (TODO:, DECISION:, FIXME:, NOTE:, HACK:, XXX:)
  → generates evidence chain with SHA-256 hashes
  → signs receipt with ed25519
  → writes WAKE.md + state.json + receipt
```

## Project Structure

```
morpheus-ai/
├── morpheus/
│   ├── cli.py           # CLI commands
│   ├── core/            # Compiler, provenance, models
│   ├── integrations/     # Filesystem, Gmail, Calendar, GitHub, Slack, Linear
│   ├── api/             # FastAPI server
│   └── training/        # Phase 3: QLoRA pipeline
├── ui/                  # Tauri desktop app
├── tests/               # pytest suite
└── scripts/             # Automation scripts
```

## License

MIT
