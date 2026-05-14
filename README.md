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
| `morpheus integrate --list` | Show available integrations |
| `morpheus consolidate --days 7` | Sessions → training dataset |
| `morpheus train --epochs 3` | QLoRA fine-tuning |
| `morpheus eval --test-file eval_questions.jsonl` | Evaluate adapter quality |
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
│   ├── integrations/     # Gmail, Calendar, GitHub
│   ├── api/             # FastAPI server
│   └── training/        # Phase 3: QLoRA pipeline
├── ui/                  # Tauri desktop app
├── tests/               # pytest suite
└── scripts/             # Automation scripts
```

## License

MIT
