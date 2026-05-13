# Morpheus AI

**Agent State Compiler with verifiable provenance.**

Stop starting AI agents from scratch. Morpheus generates `WAKE.md` — a compiled project state with a verifiable provenance trail.

## Quick Start

```bash
pip install morpheus-ai
morpheus init
morpheus compile
morpheus verify --all
```

## What is this?

Morpheus compiles your project sources, decisions, tasks, and agent history into a portable state (`WAKE.md`) with cryptographic receipts proving where each claim came from.

```
README.md     → tells humans what this is
AGENTS.md     → tells agents how to work here
WAKE.md       → tells agents where we are now
.morpheus/   → machine state, receipts, evidence
```

## Features

- **State Compilation**: Extract decisions, tasks, and facts from project files
- **Provenance Chain**: Signed receipts with SHA-256 evidence chains
- **Verification**: `morpheus verify --provenance` validates the entire chain
- **Integrations**: Gmail, Google Calendar, GitHub (more coming)
- **Daily Training Ready**: Phase 3 adds QLoRA fine-tuning for weights-as-memory

## Architecture

```
morpheus compile
  → extracts sources
  → builds claims from markers (TODO:, DECISION:, etc)
  → generates evidence chain
  → signs receipt with ed25519
  → writes WAKE.md + state.json + receipt
```

## License

MIT
