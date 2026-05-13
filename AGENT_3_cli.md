# Agent 3 — CLI + Entry Point

## Your Task

Implement CLI and Python package scaffolding in `/Users/testbot/.openclaw/workspace/morpheus-ai/`.

### Files to create:

### 1. `morpheus/__init__.py`
```python
"""
Morpheus AI - Agent State Compiler with verifiable provenance.
"""
__version__ = "0.1.0"
```

### 2. `morpheus/cli.py`
Main CLI entry point using `typer`:

```python
#!/usr/bin/env python3
"""
Morpheus CLI - morpheus <command>
"""
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

from morpheus.core.config import MorpheusConfig
from morpheus.core.compiler import compile_project
from morpheus.core.wake import generate_wake_md
from morpheus.core.provenance import compute_sha256_file, build_receipt
from morpheus.core.verify import verify_receipt_chain

app = typer.Typer(help="Morpheus AI - Agent State Compiler")
console = Console()

@app.command()
def init():
    """Initialize morpheus in current directory"""
    config = MorpheusConfig(Path.cwd())
    config.init_default()
    console.print("[green]Initialized morpheus in current directory[/green]")

@app.command()
def compile():
    """Compile sources → state.json + WAKE.md + receipt"""
    project_root = Path.cwd()
    morpheus_dir = project_root / ".morpheus"
    
    if not morpheus_dir.exists():
        console.print("[red]Not initialized. Run 'morpheus init' first.[/red]")
        raise typer.Exit(1)
    
    console.print("[blue]Compiling project state...[/blue]")
    
    # Compile
    state = compile_project(project_root)
    
    # Load config
    config = MorpheusConfig(project_root)
    cfg = config.load()
    
    # Get previous receipt hash
    receipts_dir = morpheus_dir / "receipts"
    prev_hash = None
    if receipts_dir.exists():
        existing = list(receipts_dir.glob("receipt_*.json"))
        if existing:
            import json
            last = sorted(existing)[-1]
            last_receipt = json.loads(last.read_text())
            prev_hash = last_receipt.get("receipt_id")
    
    # Build sources list
    sources_data = [{
        "id": s.id,
        "path": s.path,
        "sha256": s.sha256,
        "size_bytes": s.size_bytes,
        "line_count": s.line_count
    } for s in state.sources]
    
    # Generate WAKE.md
    wake_md = generate_wake_md(state, "pending")
    
    # Compute SHA256 of WAKE.md
    temp_wake_path = morpheus_dir / "WAKE.md.pending"
    temp_wake_path.write_text(wake_md)
    wake_md_sha = compute_sha256_file(temp_wake_path)
    temp_wake_path.unlink()
    
    # Build receipt
    private_key_path = morpheus_dir / "keys" / "local.key"
    receipt = build_receipt(
        state.model_dump(),
        wake_md_sha,
        sources_data,
        private_key_path,
        prev_hash
    )
    
    # Update WAKE.md with real receipt_id
    wake_md = wake_md.replace("pending", receipt["receipt_id"])
    wake_path = morpheus_dir / "WAKE.md"
    wake_path.write_text(wake_md)
    
    # Save state
    state_path = morpheus_dir / "state.json"
    import json
    state_path.write_text(json.dumps(state.model_dump(), indent=2, default=str))
    
    # Save receipt
    receipt_path = receipts_dir / f"receipt_{receipt['receipt_id'].split('_')[1]}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    
    # Update audit log
    audit_log = receipts_dir / "audit.log"
    with open(audit_log, "a") as f:
        f.write(f"{receipt['issued_at']} {receipt['receipt_id']}\n")
    
    console.print(f"[green]Compiled: {len(state.claims)} claims from {len(state.sources)} sources[/green]")
    console.print(f"[green]Receipt: {receipt['receipt_id']}[/green]")

@app.command()
def verify(all: bool = typer.Option(False, "--all", help="Full provenance verification")):
    """Verify receipt chain"""
    project_root = Path.cwd()
    morpheus_dir = project_root / ".morpheus"
    
    if not morpheus_dir.exists():
        console.print("[red]Not initialized.[/red]")
        raise typer.Exit(1)
    
    receipts_dir = morpheus_dir / "receipts"
    if not receipts_dir.exists():
        console.print("[red]No receipts found.[/red]")
        raise typer.Exit(1)
    
    if all:
        valid, errors = verify_receipt_chain(morpheus_dir)
        if valid:
            console.print("[green]Receipt chain valid[/green]")
        else:
            console.print("[red]Verification failed:[/red]")
            for e in errors:
                console.print(f"  [red]- {e}[/red]")
            raise typer.Exit(1)
    else:
        # Quick check - latest receipt exists
        existing = list(receipts_dir.glob("receipt_*.json"))
        if existing:
            console.print(f"[green]Latest receipt: {sorted(existing)[-1].name}[/green]")
        else:
            console.print("[yellow]No receipts found[/yellow]")

@app.command()
def status():
    """Show current state summary"""
    project_root = Path.cwd()
    morpheus_dir = project_root / ".morpheus"
    
    if not morpheus_dir.exists():
        console.print("[yellow]Not initialized[/yellow]")
        return
    
    state_path = morpheus_dir / "state.json"
    if state_path.exists():
        import json
        state = json.loads(state_path.read_text())
        
        table = Table("Metric", "Value")
        table.add_row("Sources", str(len(state.get("sources", []))))
        table.add_row("Claims", str(len(state.get("claims", []))))
        table.add_row("Evidence", str(len(state.get("evidence", []))))
        console.print(table)
    else:
        console.print("[yellow]No state yet. Run 'morpheus compile'[/yellow]")

@app.command()
def wake():
    """Print WAKE.md to stdout"""
    project_root = Path.cwd()
    morpheus_dir = project_root / ".morpheus"
    wake_path = morpheus_dir / "WAKE.md"
    
    if not wake_path.exists():
        console.print("[red]No WAKE.md found. Run 'morpheus compile'[/red]")
        raise typer.Exit(1)
    
    print(wake_path.read_text())

if __name__ == "__main__":
    app()
```

### 3. `pyproject.toml`
```toml
[project]
name = "morpheus-ai"
version = "0.1.0"
description = "Agent State Compiler with verifiable provenance"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Morpheus Team", email = "team@morpheus.ai"}
]
keywords = ["ai", "agent", "provenance", "memory", "lora"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "typer[all]>=0.12.0",
    "pydantic>=2.0.0",
    "cryptography>=42.0.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
    "toml>=0.10.0",
    "rich>=13.0.0",
    "inotify-simple>=1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.4.0",
]

[project.scripts]
morpheus = "morpheus.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py310"
```

### 4. `README.md`
```markdown
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
```

## Instructions

1. Create all files in `/Users/testbot/.openclaw/workspace/morpheus-ai/`
2. Ensure `morpheus/cli.py` is executable and has correct shebang
3. Create `README.md` with placeholder
4. Test imports work: `python -c "from morpheus.core import Source, Claim"`
