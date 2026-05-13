#!/usr/bin/env python3
"""
Morpheus CLI - morpheus <command>

Agent State Compiler with verifiable provenance.
"""
import typer
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from morpheus.core.config import MorpheusConfig
from morpheus.core.compiler import compile_project
from morpheus.core.wake import generate_wake_md
from morpheus.core.provenance import compute_sha256_file, build_receipt
from morpheus.core.verify import verify_receipt_chain
from morpheus.training.consolidate import consolidate_sessions
from morpheus.training.train import train, check_dependencies

app = typer.Typer(
    help="Morpheus AI — Agent State Compiler with verifiable provenance",
    add_completion=False
)
console = Console()

def ensure_initialized():
    """Check if morpheus is initialized in current directory."""
    morpheus_dir = Path.cwd() / ".morpheus"
    if not morpheus_dir.exists():
        console.print("[red]Not initialized. Run 'morpheus init' first.[/red]")
        raise typer.Exit(1)
    return morpheus_dir


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Reinitialize even if already initialized")
):
    """Initialize morpheus in current directory.
    
    Creates .morpheus/ with config.toml and ed25519 keys.
    """
    morpheus_dir = Path.cwd() / ".morpheus"
    
    if morpheus_dir.exists() and not force:
        console.print(f"[yellow].morpheus/ already exists. Use --force to reinitialize.[/yellow]")
        raise typer.Exit(1)
    
    config = MorpheusConfig(Path.cwd())
    config.init_default()
    
    console.print(Panel.fit(
        "[green]✓ Morpheus initialized[/green]\n"
        f"Project: [bold]{Path.cwd().name}[/bold]\n"
        "Run [bold]morpheus compile[/bold] to generate WAKE.md",
        title="Morpheus AI",
        border_style="green"
    ))


@app.command()
def compile(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output")
):
    """Compile sources → state.json + WAKE.md + signed receipt.
    
    Extracts claims (TODO:, DECISION:, FIXME:, NOTE:) from project files,
    builds evidence chain, and generates cryptographic receipt.
    """
    morpheus_dir = ensure_initialized()
    project_root = Path.cwd()
    
    console.print("[blue]Compiling project state...[/blue]")
    
    # Compile
    state = compile_project(project_root)
    
    # Get previous receipt hash
    receipts_dir = morpheus_dir / "receipts"
    prev_hash = None
    if receipts_dir.exists():
        existing = sorted(receipts_dir.glob("receipt_*.json"))
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
    receipt_placeholder = f"rcpt_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_pending"
    wake_md = generate_wake_md(state, receipt_placeholder)
    
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
    wake_md = wake_md.replace(receipt_placeholder, receipt["receipt_id"])
    wake_path = morpheus_dir / "WAKE.md"
    wake_path.write_text(wake_md)
    
    # Save state
    state_path = morpheus_dir / "state.json"
    import json
    state_path.write_text(json.dumps(state.model_dump(), indent=2, default=str))
    
    # Save receipt
    receipt_filename = f"receipt_{receipt['receipt_id'].split('_')[1]}.json"
    receipt_path = receipts_dir / receipt_filename
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    
    # Update audit log
    audit_log = receipts_dir / "audit.log"
    with open(audit_log, "a") as f:
        f.write(f"{receipt['issued_at']} {receipt['receipt_id']}\n")
    
    # Output
    if verbose:
        table = Table(title="Compilation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Sources", str(len(state.sources)))
        table.add_row("Claims", str(len(state.claims)))
        table.add_row("Evidence", str(len(state.evidence)))
        table.add_row("Receipt", receipt["receipt_id"])
        table.add_row("Signed", "✓" if receipt["signature"]["signature_b64"] else "✗")
        console.print(table)
    else:
        console.print(f"[green]✓ Compiled:[/green] {len(state.claims)} claims from {len(state.sources)} sources")
        console.print(f"[green]✓ Receipt:[/green] {receipt['receipt_id']}")


@app.command()
def verify(
    all: bool = typer.Option(False, "--all", "-a", help="Full provenance verification"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output")
):
    """Verify receipt chain integrity.
    
    Without --all: checks latest receipt exists.
    With --all: validates entire chain + signatures.
    """
    morpheus_dir = ensure_initialized()
    receipts_dir = morpheus_dir / "receipts"
    
    if not receipts_dir.exists() or not list(receipts_dir.glob("receipt_*.json")):
        console.print("[yellow]No receipts found[/yellow]")
        raise typer.Exit(1)
    
    existing = sorted(receipts_dir.glob("receipt_*.json"))
    
    if all:
        import json
        valid, errors = verify_receipt_chain(morpheus_dir)
        
        if valid:
            console.print(Panel.fit(
                "[green]✓ Receipt chain valid[/green]\n"
                f"Total receipts: {len(existing)}\n"
                "All signatures verified",
                title="Verification Passed",
                border_style="green"
            ))
        else:
            console.print(Panel.fit(
                "[red]✗ Verification failed[/red]\n" + "\n".join(f"  • {e}" for e in errors),
                title="Verification Failed",
                border_style="red"
            ))
            raise typer.Exit(1)
    else:
        # Quick check
        latest = existing[-1]
        receipt = json.loads(latest.read_text())
        
        if verbose:
            table = Table(title="Latest Receipt")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("ID", receipt.get("receipt_id", "unknown"))
            table.add_row("Issued", receipt.get("issued_at", "unknown"))
            table.add_row("Claims", str(sum(receipt.get("claim_count", {}).values())))
            table.add_row("Sources", str(len(receipt.get("sources", []))))
            console.print(table)
        else:
            console.print(f"[green]✓ Latest:[/green] {receipt.get('receipt_id', 'unknown')}")


@app.command()
def status():
    """Show current project state summary."""
    morpheus_dir = Path.cwd() / ".morpheus"
    
    if not morpheus_dir.exists():
        console.print("[yellow]Not initialized[/yellow]")
        return
    
    state_path = morpheus_dir / "state.json"
    if not state_path.exists():
        console.print("[yellow]No compilation yet. Run 'morpheus compile'[/yellow]")
        return
    
    import json
    state = json.loads(state_path.read_text())
    
    receipt_path = sorted((morpheus_dir / "receipts").glob("receipt_*.json"))[-1] if (morpheus_dir / "receipts").exists() else None
    
    table = Table(title="Project Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Sources", str(len(state.get("sources", []))))
    table.add_row("Claims", str(len(state.get("claims", []))))
    table.add_row("Evidence", str(len(state.get("evidence", []))))
    table.add_row("Last Compiled", state.get("compiled_at", "unknown")[:19] if state.get("compiled_at") else "unknown")
    table.add_row("Latest Receipt", receipt_path.name.replace("receipt_", "").replace(".json", "") if receipt_path else "none")
    console.print(table)


@app.command()
def wake():
    """Print WAKE.md to stdout."""
    morpheus_dir = Path.cwd() / ".morpheus"
    wake_path = morpheus_dir / "WAKE.md"
    
    if not wake_path.exists():
        console.print("[red]No WAKE.md found. Run 'morpheus compile'[/red]")
        raise typer.Exit(1)
    
    content = wake_path.read_text()
    syntax = Syntax(content, "markdown", theme="monokai", line_numbers=False)
    console.print(syntax)


@app.command()
def integrate(
    service: str = typer.Argument(..., help="Service: gmail, calendar, github"),
    list_services: bool = typer.Option(False, "--list", help="List available integrations")
):
    """Connect external integrations.
    
    Services:
      gmail     - Gmail API via OAuth2
      calendar  - Google Calendar API via OAuth2  
      github    - GitHub API via Personal Access Token
    """
    if list_services:
        table = Table(title="Available Integrations")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Auth", style="yellow")
        table.add_row("gmail", "[yellow]not configured[/yellow]", "OAuth2")
        table.add_row("calendar", "[yellow]not configured[/yellow]", "OAuth2")
        table.add_row("github", "[yellow]not configured[/yellow]", "PAT")
        console.print(table)
        return
    
    console.print(f"[blue]Setting up {service} integration...[/blue]")
    
    if service == "github":
        token_path = Path.home() / ".morpheus" / "github_token.txt"
        if token_path.exists():
            console.print("[green]✓ GitHub token already configured[/green]")
        else:
            console.print("[yellow]GitHub PAT required[/yellow]")
            console.print("1. Go to https://github.com/settings/tokens")
            console.print("2. Generate new token (classic) with 'repo' scope")
            console.print(f"3. Save token to: {token_path}")
            console.print(f"4. Run: echo 'YOUR_TOKEN' > {token_path}")
    else:
        console.print(f"[yellow]{service} integration not yet implemented[/yellow]")
        console.print("Use GitHub for now, more coming soon.")


@app.command()
def consolidate(
    sessions_dir: str = typer.Option(
        str(Path.home() / ".openclaw/agents/main/sessions"),
        help="OpenClaw sessions directory"
    ),
    output: str = typer.Option("dataset.jsonl", help="Output dataset file"),
    days: int = typer.Option(7, help="Process sessions from last N days"),
    stats_output: str | None = typer.Option(
        None,
        "--stats-output",
        help="Optional JSON file for consolidation counters",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show details")
):
    """Consolidate OpenClaw sessions into training dataset.
    
    Reads sessions from ~/.openclaw/agents/main/sessions/ and creates
    a Q&A dataset for LoRA fine-tuning.
    """
    consolidate_sessions(
        sessions_dir=Path(sessions_dir),
        output_path=Path(output),
        days=days,
        stats_output_path=Path(stats_output) if stats_output else None,
        verbose=verbose
    )


@app.command()
def train(
    base_model: str = typer.Option("qwen2.5:7b", help="Base model name"),
    dataset: str = typer.Option("dataset.jsonl", help="Training dataset"),
    output_dir: str = typer.Option("morpheus_adapters", help="Output directory"),
    lora_rank: int = typer.Option(64, help="LoRA rank"),
    epochs: int = typer.Option(3, help="Training epochs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate script without running")
):
    """Run QLoRA fine-tuning on session dataset.
    
    Requires llamafactory-cli installed.
    """
    from morpheus.training.train import train as run_train
    
    ok, missing = check_dependencies()
    if not ok:
        console.print(f"[red]Missing: {', '.join(missing)}[/red]")
        console.print("[yellow]Install: pip install llamafactory[/yellow]")
        raise typer.Exit(1)
    
    run_train(
        base_model=base_model,
        dataset=Path(dataset),
        output_dir=Path(output_dir),
        lora_rank=lora_rank,
        epochs=epochs,
        dry_run=dry_run
    )


@app.command()
def version():
    """Show morpheus version."""
    from morpheus import __version__
    console.print(f"Morpheus AI v{__version__}")


if __name__ == "__main__":
    app()
