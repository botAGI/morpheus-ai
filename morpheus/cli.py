#!/usr/bin/env python3
"""
Morpheus CLI - morpheus <command>

Agent State Compiler with verifiable provenance.
"""
import json
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from morpheus.core.config import MorpheusConfig
from morpheus.core.compiler import compile_project
from morpheus.core.wake import generate_wake_md
from morpheus.core.provenance import (
    compute_sha256_file,
    compute_sha256_bytes,
    build_receipt,
    evidence_jsonl_bytes,
    latest_receipt_file,
    new_receipt_id,
    receipt_file_name,
)
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.verify import verify_receipt_chain
from morpheus.training.consolidate import consolidate_sessions
from morpheus.training.train import check_dependencies

app = typer.Typer(
    help="Morpheus AI — Agent State Compiler with verifiable provenance",
    add_completion=False
)
console = Console()
WILDCARD_HOSTS = {"0.0.0.0", "::", ""}


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler that keeps CLI output focused on Morpheus URLs."""

    def log_message(self, format: str, *args) -> None:
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        if self.allow_reuse_address and hasattr(socket, "SO_REUSEADDR"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if (
            self.allow_reuse_port
            and hasattr(socket, "SO_REUSEPORT")
            and self.address_family in (socket.AF_INET, socket.AF_INET6)
        ):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def display_url(host: str, port: int, path: str = "") -> str:
    """Return a URL humans can open when a service binds to host."""
    visible_host = "127.0.0.1" if host in WILDCARD_HOSTS else host
    if ":" in visible_host and not visible_host.startswith("["):
        visible_host = f"[{visible_host}]"
    visible_path = path if not path or path.startswith("/") else f"/{path}"
    return f"http://{visible_host}:{port}{visible_path}"


def primary_lan_ip() -> str | None:
    """Best-effort LAN IP for cross-device URLs; returns None offline."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip_address = sock.getsockname()[0]
    except OSError:
        return None
    if ip_address.startswith("127."):
        return None
    return ip_address


def default_ui_root() -> Path:
    """Find the source tree that contains ui/index.html."""
    candidates = [Path.cwd(), Path(__file__).resolve().parents[1]]
    for candidate in candidates:
        if (candidate / "ui" / "index.html").is_file():
            return candidate
    return Path.cwd()


def resolve_ui_root_or_exit(ui_root: Path | None) -> Path:
    """Validate the directory served by `morpheus serve --ui`."""
    root = ui_root.expanduser() if ui_root else default_ui_root()
    try:
        reject_symlink_components(root, "UI root")
    except ValueError as exc:
        console.print(f"[red]UI root invalid:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not root.is_dir():
        console.print(f"[red]UI root not found:[/red] {root}")
        raise typer.Exit(1)

    entrypoint = root / "ui" / "index.html"
    try:
        reject_symlink_components(entrypoint, "UI entrypoint")
        reject_symlink_paths([entrypoint], "UI entrypoint")
    except ValueError as exc:
        console.print(f"[red]UI entrypoint invalid:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not entrypoint.is_file():
        console.print(f"[red]UI entrypoint not found:[/red] {entrypoint}")
        raise typer.Exit(1)
    return root


def start_static_ui_server(*, directory: Path, host: str, port: int):
    """Start the static UI server in a daemon thread and return the server."""
    handler = partial(QuietHTTPRequestHandler, directory=str(directory))
    server = ReusableThreadingHTTPServer((host, port), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def serve_summary_lines(host: str, port: int, ui_host: str | None, ui_port: int) -> list[str]:
    lines = [f"API: {display_url(host, port)}"]
    needs_lan_ip = host in WILDCARD_HOSTS or ui_host in WILDCARD_HOSTS
    lan_ip = primary_lan_ip() if needs_lan_ip else None
    if lan_ip and host in WILDCARD_HOSTS:
        lines.append(f"LAN API: {display_url(lan_ip, port)}")
    if ui_host is not None:
        lines.append(f"UI: {display_url(ui_host, ui_port, '/ui/index.html')}")
        if lan_ip and ui_host in WILDCARD_HOSTS:
            lines.append(f"LAN UI: {display_url(lan_ip, ui_port, '/ui/index.html')}")
    return lines



def ensure_initialized():
    """Check if morpheus is initialized in current directory."""
    morpheus_dir = Path.cwd() / ".morpheus"
    if morpheus_dir.is_symlink() or not morpheus_dir.is_dir():
        console.print("[red]Not initialized. Run 'morpheus init' first.[/red]")
        raise typer.Exit(1)
    return morpheus_dir


def latest_receipt_or_exit(receipts_dir: Path) -> Path | None:
    """Return the receipt chain tail or exit with a user-facing error."""
    try:
        return latest_receipt_file(receipts_dir)
    except (json.JSONDecodeError, ValueError) as exc:
        console.print(f"[red]Receipt chain invalid:[/red] {exc}")
        raise typer.Exit(1) from exc


def load_json_or_exit(path: Path, label: str) -> dict:
    """Load a JSON object or exit with a user-facing error."""
    try:
        reject_symlink_paths([path], label)
        data = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]{label} invalid:[/red] {exc}")
        raise typer.Exit(1) from exc
    if not isinstance(data, dict):
        console.print(f"[red]{label} invalid:[/red] expected JSON object")
        raise typer.Exit(1)
    return data


def list_count(value) -> int:
    """Return the length only for JSON arrays."""
    return len(value) if isinstance(value, list) else 0


def receipt_claim_total(value) -> int:
    """Return total receipt claims only for numeric claim-count mappings."""
    if not isinstance(value, dict):
        return 0
    return sum(count for count in value.values() if isinstance(count, int))


def github_token_path_error(token_path: Path) -> str | None:
    token_dir = token_path.parent
    if token_dir.is_symlink():
        return f"GitHub token directory must not be a symlink: {token_dir}"
    if token_dir.exists() and not token_dir.is_dir():
        return f"GitHub token directory is not a directory: {token_dir}"
    if token_path.is_symlink():
        return f"GitHub token path must not be a symlink: {token_path}"
    if token_path.exists() and not token_path.is_file():
        return f"GitHub token path is not a file: {token_path}"
    return None


def request_context(api_base: str):
    """Build the small request shape shared API helpers need."""
    clean_api_base = api_base.rstrip("/")
    return SimpleNamespace(
        base_url=clean_api_base + "/",
        embedded_agent_api_base=clean_api_base,
    )


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Reinitialize even if already initialized")
):
    """Initialize morpheus in current directory.
    
    Creates .morpheus/ with morpheus.toml and ed25519 keys.
    """
    morpheus_dir = Path.cwd() / ".morpheus"
    
    if morpheus_dir.is_symlink():
        console.print("[red].morpheus path must not be a symlink[/red]")
        raise typer.Exit(1)

    if morpheus_dir.exists() and not morpheus_dir.is_dir():
        console.print("[red].morpheus path is not a directory[/red]")
        raise typer.Exit(1)

    if morpheus_dir.exists() and not force:
        console.print("[yellow].morpheus/ already exists. Use --force to reinitialize.[/yellow]")
        raise typer.Exit(1)
    
    config = MorpheusConfig(project_root=Path.cwd())
    try:
        config.init_default()
    except (OSError, ValueError) as exc:
        console.print(f"[red]Initialization failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    
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
    
    Extracts claims (TODO:, DECISION:, FIXME:, NOTE:, HACK:, XXX:) from
    project files, builds evidence chain, and generates cryptographic receipt.
    """
    morpheus_dir = ensure_initialized()
    project_root = Path.cwd()
    
    console.print("[blue]Compiling project state...[/blue]")
    
    # Compile
    try:
        state = compile_project(project_root)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    
    # Get previous receipt hash
    receipts_dir = morpheus_dir / "receipts"
    prev_hash = None
    if receipts_dir.exists():
        latest = latest_receipt_or_exit(receipts_dir)
        if latest:
            try:
                prev_hash = compute_sha256_file(latest)
            except OSError as exc:
                console.print(
                    f"[red]Receipt chain invalid:[/red] {latest.name}: "
                    f"unreadable receipt ({exc})"
                )
                raise typer.Exit(1) from exc
    
    # Build sources list
    sources_data = [{
        "id": s.id,
        "path": s.path,
        "sha256": s.sha256,
        "size_bytes": s.size_bytes,
        "line_count": s.line_count
    } for s in state.sources]
    
    # Generate final WAKE.md before signing so the receipt hashes the artifact on disk.
    receipt_id = new_receipt_id()
    state.receipt_id = receipt_id
    state_dump = state.model_dump()
    state_json = json.dumps(state_dump, indent=2, default=str)
    state_json_sha = compute_sha256_bytes(state_json.encode())
    evidence_jsonl = evidence_jsonl_bytes(state_dump.get("evidence", []))
    evidence_jsonl_sha = compute_sha256_bytes(evidence_jsonl)

    wake_md = generate_wake_md(state, receipt_id)
    wake_md_sha = compute_sha256_bytes(wake_md.encode())
    
    # Build receipt
    private_key_path = morpheus_dir / "keys" / "local.key"
    try:
        receipt = build_receipt(
            state_dump,
            wake_md_sha,
            sources_data,
            private_key_path,
            prev_hash,
            receipt_id=receipt_id,
            state_json_sha=state_json_sha,
            evidence_jsonl_sha=evidence_jsonl_sha,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]Signing failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    
    # Write artifacts only after all target paths are known to be safe.
    wake_path = morpheus_dir / "WAKE.md"
    state_path = morpheus_dir / "state.json"
    evidence_path = morpheus_dir / "evidence.jsonl"
    receipt_path = receipts_dir / receipt_file_name(receipt["receipt_id"])
    audit_log = receipts_dir / "audit.log"

    try:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_paths(
            [wake_path, state_path, evidence_path, receipt_path, audit_log],
            "Output path",
        )

        wake_path.write_text(wake_md)
        
        # Save state
        state_path.write_text(state_json)

        # Save evidence
        evidence_path.write_bytes(evidence_jsonl)
        
        # Save receipt
        receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
        
        # Update audit log
        with open(audit_log, "a") as f:
            f.write(f"{receipt['issued_at']} {receipt['receipt_id']}\n")
    except (OSError, ValueError) as exc:
        console.print(f"[red]Output write failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    
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
    
    if not receipts_dir.exists():
        console.print("[yellow]No receipts found[/yellow]")
        raise typer.Exit(1)
    if not receipts_dir.is_dir():
        console.print("[red]Receipt chain invalid:[/red] receipts path is not a directory")
        raise typer.Exit(1)
    if not list(receipts_dir.glob("receipt_*.json")):
        console.print("[yellow]No receipts found[/yellow]")
        raise typer.Exit(1)
    
    existing = sorted(receipts_dir.glob("receipt_*.json"))
    
    if all:
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
        latest = latest_receipt_or_exit(receipts_dir)
        receipt = load_json_or_exit(latest, "Receipt file")
        
        if verbose:
            table = Table(title="Latest Receipt")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("ID", receipt.get("receipt_id", "unknown"))
            table.add_row("Issued", receipt.get("issued_at", "unknown"))
            table.add_row("Claims", str(receipt_claim_total(receipt.get("claim_count"))))
            table.add_row("Sources", str(list_count(receipt.get("sources"))))
            console.print(table)
        else:
            console.print(f"[green]✓ Latest:[/green] {receipt.get('receipt_id', 'unknown')}")


@app.command()
def status():
    """Show current project state summary."""
    morpheus_dir = Path.cwd() / ".morpheus"
    
    if morpheus_dir.is_symlink() or not morpheus_dir.is_dir():
        console.print("[yellow]Not initialized[/yellow]")
        return
    
    state_path = morpheus_dir / "state.json"
    if not state_path.exists():
        console.print("[yellow]No compilation yet. Run 'morpheus compile'[/yellow]")
        return
    
    state = load_json_or_exit(state_path, "State file")
    
    receipts_dir = morpheus_dir / "receipts"
    receipt_path = latest_receipt_or_exit(receipts_dir) if receipts_dir.exists() else None
    
    table = Table(title="Project Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    compiled_at = state.get("compiled_at")
    compiled_at_display = str(compiled_at)[:19] if compiled_at else "unknown"
    table.add_row("Sources", str(list_count(state.get("sources"))))
    table.add_row("Claims", str(list_count(state.get("claims"))))
    table.add_row("Evidence", str(list_count(state.get("evidence"))))
    table.add_row("Last Compiled", compiled_at_display)
    table.add_row("Latest Receipt", receipt_path.name.replace("receipt_", "").replace(".json", "") if receipt_path else "none")
    console.print(table)


@app.command("diagnostics")
def diagnostics_command(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    api_base: str = typer.Option(
        "http://127.0.0.1:8000",
        "--api-base",
        help="API base URL to embed in agent connect links",
    ),
):
    """Inspect backend-style readiness for the current project without starting a server."""
    from morpheus.api.server import diagnostics_payload

    payload = diagnostics_payload(request_context(api_base), Path.cwd())
    if json_output:
        console.out(json.dumps(payload, indent=2))
        return

    table = Table(title="Morpheus Diagnostics")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Detail", style="yellow")
    for check in payload["checks"]:
        table.add_row(
            check["label"],
            "OK" if check["ok"] else "Needs action",
            check["detail"],
        )
    console.print(table)
    next_action = payload["next_action"]
    console.print(f"Next action: {next_action['label']}")
    if next_action.get("command"):
        console.print(f"Command: {next_action['command']}")
    console.print(f"Agent connect: {payload['agent_connect_url']}")


@app.command("agent-connect")
def agent_connect_command(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    api_base: str = typer.Option(
        "http://127.0.0.1:8000",
        "--api-base",
        help="API base URL to embed in HTTP agent links",
    ),
):
    """Print the full self-connect manifest for agents without starting a server."""
    from morpheus.api.server import agent_connect_payload

    payload = agent_connect_payload(request_context(api_base), Path.cwd())
    if json_output:
        console.out(json.dumps(payload, indent=2))
        return

    state = payload["state"]
    next_action = payload["next_action"]
    console.print(Panel.fit(
        f"Project: [bold]{payload['project_root']}[/bold]\n"
        f"Initialized: [bold]{state['initialized']}[/bold]\n"
        f"Compiled: [bold]{state['compiled']}[/bold]\n"
        f"Next action: [bold]{next_action['label']}[/bold]\n"
        f"Command: [bold]{next_action['command']}[/bold]\n"
        "Machine JSON: [bold]morpheus agent-connect --json[/bold]\n"
        f"Prompt: {payload['agent_prompt']}",
        title="Morpheus Agent Connect",
        border_style="green",
    ))


@app.command("handoff")
def handoff_command(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    api_base: str = typer.Option(
        "http://127.0.0.1:8000",
        "--api-base",
        help="API base URL to embed in HTTP agent links",
    ),
):
    """Print a complete copyable bundle for handing the project to another agent."""
    from morpheus.api.server import HTTPException, agent_handoff_payload

    try:
        payload = agent_handoff_payload(request_context(api_base), Path.cwd())
    except HTTPException as exc:
        console.print(f"[red]Handoff failed:[/red] {exc.detail}")
        raise typer.Exit(1) from exc

    if json_output:
        console.out(json.dumps(payload, indent=2))
        return

    console.out(payload["markdown"])


@app.command("prepare-agent")
def prepare_agent_command(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    api_base: str = typer.Option(
        "http://127.0.0.1:8000",
        "--api-base",
        help="API base URL to embed in HTTP agent links",
    ),
):
    """Initialize, compile, bootstrap AGENTS.md, verify, and print handoff."""
    from morpheus.api.server import HTTPException, agent_prepare_payload

    try:
        payload = agent_prepare_payload(request_context(api_base), Path.cwd())
    except HTTPException as exc:
        console.print(f"[red]Prepare failed:[/red] {exc.detail}")
        raise typer.Exit(1) from exc

    if json_output:
        console.out(json.dumps(payload, indent=2))
        return

    console.out(payload["handoff"]["markdown"])


@app.command("bootstrap-agent")
def bootstrap_agent(
    api_base: str = typer.Option(
        "http://127.0.0.1:8000",
        "--api-base",
        help="API base URL to embed in AGENTS.md",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the AGENTS.md preview without writing it",
    ),
):
    """Create, refresh, or preview the Morpheus-managed AGENTS.md section."""
    from morpheus.api.server import preview_agent_bootstrap, write_agent_bootstrap
    from morpheus.api.server import HTTPException

    try:
        handler = preview_agent_bootstrap if dry_run else write_agent_bootstrap
        response = handler(request_context(api_base), Path.cwd())
    except HTTPException as exc:
        console.print(f"[red]Bootstrap failed:[/red] {exc.detail}")
        raise typer.Exit(1) from exc

    if dry_run:
        console.out(response.content)
        return

    if response.created:
        action = "Created AGENTS.md"
    elif response.updated:
        action = "Updated AGENTS.md"
    else:
        action = "AGENTS.md already current"

    console.print(Panel.fit(
        f"[green]{action}[/green]\n"
        f"Path: [bold]{response.path}[/bold]\n"
        f"Agent connect: {response.agent_connect_url}",
        title="Morpheus Agent Bootstrap",
        border_style="green",
    ))


@app.command()
def wake():
    """Print WAKE.md to stdout."""
    morpheus_dir = ensure_initialized()
    wake_path = morpheus_dir / "WAKE.md"
    
    if not wake_path.exists():
        console.print("[red]No WAKE.md found. Run 'morpheus compile'[/red]")
        raise typer.Exit(1)
    
    try:
        reject_symlink_paths([wake_path], "WAKE.md")
        content = wake_path.read_text()
    except (OSError, ValueError) as exc:
        console.print(f"[red]WAKE.md unreadable:[/red] {exc}")
        raise typer.Exit(1) from exc
    syntax = Syntax(content, "markdown", theme="monokai", line_numbers=False)
    console.print(syntax)


@app.command()
def integrate(
    service: str | None = typer.Argument(None, help="Service: gmail, calendar, github"),
    list_services: bool = typer.Option(False, "--list", help="List available integrations")
):
    """Connect external integrations.
    
    Services:
      gmail     - Gmail API via OAuth2
      calendar  - Google Calendar API via OAuth2  
      github    - GitHub API via Personal Access Token
    """
    if list_services:
        github_token_path = Path.home() / ".morpheus" / "github_token.txt"
        if github_token_path_error(github_token_path):
            github_status = "[red]invalid[/red]"
        elif github_token_path.is_file():
            github_status = "[green]configured[/green]"
        else:
            github_status = "[yellow]not configured[/yellow]"

        table = Table(title="Available Integrations")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Auth", style="yellow")
        table.add_row("gmail", "[yellow]not configured[/yellow]", "OAuth2")
        table.add_row("calendar", "[yellow]not configured[/yellow]", "OAuth2")
        table.add_row("github", github_status, "PAT")
        console.print(table)
        return

    if service is None:
        console.print("[red]Service required. Use --list to show available integrations.[/red]")
        raise typer.Exit(1)
    if service not in {"gmail", "calendar", "github"}:
        console.print(f"[red]Unknown integration service:[/red] {service}")
        console.print("[yellow]Use --list to show available integrations.[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Setting up {service} integration...[/blue]")
    
    if service == "github":
        token_path = Path.home() / ".morpheus" / "github_token.txt"
        path_error = github_token_path_error(token_path)
        if path_error:
            console.print(f"[red]{path_error}[/red]")
            raise typer.Exit(1)
        if token_path.is_file():
            console.print("[green]✓ GitHub token already configured[/green]")
        else:
            try:
                token_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                console.print(f"[red]GitHub token directory cannot be created:[/red] {exc}")
                raise typer.Exit(1) from exc
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
    min_pairs: int = typer.Option(10, help="Minimum unique Q&A pairs required"),
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
        min_pairs=min_pairs,
        stats_output_path=Path(stats_output) if stats_output else None,
        verbose=verbose
    )


@app.command()
def train(
    base_model: str = typer.Option("qwen2.5:7b", help="Base model name"),
    dataset: str = typer.Option("dataset.jsonl", help="Training dataset"),
    output_dir: str = typer.Option("morpheus_adapters", help="Output directory"),
    lora_rank: int = typer.Option(64, help="LoRA rank"),
    lora_alpha: int = typer.Option(128, help="LoRA alpha"),
    epochs: int = typer.Option(3, help="Training epochs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate script without running")
):
    """Run QLoRA fine-tuning on session dataset.
    
    Requires llamafactory-cli installed.
    """
    from morpheus.training.train import train as run_train
    
    if not dry_run:
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
        lora_alpha=lora_alpha,
        epochs=epochs,
        dry_run=dry_run
    )


@app.command("eval")
def eval_command(
    adapter_path: Path = typer.Option(Path("morpheus_adapters"), help="LoRA adapter path"),
    base_model: str = typer.Option("qwen2.5:7b", help="Base model"),
    test_file: Path = typer.Option(Path("eval_questions.jsonl"), help="Test questions"),
    output: Path = typer.Option(Path("eval_results.jsonl"), help="Results output"),
):
    """Evaluate a fine-tuned adapter on held-out questions."""
    from morpheus.training.eval import run_eval

    run_eval(
        adapter_path=adapter_path,
        base_model=base_model,
        test_file=test_file,
        output=output,
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host for the FastAPI backend"),
    port: int = typer.Option(8000, help="Port for the FastAPI backend"),
    reload: bool = typer.Option(False, "--reload", help="Reload server on code changes"),
    ui: bool = typer.Option(False, "--ui", help="Also serve the static web UI"),
    ui_port: int = typer.Option(5173, "--ui-port", help="Port for the static web UI"),
    ui_host: str | None = typer.Option(
        None,
        "--ui-host",
        help="Host for the static web UI. Defaults to --host.",
    ),
    ui_root: Path | None = typer.Option(
        None,
        "--ui-root",
        help="Directory that contains ui/index.html. Defaults to the current source tree.",
    ),
):
    """Run the FastAPI backend, optionally with the static browser UI."""
    try:
        import uvicorn
    except ImportError as exc:
        console.print("[red]uvicorn is not installed.[/red]")
        console.print("[yellow]Install the project dependencies, then run again.[/yellow]")
        raise typer.Exit(1) from exc

    static_server = None
    bound_ui_host = ui_host or host
    if ui:
        root = resolve_ui_root_or_exit(ui_root)
        try:
            static_server = start_static_ui_server(
                directory=root,
                host=bound_ui_host,
                port=ui_port,
            )
        except OSError as exc:
            console.print(f"[red]UI server failed:[/red] {exc}")
            raise typer.Exit(1) from exc

    console.print(Panel.fit(
        "\n".join(serve_summary_lines(host, port, bound_ui_host if ui else None, ui_port)),
        title="Morpheus Serve",
        border_style="green",
    ))
    try:
        uvicorn.run(
            "morpheus.api.server:app",
            host=host,
            port=port,
            reload=reload,
        )
    finally:
        if static_server is not None:
            static_server.shutdown()
            static_server.server_close()


@app.command()
def version():
    """Show morpheus version."""
    from morpheus import __version__
    console.print(f"Morpheus AI v{__version__}")


if __name__ == "__main__":
    app()
