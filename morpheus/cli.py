#!/usr/bin/env python3
"""
Morpheus CLI - morpheus <command>

Source-grounded truth layer and local learning lab for coding agents.
"""
import json
import os
import re
import socket
import sys
from fnmatch import fnmatch
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.parse import urlencode

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from morpheus.core.config import MorpheusConfig
from morpheus.core.compiler import DEFAULT_EXCLUDE_PATTERNS, compile_project
from morpheus.core.check import (
    check_exit_code,
    check_text,
    ci_mode_from_env,
    create_training_corrections,
    discover_project_root,
    render_check_annotated,
    render_check_summary,
)
from morpheus.core.learning.adapters import activate_adapter, list_adapters, rollback_adapter
from morpheus.core.learning.benchmark import write_benchmark_report
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.eval import run_learning_eval
from morpheus.core.learning.lab import (
    DEFAULT_LAB_EVAL_LIMIT,
    DEFAULT_LAB_MAX_ITERS,
    lab_auto_accept,
    run_autonomous_lab,
    run_autonomous_lab_stability,
)
from morpheus.core.learning.quality import write_quality_report
from morpheus.core.learning.registry import learning_status
from morpheus.core.learning.train import plan_training_run
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
from morpheus.core.semantic.review import (
    ReviewStore,
    accept_proposed_candidates,
    apply_accepted_candidates,
    export_review_pack,
    propose_review_candidates,
    run_semantic_review,
    strict_accept_suggestions,
    trainable_candidate,
    write_review_doctor,
    write_review_proposal,
    write_strict_accept_suggestions,
)
from morpheus.core.verify import verify_receipt_chain
from morpheus.core.providers.fake import FakeProvider
from morpheus.core.providers.local import LocalProvider
from morpheus.core.providers.null import NullProvider
from morpheus.core.providers.ollama import OllamaProvider
from morpheus.integrations.manifest import integration_cache_path_error, integration_manifest
from morpheus.training.consolidate import consolidate_sessions
from morpheus.training.train import check_dependencies

app = typer.Typer(
    help="Morpheus AI — source-grounded truth layer with a local learning lab",
    add_completion=False
)
review_app = typer.Typer(help="Review semantic candidates before they become active state.")
app.add_typer(review_app, name="review")
learn_app = typer.Typer(help="Compile reviewed state into local learning artifacts.")
app.add_typer(learn_app, name="learn")
console = Console()
WILDCARD_HOSTS = {"0.0.0.0", "::", ""}
DEFAULT_MODEL_SMOKE_MODEL = "qwen2.5:0.5b"
DEFAULT_MODEL_SMOKE_PROMPT = (
    "Reply with one short sentence confirming Morpheus model smoke test is working."
)
STALE_TEXT_SUFFIXES = {
    ".md",
    ".mdx",
    ".txt",
    ".rst",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".html",
    ".css",
}
STALE_SCAN_ROOT_FILES = {
    "AGENTS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "README.md",
    "README.ru.md",
    "SECURITY.md",
    "SPEC.md",
    "pyproject.toml",
}
STALE_POSITIONING_RULES = [
    {
        "rule_id": "personal_ai_agent",
        "pattern": re.compile(r"\bpersonal AI agent\b", re.IGNORECASE),
        "suggested_replacement": (
            "Morpheus is an Agent State Compiler that generates WAKE.md."
        ),
    },
    {
        "rule_id": "daily_lora_core",
        "pattern": re.compile(
            r"\b(daily training|daily memory consolidation)\b",
            re.IGNORECASE,
        ),
        "suggested_replacement": (
            "Truth-layer verification is the data-quality gate before any "
            "weights-as-memory experiment."
        ),
    },
    {
        "rule_id": "eu_ai_act_claim",
        "pattern": re.compile(r"\bEU AI Act compliant by design\b", re.IGNORECASE),
        "suggested_replacement": (
            "Designed for provenance, local-first operation, source attribution, "
            "and user-controlled state export."
        ),
    },
    {
        "rule_id": "memory_compiler_pitch",
        "pattern": re.compile(r"\bLocal-first memory compiler for AI agents\b", re.IGNORECASE),
        "suggested_replacement": (
            "WAKE.md for AI agents — compile project state so agents stop starting cold."
        ),
    },
]


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version_option: bool = typer.Option(False, "--version", help="Show morpheus version"),
):
    """Run morpheus commands."""
    if version_option:
        from morpheus import __version__

        console.print(f"Morpheus AI v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler that keeps CLI output focused on Morpheus URLs."""

    def log_message(self, format: str, *args) -> None:
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    allow_reuse_port = False

    def server_bind(self):
        if self.allow_reuse_address and hasattr(socket, "SO_REUSEADDR"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if (
            getattr(self, "allow_reuse_port", False)
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


def ui_entrypoint_path(api_base: str) -> str:
    """Return the static UI entrypoint with an explicit backend API hint."""
    return f"/ui/index.html?{urlencode({'api': api_base})}"


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
        console.print("[red]UI entrypoint not found[/red]")
        console.print("Expected UI file: ui/index.html")
        console.print(f"Expected path: {entrypoint}")
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
    local_api = display_url(host, port)
    lines = [f"API: {local_api}"]
    needs_lan_ip = host in WILDCARD_HOSTS or ui_host in WILDCARD_HOSTS
    lan_ip = primary_lan_ip() if needs_lan_ip else None
    if lan_ip and host in WILDCARD_HOSTS:
        lines.append(f"LAN API: {display_url(lan_ip, port)}")
    if ui_host is not None:
        lines.append(f"UI: {display_url(ui_host, ui_port, ui_entrypoint_path(local_api))}")
        if lan_ip and ui_host in WILDCARD_HOSTS:
            lan_api = display_url(lan_ip if host in WILDCARD_HOSTS else host, port)
            lines.append(f"LAN UI: {display_url(lan_ip, ui_port, ui_entrypoint_path(lan_api))}")
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


def integration_token_path_error(token_path: Path, service_label: str) -> str | None:
    token_dir = token_path.parent
    if token_dir.is_symlink():
        return f"{service_label} token directory must not be a symlink: {token_dir}"
    if token_dir.exists() and not token_dir.is_dir():
        return f"{service_label} token directory is not a directory: {token_dir}"
    if token_path.is_symlink():
        return f"{service_label} token path must not be a symlink: {token_path}"
    if token_path.exists() and not token_path.is_file():
        return f"{service_label} token path is not a file: {token_path}"
    return None


def github_token_path_error(token_path: Path) -> str | None:
    return integration_token_path_error(token_path, "GitHub")


def integration_status(token_path: Path, service_label: str) -> str:
    if integration_token_path_error(token_path, service_label):
        return "[red]invalid[/red]"
    if token_path.is_file():
        return "[green]configured[/green]"
    return "[yellow]not configured[/yellow]"


def rich_integration_status(status: str) -> str:
    labels = {
        "configured": "[green]configured[/green]",
        "cache_ready": "[green]cache ready[/green]",
        "not_configured": "[yellow]not configured[/yellow]",
        "invalid": "[red]invalid[/red]",
    }
    return labels.get(status, status)


def request_context(api_base: str):
    """Build the small request shape shared API helpers need."""
    clean_api_base = api_base.rstrip("/")
    return SimpleNamespace(
        base_url=clean_api_base + "/",
        embedded_agent_api_base=clean_api_base,
    )


def ensure_project_initialized(project_root: Path) -> tuple[Path, bool]:
    """Initialize .morpheus for a chosen project root when needed."""
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")

    morpheus_dir = project_root / ".morpheus"
    if morpheus_dir.is_symlink():
        raise ValueError(".morpheus path must not be a symlink")
    if morpheus_dir.exists() and not morpheus_dir.is_dir():
        raise ValueError(".morpheus path is not a directory")
    initialized = not morpheus_dir.exists()
    MorpheusConfig(project_root=project_root).init_default()
    return morpheus_dir, initialized


def copy_public_wake(project_root: Path, morpheus_dir: Path) -> Path:
    """Copy the compiled private WAKE.md to the project root for public handoff."""
    private_wake = morpheus_dir / "WAKE.md"
    public_wake = project_root / "WAKE.md"
    reject_symlink_paths([private_wake, public_wake], "WAKE.md")
    if public_wake.exists() and not public_wake.is_file():
        raise ValueError(f"WAKE.md path is not a file: {public_wake}")
    public_wake.write_text(private_wake.read_text())
    return public_wake


def wake_handoff_prompt() -> str:
    """Return the short prompt printed by the one-command wake flow."""
    return (
        "Read WAKE.md before editing. Treat it as current project state, then run "
        "`morpheus compile` and `morpheus verify --all` after meaningful changes."
    )


def semantic_provider_from_env():
    """Resolve the explicit semantic provider without making cloud calls."""
    provider_name = os.getenv("MORPHEUS_SEMANTIC_PROVIDER", "local").strip().lower()
    if provider_name in {"", "local"}:
        return LocalProvider()
    if provider_name == "fake":
        return FakeProvider()
    if provider_name == "null":
        return NullProvider()
    if provider_name == "ollama":
        provider = OllamaProvider()
        model = os.getenv("MORPHEUS_SEMANTIC_MODEL")
        if model:
            provider.model = model
        return provider
    raise ValueError(f"Unsupported semantic provider: {provider_name}")


def semantic_provider_display(provider) -> str:
    if isinstance(provider, LocalProvider):
        return "local (offline heuristic)"
    if isinstance(provider, NullProvider):
        return "null (no-op)"
    if isinstance(provider, FakeProvider):
        return "fake (test fixture)"
    if isinstance(provider, OllamaProvider):
        return f"ollama (explicit local model: {provider.model})"
    return f"{provider.name} ({provider.model})"


def run_semantic_review_or_exit(project_root: Path) -> dict:
    try:
        provider = semantic_provider_from_env()
        console.print(f"[blue]Semantic provider:[/blue] {semantic_provider_display(provider)}")
        report = run_semantic_review(project_root, provider=provider)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Semantic review failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        "[green]✓ Semantic review:[/green] "
        f"{report['candidates_total']} candidates, "
        f"{report['source_backed_total']} source-backed"
    )
    return report


def find_stale_positioning_claims(project_root: Path) -> list[dict[str, object]]:
    """Find launch-positioning claims that conflict with the WAKE.md framing."""
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")

    findings = []
    for path in sorted(project_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() not in STALE_TEXT_SUFFIXES:
            continue
        if stale_scan_path_excluded(path, project_root):
            continue
        if not stale_scan_path_is_launch_surface(path, project_root):
            continue

        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue

        for line_number, line in enumerate(lines, 1):
            if stale_line_is_negated_or_safe(line):
                continue
            for rule in STALE_POSITIONING_RULES:
                match = rule["pattern"].search(line)
                if not match:
                    continue
                findings.append(
                    {
                        "rule_id": rule["rule_id"],
                        "path": path.relative_to(project_root).as_posix(),
                        "line": line_number,
                        "excerpt": line.strip(),
                        "matched": match.group(0),
                        "suggested_replacement": rule["suggested_replacement"],
                    }
                )
    return findings


def stale_scan_path_excluded(path: Path, project_root: Path) -> bool:
    """Return true when a file should not be scanned by `morpheus stale`."""
    try:
        rel_path = path.relative_to(project_root)
    except ValueError:
        return True
    rel_text = rel_path.as_posix()
    for pattern in DEFAULT_EXCLUDE_PATTERNS:
        if any(part == pattern for part in rel_path.parts):
            return True
        if fnmatch(rel_text, pattern) or fnmatch(rel_path.name, pattern):
            return True
    return False


def stale_scan_path_is_launch_surface(path: Path, project_root: Path) -> bool:
    """Limit default stale scans to public positioning surfaces, not tests/code."""
    try:
        rel_path = path.relative_to(project_root)
    except ValueError:
        return False
    if len(rel_path.parts) == 1:
        return rel_path.name in STALE_SCAN_ROOT_FILES
    return rel_path.parts[0] == "docs" and path.suffix.lower() in {".md", ".mdx"}


def stale_line_is_negated_or_safe(line: str) -> bool:
    """Avoid reporting lines that intentionally reject the stale claim."""
    folded = line.casefold()
    safe_phrases = [
        "not a personal ai agent",
        "not a memory layer",
        "not a lora trainer",
        "lora is experimental",
        "lora/training is experimental",
        "not the core product path",
        "not the core launch path",
    ]
    return any(phrase in folded for phrase in safe_phrases)


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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    semantic: bool = typer.Option(False, "--semantic", help="Extract semantic review candidates"),
    review: bool = typer.Option(False, "--review", help="Write semantic candidates for review"),
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

    if semantic:
        if not review:
            console.print("[red]Semantic compile is review-gated. Pass --review.[/red]")
            raise typer.Exit(2)
        run_semantic_review_or_exit(project_root)


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
    else:
        console.print(f"Detail: {next_action['detail']}")
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
    next_action_command = next_action.get("command") or next_action["detail"]
    console.print(Panel.fit(
        f"Project: [bold]{payload['project_root']}[/bold]\n"
        f"Initialized: [bold]{state['initialized']}[/bold]\n"
        f"Compiled: [bold]{state['compiled']}[/bold]\n"
        f"Next action: [bold]{next_action['label']}[/bold]\n"
        f"Command: [bold]{next_action_command}[/bold]\n"
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
def wake(
    project: Path | None = typer.Argument(
        None,
        help="Optional project path to initialize, compile, verify, and write root WAKE.md",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Keep the compiled WAKE.md inside .morpheus/ instead of writing root WAKE.md",
    ),
    semantic: bool = typer.Option(False, "--semantic", help="Extract semantic review candidates"),
    review: bool = typer.Option(False, "--review", help="Write semantic candidates for review"),
):
    """Print WAKE.md, or run the one-command project wake flow."""
    if project is not None:
        original_cwd = Path.cwd()
        try:
            project_root = project.expanduser()
            if not project_root.is_absolute():
                project_root = original_cwd / project_root
            morpheus_dir, initialized = ensure_project_initialized(project_root)
            os.chdir(project_root)
            if initialized:
                console.print("[green]✓ Initialized .morpheus/[/green]")
            compile(verbose=False, semantic=semantic, review=review)
            verify(all=True)

            if private:
                wake_path = morpheus_dir / "WAKE.md"
                console.print(f"[green]✓ Private WAKE.md:[/green] {wake_path}")
            else:
                wake_path = copy_public_wake(project_root, morpheus_dir)
                console.print(f"[green]✓ Public WAKE.md:[/green] {wake_path}")

            console.print(Panel.fit(
                f"Agent handoff prompt:\n{wake_handoff_prompt()}",
                title="Morpheus Wake",
                border_style="green",
            ))
        except (OSError, ValueError) as exc:
            console.print(f"[red]Wake failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        finally:
            os.chdir(original_cwd)
        return

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
    console.out(content, end="")


@review_app.command("list")
def review_list(
    kind: str | None = typer.Option(None, "--kind", help="Filter by candidate kind"),
    label: str | None = typer.Option(None, "--label", help="Filter by candidate label"),
    source_backed: bool = typer.Option(False, "--source-backed", help="Show source-backed candidates only"),
    trainable: bool = typer.Option(False, "--trainable", help="Show accepted trainable candidates only"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """List semantic candidates waiting in the review store."""
    candidates = ReviewStore(Path.cwd()).load_candidates()
    if kind:
        candidates = [candidate for candidate in candidates if candidate.kind == kind]
    if label:
        candidates = [candidate for candidate in candidates if candidate.label == label]
    if source_backed:
        candidates = [candidate for candidate in candidates if candidate.label == "source_backed"]
    if trainable:
        candidates = [
            candidate
            for candidate in candidates
            if trainable_candidate(Path.cwd(), candidate)
        ]
    if json_output:
        console.out(json.dumps([candidate.model_dump(mode="json") for candidate in candidates], indent=2))
        return
    table = Table(title="Semantic Review Candidates")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Label", style="yellow")
    table.add_column("Source")
    table.add_column("Claim")
    for candidate in candidates:
        table.add_row(
            candidate.id,
            candidate.status,
            candidate.label,
            f"{candidate.source_path}:{candidate.line_start}",
            candidate.claim,
        )
    console.print(table)


@review_app.command("show")
def review_show(candidate_id: str = typer.Argument(..., help="Candidate id")):
    """Show one semantic candidate."""
    for candidate in ReviewStore(Path.cwd()).load_candidates():
        if candidate.id == candidate_id:
            console.print(Panel.fit(
                f"{candidate.claim}\n\n"
                f"Source: {candidate.source_path}:{candidate.line_start}-{candidate.line_end}\n"
                f"Kind: {candidate.kind}\n"
                f"Label: {candidate.label}\n"
                f"Status: {candidate.status}\n\n"
                f"{candidate.evidence_excerpt}",
                title=candidate.id,
                border_style="green",
            ))
            return
    console.print(f"[red]Candidate not found:[/red] {candidate_id}")
    raise typer.Exit(1)


def _read_candidate_id_file(path: Path) -> list[str]:
    reject_symlink_paths([path], "Review batch file")
    reject_symlink_components(path, "Review batch file")
    if not path.is_file():
        raise ValueError(f"candidate id file not found: {path}")
    candidate_ids = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not candidate_ids:
        raise ValueError(f"candidate id file is empty: {path}")
    return candidate_ids


@review_app.command("accept")
def review_accept(candidate_id: str = typer.Argument(..., help="Candidate id")):
    """Accept one semantic candidate."""
    try:
        candidate = ReviewStore(Path.cwd()).accept(candidate_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]✓ Accepted:[/green] {candidate.id}")


@review_app.command("reject")
def review_reject(
    candidate_id: str = typer.Argument(..., help="Candidate id"),
    reason: str = typer.Option(..., "--reason", help="Reason for rejection"),
):
    """Reject one semantic candidate."""
    try:
        candidate = ReviewStore(Path.cwd()).reject(candidate_id, reason=reason)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]✓ Rejected:[/green] {candidate.id}")


@review_app.command("accept-batch")
def review_accept_batch(
    ids_file: Path = typer.Option(..., "--file", help="Text file with one candidate id per line"),
):
    """Accept semantic candidates listed in a file."""
    try:
        candidate_ids = _read_candidate_id_file(ids_file)
        accepted = ReviewStore(Path.cwd()).accept_many(candidate_ids)
    except (KeyError, OSError, ValueError) as exc:
        console.print(f"[red]Batch accept failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]✓ Accepted batch:[/green] {len(accepted)} candidates")


@review_app.command("reject-batch")
def review_reject_batch(
    ids_file: Path = typer.Option(..., "--file", help="Text file with one candidate id per line"),
    reason: str = typer.Option(..., "--reason", help="Reason for rejection"),
):
    """Reject semantic candidates listed in a file."""
    try:
        candidate_ids = _read_candidate_id_file(ids_file)
        rejected = ReviewStore(Path.cwd()).reject_many(candidate_ids, reason=reason)
    except (KeyError, OSError, ValueError) as exc:
        console.print(f"[red]Batch reject failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]✓ Rejected batch:[/green] {len(rejected)} candidates")


@review_app.command("suggest-accept")
def review_suggest_accept(
    strict: bool = typer.Option(False, "--strict", help="Only suggest exact source-span low-risk candidates"),
):
    """Write suggested accept ids without changing candidate statuses."""
    if not strict:
        console.print("[red]Only --strict suggestions are supported.[/red]")
        raise typer.Exit(2)
    try:
        path = write_strict_accept_suggestions(Path.cwd())
        count = len(strict_accept_suggestions(Path.cwd()))
    except (OSError, ValueError) as exc:
        console.print(f"[red]Suggest accept failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]✓ Suggested accepts:[/green] {count} ids -> {path}")


@review_app.command("auto-accept")
def review_auto_accept(
    strict: bool = typer.Option(False, "--strict", help="Only accept strict machine-verifiable candidates"),
    lab_only: bool = typer.Option(False, "--lab-only", help="Required safety flag for autonomous lab acceptance"),
):
    """Accept strict machine-verifiable candidates only for autonomous lab use."""
    if not strict or not lab_only:
        console.print("[red]auto-accept requires --strict --lab-only.[/red]")
        raise typer.Exit(2)
    try:
        result = lab_auto_accept(Path.cwd())
    except (OSError, ValueError) as exc:
        console.print(f"[red]Auto-accept failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.out(json.dumps(result, indent=2, sort_keys=True))


@review_app.command("doctor")
def review_doctor():
    """Explain why candidates can or cannot be safely suggested."""
    try:
        result = write_review_doctor(Path.cwd())
    except (OSError, ValueError) as exc:
        console.print(f"[red]Review doctor failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]✓ Review doctor:[/green] {result['markdown_path']}")
    console.print(f"[green]✓ Review doctor JSON:[/green] {result['json_path']}")


@review_app.command("propose")
def review_propose(
    max_accepts: int = typer.Option(30, "--max", help="Maximum ACCEPT_SAFE ids to write"),
    threshold: float = typer.Option(0.80, "--threshold", help="Minimum confidence for ACCEPT_SAFE"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Write human review proposals without changing candidate statuses."""
    try:
        result = write_review_proposal(
            Path.cwd(),
            max_accepts=max_accepts,
            threshold=threshold,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]Review proposal failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        console.out(json.dumps({
            "counts": result["counts"],
            "proposed_accept_ids": result["proposed_accept_ids"],
            "proposed_reject_ids": result["proposed_reject_ids"],
            "paths": result["paths"],
        }, indent=2, sort_keys=True))
        return
    console.print(f"[green]✓ Proposed accepts:[/green] {len(result['proposed_accept_ids'])}")
    console.print(f"[green]✓ Proposal report:[/green] {result['paths']['report_md']}")


@review_app.command("accept-proposed")
def review_accept_proposed(
    max_accepts: int = typer.Option(30, "--max", help="Maximum ACCEPT_SAFE ids to accept"),
    threshold: float = typer.Option(0.80, "--threshold", help="Minimum confidence for ACCEPT_SAFE"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Accept freshly scored ACCEPT_SAFE proposal ids without applying active state."""
    try:
        result = accept_proposed_candidates(
            Path.cwd(),
            max_accepts=max_accepts,
            threshold=threshold,
        )
    except (KeyError, OSError, ValueError) as exc:
        console.print(f"[red]Accept proposed failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        console.out(json.dumps(result, indent=2, sort_keys=True))
        return
    console.print(f"[green]✓ Accepted proposed:[/green] {result['accepted_count']} candidates")
    console.print("[yellow]Active state not changed.[/yellow] Run `morpheus review apply` explicitly.")


@review_app.command("interactive")
def review_interactive(
    source_backed: bool = typer.Option(False, "--source-backed", help="Review source-backed candidates only"),
    proposed: bool = typer.Option(False, "--proposed", help="Review proposed ACCEPT_SAFE candidates first"),
):
    """Interactively write accept/reject id files without applying review changes."""
    if not sys.stdin.isatty():
        console.print("[red]Non-interactive terminal detected.[/red]")
        console.print("Run `morpheus review propose` or use an interactive TTY.")
        raise typer.Exit(2)
    store = ReviewStore(Path.cwd())
    candidates = store.load_candidates()
    proposal_by_id = {}
    if proposed:
        proposal = propose_review_candidates(Path.cwd())
        proposal_by_id = {item["id"]: item for item in proposal["proposals"]}
        proposed_ids = set(proposal["proposed_accept_ids"])
        candidates = [candidate for candidate in candidates if candidate.id in proposed_ids]
    if source_backed:
        candidates = [candidate for candidate in candidates if candidate.label == "source_backed"]

    accept_path = store.review_dir / "accept_ids.txt"
    reject_path = store.review_dir / "reject_ids.txt"
    store.ensure()
    accepted: list[str] = []
    rejected: list[str] = []
    for candidate in candidates:
        item = proposal_by_id.get(candidate.id, {})
        console.print(Panel.fit(
            f"Kind: {candidate.kind}\n"
            f"Claim: {candidate.claim}\n"
            f"Source: {candidate.source_path}:{candidate.line_start}-{candidate.line_end}\n"
            f"Evidence: {candidate.evidence_excerpt}\n"
            f"Confidence: {candidate.confidence}\n"
            f"Proposal: {item.get('category', 'unscored')}\n"
            f"Reason: {', '.join(item.get('reasons', [])) or 'none'}",
            title=candidate.id,
            border_style="cyan",
        ))
        while True:
            choice = input("[a]ccept / [r]eject / [s]kip / [v]iew source / [q]uit: ").strip().casefold()
            if choice == "a":
                accepted.append(candidate.id)
                break
            if choice == "r":
                rejected.append(candidate.id)
                break
            if choice == "s":
                break
            if choice == "v":
                console.print(candidate.evidence_excerpt)
                continue
            if choice == "q":
                candidates = []
                break
        if not candidates:
            break
    accept_path.write_text("\n".join(accepted) + ("\n" if accepted else ""))
    reject_path.write_text("\n".join(rejected) + ("\n" if rejected else ""))
    console.print(f"[green]✓ Wrote accepts:[/green] {accept_path}")
    console.print(f"[green]✓ Wrote rejects:[/green] {reject_path}")
    console.print("Next commands:")
    console.print("  morpheus review accept-batch --file .morpheus/review/accept_ids.txt")
    console.print("  morpheus review reject-batch --file .morpheus/review/reject_ids.txt")
    console.print("  morpheus review apply")
    console.print("  morpheus learn dataset . --from accepted --format instruction")


@review_app.command("export-pack")
def review_export_pack():
    """Write a human-reviewable semantic candidate pack."""
    try:
        path = export_review_pack(Path.cwd())
    except (OSError, ValueError) as exc:
        console.print(f"[red]Review pack export failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]✓ Review pack:[/green] {path}")


@review_app.command("diff")
def review_diff(json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON")):
    """Summarize pending review changes."""
    diff = ReviewStore(Path.cwd()).diff()
    if json_output:
        console.out(json.dumps(diff, indent=2))
        return
    table = Table(title="Semantic Review Diff")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")
    for key in ["pending", "accepted", "rejected"]:
        table.add_row(key, str(diff[key]))
    console.print(table)


@review_app.command("apply")
def review_apply():
    """Promote accepted semantic candidates into active state and sign a receipt."""
    try:
        result = apply_accepted_candidates(Path.cwd())
    except (OSError, ValueError) as exc:
        console.print(f"[red]Review apply failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        "[green]✓ Applied semantic review:[/green] "
        f"{result['accepted_applied']} accepted candidates"
    )
    console.print(f"[green]✓ Receipt:[/green] {result['receipt_id']}")


@learn_app.command("dataset")
def learn_dataset(
    project: Path = typer.Argument(Path("."), help="Project path"),
    source: str = typer.Option(
        "accepted",
        "--from",
        help="Dataset source: accepted or active-state",
    ),
    dataset_format: str = typer.Option(
        "instruction",
        "--format",
        help="Dataset format: instruction or sharegpt",
    ),
    include_corrections: bool = typer.Option(
        True,
        "--include-corrections/--no-include-corrections",
        help="Include accepted correction candidates as negative examples",
    ),
    include_refusals: bool = typer.Option(
        True,
        "--include-refusals/--no-include-refusals",
        help="Include unsupported-claim refusal eval items",
    ),
    output: Path | None = typer.Option(None, "--output", help="Optional selected dataset output path"),
):
    """Build a local dataset from accepted source-backed semantic candidates."""
    try:
        result = build_learning_dataset(
            project,
            dataset_format=dataset_format,
            source=source,
            include_corrections=include_corrections,
            include_refusals=include_refusals,
            output=output,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]Learning dataset failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    try:
        manifest = json.loads(Path(result["manifest_path"]).read_text())
    except (OSError, KeyError, json.JSONDecodeError):
        manifest = {}
    if (
        source == "accepted"
        and (
            int(manifest.get("trainable_candidate_count", 0)) < 20
            or int(result.get("examples_count", 0)) < 100
        )
    ):
        console.print("Training blocked: accepted candidates < 20 or examples < 100.")
        console.print("Run:")
        console.print("  morpheus review propose --max 30")
        console.print("  morpheus review accept-proposed --max 30")
        console.print("  morpheus review interactive --proposed")
    console.out(json.dumps(result, indent=2, sort_keys=True))


@learn_app.command("status")
def learn_status(
    project: Path = typer.Argument(Path("."), help="Project path"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Show local learning dataset status."""
    try:
        status = learning_status(project)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Learning status failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    if json_output:
        console.out(json.dumps(status, indent=2, sort_keys=True))
        return
    latest_lab = status.get("latest_lab")
    effective_dataset = status.get("effective_dataset")
    if not status["has_datasets"]:
        console.print("latest standalone dataset: none")
    else:
        manifest = status["latest_manifest"]
        console.print(
            "latest standalone dataset: "
            f"{manifest['dataset_id']} "
            f"examples={manifest['examples_count']} "
            f"skipped={manifest['skipped_count']}"
        )
    if effective_dataset:
        console.print(
            "effective dataset: "
            f"{effective_dataset.get('dataset_id')} "
            f"source={effective_dataset.get('source')} "
            f"examples={effective_dataset.get('examples_count')} "
            f"trainable={effective_dataset.get('trainable')}"
        )
    else:
        console.print("effective dataset: none")
    if latest_lab:
        console.print(
            "latest lab: "
            f"{latest_lab['lab_id']} "
            f"source={latest_lab.get('source')} "
            f"accepted={latest_lab.get('strict_accepted_candidates')} "
            f"examples={latest_lab.get('examples_count')} "
            f"verdict={latest_lab.get('verdict')} "
            f"production_ready={latest_lab.get('production_ready')}"
        )
    else:
        console.print("latest lab: none")
    active = status.get("active_adapter")
    if active:
        console.print(
            "active adapter: "
            f"{active['adapter_id']} "
            f"score={active.get('eval_score')} "
            f"created_at={active.get('created_at')}"
        )
    else:
        console.print("active adapter: none")


@learn_app.command("quality")
def learn_quality(
    project: Path = typer.Argument(Path("."), help="Project path"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Write a dataset quality report for review, routing, and train gates."""
    try:
        result = write_quality_report(project)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Learning quality failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    if json_output:
        console.out(json.dumps({
            "paths": {
                "json_path": result["json_path"],
                "markdown_path": result["markdown_path"],
            },
            **result["report"],
        }, indent=2, sort_keys=True))
        return
    review = result["report"]["review"]
    console.print(f"quality report: {result['markdown_path']}")
    console.print(
        "review: "
        f"candidates={review['candidates_total']} "
        f"accepted={review['accepted']} "
        f"pending={review['pending']} "
        f"rejected={review['rejected']}"
    )
    routing = result["report"]["routing"]
    console.print(f"routing_policy={routing['policy_version']}")
    console.print(f"audited_decisions={len(routing['decisions'])}")
    console.print(f"train_allowed={result['report']['train_allowed']}")
    if result["report"]["train_blockers"]:
        console.print("blockers: " + ", ".join(result["report"]["train_blockers"]))


@learn_app.command("benchmark")
def learn_benchmark(
    project: Path = typer.Argument(Path("."), help="Project path"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Write readiness artifacts only"),
    backend: str = typer.Option("mlx", "--backend", help="Target benchmark backend"),
    max_iters: int = typer.Option(50, "--max-iters", help="Suggested LoRA smoke iteration budget"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Write a benchmark readiness report without training or activating adapters."""
    try:
        result = write_benchmark_report(
            project,
            dry_run=dry_run,
            backend=backend,
            max_iters=max_iters,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Learning benchmark failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    payload = {
        "paths": {
            "benchmark_config_path": result["benchmark_config_path"],
            "benchmark_report_path": result["benchmark_report_path"],
            "benchmark_report_md_path": result["benchmark_report_md_path"],
        },
        **result,
    }
    if json_output:
        console.out(json.dumps(payload, indent=2, sort_keys=True))
        return
    console.print(f"benchmark report: {result['benchmark_report_md_path']}")
    console.print(f"benchmark_allowed={result['benchmark_allowed']}")
    console.print(f"activation_ready={result['activation_ready']}")
    console.print(f"critical_regressions={len(result['critical_regressions'])}")
    if result["benchmark_blockers"]:
        console.print("blockers: " + ", ".join(result["benchmark_blockers"]))
    console.print(f"next: {result['next_command']}")


@learn_app.command("lab")
def learn_lab(
    project: Path = typer.Argument(Path("."), help="Project path"),
    backend: str = typer.Option("fake", "--backend", help="Lab backend: fake or mlx"),
    model: str = typer.Option(
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "--model",
        help="Model id for MLX training",
    ),
    no_train: bool = typer.Option(False, "--no-train", help="Build dataset/eval artifacts without training"),
    fixture_only: bool = typer.Option(False, "--fixture-only", help="Use the autonomous benchmark fixture"),
    dogfood: bool = typer.Option(False, "--dogfood", help="Require dogfood source mode"),
    max_iters: int = typer.Option(
        DEFAULT_LAB_MAX_ITERS,
        "--max-iters",
        help="Maximum LoRA training iterations for autonomous lab runs",
    ),
    eval_limit: int = typer.Option(
        DEFAULT_LAB_EVAL_LIMIT,
        "--eval-limit",
        help="Maximum non-critical eval items for MLX lab eval; 0 means full eval; critical safety items are always included",
    ),
    repeat: int = typer.Option(
        1,
        "--repeat",
        help="Run repeated lab experiments and aggregate a stability report",
    ),
):
    """Run an autonomous source-grounded learning lab without activating adapters."""
    try:
        if repeat > 1:
            result = run_autonomous_lab_stability(
                project,
                repeat=repeat,
                backend=backend,
                model=model,
                no_train=no_train,
                fixture_only=fixture_only,
                dogfood=dogfood,
                max_iters=max_iters,
                eval_limit=eval_limit,
            )
        else:
            result = run_autonomous_lab(
                project,
                backend=backend,
                model=model,
                no_train=no_train,
                fixture_only=fixture_only,
                dogfood=dogfood,
                max_iters=max_iters,
                eval_limit=eval_limit,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Learning lab failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    console.out(json.dumps(result, indent=2, sort_keys=True))


@learn_app.command("train")
def learn_train(
    project: Path = typer.Argument(Path("."), help="Project path"),
    backend: str = typer.Option("llamafactory", "--backend", help="Training backend: llamafactory or peft"),
    method: str = typer.Option("qlora", "--method", help="Training method: qlora or lora"),
    base_model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", "--base-model", help="Base model id"),
    rank: int = typer.Option(16, "--rank", help="LoRA rank"),
    alpha: int = typer.Option(32, "--alpha", help="LoRA alpha"),
    dropout: float = typer.Option(0.05, "--dropout", help="LoRA dropout"),
    epochs: int = typer.Option(1, "--epochs", help="Training epochs"),
    learning_rate: str = typer.Option("2e-4", "--learning-rate", help="Learning rate"),
    max_seq_length: int = typer.Option(4096, "--max-seq-length", help="Maximum sequence length"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Generate artifacts without training"),
    execute: bool = typer.Option(False, "--execute", help="Actually run training after planning"),
    confirm_execute: bool = typer.Option(
        False,
        "--yes-i-know-this-will-train",
        help="Required with --execute",
    ),
):
    """Plan a LoRA/QLoRA training run from the latest reviewed dataset."""
    try:
        result = plan_training_run(
            project,
            backend=backend,
            method=method,
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            epochs=epochs,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            dry_run=dry_run and not execute,
            execute=execute,
            confirm_execute=confirm_execute,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Learning train failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    for warning in result["warnings"]:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    console.out(json.dumps(result, indent=2, sort_keys=True))


@learn_app.command("eval")
def learn_eval(
    project: Path = typer.Argument(Path("."), help="Project path"),
    base_only: bool = typer.Option(False, "--base-only", help="Evaluate the fake base model only"),
    adapter_id: str | None = typer.Option(None, "--adapter", help="Adapter id to evaluate"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Use deterministic fake inference"),
):
    """Evaluate a base model or planned adapter against the latest eval seed."""
    try:
        result = run_learning_eval(
            project,
            adapter_id=adapter_id,
            base_only=base_only,
            dry_run=dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Learning eval failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    console.out(json.dumps(result, indent=2, sort_keys=True))


@learn_app.command("list-adapters")
def learn_list_adapters(
    project: Path = typer.Argument(Path("."), help="Project path"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """List planned, evaluated, and active learning adapters."""
    try:
        adapters = list_adapters(project)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Learning adapters failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    if json_output:
        console.out(json.dumps(adapters, indent=2, sort_keys=True))
        return
    if not adapters:
        console.print("No learning adapters found.")
        return
    table = Table(title="Learning Adapters")
    table.add_column("Adapter ID", style="cyan")
    table.add_column("Status")
    table.add_column("Backend")
    table.add_column("Method")
    table.add_column("Eval")
    for adapter in adapters:
        table.add_row(
            adapter["adapter_id"],
            str(adapter.get("status")),
            str(adapter.get("backend")),
            str(adapter.get("method")),
            str(adapter.get("eval_score")),
        )
    console.print(table)


@learn_app.command("activate")
def learn_activate(
    adapter_id: str = typer.Argument(..., help="Adapter id to activate"),
    project: Path = typer.Option(Path("."), "--project", help="Project path"),
    force: bool = typer.Option(False, "--force", help="Bypass eval gate"),
    confirm_force: bool = typer.Option(
        False,
        "--yes-i-know-this-can-degrade",
        help="Required with --force",
    ),
):
    """Activate an adapter only after passing eval, unless explicitly forced."""
    try:
        result = activate_adapter(
            project,
            adapter_id,
            force=force,
            confirm_force=confirm_force,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Learning activate failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    console.out(json.dumps(result, indent=2, sort_keys=True))


@learn_app.command("rollback")
def learn_rollback(project: Path = typer.Option(Path("."), "--project", help="Project path")):
    """Rollback to the previously active adapter."""
    try:
        result = rollback_adapter(project)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Learning rollback failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    console.out(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def stale(
    project: Path = typer.Argument(Path("."), help="Project path to scan"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Report stale launch-positioning claims that conflict with WAKE.md framing."""
    original_cwd = Path.cwd()
    project_root = project.expanduser()
    if not project_root.is_absolute():
        project_root = original_cwd / project_root

    try:
        findings = find_stale_positioning_claims(project_root)
    except ValueError as exc:
        console.print(f"[red]Stale scan failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    payload = {
        "project_root": str(project_root.resolve()),
        "findings": findings,
    }
    if json_output:
        console.out(json.dumps(payload, indent=2))
        return

    if not findings:
        console.print("[green]No stale launch-positioning claims found.[/green]")
        return

    console.print("Outdated claims:")
    for index, finding in enumerate(findings, 1):
        console.print(
            f"{index}. {finding['path']}:{finding['line']} says "
            f"\"{finding['excerpt']}\""
        )
        console.print(f"   Suggested replacement: {finding['suggested_replacement']}")


@app.command()
def check(
    ctx: typer.Context,
    input_path: Path | None = typer.Option(None, "--input", help="File containing agent text"),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        help="Project root. Defaults to upward discovery from cwd.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    output_format: str = typer.Option(
        "summary",
        "--format",
        help="Output format: summary, annotated, or json",
    ),
    semantic: bool = typer.Option(False, "--semantic", help="Use explicit semantic provider"),
    local: bool = typer.Option(False, "--local", help="Force local checks"),
    fail_on_unknown: bool = typer.Option(False, "--fail-on-unknown", help="Exit 1 on unknown claims"),
    allow_stale_state: bool = typer.Option(
        False,
        "--allow-stale-state",
        help="Allow stale source state in CI mode",
    ),
    create_corrections: bool = typer.Option(
        False,
        "--create-training-corrections",
        help="Create pending review candidates for stale/incorrect claims",
    ),
    strict_freshness: bool = typer.Option(
        False,
        "--strict-freshness",
        help="Exit 2 on unknown freshness in CI mode",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show check configuration without input"),
    offline: bool = typer.Option(False, "--offline", help="Require local/offline operation"),
):
    """Verify agent text against local source-grounded Morpheus state."""
    root = project_root.expanduser() if project_root else discover_project_root(Path.cwd())
    if dry_run:
        payload = {
            "project_root": str(root.resolve()),
            "modes_used": ["local"],
            "semantic_requested": semantic,
            "local_forced": local or offline or not semantic,
            "offline": offline,
            "provider": None,
            "api_keys_printed": False,
        }
        console.out(json.dumps(payload, indent=2))
        return

    if semantic:
        console.print(
            "[red]Semantic check provider is not available in v0.2.0b1.[/red] "
            "Use local default or --local."
        )
        raise typer.Exit(2)

    try:
        input_text = read_check_input(input_path)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Check input failed:[/red] {exc}")
        raise typer.Exit(2) from exc

    if not input_text.strip():
        console.print(ctx.get_help())
        console.print("")
        console.print("Examples:")
        console.print("  morpheus check --input agent-output.md")
        console.print("  gh pr view 42 --json body -q .body | morpheus check")
        raise typer.Exit(2)

    try:
        result = check_text(input_text, project_root=root, fail_on_unknown=fail_on_unknown)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Check failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    if create_corrections:
        try:
            corrections = create_training_corrections(root, result)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            console.print(f"[red]Training correction creation failed:[/red] {exc}")
            raise typer.Exit(2) from exc
        result["training_corrections_created"] = len(corrections)
        result["training_correction_ids"] = [candidate.id for candidate in corrections]

    effective_format = "json" if json_output else output_format
    if effective_format not in {"summary", "annotated", "json"}:
        console.print("[red]Invalid format:[/red] use summary, annotated, or json")
        raise typer.Exit(2)
    if effective_format == "json":
        console.out(json.dumps(result, indent=2, default=str))
    elif effective_format == "annotated":
        console.print(render_check_annotated(result))
    else:
        console.print(render_check_summary(result))

    exit_code = check_exit_code(
        result,
        ci_mode=ci_mode_from_env(),
        allow_stale_state=allow_stale_state,
        strict_freshness=strict_freshness,
        fail_on_unknown=fail_on_unknown,
    )
    if exit_code:
        raise typer.Exit(exit_code)


def read_check_input(input_path: Path | None) -> str:
    if input_path is not None:
        reject_symlink_components(input_path, "Check input")
        reject_symlink_paths([input_path], "Check input")
        if not input_path.is_file():
            raise ValueError(f"input file not found: {input_path}")
        return input_path.read_text()
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read()


@app.command()
def integrate(
    service: str | None = typer.Argument(None, help="Service: gmail, calendar, github, slack, linear"),
    list_services: bool = typer.Option(False, "--list", help="List available integrations"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Connect external integrations.
    
    Services:
      gmail     - Gmail API via OAuth2
      calendar  - Google Calendar API via OAuth2  
      github    - GitHub API via Personal Access Token
      slack     - Slack cache export + optional token
      linear    - Linear cache export + optional token
    """
    if list_services:
        manifest = integration_manifest()

        if json_output:
            console.out(json.dumps(manifest, indent=2))
            return

        table = Table(title="Available Integrations")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Auth", style="yellow")
        for item in manifest["services"]:
            table.add_row(
                item["id"],
                rich_integration_status(item["status"]),
                item["auth"],
            )
        console.print(table)
        return

    if service is None:
        console.print("[red]Service required. Use --list to show available integrations.[/red]")
        raise typer.Exit(1)
    if service not in {"gmail", "calendar", "github", "slack", "linear"}:
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
    elif service in {"slack", "linear"}:
        service_label = service.title()
        token_path = Path.home() / ".morpheus" / f"{service}_token.txt"
        cache_path = Path.home() / ".morpheus" / f"{service}_cache.json"
        path_error = integration_token_path_error(token_path, service_label)
        if path_error is None:
            path_error = integration_cache_path_error(cache_path, service_label)
        if path_error:
            console.print(f"[red]{path_error}[/red]")
            raise typer.Exit(1)
        try:
            token_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            console.print(f"[red]{service_label} token directory cannot be created:[/red] {exc}")
            raise typer.Exit(1) from exc
        console.print(f"[green]{service_label} cache supported[/green]")
        console.print(f"Cache file name: {cache_path.name}")
        console.print(f"Cache file: {cache_path}")
        console.print(f"Optional token file name: {token_path.name}")
        console.print(f"Optional token file: {token_path}")
        console.print("Drop exported JSON rows there, then compile or call the integration directly.")
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
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Generate script without running"),
    execute: bool = typer.Option(False, "--execute", help="Run legacy raw-dataset training"),
    confirm_legacy_raw_training: bool = typer.Option(
        False,
        "--yes-i-know-this-is-legacy-raw-training",
        help="Required with --execute for the deprecated root train command",
    ),
):
    """Deprecated legacy QLoRA fine-tuning on a raw dataset.
    
    Prefer `morpheus learn train . --dry-run`, which uses reviewed source-backed
    datasets. This command is kept for old dry-run script generation only.
    """
    from morpheus.training.train import train as run_train

    resolved_dry_run = dry_run and not execute
    if execute and not confirm_legacy_raw_training:
        console.print(
            "[red]legacy raw-dataset training is blocked by default.[/red]\n"
            "Use `morpheus learn train . --dry-run` for reviewed source-backed "
            "datasets. To run this deprecated command anyway, pass "
            "`--execute --yes-i-know-this-is-legacy-raw-training`."
        )
        raise typer.Exit(2)

    if not resolved_dry_run:
        ok, missing = check_dependencies()
        if not ok:
            console.print(f"[red]Missing: {', '.join(missing)}[/red]")
            console.print("[yellow]Install: pip install llamafactory[/yellow]")
            raise typer.Exit(1)
    else:
        console.print(
            "[yellow]warning:[/yellow] root `morpheus train` is deprecated. "
            "Use `morpheus learn train . --dry-run` for reviewed source-backed datasets."
        )
    
    run_train(
        base_model=base_model,
        dataset=Path(dataset),
        output_dir=Path(output_dir),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        epochs=epochs,
        dry_run=resolved_dry_run,
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


@app.command("model-smoke")
def model_smoke_command(
    base_model: str = typer.Option(
        DEFAULT_MODEL_SMOKE_MODEL,
        "--base-model",
        help="Ollama model to smoke-test",
    ),
    prompt: str = typer.Option(
        DEFAULT_MODEL_SMOKE_PROMPT,
        "--prompt",
        help="Prompt to send to the model",
    ),
):
    """Run a direct Ollama smoke test through Morpheus."""
    from morpheus.training.eval import query_model

    base_model = base_model.strip() or DEFAULT_MODEL_SMOKE_MODEL
    prompt = prompt.strip() or DEFAULT_MODEL_SMOKE_PROMPT

    answer = query_model(prompt, base_model=base_model)
    if answer.startswith("Error:"):
        console.print(f"[red]{answer}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"Model: [bold]{base_model}[/bold]\nAnswer:\n{answer}",
            title="Morpheus Model Smoke",
            border_style="green",
        )
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
    previous_ui_env = {
        "MORPHEUS_UI_HOST": os.environ.get("MORPHEUS_UI_HOST"),
        "MORPHEUS_UI_PORT": os.environ.get("MORPHEUS_UI_PORT"),
    }
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
        os.environ["MORPHEUS_UI_PORT"] = str(ui_port)
        if ui_host is None:
            os.environ.pop("MORPHEUS_UI_HOST", None)
        else:
            os.environ["MORPHEUS_UI_HOST"] = bound_ui_host

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
        for name, previous_value in previous_ui_env.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value
        if static_server is not None:
            static_server.shutdown()
            static_server.server_close()


@app.command()
def version(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Show morpheus version."""
    from morpheus import __version__

    if json_output:
        console.print(json.dumps({"service": "morpheus", "version": __version__}))
        return

    console.print(f"Morpheus AI v{__version__}")


def main() -> None:
    """Run the Morpheus CLI application."""
    app()


if __name__ == "__main__":
    main()
