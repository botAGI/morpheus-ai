"""
Session Consolidation — Convert OpenClaw sessions to training dataset.

Reads session JSONL files from OpenClaw agents/main/sessions/,
extracts meaningful conversations, and produces a ShareGPT-format dataset
for LoRA fine-tuning.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Iterator
import typer
from rich.console import Console
from rich.progress import Progress

console = Console()

# Markers that indicate important content
IMPORTANT_MARKERS = [
    "TODO:", "FIXME:", "DECISION:", "NOTE:", "HACK:", "XXX:",
    "сделано:", "создано:", "решено:", "результат:", "выполнено:",
    "completed", "created", "decided", "implemented", "fixed",
    "saved to", "generated", "wrote", "updated"
]

# System prompt to filter out
SYSTEM_PROMPTS = [
    "Read HEARTBEAT.md",
    "You are a personal assistant",
    "You are an AI assistant",
    "SOUL.md",
    "You are Gbot",
    "Subagent Context",
    "Inter-session message",
    "sourceSession=",
    "<<<BEGIN_OPENCLAW",
    "[Subagent Context]",
    "You are running as a subagent",
    "Exec completed",
    "sourceChannel=",
    "Session Context",
    "Session Info",
    "untrusted metadata",
    "timestamp"
]


def is_useful_message(content: str) -> bool:
    """Check if message has useful content."""
    if not content or len(content) < 50:
        return False
    
    first_200 = content[:200].lower()
    
    # Filter system prompts
    for sp in SYSTEM_PROMPTS:
        if sp.lower() in first_200:
            return False
    
    # Skip very short messages
    if len(content) < 50:
        return False
    
    # Skip messages that look like system/infrastructure
    skip_patterns = [
        "metadata", "timestamp", "untrusted", "session_id",
        "chat_id", "message_id", "sender", "inter-session",
        "begin_openclaw", "end_openclaw"
    ]
    for pat in skip_patterns:
        if pat in first_200:
            return False
    
    return True


def extract_text_from_content(content_blocks: list) -> str:
    """Extract text from OpenClaw content blocks format."""
    if isinstance(content_blocks, str):
        return content_blocks
    
    texts = []
    for block in content_blocks:
        if isinstance(block, dict):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "image":
                texts.append("[image]")
        elif isinstance(block, str):
            texts.append(block)
    return " ".join(texts)


def parse_session_file(session_path: Path) -> list[dict]:
    """Parse OpenClaw session JSONL into messages format.
    
    OpenClaw session format:
    - type: "message"
    - message.role: "user" | "assistant"
    - message.content: list of content blocks [{type: "text", text: "..."}]
    """
    messages = []
    
    try:
        for line in session_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") != "message":
                    continue
                
                msg = entry.get("message", {})
                role = msg.get("role", "")
                content_raw = msg.get("content", [])
                
                if role not in ("user", "assistant"):
                    continue
                
                content = extract_text_from_content(content_raw)
                
                if not is_useful_message(content):
                    continue
                
                messages.append({
                    "role": role,
                    "content": content[:2000]  # Truncate long messages
                })
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    return messages


def messages_to_qa_pairs(messages: list[dict]) -> list[dict]:
    """Convert conversation messages to instruction-tuning Q&A format."""
    pairs = []
    
    # Group into conversations: user question → assistant answer
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user":
            user_msg = messages[i]["content"]
            assistant_msg = ""
            
            # Find next assistant response
            for j in range(i + 1, len(messages)):
                if messages[j]["role"] == "assistant":
                    assistant_msg = messages[j]["content"]
                    break
            
            if assistant_msg and len(user_msg) > 10:
                # Check if contains useful info
                has_marker = any(m.lower() in user_msg.lower() or m.lower() in assistant_msg.lower() 
                               for m in IMPORTANT_MARKERS)
                has_substantial_content = len(assistant_msg) > 50
                
                if has_marker or has_substantial_content:
                    pairs.append({
                        "instruction": user_msg[:500],
                        "input": "",
                        "output": assistant_msg[:1000]
                    })
            
            i += 1
        else:
            i += 1
    
    return pairs


def consolidate_sessions(
    sessions_dir: Path = typer.Option(
        Path.home() / ".openclaw/agents/main/sessions",
        help="OpenClaw sessions directory"
    ),
    output_path: Path = typer.Option(
        Path("dataset.jsonl"),
        help="Output dataset file"
    ),
    days: int = typer.Option(7, help="Process sessions from last N days"),
    min_pairs: int = typer.Option(10, help="Minimum Q&A pairs to consider useful"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show details")
):
    """Find sessions from last N days and create training dataset."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    all_pairs = []
    processed_files = 0
    
    if not sessions_dir.exists():
        console.print(f"[red]Sessions directory not found: {sessions_dir}[/red]")
        raise typer.Exit(1)
    
    session_files = list(sessions_dir.glob("*.jsonl"))
    
    if not session_files:
        console.print(f"[yellow]No session files found in {sessions_dir}[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Found {len(session_files)} session files[/blue]")
    console.print(f"[blue]Processing sessions from last {days} day(s)...[/blue]\n")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing sessions...", total=len(session_files))
        
        for session_path in session_files:
            # Check file modification time
            try:
                file_time = datetime.fromtimestamp(session_path.stat().st_mtime, tz=timezone.utc)
                if file_time < cutoff:
                    progress.update(task, advance=1)
                    continue
            except OSError:
                pass
            
            messages = parse_session_file(session_path)
            
            if len(messages) >= 2:
                pairs = messages_to_qa_pairs(messages)
                all_pairs.extend(pairs)
                processed_files += 1
            
            progress.update(task, advance=1)
    
    if verbose:
        console.print(f"\n[cyan]Statistics:[/cyan]")
        console.print(f"  Files processed: {processed_files}")
        console.print(f"  Total Q&A pairs: {len(all_pairs)}")
    
    if len(all_pairs) < min_pairs:
        console.print(f"[yellow]Not enough Q&A pairs ({len(all_pairs)} < {min_pairs})[/yellow]")
        console.print("[yellow]Try increasing --days or check session directory[/yellow]")
        raise typer.Exit(1)
    
    # Remove duplicates by hashing instruction
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        h = hashlib.md5(pair["instruction"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_pairs.append(pair)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    console.print(f"\n[green]✓ Dataset created:[/green] {len(unique_pairs)} Q&A pairs")
    console.print(f"[green]✓ Saved to:[/green] {output_path}")


def main():
    typer.run(consolidate_sessions)


if __name__ == "__main__":
    main()
