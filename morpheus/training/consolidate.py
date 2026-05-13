"""
Session Consolidation — Convert OpenClaw sessions to training dataset.

Reads session JSONL files from OpenClaw agents/main/sessions/,
extracts meaningful conversations, and produces a ShareGPT-format dataset
for LoRA fine-tuning.
"""
import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Annotated
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

NOISE_MARKERS = [
    "HEARTBEAT_OK",
    "BEGIN_OPENCLAW",
    "END_OPENCLAW",
    "sourceSession=",
    "sourceChannel=",
    "untrusted metadata",
    "Session Context",
    "Session Info",
    "Exec completed",
    "ToolCall",
]

TOOL_CONTENT_TYPES = {
    "tool_use",
    "tool_result",
    "function_call",
    "function_result",
    "server_tool_call",
    "computer_call",
    "reasoning",
}

LOW_VALUE_USER_MESSAGES = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "continue",
    "go on",
    "yes",
    "no",
    "done",
}


@dataclass
class ConsolidationStats:
    """Operational counters for session consolidation."""

    files_found: int = 0
    files_processed: int = 0
    files_skipped_old: int = 0
    files_unreadable: int = 0
    malformed_lines: int = 0
    messages_seen: int = 0
    messages_kept: int = 0
    messages_filtered: int = 0
    pairs_extracted: int = 0
    pairs_unique: int = 0
    pairs_duplicate: int = 0


def normalize_whitespace(content: str) -> str:
    """Collapse noisy session whitespace without changing the text meaning."""
    return re.sub(r"\s+", " ", content).strip()


def truncate_text(content: str, max_chars: int) -> str:
    """Truncate at a readable boundary when possible."""
    if len(content) <= max_chars:
        return content

    clipped = content[:max_chars].rsplit(" ", 1)[0].rstrip()
    return clipped if clipped else content[:max_chars].rstrip()


def is_useful_message(content: str, role: str | None = None) -> bool:
    """Check if message has useful content."""
    content = normalize_whitespace(content)
    if not content:
        return False

    lowered = content.lower()
    first_200 = content[:200].lower()

    if lowered in LOW_VALUE_USER_MESSAGES:
        return False

    # Filter system prompts
    for sp in SYSTEM_PROMPTS:
        if sp.lower() in first_200:
            return False

    for marker in NOISE_MARKERS:
        marker_lower = marker.lower()
        if marker_lower == "heartbeat_ok":
            if lowered.strip() == marker_lower or lowered.startswith(f"{marker_lower} "):
                return False
        elif marker_lower in first_200:
            return False

    if lowered.startswith((
        "<environment_context>",
        "<system",
        "<developer",
        "<tool",
        "<function",
    )):
        return False

    if content.startswith(("{", "[")) and any(
        key in first_200
        for key in (
            "tool_uses",
            "tool_calls",
            "recipient_name",
            "function_call",
            "tool_result",
            "cmd",
            "session_id",
        )
    ):
        return False

    if any(
        marker in content
        for marker in (
            "Chunk ID:",
            "Wall time:",
            "Process exited with code",
            "Original token count:",
            "Exit code:",
        )
    ):
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

    min_len = 8 if role == "user" else 30
    if len(content) < min_len:
        return False

    if role == "user" and len(re.findall(r"[A-Za-zА-Яа-я0-9]{2,}", content)) < 2:
        return False

    return True


def extract_text_from_content(content_blocks: list) -> str:
    """Extract text from OpenClaw content blocks format."""
    if isinstance(content_blocks, str):
        return normalize_whitespace(content_blocks)
    
    texts = []
    for block in content_blocks:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type in TOOL_CONTENT_TYPES:
                continue
            if block_type in ("text", "input_text", "output_text"):
                texts.append(block.get("text", ""))
            elif "text" in block and not any(key in block for key in ("name", "input", "call_id")):
                texts.append(block.get("text", ""))
            elif block.get("type") == "image":
                texts.append("[image]")
        elif isinstance(block, str):
            texts.append(block)
    return normalize_whitespace(" ".join(texts))


def parse_session_file(session_path: Path, stats: ConsolidationStats | None = None) -> list[dict]:
    """Parse OpenClaw session JSONL into messages format.
    
    OpenClaw session format:
    - type: "message"
    - message.role: "user" | "assistant"
    - message.content: list of content blocks [{type: "text", text: "..."}]
    """
    messages = []
    
    try:
        session_file = session_path.open(encoding="utf-8", errors="replace")
    except OSError:
        if stats:
            stats.files_unreadable += 1
        return messages

    with session_file:
        for line in session_file:
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

                if stats:
                    stats.messages_seen += 1

                content = extract_text_from_content(content_raw)

                if not is_useful_message(content, role):
                    if stats:
                        stats.messages_filtered += 1
                    continue

                messages.append({
                    "role": role,
                    "content": truncate_text(content, 2000)
                })
                if stats:
                    stats.messages_kept += 1
            except json.JSONDecodeError:
                if stats:
                    stats.malformed_lines += 1
                continue
    
    return messages


def is_high_quality_pair(instruction: str, output: str) -> bool:
    """Check whether a user/assistant turn is useful enough for training."""
    instruction = normalize_whitespace(instruction)
    output = normalize_whitespace(output)
    if not instruction or not output:
        return False

    if not is_useful_message(instruction, "user") or not is_useful_message(output, "assistant"):
        return False

    instruction_words = re.findall(r"[A-Za-zА-Яа-я0-9]{2,}", instruction)
    output_words = re.findall(r"[A-Za-zА-Яа-я0-9]{2,}", output)
    if len(instruction_words) < 2 or len(output_words) < 8:
        return False

    low_signal_outputs = (
        "i'll get started",
        "i will get started",
        "i'm going to",
        "i am going to",
        "working on it",
        "let me check",
    )
    if output.lower() in low_signal_outputs:
        return False

    has_marker = any(
        marker.lower() in instruction.lower() or marker.lower() in output.lower()
        for marker in IMPORTANT_MARKERS
    )
    has_substantial_content = len(output) >= 80
    return has_marker or has_substantial_content


def messages_to_qa_pairs(messages: list[dict]) -> list[dict]:
    """Convert conversation messages to instruction-tuning Q&A format."""
    pairs = []

    # Group only adjacent user -> assistant turns. This avoids joining a user
    # prompt with an unrelated later assistant response after another user turn.
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user":
            user_msg = normalize_whitespace(messages[i]["content"])
            assistant_parts = []
            j = i + 1

            while j < len(messages) and messages[j]["role"] == "assistant":
                assistant_parts.append(messages[j]["content"])
                j += 1

            assistant_msg = normalize_whitespace("\n\n".join(assistant_parts))

            if is_high_quality_pair(user_msg, assistant_msg):
                pairs.append({
                    "instruction": truncate_text(user_msg, 500),
                    "input": "",
                    "output": truncate_text(assistant_msg, 1000)
                })

            i = max(j, i + 1)
        else:
            i += 1
    
    return pairs


def consolidate_sessions(
    sessions_dir: Annotated[Path, typer.Option(
        help="OpenClaw sessions directory"
    )] = Path.home() / ".openclaw/agents/main/sessions",
    output_path: Annotated[Path, typer.Option(
        help="Output dataset file"
    )] = Path("dataset.jsonl"),
    days: Annotated[int, typer.Option(help="Process sessions from last N days")] = 7,
    min_pairs: Annotated[int, typer.Option(help="Minimum Q&A pairs to consider useful")] = 10,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show details")] = False,
):
    """Find sessions from last N days and create training dataset."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    all_pairs = []
    stats = ConsolidationStats()

    if not sessions_dir.exists():
        console.print(f"[red]Sessions directory not found: {sessions_dir}[/red]")
        raise typer.Exit(1)

    session_files = list(sessions_dir.glob("*.jsonl"))
    stats.files_found = len(session_files)

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
            except OSError:
                stats.files_unreadable += 1
                progress.update(task, advance=1)
                continue

            if file_time < cutoff:
                stats.files_skipped_old += 1
                progress.update(task, advance=1)
                continue

            messages = parse_session_file(session_path, stats)

            if len(messages) >= 2:
                pairs = messages_to_qa_pairs(messages)
                all_pairs.extend(pairs)
                stats.pairs_extracted += len(pairs)
                stats.files_processed += 1

            progress.update(task, advance=1)

    # Remove duplicates by hashing instruction
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        h = hashlib.md5(pair["instruction"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_pairs.append(pair)
        else:
            stats.pairs_duplicate += 1
    stats.pairs_unique = len(unique_pairs)

    if verbose:
        console.print(f"\n[cyan]Statistics:[/cyan]")
        console.print(f"  Files found: {stats.files_found}")
        console.print(f"  Files processed: {stats.files_processed}")
        console.print(f"  Files skipped by age: {stats.files_skipped_old}")
        console.print(f"  Files unreadable: {stats.files_unreadable}")
        console.print(f"  Malformed JSONL lines: {stats.malformed_lines}")
        console.print(f"  Messages kept/seen: {stats.messages_kept}/{stats.messages_seen}")
        console.print(f"  Messages filtered: {stats.messages_filtered}")
        console.print(f"  Total Q&A pairs: {len(all_pairs)}")
        console.print(f"  Unique Q&A pairs: {stats.pairs_unique}")
        console.print(f"  Duplicate Q&A pairs: {stats.pairs_duplicate}")

    if len(unique_pairs) < min_pairs:
        console.print(f"[yellow]Not enough unique Q&A pairs ({len(unique_pairs)} < {min_pairs})[/yellow]")
        console.print("[yellow]Try increasing --days or check session directory[/yellow]")
        raise typer.Exit(1)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    console.print(f"\n[green]✓ Dataset created:[/green] {len(unique_pairs)} Q&A pairs")
    console.print(f"[green]✓ Saved to:[/green] {output_path}")
    return stats


def main():
    typer.run(consolidate_sessions)


if __name__ == "__main__":
    main()
