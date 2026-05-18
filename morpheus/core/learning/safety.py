"""Safety filters for reviewed learning datasets."""
from fnmatch import fnmatch
import re
from pathlib import Path

from morpheus.core.compiler import DEFAULT_EXCLUDE_PATTERNS
from morpheus.core.semantic.scanner import SECRET_PATTERNS


SECRET_REGEXES = [
    re.compile(
        r"(?i)\b(api[_ -]?key|secret[_ -]?key|oauth|token|cookie|password)\b"
        r"\s*(?:is|=|:)\s*[\"']?[^\"'\s]+"
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b[A-Z0-9_]*(SECRET|TOKEN|PASSWORD|API_KEY)[A-Z0-9_]{16,}\b"),
    re.compile(r"\b(?:sk|ghp|xoxb|xoxp)-[A-Za-z0-9_-]{16,}\b"),
]
PERSONAL_JOURNAL_PATTERNS = {"journal", "diary", "daily-notes", "private-notes"}


def load_morpheusignore(project_root: Path) -> set[str]:
    ignore_path = project_root / ".morpheusignore"
    if ignore_path.is_symlink() or not ignore_path.is_file():
        return set()
    try:
        return {
            line.strip()
            for line in ignore_path.read_text(errors="ignore").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError:
        return set()


def path_is_ignored(rel_path: Path, ignore_patterns: set[str]) -> bool:
    if rel_path.parts[:3] == (".morpheus", "review", "check_corrections"):
        return False
    rel_text = rel_path.as_posix()
    patterns = DEFAULT_EXCLUDE_PATTERNS | SECRET_PATTERNS | ignore_patterns
    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        if any(part == pattern for part in rel_path.parts):
            return True
        if fnmatch(rel_text, pattern) or fnmatch(rel_path.name, pattern):
            return True
    return any(part.casefold() in PERSONAL_JOURNAL_PATTERNS for part in rel_path.parts)


def contains_secret_like_text(value: str) -> bool:
    return any(regex.search(value) for regex in SECRET_REGEXES)


def redact_secret_text(value: str) -> tuple[str, int]:
    redacted = value
    count = 0
    for regex in SECRET_REGEXES:
        redacted, replacements = regex.subn("[REDACTED]", redacted)
        count += replacements
    return redacted, count
