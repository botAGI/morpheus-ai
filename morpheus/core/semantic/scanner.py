"""High-signal source scanner for semantic review runs."""
from fnmatch import fnmatch
import math
from datetime import datetime, timezone
from pathlib import Path

from morpheus.core.compiler import DEFAULT_EXCLUDE_PATTERNS, compute_sha256
from morpheus.core.safe_io import reject_symlink_components
from morpheus.core.semantic.models import SemanticSource


DOC_ROOT_FILES = {
    "README.md",
    "README.ru.md",
    "SPEC.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CHANGELOG.md",
    "WAKE.md",
}
BUILD_MANIFEST_FILES = {
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
}
SECRET_PATTERNS = {
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
SEMANTIC_EXCLUDE_PATTERNS = DEFAULT_EXCLUDE_PATTERNS - {"WAKE.md"}


def scan_semantic_sources(project_root: Path) -> list[SemanticSource]:
    """Return high-signal sources eligible for semantic candidate extraction."""
    project_root = project_root.expanduser()
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    ignore_patterns = _load_morpheusignore(project_root)

    sources = []
    for path in sorted(project_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            rel_path = path.relative_to(project_root)
        except ValueError:
            continue
        rel_text = rel_path.as_posix()
        if _matches_any(rel_path, rel_text, SEMANTIC_EXCLUDE_PATTERNS | SECRET_PATTERNS):
            continue
        if _matches_any(rel_path, rel_text, ignore_patterns):
            continue

        category = _category_for_path(rel_path)
        if category is None:
            continue

        try:
            content = path.read_text(errors="ignore")
        except OSError:
            continue
        if _looks_sensitive(content):
            continue
        try:
            stat = path.stat()
            sha = compute_sha256(path)
        except (OSError, ValueError):
            continue
        sources.append(
            SemanticSource(
                path=rel_text,
                category=category,
                sha256=sha,
                size_bytes=stat.st_size,
                line_count=len(content.splitlines()),
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                content=content,
            )
        )
    return sources


def _category_for_path(rel_path: Path) -> str | None:
    rel_text = rel_path.as_posix()
    if len(rel_path.parts) == 1 and rel_path.name in DOC_ROOT_FILES:
        return "docs_state_sources"
    if rel_path.parts[:1] in [("docs",), (".github",)] and rel_path.suffix.lower() in {
        ".md",
        ".mdx",
    }:
        return "docs_state_sources"
    if rel_path.name in BUILD_MANIFEST_FILES:
        return "build_manifest_sources"
    if len(rel_path.parts) >= 3 and rel_path.parts[:2] == (".github", "workflows"):
        if rel_path.suffix.lower() in {".yml", ".yaml"}:
            return "workflow_sources"
    if rel_path.suffix == ".py" and (
        rel_path.name in {"__init__.py", "cli.py"} or "/api/" in f"/{rel_text}"
    ):
        if rel_path.parts[0] not in {"tests", ".worktrees"}:
            return "cli_api_sources"
    return None


def _load_morpheusignore(project_root: Path) -> set[str]:
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


def _matches_any(rel_path: Path, rel_text: str, patterns: set[str]) -> bool:
    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        if any(part == pattern for part in rel_path.parts):
            return True
        if fnmatch(rel_text, pattern) or fnmatch(rel_path.name, pattern):
            return True
    return False


def _looks_sensitive(content: str) -> bool:
    lowered = content.casefold()
    if "private key" in lowered or "api_key" in lowered or "secret_key" in lowered:
        return True
    if "sk-" in lowered or "xoxb-" in lowered or "ghp_" in lowered:
        return True
    for token in content.replace('"', " ").replace("'", " ").split():
        if len(token) >= 64 and _entropy(token) >= 4.5:
            return True
    return False


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    chars = set(value)
    return -sum((value.count(char) / len(value)) * math.log2(value.count(char) / len(value)) for char in chars)
