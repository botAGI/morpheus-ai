"""Source span verification for semantic candidates."""
from difflib import SequenceMatcher
import hashlib
from pathlib import Path

from morpheus.core.compiler import compute_sha256
from morpheus.core.semantic.models import SemanticCandidate


def verify_candidate_span(
    project_root: Path,
    candidate: SemanticCandidate,
    *,
    fuzzy_threshold: float = 0.85,
) -> SemanticCandidate:
    """Mark a candidate source-backed only when its span is valid."""
    if candidate.source_path == "WAKE.md":
        return _needs_review(candidate)
    path = project_root / candidate.source_path
    try:
        if not path.is_file():
            return _needs_review(candidate)
        source_sha = compute_sha256(path)
        lines = path.read_text(errors="ignore").splitlines()
    except (OSError, ValueError):
        return _needs_review(candidate)

    if source_sha != candidate.source_sha256:
        return _needs_review(candidate)
    if candidate.line_start > candidate.line_end:
        return _needs_review(candidate)
    if candidate.line_end > len(lines):
        return _needs_review(candidate)

    actual = "\n".join(lines[candidate.line_start - 1 : candidate.line_end]).strip()
    expected = candidate.evidence_excerpt.strip()
    ratio = SequenceMatcher(None, _normalize(actual), _normalize(expected)).ratio()
    if expected in actual or ratio >= fuzzy_threshold:
        verified = candidate.model_copy()
        verified.label = "source_backed"
        verified.evidence_sha256 = hashlib.sha256(actual.encode()).hexdigest()
        return verified
    return _needs_review(candidate)


def _needs_review(candidate: SemanticCandidate) -> SemanticCandidate:
    updated = candidate.model_copy()
    updated.label = "needs_review"
    return updated


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
