"""Deterministic product-class classifier for semantic candidates."""
from pathlib import PurePosixPath

from morpheus.core.semantic.models import CandidateClass, SemanticCandidate


SECURITY_TERMS = {
    "activate adapter",
    "activation",
    "cloud",
    "credential",
    "cookie",
    "eval pass",
    "private",
    "raw markdown",
    "secret",
    "source span",
    "token",
    "unsafe",
    "127.0.0.1",
}
INTEGRATION_TERMS = {"api", "endpoint", "mcp", "a2a", "server", "serve", "truth tools"}
COMMAND_TERMS = {
    "morpheus check",
    "morpheus learn",
    "morpheus review",
    "morpheus serve",
    "morpheus wake",
    "uvx ",
    "pipx ",
    "pytest",
    "ruff",
    "make ",
    "python ",
    "cli",
    "--",
}
ARCHITECTURE_TERMS = {
    "adapter",
    "classifier",
    "compiled project state",
    "dataset",
    "evidence",
    "learning core",
    "pipeline",
    "receipt",
    "semantic",
    "source-backed",
    "state",
    "truth layer",
    "wake.md",
    "weights",
}
PRODUCT_TERMS = {
    "agent",
    "first verify",
    "learning layer",
    "morpheus builds",
    "morpheus checks",
    "morpheus generates",
    "morpheus is",
    "product",
    "verified learning",
}
TEMPORARY_TERMS = {
    "expires after",
    "for now",
    "short-lived",
    "temporary",
    "until rollout",
}


def classify_candidate(candidate: SemanticCandidate) -> CandidateClass:
    """Return a stable class for review, dataset, and eval routing."""
    return classify_claim(
        kind=candidate.kind,
        claim=candidate.claim,
        source_path=candidate.source_path,
    )


def classify_claim(*, kind: str, claim: str, source_path: str) -> CandidateClass:
    """Classify a source-backed claim without requiring a full candidate model."""
    folded_claim = claim.casefold()
    path = PurePosixPath(source_path)
    path_text = source_path.casefold()

    if kind == "outdated_claim":
        return "stale"
    if kind == "open_task":
        return "open_task"
    if kind == "agent_rule":
        return "convention"
    if _has_any(folded_claim, TEMPORARY_TERMS):
        return "temporary"
    if _has_any(folded_claim, SECURITY_TERMS):
        return "security"
    if _path_in(path, "docs/architecture"):
        return "architecture"
    if _is_integration_source(path_text) or _has_any(folded_claim, INTEGRATION_TERMS):
        return "integration"
    if _is_command_source(path_text) or _has_any(folded_claim, COMMAND_TERMS):
        return "command"
    if _has_any(folded_claim, PRODUCT_TERMS):
        return "product"
    if _has_any(folded_claim, ARCHITECTURE_TERMS):
        return "architecture"
    if path_text.startswith("morpheus/"):
        return "implementation"
    if kind == "source_reference":
        return "implementation"
    if kind in {"current_state", "active_decision"}:
        return "product"
    return "unknown"


def with_candidate_class(candidate: SemanticCandidate) -> SemanticCandidate:
    return candidate.model_copy(update={"semantic_class": classify_candidate(candidate)})


def classify_candidates(candidates: list[SemanticCandidate]) -> list[SemanticCandidate]:
    return [with_candidate_class(candidate) for candidate in candidates]


def _has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _path_in(path: PurePosixPath, prefix: str) -> bool:
    return path.as_posix().casefold().startswith(prefix)


def _is_command_source(path_text: str) -> bool:
    return (
        path_text == "pyproject.toml"
        or path_text == "makefile"
        or path_text.startswith(".github/workflows/")
        or path_text.endswith("/cli.py")
    )


def _is_integration_source(path_text: str) -> bool:
    return path_text.startswith("morpheus/api/") or path_text.startswith("morpheus/integrations/")
