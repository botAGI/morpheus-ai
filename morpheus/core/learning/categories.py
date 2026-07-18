"""Canonical category contract for adapter memory benchmarks."""

from morpheus.core.semantic.models import SemanticCandidate


BENCHMARK_CATEGORY_SCHEMA = "morpheus-benchmark-categories/1"
CANONICAL_BENCHMARK_CATEGORIES = frozenset({
    "product_identity", "commands_and_cli_behavior", "architecture",
    "safety_rules", "team_conventions", "stale_claim_correction",
    "unsupported_claim_refusal",
})
CRITICAL_BENCHMARK_CATEGORIES = frozenset({
    "safety_rules", "stale_claim_correction", "unsupported_claim_refusal",
})
DIAGNOSTIC_BENCHMARK_CATEGORIES = frozenset({"project_recall"})
KNOWN_BENCHMARK_CATEGORIES = (
    CANONICAL_BENCHMARK_CATEGORIES | DIAGNOSTIC_BENCHMARK_CATEGORIES
)


def benchmark_category_for_candidate(candidate: SemanticCandidate) -> str:
    """Return the canonical benchmark category for a routed candidate."""
    if candidate.kind == "outdated_claim" or candidate.semantic_class == "stale":
        return "stale_claim_correction"

    by_class = {
        "product": "product_identity",
        "command": "commands_and_cli_behavior",
        "architecture": "architecture",
        "security": "safety_rules",
        "convention": "team_conventions",
    }
    category = by_class.get(candidate.semantic_class)
    if category is not None:
        return category
    return "project_recall"
