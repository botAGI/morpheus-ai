"""Deterministic trainability and memory-route decisions for candidates."""
import re

from morpheus.core.semantic.classifier import classify_candidate
from morpheus.core.semantic.models import MemoryRoute, SemanticCandidate, TrainabilityStatus


STABLE_TRAINING_KINDS = {
    "current_state",
    "active_decision",
    "agent_rule",
}
ROUTING_POLICY_VERSION = "morpheus-memory-routing/1"
SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_ -]?key|secret|token|cookie|password|credential)\b"
)


def route_candidate(candidate: SemanticCandidate) -> SemanticCandidate:
    semantic_class = candidate.semantic_class
    if semantic_class == "unknown":
        semantic_class = classify_candidate(candidate)
    classified = candidate.model_copy(update={"semantic_class": semantic_class})
    status, route, reason = _route(classified)
    return classified.model_copy(update={
        "trainability_status": status,
        "trainability_reason": reason,
        "memory_route": route,
    })


def route_candidates(candidates: list[SemanticCandidate]) -> list[SemanticCandidate]:
    return [route_candidate(candidate) for candidate in candidates]


def route_check_result(status: str, semantic_class: str) -> tuple[MemoryRoute, str]:
    """Route an unreviewed check result without promoting it to training."""
    if status == "verified":
        if semantic_class in {"security", "convention", "open_task", "temporary"}:
            return "prompt_context", "verified_immediate_agent_context"
        return "retrieval", "verified_project_knowledge"
    if status == "stale":
        return "stale_archive", "checked_claim_is_stale"
    return "human_review", f"checked_claim_{status}_requires_review"


def _route(candidate: SemanticCandidate) -> tuple[TrainabilityStatus, MemoryRoute, str]:
    if _secret_like(candidate):
        return "unsafe", "excluded", "secret_like_content"
    if candidate.status == "rejected":
        return "excluded", "excluded", "status_rejected"
    if candidate.label != "source_backed":
        if candidate.label == "inferred":
            return "excluded", "excluded", "label_inferred"
        return "needs_review", "human_review", f"label_{candidate.label}"
    if candidate.semantic_class == "stale" and candidate.kind != "outdated_claim":
        return (
            "needs_review",
            "stale_archive",
            "stale_class_requires_outdated_claim_kind",
        )
    if candidate.kind == "outdated_claim":
        if candidate.status == "accepted":
            return "negative_example", "negative_example", "outdated_claim_correction"
        return "needs_review", "stale_archive", "outdated_claim_pending_review"
    if candidate.status != "accepted":
        return "needs_review", "human_review", f"status_{candidate.status}"
    if candidate.kind == "open_task":
        return "retrievable", "prompt_context", f"{candidate.kind}_current_context"
    if candidate.kind == "source_reference":
        return "retrievable", "retrieval", "source_reference_lookup"
    if candidate.semantic_class == "temporary":
        return "eval_only", "eval_only", "temporary_fact_not_stable_training_fact"
    if candidate.kind in STABLE_TRAINING_KINDS:
        return "trainable", "adapter_training", "accepted_source_backed_stable_claim"
    return "retrievable", "retrieval", f"kind_{candidate.kind}"


def _secret_like(candidate: SemanticCandidate) -> bool:
    return bool(SECRET_PATTERN.search(candidate.claim) or SECRET_PATTERN.search(candidate.evidence_excerpt))
