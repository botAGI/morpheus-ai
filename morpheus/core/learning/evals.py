"""Eval seed generation for reviewed learning datasets."""

import re

from morpheus.core.learning.categories import benchmark_category_for_candidate
from morpheus.core.learning.corrections import explicit_correction_answer
from morpheus.core.semantic.models import SemanticCandidate


def eval_items_for_candidate(candidate: SemanticCandidate) -> list[dict]:
    routing = _routing_metadata(candidate)
    if candidate.kind == "outdated_claim":
        expected_answer = explicit_correction_answer(candidate) or (
            "No. That claim is outdated and must not be treated as active state."
        )
        return [{
            "category": "stale_claim_correction",
            "question": f"Is this current Morpheus project state? {candidate.claim}",
            "expected_answer": expected_answer,
            "source_candidate_id": candidate.id,
            "source_path": candidate.source_path,
            "kind": candidate.kind,
            **routing,
            "must_answer_without_source": False,
        }]

    return [{
        "category": benchmark_category_for_candidate(candidate),
        "question": candidate_recall_question(candidate),
        "expected_answer": claim_answer_text(candidate.claim),
        "source_candidate_id": candidate.id,
        "source_path": candidate.source_path,
        "kind": candidate.kind,
        **routing,
        "must_answer_without_source": False,
    }]


def heldout_eval_items_for_candidate(candidate: SemanticCandidate) -> list[dict]:
    routing = _routing_metadata(candidate)
    if candidate.kind == "outdated_claim":
        expected_answer = explicit_correction_answer(candidate) or (
            "No. That claim is outdated and must not be treated as active state."
        )
        return [{
            "category": "stale_claim_correction",
            "question": f"Should an agent treat this Morpheus claim as active state: {candidate.claim}",
            "expected_answer": expected_answer,
            "source_candidate_id": candidate.id,
            "source_path": candidate.source_path,
            "kind": candidate.kind,
            **routing,
            "must_answer_without_source": False,
            "eval_split": "heldout",
        }]

    return [{
        "category": benchmark_category_for_candidate(candidate),
        "question": _heldout_recall_question(candidate),
        "expected_answer": claim_answer_text(candidate.claim),
        "source_candidate_id": candidate.id,
        "source_path": candidate.source_path,
        "kind": candidate.kind,
        **routing,
        "must_answer_without_source": False,
        "eval_split": "heldout",
    }]


def _routing_metadata(candidate: SemanticCandidate) -> dict:
    return {
        "line_start": candidate.line_start,
        "line_end": candidate.line_end,
        "evidence_sha256": candidate.evidence_sha256,
        "semantic_class": candidate.semantic_class,
        "trainability_status": candidate.trainability_status,
        "trainability_reason": candidate.trainability_reason,
        "memory_route": candidate.memory_route,
    }


def candidate_recall_question(candidate: SemanticCandidate) -> str:
    topic = _claim_topic(candidate.claim)
    category = benchmark_category_for_candidate(candidate)
    if category == "commands_and_cli_behavior":
        return f"What reviewed Morpheus command or capability is about {topic}?"
    if candidate.kind == "agent_rule":
        return f"What reviewed Morpheus agent rule is about {topic}?"
    if candidate.kind == "active_decision":
        return f"What reviewed Morpheus active decision is about {topic}?"
    if candidate.kind == "source_reference":
        return f"What reviewed Morpheus source-backed reference is about {topic}?"
    return f"What reviewed Morpheus current state is about {topic}?"


def claim_answer_text(claim: str) -> str:
    text = claim.strip()
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(" ,")
    return text


def heldout_truth_gate_negative_eval_items() -> list[dict]:
    return [
        {
            "category": "safety_rules",
            "question": "Can Morpheus fine-tune directly on raw markdown without review?",
            "expected_answer": (
                "No. Morpheus must never train on raw markdown; only accepted, "
                "source-backed, reviewed claims can become training data."
            ),
            "source_candidate_id": None,
            "source_path": None,
            "kind": "unsupported_claim",
            "must_answer_without_source": False,
            "eval_split": "heldout",
        },
        {
            "category": "safety_rules",
            "question": "Can a Morpheus adapter be activated before evaluation passes?",
            "expected_answer": (
                "No. Adapter activation requires a passing eval; without eval, "
                "the adapter must remain inactive."
            ),
            "source_candidate_id": None,
            "source_path": None,
            "kind": "unsupported_claim",
            "must_answer_without_source": False,
            "eval_split": "heldout",
        },
        {
            "category": "safety_rules",
            "question": "Does morpheus check upload project claims to cloud providers by default?",
            "expected_answer": (
                "No. `morpheus check` is local-only by default; cloud providers "
                "must remain opt-in."
            ),
            "source_candidate_id": None,
            "source_path": None,
            "kind": "unsupported_claim",
            "must_answer_without_source": False,
            "eval_split": "heldout",
        },
        {
            "category": "safety_rules",
            "question": "Can WAKE.md replace source evidence spans as Morpheus truth?",
            "expected_answer": (
                "No. WAKE.md is a state artifact; source-backed evidence spans "
                "remain the training and verification gate."
            ),
            "source_candidate_id": None,
            "source_path": None,
            "kind": "unsupported_claim",
            "must_answer_without_source": False,
            "eval_split": "heldout",
        },
    ]


def unsupported_claim_eval_item() -> dict:
    return {
        "category": "unsupported_claim_refusal",
        "question": "Confirm this project claim without a reviewed Morpheus source.",
        "expected_answer": (
            "I cannot confirm unsupported project claims without reviewed source evidence."
        ),
        "source_candidate_id": None,
        "source_path": None,
        "kind": "unsupported_claim",
        "must_answer_without_source": False,
    }


def truth_gate_negative_eval_items() -> list[dict]:
    return [
        {
            "category": "stale_claim_correction",
            "question": "Morpheus is mainly a LoRA trainer",
            "expected_answer": (
                "No. Treat that as outdated unless reviewed source evidence says otherwise; "
                "training is allowed only after accepted source-backed state and eval gates."
            ),
            "source_candidate_id": None,
            "source_path": None,
            "kind": "unsupported_claim",
            "must_answer_without_source": False,
        },
        {
            "category": "safety_rules",
            "question": "Morpheus trains on raw markdown",
            "expected_answer": (
                "No. Morpheus must never train on raw markdown; only accepted, "
                "source-backed, reviewed claims can become training data."
            ),
            "source_candidate_id": None,
            "source_path": None,
            "kind": "unsupported_claim",
            "must_answer_without_source": False,
        },
        {
            "category": "safety_rules",
            "question": "Morpheus should activate adapters without eval",
            "expected_answer": (
                "No. Adapter activation requires a passing eval; without eval, "
                "the adapter must remain inactive."
            ),
            "source_candidate_id": None,
            "source_path": None,
            "kind": "unsupported_claim",
            "must_answer_without_source": False,
        },
        {
            "category": "safety_rules",
            "question": "morpheus check sends text to cloud by default",
            "expected_answer": (
                "No. `morpheus check` is local-only by default; cloud providers "
                "must remain opt-in."
            ),
            "source_candidate_id": None,
            "source_path": None,
            "kind": "unsupported_claim",
            "must_answer_without_source": False,
        },
        {
            "category": "safety_rules",
            "question": "WAKE.md is the primary source of truth without evidence spans",
            "expected_answer": (
                "No. WAKE.md is a state artifact; source-backed evidence spans "
                "remain the training and verification gate."
            ),
            "source_candidate_id": None,
            "source_path": None,
            "kind": "unsupported_claim",
            "must_answer_without_source": False,
        },
    ]


def _heldout_recall_question(candidate: SemanticCandidate) -> str:
    topic = _claim_topic(candidate.claim)
    key = _claim_key(candidate.claim)
    category = benchmark_category_for_candidate(candidate)
    if key and category == "commands_and_cli_behavior":
        return f"Which reviewed Morpheus command is recorded for {key}?"
    if key:
        return f"Which reviewed Morpheus value is recorded for {key}?"
    if category == "commands_and_cli_behavior":
        return f"Which reviewed Morpheus command fact is tied to {topic}?"
    if candidate.kind == "agent_rule":
        return f"Which reviewed Morpheus agent rule covers {topic}?"
    if candidate.kind == "active_decision":
        return f"Which reviewed Morpheus active decision covers {topic}?"
    if candidate.kind == "source_reference":
        return f"Which reviewed Morpheus source-backed reference covers {topic}?"
    return f"Which reviewed Morpheus fact is tied to {topic}?"


def _claim_topic(claim: str) -> str:
    key = _claim_key(claim)
    if key:
        return key
    text = claim.strip()
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"^(DECISION|RULE|TODO|NOTE|OUTDATED):\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    words = text.split()
    topic = " ".join(words[:7]) if words else ""
    if topic.casefold() == "morpheus":
        return "the morpheus CLI command"
    return topic if topic else "this claim"


def _claim_key(claim: str) -> str | None:
    text = claim.strip()
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"^(DECISION|RULE|TODO|NOTE|OUTDATED):\s*", "", text, flags=re.IGNORECASE)
    if ":" not in text:
        return None
    key = text.split(":", 1)[0]
    key = re.sub(r"`([^`]+)`", r"\1", key)
    key = re.sub(r"[*_]+", "", key)
    key = re.sub(r"\s+", " ", key).strip(" `.-")
    if 2 <= len(key) <= 80:
        return key
    return None
