"""Eval seed generation for reviewed learning datasets."""

from morpheus.core.semantic.models import SemanticCandidate


def eval_items_for_candidate(candidate: SemanticCandidate) -> list[dict]:
    if candidate.kind == "outdated_claim":
        return [{
            "category": "outdated_claim_correction",
            "question": f"Is this current Morpheus project state? {candidate.claim}",
            "expected_answer": (
                "No. That claim is outdated and must not be treated as active state."
            ),
            "source_candidate_id": candidate.id,
            "source_path": candidate.source_path,
            "kind": candidate.kind,
            "must_answer_without_source": False,
        }]

    return [{
        "category": _category(
            candidate.kind,
            source_path=candidate.source_path,
            claim=candidate.claim,
        ),
        "question": f"What reviewed project state is supported by {candidate.source_path}:{candidate.line_start}?",
        "expected_answer": candidate.claim,
        "source_candidate_id": candidate.id,
        "source_path": candidate.source_path,
        "kind": candidate.kind,
        "must_answer_without_source": False,
    }]


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
            "category": "outdated_claim_correction",
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
            "category": "unsupported_claim_refusal",
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
            "category": "agent_rule_adherence",
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
            "category": "command_cli_capability_claims",
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
            "category": "unsupported_claim_refusal",
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


def _category(kind: str, *, source_path: str = "", claim: str = "") -> str:
    if source_path == "pyproject.toml":
        return "package_metadata_claims"
    if "morpheus " in claim.casefold():
        return "command_cli_capability_claims"
    return {
        "active_decision": "active_decision_recall",
        "agent_rule": "agent_rule_adherence",
        "open_task": "project_recall",
        "source_reference": "project_recall",
    }.get(kind, "project_recall")
