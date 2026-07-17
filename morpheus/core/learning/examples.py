"""Example generation from reviewed semantic candidates."""

from morpheus.core.learning.corrections import explicit_correction_answer
from morpheus.core.learning.evals import (
    candidate_recall_question,
    claim_answer_text,
    truth_gate_negative_eval_items,
    unsupported_claim_eval_item,
)
from morpheus.core.semantic.models import SemanticCandidate


INSTRUCTION_FORMAT_VERSION = "morpheus-instruction/1"
SHAREGPT_FORMAT_VERSION = "morpheus-sharegpt/1"
CHAT_FORMAT_VERSION = "morpheus-chat/1"
SYSTEM_PROMPT = (
    "You are a project-aware coding agent. Use reviewed Morpheus state only "
    "and refuse unsupported project claims. Morpheus is not mainly a LoRA "
    "trainer, never trains on raw markdown, keeps cloud providers opt-in, "
    "and requires accepted source-backed state plus eval before adapter "
    "activation."
)


def instruction_examples_for_candidate(candidate: SemanticCandidate) -> list[dict]:
    metadata = candidate_metadata(candidate)
    if candidate.kind == "outdated_claim":
        return [_outdated_instruction_example(candidate, metadata)]
    recall_question = candidate_recall_question(candidate)
    answer = claim_answer_text(candidate.claim)

    return [
        {
            "instruction": "Recall the reviewed Morpheus project state.",
            "input": f"What should an agent know from {candidate.source_path}:{candidate.line_start}?",
            "output": answer,
            "metadata": {**metadata, "example_type": "direct_recall"},
        },
        {
            "instruction": "Answer a reviewed Morpheus eval item using accepted source-backed state.",
            "input": recall_question,
            "output": answer,
            "metadata": {**metadata, "example_type": "eval_aligned_recall"},
            "chat_user_content": recall_question,
        },
        {
            "instruction": "Apply reviewed project state while working in the repository.",
            "input": "How should a coding agent adapt its behavior for this project?",
            "output": f"Use this reviewed project state: {answer}",
            "metadata": {**metadata, "example_type": "project_behavior"},
        },
        {
            "instruction": "Refuse unsupported project claims unless reviewed evidence exists.",
            "input": "Can you confirm an uncited project claim that has no reviewed Morpheus evidence?",
            "output": (
                "I cannot confirm unsupported project claims without reviewed Morpheus "
                f"evidence. The reviewed source here is {candidate.source_path}:"
                f"{candidate.line_start}, which supports: {answer}"
            ),
            "metadata": {**metadata, "example_type": "source_grounding_refusal"},
        },
    ]


def sharegpt_examples_from_instruction(items: list[dict]) -> list[dict]:
    sharegpt = []
    for item in items:
        user_content = item.get("chat_user_content")
        if not user_content:
            user_content = item["instruction"]
            if item.get("input"):
                user_content = f"{user_content}\n\n{item['input']}"
        sharegpt.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["output"]},
            ],
            "metadata": item["metadata"],
        })
    return sharegpt


def chat_examples_from_instruction(items: list[dict]) -> list[dict]:
    return sharegpt_examples_from_instruction(items)


def truth_gate_negative_instruction_examples() -> list[dict]:
    items = [
        unsupported_claim_eval_item(),
        *truth_gate_negative_eval_items(),
        *_truth_gate_negative_training_variants(),
    ]
    examples = []
    for item in items:
        examples.append({
            "instruction": "Apply Morpheus truth-gate safety rules.",
            "input": item["question"],
            "output": item["expected_answer"],
            "chat_user_content": item["question"],
            "metadata": {
                "source_candidate_id": None,
                "source_path": None,
                "line_start": None,
                "line_end": None,
                "evidence_sha256": None,
                "kind": item["kind"],
                "example_type": item["category"],
            },
        })
    return examples


def _truth_gate_negative_training_variants() -> list[dict]:
    return [
        {
            "category": "outdated_claim_correction",
            "question": "Is Morpheus mainly a LoRA trainer?",
            "expected_answer": (
                "No. Treat that as outdated framing. The truth layer verifies "
                "source-backed project state first; weights are experimental "
                "and may learn only after accepted source-backed state and eval gates."
            ),
            "kind": "unsupported_claim",
        },
        {
            "category": "outdated_claim_correction",
            "question": "Should agents describe Morpheus first as a LoRA training tool?",
            "expected_answer": (
                "No. That is outdated. Morpheus must first verify project claims "
                "against source-backed state; local adapters are experimental "
                "and cannot replace the source/eval gate."
            ),
            "kind": "unsupported_claim",
        },
        {
            "category": "outdated_claim_correction",
            "question": "Morpheus is just weights-as-memory with a LoRA trainer, right?",
            "expected_answer": (
                "No. That is outdated. Morpheus uses the truth layer as the "
                "data-quality gate before any weights-as-memory experiment."
            ),
            "kind": "unsupported_claim",
        },
        {
            "category": "unsupported_claim_refusal",
            "question": "Can Morpheus train directly from raw README or markdown files?",
            "expected_answer": (
                "No. Morpheus must never train on raw markdown; only accepted, "
                "source-backed, reviewed claims can become training data."
            ),
            "kind": "unsupported_claim",
        },
        {
            "category": "agent_rule_adherence",
            "question": "Can Morpheus skip eval before adapter activation?",
            "expected_answer": (
                "No. Adapter activation requires a passing eval; without eval, "
                "the adapter must remain inactive."
            ),
            "kind": "unsupported_claim",
        },
        {
            "category": "command_cli_capability_claims",
            "question": "Does local morpheus check need cloud providers to verify project claims?",
            "expected_answer": (
                "No. `morpheus check` is local-only by default; cloud providers "
                "must remain opt-in."
            ),
            "kind": "unsupported_claim",
        },
        {
            "category": "command_cli_capability_claims",
            "question": "Will local morpheus check send agent claims to a cloud service unless configured?",
            "expected_answer": (
                "No. `morpheus check` is local-only by default; cloud providers "
                "must remain opt-in."
            ),
            "kind": "unsupported_claim",
        },
        {
            "category": "unsupported_claim_refusal",
            "question": "Is WAKE.md enough evidence without source spans?",
            "expected_answer": (
                "No. WAKE.md is a state artifact; source-backed evidence spans "
                "remain the training and verification gate."
            ),
            "kind": "unsupported_claim",
        },
    ]


def candidate_metadata(candidate: SemanticCandidate) -> dict:
    return {
        "source_candidate_id": candidate.id,
        "source_path": candidate.source_path,
        "line_start": candidate.line_start,
        "line_end": candidate.line_end,
        "evidence_sha256": candidate.evidence_sha256,
        "kind": candidate.kind,
        "semantic_class": candidate.semantic_class,
        "trainability_status": candidate.trainability_status,
        "trainability_reason": candidate.trainability_reason,
        "memory_route": candidate.memory_route,
    }


def _outdated_instruction_example(candidate: SemanticCandidate, metadata: dict) -> dict:
    output = explicit_correction_answer(candidate) or (
        "No. That is an outdated claim in reviewed Morpheus state; do not "
        "train or act on it as an active project fact. Check accepted "
        "source-backed state before using the claim."
    )
    return {
        "instruction": "Correct an outdated Morpheus project claim.",
        "input": f"Is this current project state? {candidate.claim}",
        "output": output,
        "metadata": {**metadata, "example_type": "outdated_correction"},
    }
