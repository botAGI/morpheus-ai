"""Example generation from reviewed semantic candidates."""

from morpheus.core.learning.corrections import explicit_correction_answer
from morpheus.core.learning.evals import candidate_recall_question, claim_answer_text
from morpheus.core.semantic.models import SemanticCandidate


INSTRUCTION_FORMAT_VERSION = "morpheus-instruction/1"
SHAREGPT_FORMAT_VERSION = "morpheus-sharegpt/1"
CHAT_FORMAT_VERSION = "morpheus-chat/1"
SYSTEM_PROMPT = (
    "You are a project-aware coding agent. Use only reviewed, source-bound "
    "project knowledge and refuse unsupported project claims."
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
