"""Shared rendering for explicitly reviewed correction text."""

import re

from morpheus.core.semantic.models import SemanticCandidate


def explicit_correction_answer(candidate: SemanticCandidate) -> str | None:
    if not candidate.correction_text:
        return None
    correction = re.sub(r"\s+", " ", candidate.correction_text).strip()
    if not correction:
        return None
    if correction[-1] not in ".!?":
        correction += "."
    return f"No. {correction}"
