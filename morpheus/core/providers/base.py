"""Provider protocol for semantic candidate extraction."""
from typing import Protocol

from morpheus.core.semantic.models import SemanticCandidate, SemanticSource


class SemanticProvider(Protocol):
    name: str
    model: str

    def extract_candidates(
        self,
        source: SemanticSource,
        *,
        run_id: str,
        prompt_sha256: str,
        source_revision: str,
    ) -> list[SemanticCandidate]:
        """Extract semantic candidates from one source."""
        ...

