"""Null semantic provider: explicit no-op default."""
from morpheus.core.semantic.models import SemanticCandidate, SemanticSource


class NullProvider:
    name = "null"
    model = "none"

    def extract_candidates(
        self,
        source: SemanticSource,
        *,
        run_id: str,
        prompt_sha256: str,
        source_revision: str,
    ) -> list[SemanticCandidate]:
        return []

