"""Local deterministic semantic provider for alpha dogfood."""
from morpheus.core.providers.fake import FakeProvider
from morpheus.core.semantic.models import SemanticCandidate, SemanticSource


class LocalProvider(FakeProvider):
    name = "local"
    model = "heuristic"

    def extract_candidates(
        self,
        source: SemanticSource,
        *,
        run_id: str,
        prompt_sha256: str,
        source_revision: str,
    ) -> list[SemanticCandidate]:
        if source.path == "WAKE.md":
            return []
        candidates = super().extract_candidates(
            source,
            run_id=run_id,
            prompt_sha256=prompt_sha256,
            source_revision=source_revision,
        )
        if source.category == "docs_state_sources":
            return candidates[:8]
        if source.category == "build_manifest_sources":
            return [candidate for candidate in candidates if _is_manifest_signal(candidate.claim)][:3]
        if source.category == "workflow_sources":
            return [candidate for candidate in candidates if _is_workflow_signal(candidate.claim)][:3]
        if source.category == "cli_api_sources":
            return [candidate for candidate in candidates if _is_cli_api_signal(candidate.claim)][:3]
        return candidates[:3]


def _is_manifest_signal(claim: str) -> bool:
    lowered = claim.casefold()
    return (
        "morpheus-wake" in lowered
        or "morpheus.cli" in lowered
        or "requires-python" in lowered
        or "version" in lowered
    )


def _is_workflow_signal(claim: str) -> bool:
    lowered = claim.casefold()
    return "pypi" in lowered or "trusted" in lowered or "morpheus-wake" in lowered


def _is_cli_api_signal(claim: str) -> bool:
    lowered = claim.casefold()
    return (
        "morpheus ai" in lowered
        or "morpheus wake" in lowered
        or "semantic" in lowered
        or "version" in lowered
    )
