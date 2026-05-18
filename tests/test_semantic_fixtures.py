import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from morpheus.core.providers.local import LocalProvider
from morpheus.core.semantic.models import SemanticCandidate, SemanticSource
from morpheus.core.semantic.review import ReviewStore, run_semantic_review


FIXTURES = Path(__file__).parent / "fixtures" / "semantic"


def copy_fixture(tmp_path: Path, name: str) -> Path:
    project_root = tmp_path / name
    shutil.copytree(FIXTURES / name, project_root)
    return project_root


def test_unmarked_repo_fixture_produces_source_backed_candidates(tmp_path):
    project_root = copy_fixture(tmp_path, "unmarked_repo")

    report = run_semantic_review(project_root, provider=LocalProvider())

    assert report["candidates_total"] >= 1
    assert report["source_backed_total"] >= 1
    draft = ReviewStore(project_root).draft_wake_path.read_text()
    assert "Morpheus generates WAKE.md" in draft


def test_prompt_injection_repo_fixture_ignores_instructions(tmp_path):
    project_root = copy_fixture(tmp_path, "prompt_injection_repo")

    report = run_semantic_review(project_root, provider=LocalProvider())
    claims = [candidate.claim for candidate in ReviewStore(project_root).load_candidates()]

    assert report["source_backed_total"] >= 1
    assert not any("uploads secrets" in claim for claim in claims)
    assert not any("ignore previous instructions" in claim.casefold() for claim in claims)


def test_bad_source_span_repo_fixture_marks_invalid_span_needs_review(tmp_path):
    project_root = copy_fixture(tmp_path, "bad_source_span_repo")

    report = run_semantic_review(project_root, provider=BadSpanProvider())
    candidates = ReviewStore(project_root).load_candidates()

    assert report["candidates_total"] == 1
    assert report["source_backed_total"] == 0
    assert candidates[0].label == "needs_review"


class BadSpanProvider:
    name = "bad_span"
    model = "fixture"

    def extract_candidates(
        self,
        source: SemanticSource,
        *,
        run_id: str,
        prompt_sha256: str,
        source_revision: str,
    ) -> list[SemanticCandidate]:
        return [
            SemanticCandidate(
                id="cand_bad_span",
                run_id=run_id,
                kind="current_state",
                claim="Morpheus generates WAKE.md from project state.",
                source_path=source.path,
                source_sha256=source.sha256,
                source_mtime=source.modified_at,
                source_revision=source_revision,
                line_start=99,
                line_end=99,
                evidence_excerpt="Morpheus generates WAKE.md from project state.",
                evidence_sha256=hashlib.sha256(
                    b"Morpheus generates WAKE.md from project state."
                ).hexdigest(),
                confidence=0.9,
                label="needs_review",
                status="pending",
                created_at=datetime.now(timezone.utc),
                provider={"name": self.name, "model": self.model},
                prompt_sha256=prompt_sha256,
            )
        ]
