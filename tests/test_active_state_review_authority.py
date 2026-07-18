import hashlib
from datetime import datetime, timezone

import pytest

from morpheus.core.models import Claim, Evidence, Source
from morpheus.core.semantic.active_authority import (
    build_active_state_review_authority,
    project_active_state_review_candidates,
)
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.routing import route_candidate


def _candidate(**updates) -> SemanticCandidate:
    timestamp = datetime(2026, 7, 18, tzinfo=timezone.utc)
    excerpt = "DECISION: reviewed state is explicit authority."
    candidate = SemanticCandidate(
        id="candidate_reviewed",
        run_id="semrun_authority",
        kind="active_decision",
        claim=excerpt,
        source_path="README.md",
        source_sha256="a" * 64,
        source_mtime=timestamp,
        source_revision="git:test",
        line_start=3,
        line_end=3,
        evidence_excerpt=excerpt,
        evidence_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
        confidence=0.98,
        label="source_backed",
        status="accepted",
        created_at=timestamp,
        provider={"name": "local", "model": "fixture"},
        prompt_sha256="b" * 64,
        reviewed_by="tester",
        reviewed_at=timestamp,
    )
    return route_candidate(candidate).model_copy(update=updates)


def _binding(
    *,
    candidate: SemanticCandidate | None = None,
    claim_id: str = "clm_sem_0001",
    evidence_id: str = "ev_sem_0001",
) -> tuple[Claim, Evidence, SemanticCandidate]:
    candidate = candidate or _candidate()
    claim = Claim(
        id=claim_id,
        source_id="src_readme",
        line_start=candidate.line_start,
        line_end=candidate.line_end,
        excerpt=candidate.claim,
        status="active",
        category="decision",
        inference=False,
    )
    evidence = Evidence(
        id=evidence_id,
        claim_id=claim_id,
        source_id="src_readme",
        path=candidate.source_path,
        line_start=candidate.line_start,
        line_end=candidate.line_end,
        excerpt=candidate.evidence_excerpt,
        source_sha256=candidate.source_sha256,
        excerpt_sha256=candidate.evidence_sha256,
    )
    return claim, evidence, candidate


def _state_and_evidence(bindings):
    return (
        {
            "sources": [
                Source(
                    id="src_readme",
                    path="README.md",
                    sha256="a" * 64,
                ).model_dump(mode="json")
            ],
            "claims": [claim.model_dump(mode="json") for claim, _, _ in bindings],
        },
        [evidence.model_dump(mode="json") for _, evidence, _ in bindings],
    )


def test_active_state_review_authority_round_trips_exact_candidate():
    bindings = [_binding()]
    authority = build_active_state_review_authority(bindings)
    state, evidence_rows = _state_and_evidence(bindings)

    candidates = project_active_state_review_candidates(
        authority,
        state,
        evidence_rows,
    )

    assert candidates == [bindings[0][2]]
    assert authority["binding_count"] == 1
    assert len(authority["sha256"]) == 64


def test_active_state_review_authority_requires_reviewer_identity():
    with pytest.raises(ValueError, match="reviewer_missing"):
        build_active_state_review_authority([
            _binding(candidate=_candidate(reviewed_by=None))
        ])


def test_active_state_review_authority_requires_review_time():
    with pytest.raises(ValueError, match="review_time_missing"):
        build_active_state_review_authority([
            _binding(candidate=_candidate(reviewed_at=None))
        ])


def test_active_state_review_authority_rejects_stale_route_fields():
    with pytest.raises(ValueError, match="route_mismatch"):
        build_active_state_review_authority([
            _binding(candidate=_candidate(memory_route="human_review"))
        ])


def test_active_state_review_authority_recomputes_semantic_class_before_routing():
    excerpt = "DECISION: temporary rollout state expires after this test."
    forged = route_candidate(_candidate().model_copy(update={
        "claim": excerpt,
        "evidence_excerpt": excerpt,
        "evidence_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "semantic_class": "product",
    }))
    assert forged.memory_route == "adapter_training"

    with pytest.raises(ValueError, match="semantic_class_mismatch"):
        build_active_state_review_authority([_binding(candidate=forged)])


def test_active_state_review_authority_rejects_duplicate_candidate_ids():
    with pytest.raises(ValueError, match="duplicate_candidate_id"):
        build_active_state_review_authority([
            _binding(),
            _binding(claim_id="clm_sem_0002", evidence_id="ev_sem_0002"),
        ])


@pytest.mark.parametrize(
    ("duplicate_field", "error_code"),
    [
        ("claim", "duplicate_claim_id"),
        ("evidence", "duplicate_evidence_id"),
    ],
)
def test_active_state_review_authority_rejects_duplicate_binding_ids(
    duplicate_field,
    error_code,
):
    second_claim_id = (
        "clm_sem_0001" if duplicate_field == "claim" else "clm_sem_0002"
    )
    second_evidence_id = (
        "ev_sem_0001" if duplicate_field == "evidence" else "ev_sem_0002"
    )
    with pytest.raises(ValueError, match=error_code):
        build_active_state_review_authority([
            _binding(),
            _binding(
                candidate=_candidate(id="candidate_second"),
                claim_id=second_claim_id,
                evidence_id=second_evidence_id,
            ),
        ])


def test_active_state_review_authority_rejects_state_binding_mismatch():
    bindings = [_binding()]
    authority = build_active_state_review_authority(bindings)
    state, evidence_rows = _state_and_evidence(bindings)
    state["claims"][0]["excerpt"] = "DECISION: unsigned replacement."

    with pytest.raises(ValueError, match="claim_mismatch"):
        project_active_state_review_candidates(authority, state, evidence_rows)


def test_active_state_review_authority_rejects_non_string_source_identity():
    bindings = [_binding()]
    authority = build_active_state_review_authority(bindings)
    state, evidence_rows = _state_and_evidence(bindings)
    state["claims"][0]["source_id"] = []
    evidence_rows[0]["source_id"] = []

    with pytest.raises(ValueError, match="claim_mismatch"):
        project_active_state_review_candidates(authority, state, evidence_rows)


def test_active_state_review_authority_rejects_evidence_binding_mismatch():
    bindings = [_binding()]
    authority = build_active_state_review_authority(bindings)
    state, evidence_rows = _state_and_evidence(bindings)
    evidence_rows[0]["excerpt_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="evidence_mismatch"):
        project_active_state_review_candidates(authority, state, evidence_rows)


def test_active_state_review_authority_rejects_source_binding_mismatch():
    bindings = [_binding()]
    authority = build_active_state_review_authority(bindings)
    state, evidence_rows = _state_and_evidence(bindings)
    state["sources"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source_mismatch"):
        project_active_state_review_candidates(authority, state, evidence_rows)


def test_active_state_review_authority_rejects_digest_tamper():
    bindings = [_binding()]
    authority = build_active_state_review_authority(bindings)
    state, evidence_rows = _state_and_evidence(bindings)
    authority["bindings"][0]["candidate"]["reviewed_by"] = "mallory"

    with pytest.raises(ValueError, match="digest_mismatch"):
        project_active_state_review_candidates(authority, state, evidence_rows)
