"""Signed semantic-review authority for active-state learning."""

from collections import Counter
from collections.abc import Iterable
import hashlib
import json

from morpheus.core.models import Claim, Evidence
from morpheus.core.semantic.classifier import classify_candidate
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.routing import ROUTING_POLICY_VERSION, route_candidate


ACTIVE_STATE_REVIEW_AUTHORITY_SCHEMA = (
    "morpheus-active-state-review-authority/1"
)
ACTIVE_STATE_REVIEW_AUTHORITY_PRODUCER = "semantic_review_apply"
_AUTHORITY_KEYS = {
    "schema",
    "producer",
    "routing_policy_version",
    "binding_count",
    "bindings",
    "sha256",
}
_BINDING_KEYS = {"claim_id", "evidence_id", "candidate"}
_ROUTE_FIELDS = (
    "semantic_class",
    "trainability_status",
    "trainability_reason",
    "memory_route",
)


def build_active_state_review_authority(
    bindings: Iterable[tuple[Claim, Evidence, SemanticCandidate]],
) -> dict:
    """Build the canonical authority signed by a semantic review-apply receipt."""
    records = []
    for claim, evidence, candidate in bindings:
        candidate = _validated_candidate(candidate)
        _validate_claim_evidence(candidate, claim.model_dump(), evidence.model_dump())
        records.append({
            "claim_id": claim.id,
            "evidence_id": evidence.id,
            "candidate": candidate.model_dump(mode="json"),
        })
    records = sorted(records, key=_binding_sort_key)
    _reject_duplicate_binding_ids(records)
    payload = {
        "schema": ACTIVE_STATE_REVIEW_AUTHORITY_SCHEMA,
        "producer": ACTIVE_STATE_REVIEW_AUTHORITY_PRODUCER,
        "routing_policy_version": ROUTING_POLICY_VERSION,
        "binding_count": len(records),
        "bindings": records,
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def project_active_state_review_candidates(
    authority: object,
    state: object,
    evidence_rows: object,
) -> list[SemanticCandidate]:
    """Validate signed bindings against state artifacts and return candidates."""
    records, candidates = _validated_authority(authority)
    if not isinstance(state, dict):
        _fail("active_state_review_authority_state_invalid", "state must be an object")
    claims = state.get("claims")
    sources = state.get("sources")
    if not isinstance(claims, list) or any(not isinstance(item, dict) for item in claims):
        _fail("active_state_review_authority_state_invalid", "claims must be objects")
    if not isinstance(sources, list) or any(not isinstance(item, dict) for item in sources):
        _fail("active_state_review_authority_state_invalid", "sources must be objects")
    if not isinstance(evidence_rows, list) or any(
        not isinstance(item, dict) for item in evidence_rows
    ):
        _fail(
            "active_state_review_authority_evidence_invalid",
            "evidence rows must be objects",
        )

    claim_by_id = _unique_records_by_id(
        claims,
        "active_state_review_authority_duplicate_state_claim_id",
    )
    evidence_by_id = _unique_records_by_id(
        evidence_rows,
        "active_state_review_authority_duplicate_state_evidence_id",
    )
    source_by_id = _unique_records_by_id(
        sources,
        "active_state_review_authority_duplicate_state_source_id",
    )
    for record, candidate in zip(records, candidates):
        claim = claim_by_id.get(record["claim_id"])
        if claim is None:
            _fail(
                "active_state_review_authority_claim_missing",
                record["claim_id"],
            )
        evidence = evidence_by_id.get(record["evidence_id"])
        if evidence is None:
            _fail(
                "active_state_review_authority_evidence_missing",
                record["evidence_id"],
            )
        _validate_claim_evidence(candidate, claim, evidence)
        source_id = claim.get("source_id")
        source = source_by_id.get(source_id)
        if (
            source is None
            or source.get("path") != candidate.source_path
            or source.get("sha256") != candidate.source_sha256
        ):
            _fail(
                "active_state_review_authority_source_mismatch",
                candidate.id,
            )
    return candidates


def active_state_review_authority_summary(authority: object) -> dict:
    """Return the dataset-provenance identity of one validated authority."""
    _validated_authority(authority)
    return {
        "schema": authority["schema"],
        "routing_policy_version": authority["routing_policy_version"],
        "binding_count": authority["binding_count"],
        "sha256": authority["sha256"],
    }


def claim_category_for_candidate_kind(kind: str) -> str:
    """Map a semantic candidate kind to its compiled state claim category."""
    return {
        "active_decision": "decision",
        "open_task": "task",
        "outdated_claim": "outdated",
        "agent_rule": "agent_rule",
        "source_reference": "source_reference",
    }.get(kind, "note")


def _validated_authority(
    authority: object,
) -> tuple[list[dict], list[SemanticCandidate]]:
    if not isinstance(authority, dict):
        _fail("active_state_review_authority_invalid", "authority must be an object")
    if set(authority) != _AUTHORITY_KEYS:
        _fail("active_state_review_authority_invalid", "authority fields are invalid")
    payload = {key: value for key, value in authority.items() if key != "sha256"}
    if authority.get("sha256") != _canonical_sha256(payload):
        _fail("active_state_review_authority_digest_mismatch", "sha256 mismatch")
    if authority.get("schema") != ACTIVE_STATE_REVIEW_AUTHORITY_SCHEMA:
        _fail("active_state_review_authority_schema_invalid", "unsupported schema")
    if authority.get("producer") != ACTIVE_STATE_REVIEW_AUTHORITY_PRODUCER:
        _fail("active_state_review_authority_producer_invalid", "unsupported producer")
    if authority.get("routing_policy_version") != ROUTING_POLICY_VERSION:
        _fail("active_state_review_authority_policy_mismatch", "routing policy changed")
    records = authority.get("bindings")
    count = authority.get("binding_count")
    if (
        not isinstance(records, list)
        or type(count) is not int
        or count < 0
        or count != len(records)
    ):
        _fail("active_state_review_authority_count_mismatch", "binding count mismatch")
    if any(not isinstance(record, dict) or set(record) != _BINDING_KEYS for record in records):
        _fail("active_state_review_authority_binding_invalid", "binding fields are invalid")
    if records != sorted(records, key=_binding_sort_key):
        _fail("active_state_review_authority_order_invalid", "bindings are not canonical")
    _reject_duplicate_binding_ids(records)

    candidates = []
    for record in records:
        if (
            not isinstance(record.get("claim_id"), str)
            or not record["claim_id"]
            or not isinstance(record.get("evidence_id"), str)
            or not record["evidence_id"]
            or not isinstance(record.get("candidate"), dict)
        ):
            _fail("active_state_review_authority_binding_invalid", "binding identity invalid")
        try:
            candidate = SemanticCandidate.model_validate(record["candidate"])
        except ValueError as exc:
            _fail("active_state_review_authority_candidate_invalid", str(exc))
        candidate = _validated_candidate(candidate)
        if record["candidate"] != candidate.model_dump(mode="json"):
            _fail(
                "active_state_review_authority_candidate_noncanonical",
                candidate.id,
            )
        candidates.append(candidate)
    return records, candidates


def _validated_candidate(candidate: SemanticCandidate) -> SemanticCandidate:
    if candidate.status != "accepted" or candidate.label != "source_backed":
        _fail(
            "active_state_review_authority_candidate_unaccepted",
            candidate.id,
        )
    if not isinstance(candidate.reviewed_by, str) or not candidate.reviewed_by.strip():
        _fail("active_state_review_authority_reviewer_missing", candidate.id)
    if candidate.reviewed_at is None:
        _fail("active_state_review_authority_review_time_missing", candidate.id)
    semantic_class = classify_candidate(candidate)
    if candidate.semantic_class != semantic_class:
        _fail(
            "active_state_review_authority_semantic_class_mismatch",
            candidate.id,
        )
    canonical = route_candidate(candidate.model_copy(update={
        "semantic_class": semantic_class,
    }))
    if any(
        getattr(candidate, field) != getattr(canonical, field)
        for field in _ROUTE_FIELDS
    ):
        _fail("active_state_review_authority_route_mismatch", candidate.id)
    return candidate


def _validate_claim_evidence(
    candidate: SemanticCandidate,
    claim: dict,
    evidence: dict,
) -> None:
    claim_source_id = claim.get("source_id")
    evidence_source_id = evidence.get("source_id")
    claim_matches = bool(
        claim.get("id") == evidence.get("claim_id")
        and isinstance(claim_source_id, str)
        and bool(claim_source_id)
        and isinstance(evidence_source_id, str)
        and claim_source_id == evidence_source_id
        and claim.get("line_start") == candidate.line_start
        and claim.get("line_end") == candidate.line_end
        and claim.get("excerpt") == candidate.claim
        and claim.get("status") == "active"
        and claim.get("category") == claim_category_for_candidate_kind(candidate.kind)
        and claim.get("inference") is False
    )
    if not claim_matches:
        _fail("active_state_review_authority_claim_mismatch", candidate.id)
    evidence_matches = bool(
        evidence.get("path") == candidate.source_path
        and evidence.get("line_start") == candidate.line_start
        and evidence.get("line_end") == candidate.line_end
        and evidence.get("excerpt") == candidate.evidence_excerpt
        and evidence.get("source_sha256") == candidate.source_sha256
        and evidence.get("excerpt_sha256") == candidate.evidence_sha256
    )
    if not evidence_matches:
        _fail("active_state_review_authority_evidence_mismatch", candidate.id)


def _reject_duplicate_binding_ids(records: list[dict]) -> None:
    for key, code in (
        ("claim_id", "active_state_review_authority_duplicate_claim_id"),
        ("evidence_id", "active_state_review_authority_duplicate_evidence_id"),
    ):
        duplicates = _duplicates(record.get(key) for record in records)
        if duplicates:
            _fail(code, ", ".join(duplicates))
    duplicates = _duplicates(
        record.get("candidate", {}).get("id")
        if isinstance(record.get("candidate"), dict)
        else None
        for record in records
    )
    if duplicates:
        _fail(
            "active_state_review_authority_duplicate_candidate_id",
            ", ".join(duplicates),
        )


def _unique_records_by_id(records: list[dict], duplicate_code: str) -> dict:
    ids = [record.get("id") for record in records]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        _fail("active_state_review_authority_state_identity_invalid", "missing id")
    duplicates = _duplicates(ids)
    if duplicates:
        _fail(duplicate_code, ", ".join(duplicates))
    return {record["id"]: record for record in records}


def _duplicates(values: Iterable[object]) -> list[str]:
    counts = Counter(value for value in values if isinstance(value, str))
    return sorted(value for value, count in counts.items() if count > 1)


def _binding_sort_key(record: dict) -> tuple[str, str, str]:
    candidate = record.get("candidate")
    candidate_id = candidate.get("id", "") if isinstance(candidate, dict) else ""
    return (
        str(candidate_id),
        str(record.get("claim_id", "")),
        str(record.get("evidence_id", "")),
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _fail(code: str, detail: str) -> None:
    raise ValueError(f"{code}: {detail}")
