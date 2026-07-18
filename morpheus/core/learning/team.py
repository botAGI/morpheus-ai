"""Local, review-gated orchestration for team-learning inputs."""

import base64
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, field_validator

from morpheus.core.learning.safety import contains_secret_like_text
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.semantic.routing import route_candidate
from morpheus.core.semantic.verifier import verify_candidate_span


TEAM_LOOP_POLICY_VERSION = "morpheus-team-learning/2"
TEAM_FEEDBACK_PROMPT_SHA256 = hashlib.sha256(b"morpheus-team-feedback-v1").hexdigest()
CHECK_CORRECTION_PROMPT_SHA256 = hashlib.sha256(
    b"morpheus-check-training-correction-v1"
).hexdigest()
TEAM_FEEDBACK_PROVIDER = "morpheus-team-loop"
CHECK_CORRECTION_PROVIDER = "morpheus-check"
TEAM_LEARNING_SOURCE_TYPES = (
    "pr_comment",
    "rejected_agent_claim",
    "human_correction",
    "accepted_review_candidate",
    "check_result",
    "stale_claim_correction",
)
TeamFeedbackSource = Literal[
    "pr_comment",
    "rejected_agent_claim",
    "human_correction",
]
CheckResultStatus = Literal["verified", "stale", "incorrect", "unknown"]
CHECK_CORRECTION_STATUSES = {"stale", "incorrect"}
TEAM_TRANSACTION_SCHEMA = "morpheus-team-learning-transaction/1"
CHECK_MARKER_RE = re.compile(r"^(TODO|DECISION|FIXME|NOTE|HACK|XXX):\s*", re.IGNORECASE)
CHECK_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")


class TeamFeedbackEvent(BaseModel):
    """Compatibility contract for the original three direct feedback sources."""

    model_config = ConfigDict(extra="forbid")

    source_type: TeamFeedbackSource
    external_id: str = Field(min_length=1, max_length=240)
    claim: str = Field(min_length=1, max_length=4000)
    correction: str | None = Field(default=None, max_length=4000)
    author: str | None = Field(default=None, max_length=240)
    url: str | None = Field(default=None, max_length=2000)

    @field_validator("external_id", "claim")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("correction", "author", "url")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class StaleClaimCorrectionEvent(BaseModel):
    """Explicit reviewed-input proposal for a stale claim correction."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["stale_claim_correction"]
    external_id: str = Field(min_length=1, max_length=240)
    claim: str = Field(min_length=1, max_length=4000)
    correction: str = Field(min_length=1, max_length=4000)
    author: str | None = Field(default=None, max_length=240)
    url: str | None = Field(default=None, max_length=2000)

    @field_validator("external_id", "claim", "correction")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("author", "url")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class AcceptedReviewCandidateEvent(BaseModel):
    """Reference to authority that was already accepted by the review flow."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["accepted_review_candidate"]
    candidate_id: str = Field(min_length=1, max_length=240)
    candidate_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator("candidate_id")
    @classmethod
    def strip_candidate_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class CheckEvidence(BaseModel):
    """Minimal source label required to preserve check-correction identity."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=2000)
    line_start: StrictInt = Field(ge=1)

    @field_validator("path")
    @classmethod
    def strip_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        if "\n" in value or "\r" in value:
            raise ValueError("value must be a single line")
        return value


class CheckResultEvent(BaseModel):
    """One stable result from a local Morpheus check run."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["check_result"]
    claim: str = Field(min_length=1, max_length=4000)
    status: CheckResultStatus
    reason: str = Field(min_length=1, max_length=4000)
    evidence: CheckEvidence | None = None
    active_state_receipt: str | None = Field(default=None, max_length=240)
    input_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )

    @field_validator("claim", "reason")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        if "\n" in value or "\r" in value:
            raise ValueError("value must be a single line")
        return value

    @field_validator("active_state_receipt", "input_hash")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


DirectFeedbackEvent = TeamFeedbackEvent | StaleClaimCorrectionEvent
TeamLearningInput = Annotated[
    TeamFeedbackEvent
    | StaleClaimCorrectionEvent
    | AcceptedReviewCandidateEvent
    | CheckResultEvent,
    Field(discriminator="source_type"),
]
_TEAM_LEARNING_INPUTS = TypeAdapter(list[TeamLearningInput])
_DIRECT_FEEDBACK_INPUT = TypeAdapter(Annotated[
    TeamFeedbackEvent | StaleClaimCorrectionEvent,
    Field(discriminator="source_type"),
])


@dataclass(frozen=True)
class _InputReceiptPlan:
    receipt_id: str
    artifact_line: str
    artifact_path: Path


@dataclass(frozen=True)
class _FeedbackCandidatePlan:
    event: DirectFeedbackEvent
    candidate_id: str
    artifact_line: str
    artifact_path: Path


@dataclass(frozen=True)
class _CheckCandidatePlan:
    event: CheckResultEvent
    candidate_id: str
    artifact_line: str
    artifact_path: Path
    source_label: str


CandidatePlan = _FeedbackCandidatePlan | _CheckCandidatePlan


@dataclass(frozen=True)
class _TransactionEntry:
    path: Path
    before: bytes | None
    after: bytes | None


def parse_team_learning_items(items: list[dict]) -> list[TeamLearningInput]:
    """Validate and secret-screen a complete mixed batch before local writes."""
    events = _TEAM_LEARNING_INPUTS.validate_python(items)
    for event in events:
        if isinstance(event, (TeamFeedbackEvent, StaleClaimCorrectionEvent)):
            values = (
                event.external_id,
                event.claim,
                event.correction,
                event.author,
                event.url,
            )
        elif isinstance(event, CheckResultEvent):
            values = (
                event.claim,
                event.reason,
                event.evidence.path if event.evidence is not None else None,
                event.active_state_receipt,
            )
        else:
            values = (event.candidate_id,)
        if any(value and contains_secret_like_text(value) for value in values):
            raise ValueError(
                f"Team learning input {event.source_type!r} contains secret-like content"
            )
    return events


def parse_team_feedback_items(items: list[dict]) -> list[TeamFeedbackEvent]:
    """Validate the original three-source feedback contract for compatibility."""
    events = parse_team_learning_items(items)
    if not all(isinstance(event, TeamFeedbackEvent) for event in events):
        raise ValueError("Expected only direct team feedback events")
    return [event for event in events if isinstance(event, TeamFeedbackEvent)]


def team_feedback_projection_error(candidate: SemanticCandidate) -> str | None:
    """Return a stable blocker when a team candidate diverges from its event span."""
    if candidate.provider.get("name") != TEAM_FEEDBACK_PROVIDER:
        return None
    try:
        event = _DIRECT_FEEDBACK_INPUT.validate_json(candidate.evidence_excerpt)
    except ValueError:
        return "team_feedback_projection_mismatch"
    _, expected_id, expected_line = _prepare_feedback_event(event)
    expected_source_path = f".morpheus/review/team_feedback/{expected_id}.jsonl"
    expected_provider = {
        "name": TEAM_FEEDBACK_PROVIDER,
        "model": "local",
        "feedback_source": event.source_type,
        "external_id": event.external_id,
        "author": event.author,
        "url": event.url,
    }
    expected_evidence_sha256 = hashlib.sha256(expected_line.encode()).hexdigest()
    if any([
        candidate.id != expected_id,
        candidate.kind != "outdated_claim",
        candidate.claim != event.claim,
        candidate.correction_text != event.correction,
        candidate.source_path != expected_source_path,
        candidate.source_revision != f"feedback:{event.source_type}:{event.external_id}",
        candidate.line_start != 1,
        candidate.line_end != 1,
        candidate.evidence_excerpt != expected_line,
        candidate.evidence_sha256 != expected_evidence_sha256,
        candidate.provider != expected_provider,
        candidate.prompt_sha256 != TEAM_FEEDBACK_PROMPT_SHA256,
    ]):
        return "team_feedback_projection_mismatch"
    return None


def check_correction_projection_error(candidate: SemanticCandidate) -> str | None:
    """Return a stable blocker when a check candidate diverges from its artifact."""
    if candidate.provider.get("name") != CHECK_CORRECTION_PROVIDER:
        return None
    source_label = candidate.provider.get("source_label")
    if not isinstance(source_label, str) or not source_label.strip():
        return "check_correction_projection_mismatch"
    source_label = source_label.strip()
    expected_id = _check_correction_id(candidate.claim, source_label)
    expected_source_path = (
        Path(".morpheus")
        / "review"
        / "check_corrections"
        / f"{expected_id}.md"
    ).as_posix()
    expected_provider = {
        "name": CHECK_CORRECTION_PROVIDER,
        "model": "local",
        "source_label": source_label,
    }
    if any([
        candidate.id != expected_id,
        candidate.kind != "outdated_claim",
        candidate.correction_text is not None,
        candidate.source_path != expected_source_path,
        not candidate.source_revision.startswith("check:"),
        candidate.line_start != 1,
        candidate.line_end != 1,
        candidate.label != "source_backed",
        not _valid_check_artifact_projection(
            candidate.evidence_excerpt,
            claim=candidate.claim,
            source_label=source_label,
        ),
        candidate.evidence_sha256
        != hashlib.sha256(candidate.evidence_excerpt.encode()).hexdigest(),
        candidate.provider != expected_provider,
        candidate.prompt_sha256 != CHECK_CORRECTION_PROMPT_SHA256,
    ]):
        return "check_correction_projection_mismatch"
    return None


def reviewed_input_projection_error(candidate: SemanticCandidate) -> str | None:
    """Return the source-specific projection blocker for reviewed learning input."""
    return (
        team_feedback_projection_error(candidate)
        or check_correction_projection_error(candidate)
    )


def run_team_learning_loop(
    project_root: Path,
    items: list[dict] | None = None,
) -> dict:
    """Ingest six reviewed-input sources and write an audit-only loop report."""
    project_root = _safe_project_root(project_root)
    events = parse_team_learning_items(items or [])
    store = ReviewStore(project_root)
    with store.transaction():
        return _run_team_learning_loop_locked(project_root, store, events)


def _run_team_learning_loop_locked(
    project_root: Path,
    store: ReviewStore,
    events: list[TeamLearningInput],
) -> dict:
    """Preflight and commit one mixed batch under the shared review-store lock."""
    recover_team_learning_transaction(project_root)
    existing = store.load_candidates()
    _reject_duplicate_candidate_ids(existing)
    existing_by_id = {candidate.id: candidate for candidate in existing}

    accepted_by_index: dict[int, SemanticCandidate] = {}
    reconciled_ids: list[str] = []
    for index, event in enumerate(events):
        if not isinstance(event, AcceptedReviewCandidateEvent):
            continue
        candidate = existing_by_id.get(event.candidate_id)
        if candidate is None:
            raise ValueError(
                f"Accepted review candidate not found: {event.candidate_id}"
            )
        _validate_accepted_candidate(project_root, candidate)
        current_sha256 = _candidate_sha256(candidate)
        if (
            event.candidate_sha256 is not None
            and event.candidate_sha256 != current_sha256
        ):
            raise ValueError(
                f"Accepted review candidate digest mismatch: {event.candidate_id}"
            )
        accepted_by_index[index] = candidate
        reconciled_ids.append(candidate.id)

    input_dir = store.review_dir / "team_inputs"
    receipt_plans: list[_InputReceiptPlan] = []
    created_receipt_ids: list[str] = []
    existing_receipt_ids: list[str] = []
    planned_receipt_lines: dict[str, str] = {}
    if events:
        _preflight_directory(input_dir, "Team input receipt path")
    for index, event in enumerate(events):
        receipt_id, artifact_line = _prepare_input_receipt(
            event,
            accepted_candidate=accepted_by_index.get(index),
        )
        artifact_path = input_dir / f"{receipt_id}.jsonl"
        planned_line = planned_receipt_lines.get(receipt_id)
        if planned_line is not None:
            if planned_line != artifact_line:
                raise ValueError(f"Team input receipt id collision: {receipt_id}")
            existing_receipt_ids.append(receipt_id)
            continue
        if artifact_path.exists():
            _verify_exact_artifact(
                project_root,
                artifact_path,
                artifact_line,
                "Team input receipt",
            )
            planned_receipt_lines[receipt_id] = artifact_line
            existing_receipt_ids.append(receipt_id)
            continue
        _preflight_exact_artifact(
            project_root,
            artifact_path,
            artifact_line,
            "Team input receipt",
        )
        planned_receipt_lines[receipt_id] = artifact_line
        created_receipt_ids.append(receipt_id)
        receipt_plans.append(_InputReceiptPlan(
            receipt_id=receipt_id,
            artifact_line=artifact_line,
            artifact_path=artifact_path,
        ))

    feedback_dir = store.review_dir / "team_feedback"
    check_dir = store.review_dir / "check_corrections"
    candidate_plans: list[CandidatePlan] = []
    existing_candidate_ids: list[str] = []
    planned_feedback_lines: dict[str, str] = {}
    planned_check_keys: dict[str, tuple[str, str]] = {}
    if any(isinstance(event, (TeamFeedbackEvent, StaleClaimCorrectionEvent)) for event in events):
        _preflight_directory(feedback_dir, "Team feedback path")
    if any(
        isinstance(event, CheckResultEvent) and event.status in CHECK_CORRECTION_STATUSES
        for event in events
    ):
        _preflight_directory(check_dir, "Check corrections path")

    for event in events:
        if isinstance(event, (TeamFeedbackEvent, StaleClaimCorrectionEvent)):
            _, candidate_id, artifact_line = _prepare_feedback_event(event)
            artifact_path = feedback_dir / f"{candidate_id}.jsonl"
            current = existing_by_id.get(candidate_id)
            if current is not None:
                if (
                    current.provider.get("name") != TEAM_FEEDBACK_PROVIDER
                    or team_feedback_projection_error(current)
                ):
                    raise ValueError(
                        f"Team feedback candidate projection mismatch: {candidate_id}"
                    )
                artifact_bytes = _verify_exact_artifact(
                    project_root,
                    artifact_path,
                    artifact_line,
                    "Team feedback artifact",
                )
                if current.source_sha256 != hashlib.sha256(artifact_bytes).hexdigest():
                    raise ValueError(
                        f"Team feedback candidate source hash mismatch: {candidate_id}"
                    )
                existing_candidate_ids.append(candidate_id)
                continue
            planned_line = planned_feedback_lines.get(candidate_id)
            if planned_line is not None:
                if planned_line != artifact_line:
                    raise ValueError(f"Team feedback id collision: {candidate_id}")
                existing_candidate_ids.append(candidate_id)
                continue
            _preflight_exact_artifact(
                project_root,
                artifact_path,
                artifact_line,
                "Team feedback artifact",
            )
            planned_feedback_lines[candidate_id] = artifact_line
            candidate_plans.append(_FeedbackCandidatePlan(
                event=event,
                candidate_id=candidate_id,
                artifact_line=artifact_line,
                artifact_path=artifact_path,
            ))
            continue

        if not isinstance(event, CheckResultEvent):
            continue
        if event.status not in CHECK_CORRECTION_STATUSES:
            continue
        candidate_id, artifact_line, source_label = _prepare_check_result(event)
        artifact_path = check_dir / f"{candidate_id}.md"
        current = existing_by_id.get(candidate_id)
        if current is not None:
            _verify_existing_check_candidate(
                current,
                project_root=project_root,
                event=event,
                source_label=source_label,
                artifact_path=artifact_path,
            )
            existing_candidate_ids.append(candidate_id)
            continue
        projection_key = (_normalize_check_claim(event.claim), source_label)
        planned_key = planned_check_keys.get(candidate_id)
        if planned_key is not None:
            if planned_key != projection_key:
                raise ValueError(f"Check correction id collision: {candidate_id}")
            existing_candidate_ids.append(candidate_id)
            continue
        _preflight_exact_artifact(
            project_root,
            artifact_path,
            artifact_line,
            "Check correction artifact",
        )
        planned_check_keys[candidate_id] = projection_key
        candidate_plans.append(_CheckCandidatePlan(
            event=event,
            candidate_id=candidate_id,
            artifact_line=artifact_line,
            artifact_path=artifact_path,
            source_label=source_label,
        ))

    _preflight_team_report(project_root)
    if receipt_plans:
        _secure_ensure_project_directory(
            project_root,
            input_dir,
            "Team input receipt path",
        )
    if any(isinstance(plan, _FeedbackCandidatePlan) for plan in candidate_plans):
        _secure_ensure_project_directory(
            project_root,
            feedback_dir,
            "Team feedback path",
        )
    if any(isinstance(plan, _CheckCandidatePlan) for plan in candidate_plans):
        _secure_ensure_project_directory(
            project_root,
            check_dir,
            "Check corrections path",
        )
    learning_dir = project_root / ".morpheus" / "learning"
    _secure_ensure_project_directory(
        project_root,
        learning_dir,
        "Team learning report directory",
    )

    timestamp = datetime.now(timezone.utc)
    timestamp_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    created: list[SemanticCandidate] = []
    for plan in candidate_plans:
        if isinstance(plan, _FeedbackCandidatePlan):
            candidate = _feedback_candidate(plan, timestamp, timestamp_id, project_root)
        else:
            candidate = _check_candidate(plan, timestamp, timestamp_id, project_root)
        created.append(candidate)
        existing_by_id[candidate.id] = candidate

    candidates = [*existing, *created]
    report = _team_loop_report(
        project_root,
        candidates,
        events=events,
        created=created,
        existing_candidate_ids=existing_candidate_ids,
        created_receipt_ids=created_receipt_ids,
        existing_receipt_ids=existing_receipt_ids,
        reconciled_ids=reconciled_ids,
    )
    json_path = learning_dir / "team_loop_report.json"
    markdown_path = learning_dir / "team_loop_report.md"
    writes: list[tuple[Path, bytes]] = []
    immutable_paths: set[Path] = set()
    for plan in receipt_plans:
        writes.append((plan.artifact_path, (plan.artifact_line + "\n").encode()))
        immutable_paths.add(plan.artifact_path)
    for plan in candidate_plans:
        writes.append((plan.artifact_path, (plan.artifact_line + "\n").encode()))
        immutable_paths.add(plan.artifact_path)
    if created:
        writes.append((store.candidates_path, _candidate_store_bytes(candidates)))
    writes.extend([
        (json_path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()),
        (markdown_path, _render_team_loop_report(report).encode()),
    ])
    _commit_team_learning_transaction(
        project_root,
        writes,
        immutable_paths=immutable_paths,
    )
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "report": report,
    }


def _prepare_feedback_event(
    event: DirectFeedbackEvent,
) -> tuple[DirectFeedbackEvent, str, str]:
    artifact_line = _canonical_json(event.model_dump(mode="json", exclude_none=True))
    digest = hashlib.sha256(artifact_line.encode()).hexdigest()
    return event, f"teamfb_{digest[:24]}", artifact_line


def _prepare_input_receipt(
    event: TeamLearningInput,
    *,
    accepted_candidate: SemanticCandidate | None,
) -> tuple[str, str]:
    payload = event.model_dump(mode="json", exclude_none=True)
    if isinstance(event, AcceptedReviewCandidateEvent):
        if accepted_candidate is None:
            raise ValueError(
                f"Accepted review candidate not resolved: {event.candidate_id}"
            )
        payload["candidate_sha256"] = _candidate_sha256(accepted_candidate)
    artifact_line = _canonical_json(payload)
    digest = hashlib.sha256(artifact_line.encode()).hexdigest()
    return f"teamin_{digest[:24]}", artifact_line


def _prepare_check_result(event: CheckResultEvent) -> tuple[str, str, str]:
    source_label = _check_source_label(event.evidence)
    candidate_id = _check_correction_id(event.claim, source_label)
    artifact_line = (
        f"Correction candidate: {event.status} claim {json.dumps(event.claim)} "
        f"was flagged by morpheus check because {event.reason}. "
        f"Source: {source_label}."
    )
    return candidate_id, artifact_line, source_label


def _feedback_candidate(
    plan: _FeedbackCandidatePlan,
    timestamp: datetime,
    timestamp_id: str,
    project_root: Path,
) -> SemanticCandidate:
    event = plan.event
    return route_candidate(SemanticCandidate(
        id=plan.candidate_id,
        run_id=f"teamloop_{timestamp_id}",
        kind="outdated_claim",
        claim=event.claim,
        correction_text=event.correction,
        source_path=plan.artifact_path.relative_to(project_root).as_posix(),
        source_sha256=hashlib.sha256((plan.artifact_line + "\n").encode()).hexdigest(),
        source_mtime=timestamp,
        source_revision=f"feedback:{event.source_type}:{event.external_id}",
        line_start=1,
        line_end=1,
        evidence_excerpt=plan.artifact_line,
        evidence_sha256=hashlib.sha256(plan.artifact_line.encode()).hexdigest(),
        confidence=1.0,
        label="source_backed",
        status="pending",
        created_at=timestamp,
        provider={
            "name": TEAM_FEEDBACK_PROVIDER,
            "model": "local",
            "feedback_source": event.source_type,
            "external_id": event.external_id,
            "author": event.author,
            "url": event.url,
        },
        prompt_sha256=TEAM_FEEDBACK_PROMPT_SHA256,
    ))


def _check_candidate(
    plan: _CheckCandidatePlan,
    timestamp: datetime,
    timestamp_id: str,
    project_root: Path,
) -> SemanticCandidate:
    event = plan.event
    return route_candidate(SemanticCandidate(
        id=plan.candidate_id,
        run_id=f"check_{timestamp_id}",
        kind="outdated_claim",
        claim=event.claim,
        source_path=plan.artifact_path.relative_to(project_root).as_posix(),
        source_sha256=hashlib.sha256((plan.artifact_line + "\n").encode()).hexdigest(),
        source_mtime=timestamp,
        source_revision=f"check:{event.active_state_receipt or 'unknown'}",
        line_start=1,
        line_end=1,
        evidence_excerpt=plan.artifact_line,
        evidence_sha256=hashlib.sha256(plan.artifact_line.encode()).hexdigest(),
        confidence=1.0,
        label="source_backed",
        status="pending",
        created_at=timestamp,
        provider={
            "name": CHECK_CORRECTION_PROVIDER,
            "model": "local",
            "source_label": plan.source_label,
        },
        prompt_sha256=CHECK_CORRECTION_PROMPT_SHA256,
    ))


def _validate_accepted_candidate(
    project_root: Path,
    candidate: SemanticCandidate,
) -> None:
    if candidate.status != "accepted":
        raise ValueError(
            f"Review candidate is not accepted: {candidate.id} ({candidate.status})"
        )
    if not candidate.reviewed_by or candidate.reviewed_at is None:
        raise ValueError(f"Accepted review candidate lacks review authority: {candidate.id}")
    if candidate.label != "source_backed":
        raise ValueError(
            f"Accepted review candidate is not source-backed: {candidate.id}"
        )
    if any(contains_secret_like_text(value) for value in (
        candidate.claim,
        candidate.evidence_excerpt,
        candidate.correction_text or "",
    )):
        raise ValueError(
            f"Accepted review candidate contains secret-like content: {candidate.id}"
        )
    relative_path = Path(candidate.source_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"Accepted review candidate source path is unsafe: {candidate.id}"
        )
    source_path = project_root / relative_path
    reject_symlink_paths([source_path], "Accepted review candidate source")
    reject_symlink_components(source_path, "Accepted review candidate source")
    if not source_path.is_file():
        raise ValueError(
            f"Accepted review candidate source is missing: {candidate.id}"
        )
    if verify_candidate_span(project_root, candidate).label != "source_backed":
        raise ValueError(
            f"Accepted review candidate has no live source span: {candidate.id}"
        )
    projection_error = reviewed_input_projection_error(candidate)
    if projection_error:
        raise ValueError(
            f"Accepted review candidate projection mismatch: {candidate.id}"
        )


def _verify_existing_check_candidate(
    candidate: SemanticCandidate,
    *,
    project_root: Path,
    event: CheckResultEvent,
    source_label: str,
    artifact_path: Path,
) -> None:
    if any([
        candidate.provider.get("name") != CHECK_CORRECTION_PROVIDER,
        check_correction_projection_error(candidate) is not None,
        _normalize_check_claim(candidate.claim) != _normalize_check_claim(event.claim),
        candidate.provider.get("source_label") != source_label,
    ]):
        raise ValueError(
            f"Check correction candidate projection mismatch: {candidate.id}"
        )
    artifact_bytes = _verify_exact_artifact(
        project_root,
        artifact_path,
        candidate.evidence_excerpt,
        "Check correction artifact",
    )
    if candidate.source_sha256 != hashlib.sha256(artifact_bytes).hexdigest():
        raise ValueError(
            f"Check correction candidate projection mismatch: {candidate.id}"
        )


def _reject_duplicate_candidate_ids(candidates: list[SemanticCandidate]) -> None:
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(candidate.id for candidate in candidates).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "Review store contains duplicate candidate ids: " + ", ".join(duplicate_ids)
        )


def _candidate_sha256(candidate: SemanticCandidate) -> str:
    payload = candidate.model_dump(mode="json")
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _check_source_label(evidence: CheckEvidence | None) -> str:
    if evidence is None:
        return "no source span"
    return f"{evidence.path}:{evidence.line_start}"


def _check_correction_id(claim: str, source_label: str) -> str:
    key = hashlib.sha256(
        f"{_normalize_check_claim(claim)}\n{source_label}".encode()
    ).hexdigest()
    return f"corr_{key[:24]}"


def _valid_check_artifact_projection(
    artifact_line: str,
    *,
    claim: str,
    source_label: str,
) -> bool:
    suffix = f". Source: {source_label}."
    marker = " was flagged by morpheus check because "
    if "\n" in artifact_line or "\r" in artifact_line or not artifact_line.endswith(suffix):
        return False
    body = artifact_line[: -len(suffix)]
    for status in sorted(CHECK_CORRECTION_STATUSES):
        prefix = f"Correction candidate: {status} claim "
        if not body.startswith(prefix):
            continue
        projection = body[len(prefix):]
        try:
            projected_claim, claim_end = json.JSONDecoder().raw_decode(projection)
        except json.JSONDecodeError:
            return False
        remainder = projection[claim_end:]
        if not remainder.startswith(marker):
            return False
        reason = remainder[len(marker):]
        return projected_claim == claim and bool(reason.strip())
    return False


def _normalize_check_claim(text: str) -> str:
    text = CHECK_MARKER_RE.sub("", text).strip().casefold()
    return " ".join(CHECK_WORD_RE.findall(text))


def _canonical_json(payload: dict) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _verify_exact_artifact(
    project_root: Path,
    path: Path,
    expected: str,
    label: str,
) -> bytes:
    payload = _secure_project_read_optional(project_root, path, label)
    if payload is None:
        raise ValueError(f"{label} is missing: {path}")
    expected_bytes = (expected + "\n").encode()
    if payload != expected_bytes:
        raise ValueError(f"{label} content changed: {path}")
    return payload


def _preflight_exact_artifact(
    project_root: Path,
    path: Path,
    content: str,
    label: str,
) -> None:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    if not path.parent.exists():
        return
    payload = _secure_project_read_optional(project_root, path, label)
    if payload is not None and payload != (content + "\n").encode():
        raise ValueError(f"{label} already exists with different content: {path}")


def _candidate_store_bytes(candidates: list[SemanticCandidate]) -> bytes:
    routed = [route_candidate(candidate) for candidate in candidates]
    content = "\n".join(candidate.model_dump_json() for candidate in routed)
    if routed:
        content += "\n"
    return content.encode()


def _team_transaction_journal_path(project_root: Path) -> Path:
    return project_root / ".morpheus" / "review" / ".team-learning-transaction.json"


def _commit_team_learning_transaction(
    project_root: Path,
    writes: list[tuple[Path, bytes]],
    *,
    immutable_paths: set[Path],
) -> None:
    """Journal and atomically apply a complete team-learning state transition."""
    duplicate_paths = sorted(
        str(path)
        for path, count in Counter(path for path, _ in writes).items()
        if count > 1
    )
    if duplicate_paths:
        raise ValueError(
            "Team learning transaction contains duplicate paths: "
            + ", ".join(duplicate_paths)
        )
    transaction_id = f"teamtxn_{secrets.token_hex(16)}"
    entries = []
    for path, after in writes:
        _validate_team_transaction_path(project_root, path)
        before = _secure_project_read_optional(
            project_root,
            path,
            "Team learning transaction artifact",
        )
        if path in immutable_paths and before is not None:
            raise ValueError(
                f"Immutable team learning artifact appeared after preflight: {path}"
            )
        entries.append(_TransactionEntry(path=path, before=before, after=after))

    prepared = _team_transaction_payload(
        project_root,
        transaction_id=transaction_id,
        phase="prepared",
        entries=entries,
    )
    committed = {**prepared, "phase": "committed"}
    journal_path = _team_transaction_journal_path(project_root)
    prepared_bytes = _journal_bytes(prepared)
    committed_bytes = _journal_bytes(committed)
    try:
        _atomic_project_write(
            project_root,
            journal_path,
            prepared_bytes,
            expected=None,
            transaction_id=transaction_id,
            label="Team learning transaction journal",
        )
    except Exception:
        try:
            recover_team_learning_transaction(project_root)
        except Exception as rollback_exc:
            raise RuntimeError(
                "Team learning journal publication failed and cleanup was incomplete"
            ) from rollback_exc
        raise
    try:
        for entry in entries:
            _apply_team_transaction_entry(
                project_root,
                entry,
                use_after=True,
                transaction_id=transaction_id,
            )
        _atomic_project_write(
            project_root,
            journal_path,
            committed_bytes,
            expected=prepared_bytes,
            transaction_id=transaction_id,
            label="Team learning transaction journal",
        )
    except Exception:
        try:
            recovered_after = recover_team_learning_transaction(project_root)
        except Exception as rollback_exc:
            raise RuntimeError(
                "Team learning transaction failed and rollback was incomplete"
            ) from rollback_exc
        if recovered_after is True:
            return
        raise
    try:
        _secure_project_unlink(
            project_root,
            journal_path,
            expected=committed_bytes,
            label="Team learning transaction journal",
        )
    except Exception:
        recover_team_learning_transaction(project_root)


def recover_team_learning_transaction(project_root: Path) -> bool | None:
    """Recover a prepared transaction backward or a committed one forward."""
    journal_path = _team_transaction_journal_path(project_root)
    raw = _secure_project_read_optional(
        project_root,
        journal_path,
        "Team learning transaction journal",
    )
    if raw is None:
        return None
    try:
        journal = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Team learning transaction journal is invalid JSON") from exc
    if not isinstance(journal, dict):
        raise ValueError("Team learning transaction journal must be a JSON object")
    if journal.get("schema") != TEAM_TRANSACTION_SCHEMA:
        raise ValueError("Team learning transaction journal schema invalid")
    transaction_id = journal.get("transaction_id")
    phase = journal.get("phase")
    raw_entries = journal.get("entries")
    if (
        not isinstance(transaction_id, str)
        or not re.fullmatch(r"teamtxn_[0-9a-f]{32}", transaction_id)
        or phase not in {"prepared", "committed"}
        or not isinstance(raw_entries, list)
    ):
        raise ValueError("Team learning transaction journal structure invalid")
    entries: list[_TransactionEntry] = []
    seen_paths: set[Path] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {"path", "before", "after"}:
            raise ValueError("Team learning transaction journal entry invalid")
        raw_path = raw_entry["path"]
        if not isinstance(raw_path, str):
            raise ValueError("Team learning transaction journal path invalid")
        path = project_root / raw_path
        _validate_team_transaction_path(project_root, path)
        if path in seen_paths:
            raise ValueError("Team learning transaction journal paths are not unique")
        seen_paths.add(path)
        entries.append(_TransactionEntry(
            path=path,
            before=_decode_journal_bytes(raw_entry["before"]),
            after=_decode_journal_bytes(raw_entry["after"]),
        ))
    _cleanup_team_transaction_temp(
        project_root,
        journal_path,
        transaction_id=transaction_id,
        label="Team learning transaction journal",
    )
    for entry in entries:
        _cleanup_team_transaction_temp(
            project_root,
            entry.path,
            transaction_id=transaction_id,
            label="Team learning transaction artifact",
        )
    use_after = phase == "committed"
    for entry in entries:
        current = _secure_project_read_optional(
            project_root,
            entry.path,
            "Team learning transaction artifact",
        )
        if current not in {entry.before, entry.after}:
            raise ValueError(
                f"Team learning transaction artifact diverged: {entry.path}"
            )
    for entry in entries:
        _apply_team_transaction_entry(
            project_root,
            entry,
            use_after=use_after,
            transaction_id=transaction_id,
        )
    expected_journal = _journal_bytes(journal)
    _secure_project_unlink(
        project_root,
        journal_path,
        expected=expected_journal,
        label="Team learning transaction journal",
    )
    return use_after


def _apply_team_transaction_entry(
    project_root: Path,
    entry: _TransactionEntry,
    *,
    use_after: bool,
    transaction_id: str,
) -> None:
    target = entry.after if use_after else entry.before
    other = entry.before if use_after else entry.after
    current = _secure_project_read_optional(
        project_root,
        entry.path,
        "Team learning transaction artifact",
    )
    if current == target:
        return
    if current != other:
        raise ValueError(
            f"Team learning transaction artifact changed: {entry.path}"
        )
    if target is None:
        _secure_project_unlink(
            project_root,
            entry.path,
            expected=current,
            label="Team learning transaction artifact",
        )
        return
    _atomic_project_write(
        project_root,
        entry.path,
        target,
        expected=current,
        transaction_id=transaction_id,
        label="Team learning transaction artifact",
    )


def _team_transaction_payload(
    project_root: Path,
    *,
    transaction_id: str,
    phase: str,
    entries: list[_TransactionEntry],
) -> dict:
    return {
        "schema": TEAM_TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "phase": phase,
        "entries": [
            {
                "path": entry.path.relative_to(project_root).as_posix(),
                "before": _encode_journal_bytes(entry.before),
                "after": _encode_journal_bytes(entry.after),
            }
            for entry in entries
        ],
    }


def _journal_bytes(journal: dict) -> bytes:
    return (json.dumps(journal, indent=2, sort_keys=True) + "\n").encode()


def _encode_journal_bytes(payload: bytes | None) -> dict | None:
    if payload is None:
        return None
    return {
        "base64": base64.b64encode(payload).decode("ascii"),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _decode_journal_bytes(encoded: object) -> bytes | None:
    if encoded is None:
        return None
    if not isinstance(encoded, dict) or set(encoded) != {"base64", "sha256"}:
        raise ValueError("Team learning transaction journal bytes invalid")
    raw_base64 = encoded["base64"]
    expected_sha256 = encoded["sha256"]
    if not isinstance(raw_base64, str) or not isinstance(expected_sha256, str):
        raise ValueError("Team learning transaction journal bytes invalid")
    try:
        payload = base64.b64decode(raw_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Team learning transaction journal bytes invalid") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Team learning transaction journal bytes digest mismatch")
    return payload


def _validate_team_transaction_path(project_root: Path, path: Path) -> None:
    relative = _project_relative_path(project_root, path, "Team learning transaction path")
    parts = relative.parts
    fixed_paths = {
        (".morpheus", "review", "semantic_candidates.jsonl"),
        (".morpheus", "learning", "team_loop_report.json"),
        (".morpheus", "learning", "team_loop_report.md"),
    }
    variable_paths = bool(
        len(parts) == 4
        and parts[:3] == (".morpheus", "review", "team_inputs")
        and re.fullmatch(r"teamin_[0-9a-f]{24}\.jsonl", parts[3])
        or len(parts) == 4
        and parts[:3] == (".morpheus", "review", "team_feedback")
        and re.fullmatch(r"teamfb_[0-9a-f]{24}\.jsonl", parts[3])
        or len(parts) == 4
        and parts[:3] == (".morpheus", "review", "check_corrections")
        and re.fullmatch(r"corr_[0-9a-f]{24}\.md", parts[3])
    )
    if parts not in fixed_paths and not variable_paths:
        raise ValueError(f"Team learning transaction path is not allowed: {relative}")


def _secure_ensure_project_directory(
    project_root: Path,
    path: Path,
    label: str,
) -> None:
    relative = _project_relative_path(project_root, path, label)
    _require_secure_descriptor_operations()
    flags = _directory_open_flags()
    descriptors: list[int] = []
    descriptor = os.open(project_root, flags)
    descriptors.append(descriptor)
    try:
        for component in relative.parts:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise ValueError(f"{label} must be a directory: {path}")
            descriptors.append(child)
            descriptor = child
    finally:
        for opened in reversed(descriptors):
            os.close(opened)


@contextmanager
def _project_parent_descriptor(
    project_root: Path,
    path: Path,
    label: str,
) -> Iterator[tuple[int | None, str]]:
    relative = _project_relative_path(project_root, path, label)
    _require_secure_descriptor_operations()
    flags = _directory_open_flags()
    descriptors: list[int] = []
    descriptor = os.open(project_root, flags)
    descriptors.append(descriptor)
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise ValueError(f"{label} parent must be a directory: {path.parent}")
            descriptors.append(child)
            descriptor = child
        yield descriptor, relative.name
    finally:
        for opened in reversed(descriptors):
            os.close(opened)


def _secure_project_read_optional(
    project_root: Path,
    path: Path,
    label: str,
) -> bytes | None:
    with _project_parent_descriptor(project_root, path, label) as (parent_fd, name):
        return _read_project_entry(parent_fd, name, path, label)


def _read_project_entry(
    parent_fd: int | None,
    name: str,
    path: Path,
    label: str,
) -> bytes | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = (
            os.open(path, flags)
            if parent_fd is None
            else os.open(name, flags, dir_fd=parent_fd)
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _atomic_project_write(
    project_root: Path,
    path: Path,
    payload: bytes,
    *,
    expected: bytes | None,
    transaction_id: str,
    label: str,
) -> None:
    with _project_parent_descriptor(project_root, path, label) as (parent_fd, name):
        current = _read_project_entry(parent_fd, name, path, label)
        if current != expected:
            raise ValueError(f"{label} changed before commit: {path}")
        temp_name = _team_transaction_temp_name(path, transaction_id)
        temp_path = path.parent / temp_name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = (
                os.open(temp_path, flags, 0o600)
                if parent_fd is None
                else os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"{label} staging file is not regular: {temp_path}")
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            if _read_project_entry(parent_fd, name, path, label) != expected:
                raise ValueError(f"{label} changed during commit: {path}")
            if parent_fd is None:
                if expected is None and path.exists():
                    raise ValueError(f"{label} appeared during commit: {path}")
                os.replace(temp_path, path)
            elif expected is None and os.link in os.supports_dir_fd:
                os.link(
                    temp_name,
                    name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                os.unlink(temp_name, dir_fd=parent_fd)
            elif os.rename in os.supports_dir_fd:
                os.rename(
                    temp_name,
                    name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
            else:  # pragma: no cover - supported POSIX hosts expose dir_fd rename.
                os.replace(temp_path, path)
            if parent_fd is not None:
                os.fsync(parent_fd)
            if _read_project_entry(parent_fd, name, path, label) != payload:
                raise ValueError(f"{label} verification failed after commit: {path}")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                if parent_fd is None:
                    if temp_path.exists() and not temp_path.is_symlink():
                        temp_path.unlink()
                else:
                    os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def _cleanup_team_transaction_temp(
    project_root: Path,
    path: Path,
    *,
    transaction_id: str,
    label: str,
) -> None:
    temp_name = _team_transaction_temp_name(path, transaction_id)
    temp_path = path.parent / temp_name
    with _project_parent_descriptor(project_root, path, label) as (parent_fd, _):
        payload = _read_project_entry(parent_fd, temp_name, temp_path, label)
        if payload is None:
            return
        if parent_fd is None:
            temp_path.unlink()
        else:
            os.unlink(temp_name, dir_fd=parent_fd)
            os.fsync(parent_fd)


def _team_transaction_temp_name(path: Path, transaction_id: str) -> str:
    return (
        f".{path.name}.{transaction_id}."
        f"{hashlib.sha256(path.as_posix().encode()).hexdigest()[:12]}.tmp"
    )


def _secure_project_unlink(
    project_root: Path,
    path: Path,
    *,
    expected: bytes,
    label: str,
) -> None:
    with _project_parent_descriptor(project_root, path, label) as (parent_fd, name):
        current = _read_project_entry(parent_fd, name, path, label)
        if current != expected:
            raise ValueError(f"{label} changed before removal: {path}")
        if parent_fd is None:
            path.unlink()
        else:
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)


def _project_relative_path(project_root: Path, path: Path, label: str) -> Path:
    try:
        relative = path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside the project: {path}") from exc
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} is invalid: {path}")
    return relative


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _require_secure_descriptor_operations() -> None:
    required_dir_fd = (os.open, os.mkdir, os.link, os.rename, os.unlink)
    if (
        any(operation not in os.supports_dir_fd for operation in required_dir_fd)
        or os.link not in os.supports_follow_symlinks
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise RuntimeError(
            "Team learning requires descriptor-relative filesystem operations"
        )


def _team_loop_report(
    project_root: Path,
    candidates: list[SemanticCandidate],
    *,
    events: list[TeamLearningInput],
    created: list[SemanticCandidate],
    existing_candidate_ids: list[str],
    created_receipt_ids: list[str],
    existing_receipt_ids: list[str],
    reconciled_ids: list[str],
) -> dict:
    routed = [route_candidate(candidate) for candidate in candidates]
    review_counts = Counter(candidate.status for candidate in routed)
    route_counts = Counter(candidate.memory_route for candidate in routed)
    feedback_source_counts = Counter(
        str(candidate.provider.get("feedback_source"))
        for candidate in routed
        if candidate.provider.get("name") == TEAM_FEEDBACK_PROVIDER
        and candidate.provider.get("feedback_source")
    )
    input_source_counts = Counter(event.source_type for event in events)
    no_candidate_input_count = sum(
        1
        for event in events
        if isinstance(event, CheckResultEvent)
        and event.status not in CHECK_CORRECTION_STATUSES
    )
    return {
        "policy_version": TEAM_LOOP_POLICY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "input_count": len(events),
        "input_source_counts": dict(sorted(input_source_counts.items())),
        "created_input_receipt_count": len(created_receipt_ids),
        "existing_input_receipt_count": len(existing_receipt_ids),
        "created_input_receipt_ids": created_receipt_ids,
        "existing_input_receipt_ids": existing_receipt_ids,
        "created_count": len(created),
        "existing_count": len(existing_candidate_ids),
        "created_candidate_ids": [candidate.id for candidate in created],
        "existing_candidate_ids": existing_candidate_ids,
        "reconciled_count": len(reconciled_ids),
        "reconciled_candidate_ids": reconciled_ids,
        "no_candidate_input_count": no_candidate_input_count,
        "review_counts": {
            "accepted": review_counts.get("accepted", 0),
            "pending": review_counts.get("pending", 0),
            "rejected": review_counts.get("rejected", 0),
        },
        "route_counts": dict(sorted(route_counts.items())),
        "feedback_source_counts": dict(sorted(feedback_source_counts.items())),
        "accepted_review_candidate_count": review_counts.get("accepted", 0),
        "check_correction_count": sum(
            1
            for candidate in routed
            if candidate.provider.get("name") == CHECK_CORRECTION_PROVIDER
        ),
        "stale_correction_count": sum(
            1 for candidate in routed if candidate.kind == "outdated_claim"
        ),
        "actions": {
            "dataset_generation_attempted": False,
            "training_attempted": False,
            "evaluation_attempted": False,
            "adapter_activation_attempted": False,
        },
    }


def _preflight_team_report(project_root: Path) -> None:
    learning_dir = project_root / ".morpheus" / "learning"
    _preflight_directory(learning_dir, "Team learning report directory")
    paths = [
        learning_dir / "team_loop_report.json",
        learning_dir / "team_loop_report.md",
    ]
    reject_symlink_paths(paths, "Team learning report")
    for path in paths:
        reject_symlink_components(path, "Team learning report")


def _render_team_loop_report(report: dict) -> str:
    lines = [
        "# Morpheus Team Learning Loop",
        "",
        f"- Policy: `{report['policy_version']}`",
        f"- Input events: {report['input_count']}",
        f"- Input receipts created: {report['created_input_receipt_count']}",
        f"- Input receipts existing: {report['existing_input_receipt_count']}",
        f"- Candidates created: {report['created_count']}",
        f"- Candidates existing: {report['existing_count']}",
        f"- Accepted candidates reconciled: {report['reconciled_count']}",
        f"- Accepted review candidates: {report['accepted_review_candidate_count']}",
        f"- Check corrections: {report['check_correction_count']}",
        f"- Stale corrections: {report['stale_correction_count']}",
        "",
        "## Current Input Batch",
        "",
    ]
    for source_type, count in report["input_source_counts"].items():
        lines.append(f"- `{source_type}`: {count}")
    lines.extend(["", "## Review State", ""])
    for status, count in report["review_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Memory Routes", ""])
    for route, count in report["route_counts"].items():
        lines.append(f"- `{route}`: {count}")
    lines.extend([
        "",
        "## Safety",
        "",
        "- Dataset generation attempted: false",
        "- Training attempted: false",
        "- Evaluation attempted: false",
        "- Adapter activation attempted: false",
        "",
    ])
    return "\n".join(lines)


def _preflight_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    reject_symlink_components(path.parent, label)
    reject_symlink_components(path, label)


def _safe_project_root(project_root: Path) -> Path:
    expanded = project_root.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {expanded}")
    reject_symlink_components(expanded, "Project root")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Project root is not a directory: {resolved}")
    return resolved
