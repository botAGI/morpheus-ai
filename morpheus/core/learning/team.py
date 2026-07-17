"""Local, review-gated ingestion for team corrections."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from morpheus.core.learning.safety import contains_secret_like_text
from morpheus.core.provenance import compute_sha256_file
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.semantic.routing import route_candidate


TEAM_LOOP_POLICY_VERSION = "morpheus-team-learning/1"
TEAM_FEEDBACK_PROMPT_SHA256 = hashlib.sha256(b"morpheus-team-feedback-v1").hexdigest()
TEAM_FEEDBACK_PROVIDER = "morpheus-team-loop"
TeamFeedbackSource = Literal[
    "pr_comment",
    "rejected_agent_claim",
    "human_correction",
]


class TeamFeedbackEvent(BaseModel):
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


def parse_team_feedback_items(items: list[dict]) -> list[TeamFeedbackEvent]:
    """Validate an entire feedback batch before any local state is written."""
    events = [TeamFeedbackEvent.model_validate(item) for item in items]
    for event in events:
        for value in (event.claim, event.correction):
            if value and contains_secret_like_text(value):
                raise ValueError(
                    f"Team feedback {event.external_id!r} contains secret-like content"
                )
    return events


def team_feedback_projection_error(candidate: SemanticCandidate) -> str | None:
    """Return a stable blocker when a team candidate diverges from its event span."""
    if candidate.provider.get("name") != TEAM_FEEDBACK_PROVIDER:
        return None
    try:
        event = TeamFeedbackEvent.model_validate_json(candidate.evidence_excerpt)
    except ValueError:
        return "team_feedback_projection_mismatch"
    _, expected_id, expected_line = _prepare_event(event)
    expected_source_path = f".morpheus/review/team_feedback/{expected_id}.jsonl"
    expected_provider = {
        "name": TEAM_FEEDBACK_PROVIDER,
        "model": "local",
        "feedback_source": event.source_type,
        "external_id": event.external_id,
        "author": event.author,
        "url": event.url,
    }
    if any([
        candidate.id != expected_id,
        candidate.kind != "outdated_claim",
        candidate.claim != event.claim,
        candidate.correction_text != event.correction,
        candidate.source_path != expected_source_path,
        candidate.source_revision != f"feedback:{event.source_type}:{event.external_id}",
        candidate.evidence_excerpt != expected_line,
        candidate.provider != expected_provider,
        candidate.prompt_sha256 != TEAM_FEEDBACK_PROMPT_SHA256,
    ]):
        return "team_feedback_projection_mismatch"
    return None


def run_team_learning_loop(
    project_root: Path,
    items: list[dict] | None = None,
) -> dict:
    """Ingest pending feedback candidates and write an audit-only loop report."""
    project_root = _safe_project_root(project_root)
    events = parse_team_feedback_items(items or [])
    prepared = [_prepare_event(event) for event in events]

    store = ReviewStore(project_root)
    with store.transaction():
        return _run_team_learning_loop_locked(project_root, store, events, prepared)


def _run_team_learning_loop_locked(
    project_root: Path,
    store: ReviewStore,
    events: list[TeamFeedbackEvent],
    prepared: list[tuple[TeamFeedbackEvent, str, str]],
) -> dict:
    """Run one feedback projection while holding the shared review-store lock."""
    existing = store.load_candidates()
    existing_by_id = {candidate.id: candidate for candidate in existing}
    feedback_dir = store.review_dir / "team_feedback"

    created: list[SemanticCandidate] = []
    existing_ids: list[str] = []
    planned: list[tuple[TeamFeedbackEvent, str, str, Path]] = []
    planned_by_id: dict[str, str] = {}
    if prepared:
        _preflight_directory(feedback_dir, "Team feedback path")
    for event, candidate_id, artifact_line in prepared:
        artifact_path = feedback_dir / f"{candidate_id}.jsonl"
        current = existing_by_id.get(candidate_id)
        if current is not None:
            if team_feedback_projection_error(current):
                raise ValueError(
                    f"Team feedback candidate projection mismatch: {candidate_id}"
                )
            _verify_existing_artifact(artifact_path, artifact_line)
            existing_ids.append(candidate_id)
            continue
        if candidate_id in planned_by_id:
            if planned_by_id[candidate_id] != artifact_line:
                raise ValueError(f"Team feedback id collision: {candidate_id}")
            existing_ids.append(candidate_id)
            continue
        _preflight_immutable_artifact(artifact_path, artifact_line)
        planned.append((event, candidate_id, artifact_line, artifact_path))
        planned_by_id[candidate_id] = artifact_line

    if planned:
        _ensure_directory(feedback_dir, "Team feedback path")
    run_id = f"teamloop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    for event, candidate_id, artifact_line, artifact_path in planned:
        _write_immutable_artifact(artifact_path, artifact_line)
        timestamp = datetime.fromtimestamp(artifact_path.stat().st_mtime, timezone.utc)
        candidate = route_candidate(SemanticCandidate(
            id=candidate_id,
            run_id=run_id,
            kind="outdated_claim",
            claim=event.claim,
            correction_text=event.correction,
            source_path=artifact_path.relative_to(project_root).as_posix(),
            source_sha256=compute_sha256_file(artifact_path),
            source_mtime=timestamp,
            source_revision=f"feedback:{event.source_type}:{event.external_id}",
            line_start=1,
            line_end=1,
            evidence_excerpt=artifact_line,
            evidence_sha256=hashlib.sha256(artifact_line.encode()).hexdigest(),
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
        created.append(candidate)
        existing_by_id[candidate_id] = candidate

    if created:
        store.save_candidates([*existing, *created])
    candidates = store.load_candidates()
    report = _team_loop_report(
        project_root,
        candidates,
        input_count=len(events),
        created=created,
        existing_ids=existing_ids,
    )
    return _write_team_loop_report(project_root, report)


def _prepare_event(event: TeamFeedbackEvent) -> tuple[TeamFeedbackEvent, str, str]:
    payload = event.model_dump(mode="json", exclude_none=True)
    artifact_line = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(artifact_line.encode()).hexdigest()
    return event, f"teamfb_{digest[:24]}", artifact_line


def _verify_existing_artifact(path: Path, expected: str) -> None:
    reject_symlink_paths([path], "Team feedback artifact")
    reject_symlink_components(path, "Team feedback artifact")
    if not path.is_file():
        raise ValueError(f"Team feedback artifact is missing: {path}")
    if path.read_text() != expected + "\n":
        raise ValueError(f"Team feedback artifact content changed: {path}")


def _preflight_immutable_artifact(path: Path, content: str) -> None:
    reject_symlink_paths([path], "Team feedback artifact")
    reject_symlink_components(path, "Team feedback artifact")
    if path.exists() and (not path.is_file() or path.read_text() != content + "\n"):
        raise ValueError(f"Team feedback artifact already exists with different content: {path}")


def _write_immutable_artifact(path: Path, content: str) -> None:
    reject_symlink_paths([path], "Team feedback artifact")
    reject_symlink_components(path, "Team feedback artifact")
    if path.exists():
        if path.is_file() and path.read_text() == content + "\n":
            return
        raise ValueError(f"Team feedback artifact already exists with different content: {path}")
    path.write_text(content + "\n")


def _team_loop_report(
    project_root: Path,
    candidates: list[SemanticCandidate],
    *,
    input_count: int,
    created: list[SemanticCandidate],
    existing_ids: list[str],
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
    return {
        "policy_version": TEAM_LOOP_POLICY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "input_count": input_count,
        "created_count": len(created),
        "existing_count": len(existing_ids),
        "created_candidate_ids": [candidate.id for candidate in created],
        "existing_candidate_ids": existing_ids,
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
            if candidate.provider.get("name") == "morpheus-check"
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


def _write_team_loop_report(project_root: Path, report: dict) -> dict:
    learning_dir = project_root / ".morpheus" / "learning"
    _ensure_directory(learning_dir, "Team learning report directory")
    json_path = learning_dir / "team_loop_report.json"
    markdown_path = learning_dir / "team_loop_report.md"
    reject_symlink_paths([json_path, markdown_path], "Team learning report")
    reject_symlink_components(json_path, "Team learning report")
    reject_symlink_components(markdown_path, "Team learning report")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_team_loop_report(report))
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "report": report,
    }


def _render_team_loop_report(report: dict) -> str:
    lines = [
        "# Morpheus Team Learning Loop",
        "",
        f"- Policy: `{report['policy_version']}`",
        f"- Input events: {report['input_count']}",
        f"- Created: {report['created_count']}",
        f"- Existing: {report['existing_count']}",
        f"- Accepted review candidates: {report['accepted_review_candidate_count']}",
        f"- Check corrections: {report['check_correction_count']}",
        f"- Stale corrections: {report['stale_correction_count']}",
        "",
        "## Review State",
        "",
    ]
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


def _ensure_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    reject_symlink_components(path.parent, label)
    path.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path, label)


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
