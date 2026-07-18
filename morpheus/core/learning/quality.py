"""Dataset quality reports for reviewed Morpheus learning state."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from morpheus.core.learning.dataset_validation import (
    manifest_count,
    validation_blocker_messages,
)
from morpheus.core.learning.readiness import benchmark_readiness_gate
from morpheus.core.learning.registry import dataset_manifest, latest_effective_dataset
from morpheus.core.learning.safety import (
    contains_secret_like_text,
    load_morpheusignore,
    path_is_ignored,
)
from morpheus.core.learning.team import reviewed_input_projection_error
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.semantic.routing import ROUTING_POLICY_VERSION, route_candidate
from morpheus.core.semantic.verifier import verify_candidate_span


TRAIN_MIN_ACCEPTED = 20
TRAIN_MIN_EXAMPLES = 100
def build_quality_report(project_root: Path) -> dict:
    project_root = project_root.expanduser().resolve()
    candidates = ReviewStore(project_root).load_candidates()
    ignore_patterns = load_morpheusignore(project_root)
    routed = sorted(
        (_quality_candidate(project_root, candidate, ignore_patterns) for candidate in candidates),
        key=lambda item: item["id"],
    )
    route_counts = _counts(item["memory_route"] for item in routed)
    effective_dataset = latest_effective_dataset(project_root)
    latest = (
        Path(str(effective_dataset["dataset_dir"]))
        if effective_dataset is not None
        else None
    )
    try:
        raw_latest_manifest = dataset_manifest(latest) if latest is not None else None
    except (OSError, ValueError, json.JSONDecodeError):
        raw_latest_manifest = None
    validation = (
        effective_dataset["validation"]
        if latest is not None
        else {
            "available": False,
            "valid": False,
            "blockers": [],
            "source_freshness": {
                "available": False,
                "fresh": False,
                "checked_paths": 0,
                "changed_paths": [],
                "missing_paths": [],
                "missing_hash_paths": [],
                "invalid_paths": [],
            },
        }
    )
    freshness = validation["source_freshness"]
    latest_manifest = raw_latest_manifest if validation["valid"] else None
    root_accepted_trainable = sum(
        1 for item in routed if item["trainability_status"] == "trainable"
    )
    dataset_trainable = (
        manifest_count(raw_latest_manifest, "trainable_candidate_count")
        if latest is not None
        else root_accepted_trainable
    )
    examples_count = manifest_count(raw_latest_manifest, "examples_count")
    benchmark_gate = benchmark_readiness_gate(
        raw_latest_manifest,
        validation,
    )
    train_blockers = []
    if dataset_trainable < TRAIN_MIN_ACCEPTED:
        train_blockers.append("accepted candidates < 20")
    if examples_count < TRAIN_MIN_EXAMPLES:
        train_blockers.append("examples < 100")
    if effective_dataset is not None and not validation["valid"]:
        train_blockers.extend(validation_blocker_messages(validation))
    next_actions = []
    if dataset_trainable < TRAIN_MIN_ACCEPTED:
        next_actions.extend([
            "morpheus review propose --max 30",
            "morpheus review accept-proposed --max 30",
        ])
    if train_blockers:
        next_actions.append("morpheus learn dataset . --from accepted --format instruction")
    return {
        "review": {
            "candidates_total": len(routed),
            "accepted": sum(1 for item in routed if item["status"] == "accepted"),
            "pending": sum(1 for item in routed if item["status"] == "pending"),
            "rejected": sum(1 for item in routed if item["status"] == "rejected"),
            "source_backed": sum(1 for item in routed if item["label"] == "source_backed"),
            "source_path_count": len({item["source_path"] for item in routed}),
            "by_class": _counts(item["semantic_class"] for item in routed),
            "by_trainability": _counts(item["trainability_status"] for item in routed),
            "by_route": route_counts,
            "top_blockers": _counts(
                item["trainability_reason"]
                for item in routed
                if item["trainability_status"] != "trainable"
            ),
            "source_paths": _counts(item["source_path"] for item in routed),
        },
        "dataset": {
            "latest_dataset_dir": str(latest) if latest is not None else None,
            "latest_manifest": latest_manifest,
            "freshness": freshness,
            "validation": validation,
            "effective_dataset": effective_dataset,
            "trainable_candidate_count": dataset_trainable,
        },
        "routing": {
            "policy_version": ROUTING_POLICY_VERSION,
            "decisions": routed,
            "by_route": route_counts,
            "prompt_context": [
                item for item in routed if item["memory_route"] == "prompt_context"
            ],
        },
        "train_allowed": not train_blockers,
        "train_blockers": train_blockers,
        "benchmark_allowed": benchmark_gate["allowed"],
        "benchmark_blockers": benchmark_gate["blockers"],
        "benchmark_gate": benchmark_gate,
        "next_actions": next_actions,
    }


def write_quality_report(project_root: Path) -> dict:
    project_root = project_root.expanduser().resolve()
    report = build_quality_report(project_root)
    quality_dir = project_root / ".morpheus" / "training" / "quality"
    reject_symlink_components(quality_dir, "Learning quality report directory")
    quality_dir.mkdir(parents=True, exist_ok=True)
    json_path = quality_dir / "quality_report.json"
    markdown_path = quality_dir / "quality_report.md"
    reject_symlink_paths([json_path, markdown_path], "Learning quality report")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_quality_report(report))
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "report": report,
    }


def render_quality_report(report: dict) -> str:
    review = report["review"]
    manifest = (report["dataset"].get("latest_manifest") or {})
    freshness = report["dataset"]["freshness"]
    lines = [
        "# Morpheus Dataset Quality",
        "",
        "## Review State",
        "",
        f"- Candidates: {review['candidates_total']}",
        f"- Accepted: {review['accepted']}",
        f"- Pending: {review['pending']}",
        f"- Rejected: {review['rejected']}",
        f"- Source paths: {review['source_path_count']}",
        "",
        "## Trainability",
        "",
    ]
    for key, value in review["by_trainability"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Memory Routes", ""])
    lines.append(f"- Policy: `{report['routing']['policy_version']}`")
    lines.append(f"- Audited decisions: {len(report['routing']['decisions'])}")
    for key, value in review["by_route"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Top Blockers", ""])
    for key, value in review["top_blockers"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Latest Dataset",
        "",
        f"- Dataset id: {manifest.get('dataset_id') or 'none'}",
        f"- Examples: {manifest_count(manifest, 'examples_count')}",
        f"- Eval items: {manifest_count(manifest, 'eval_items_count')}",
        f"- Skipped: {manifest_count(manifest, 'skipped_count')}",
        "",
        "## Dataset Freshness",
        "",
        f"- Available: {freshness['available']}",
        f"- Fresh: {freshness['fresh']}",
        f"- Checked paths: {freshness['checked_paths']}",
    ])
    freshness_labels = {
        "changed_paths": "Changed",
        "missing_paths": "Missing",
        "missing_hash_paths": "Missing hash",
        "invalid_paths": "Invalid",
    }
    for key, label in freshness_labels.items():
        for path in freshness[key]:
            lines.append(f"- {label}: `{path}`")
    lines.extend([
        "",
        "## Train Gate",
        "",
        f"- Train allowed: {report['train_allowed']}",
    ])
    for blocker in report["train_blockers"]:
        lines.append(f"- Blocker: {blocker}")
    gate = report["benchmark_gate"]
    lines.extend([
        "",
        "## Benchmark Gate",
        "",
        f"- Benchmark allowed: {gate['allowed']}",
    ])
    for blocker in gate["blockers"]:
        lines.append(f"- Blocker: {blocker}")
    if report["next_actions"]:
        lines.extend(["", "## Next Actions", ""])
        for action in report["next_actions"]:
            lines.append(f"- `{action}`")
    return "\n".join(lines).rstrip() + "\n"


def _quality_candidate(project_root: Path, candidate, ignore_patterns: set[str]) -> dict:
    routed = route_candidate(candidate)
    reason = routed.trainability_reason
    status = routed.trainability_status
    route = routed.memory_route
    label = routed.label
    rel_path = Path(routed.source_path)
    projection_error = reviewed_input_projection_error(routed)
    if contains_secret_like_text(routed.correction_text or ""):
        status = "unsafe"
        route = "excluded"
        reason = "secret_like_content"
    elif projection_error:
        status = "needs_review"
        route = "human_review"
        reason = projection_error
        label = "needs_review"
    elif rel_path.is_absolute() or ".." in rel_path.parts:
        status = "excluded"
        route = "excluded"
        reason = "invalid_source_path"
    elif path_is_ignored(rel_path, ignore_patterns):
        status = "excluded"
        route = "excluded"
        reason = "ignored_path"
    elif route != "excluded":
        source_path = project_root / rel_path
        try:
            if source_path.is_symlink():
                raise ValueError("source path is a symlink")
            reject_symlink_components(source_path, "Routing source path")
        except ValueError:
            status = "excluded"
            route = "excluded"
            reason = "unsafe_source_path"
        else:
            if not source_path.is_file():
                status = "excluded"
                route = "excluded"
                reason = "missing_source_path"
            else:
                try:
                    actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
                except OSError:
                    status = "needs_review"
                    route = "human_review"
                    reason = "unreadable_source_path"
                    label = "needs_review"
                else:
                    if actual_sha != routed.source_sha256:
                        status = "needs_review"
                        route = "human_review"
                        reason = "source_sha256_mismatch"
                        label = "needs_review"
                    elif verify_candidate_span(project_root, routed).label != "source_backed":
                        status = "needs_review"
                        route = "human_review"
                        reason = "invalid_source_span"
                        label = "needs_review"
    return {
        "id": routed.id,
        "claim": routed.claim,
        "status": routed.status,
        "label": label,
        "kind": routed.kind,
        "semantic_class": routed.semantic_class,
        "trainability_status": status,
        "trainability_reason": reason,
        "memory_route": route,
        "source_path": routed.source_path,
        "line_start": routed.line_start,
        "line_end": routed.line_end,
    }


def _counts(values) -> dict:
    return dict(sorted(Counter(values).items()))
