"""Dataset quality reports for reviewed Morpheus learning state."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from morpheus.core.learning.registry import dataset_manifest, latest_dataset_dir
from morpheus.core.learning.safety import (
    contains_secret_like_text,
    load_morpheusignore,
    path_is_ignored,
)
from morpheus.core.learning.team import team_feedback_projection_error
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.semantic.routing import ROUTING_POLICY_VERSION, route_candidate
from morpheus.core.semantic.verifier import verify_candidate_span


TRAIN_MIN_ACCEPTED = 20
TRAIN_MIN_EXAMPLES = 100
BENCHMARK_MIN_TRAINABLE = 20
BENCHMARK_MIN_EXAMPLES = 100
BENCHMARK_MIN_EVAL_ITEMS = 30
BENCHMARK_MIN_SOURCE_PATHS = 3
BENCHMARK_CLASS_MINIMUMS = {
    "product": 1,
    "command": 2,
    "architecture": 1,
}
BENCHMARK_CLASS_GROUPS = {
    "security_or_convention": {
        "classes": ("security", "convention"),
        "minimum": 1,
    },
}
BENCHMARK_EVAL_CATEGORY_MINIMUMS = {
    "unsupported_claim_refusal": 1,
    "outdated_claim_correction": 1,
}


def build_quality_report(project_root: Path) -> dict:
    project_root = project_root.expanduser().resolve()
    candidates = ReviewStore(project_root).load_candidates()
    ignore_patterns = load_morpheusignore(project_root)
    routed = sorted(
        (_quality_candidate(project_root, candidate, ignore_patterns) for candidate in candidates),
        key=lambda item: item["id"],
    )
    route_counts = _counts(item["memory_route"] for item in routed)
    latest = latest_dataset_dir(project_root)
    latest_manifest = dataset_manifest(latest) if latest is not None else None
    freshness = _dataset_freshness(project_root, latest_manifest)
    accepted_trainable = sum(
        1 for item in routed if item["trainability_status"] == "trainable"
    )
    examples_count = int((latest_manifest or {}).get("examples_count") or 0)
    eval_category_counts = _eval_category_counts(latest / "eval.seed.jsonl") if latest is not None else {}
    benchmark_gate = _benchmark_gate(latest_manifest, eval_category_counts, freshness)
    train_blockers = []
    if accepted_trainable < TRAIN_MIN_ACCEPTED:
        train_blockers.append("accepted candidates < 20")
    if examples_count < TRAIN_MIN_EXAMPLES:
        train_blockers.append("examples < 100")
    if latest_manifest is not None and not freshness["fresh"]:
        train_blockers.append("dataset sources changed")
    next_actions = []
    if accepted_trainable < TRAIN_MIN_ACCEPTED:
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
        f"- Examples: {manifest.get('examples_count') or 0}",
        f"- Eval items: {manifest.get('eval_items_count') or 0}",
        f"- Skipped: {manifest.get('skipped_count') or 0}",
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


def _dataset_freshness(project_root: Path, manifest: dict | None) -> dict:
    result = {
        "available": manifest is not None,
        "fresh": manifest is not None,
        "checked_paths": 0,
        "changed_paths": [],
        "missing_paths": [],
        "missing_hash_paths": [],
        "invalid_paths": [],
    }
    if manifest is None:
        return result

    source_paths = manifest.get("source_paths")
    if not isinstance(source_paths, list):
        result["invalid_paths"].append("source_paths")
        result["fresh"] = False
        return result
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        source_hashes = {}

    for raw_path in source_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            result["invalid_paths"].append(_invalid_path_label(raw_path))
            continue
        rel_path = Path(raw_path)
        if not rel_path.parts or rel_path.is_absolute() or ".." in rel_path.parts:
            result["invalid_paths"].append(raw_path)
            continue

        source_path = project_root / rel_path
        try:
            if source_path.is_symlink():
                raise ValueError("dataset source is a symlink")
            reject_symlink_components(source_path, "Dataset source")
        except ValueError:
            result["invalid_paths"].append(raw_path)
            continue
        if not source_path.is_file():
            result["missing_paths"].append(raw_path)
            continue

        expected_sha = source_hashes.get(raw_path)
        if not _valid_sha256(expected_sha):
            result["missing_hash_paths"].append(raw_path)
            continue

        actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        result["checked_paths"] += 1
        if actual_sha != expected_sha:
            result["changed_paths"].append(raw_path)

    for key in ("changed_paths", "missing_paths", "missing_hash_paths", "invalid_paths"):
        result[key] = sorted(set(result[key]))
    result["fresh"] = not any(
        result[key]
        for key in ("changed_paths", "missing_paths", "missing_hash_paths", "invalid_paths")
    )
    return result


def _invalid_path_label(value: object) -> str:
    if value == "":
        return "<empty>"
    return json.dumps(value, sort_keys=True, default=str)


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _quality_candidate(project_root: Path, candidate, ignore_patterns: set[str]) -> dict:
    routed = route_candidate(candidate)
    reason = routed.trainability_reason
    status = routed.trainability_status
    route = routed.memory_route
    label = routed.label
    rel_path = Path(routed.source_path)
    projection_error = team_feedback_projection_error(routed)
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


def _benchmark_gate(
    manifest: dict | None,
    eval_category_counts: dict[str, int],
    freshness: dict,
) -> dict:
    manifest = manifest or {}
    class_counts = manifest.get("class_counts") or {}
    route_counts = manifest.get("route_counts") or {}
    source_paths = manifest.get("source_paths") or []
    trainable_count = int(manifest.get("trainable_candidate_count") or 0)
    examples_count = int(manifest.get("examples_count") or 0)
    eval_items_count = int(manifest.get("eval_items_count") or 0)
    blockers = []
    if trainable_count < BENCHMARK_MIN_TRAINABLE:
        blockers.append("trainable_candidate_count < 20")
    if examples_count < BENCHMARK_MIN_EXAMPLES:
        blockers.append("examples < 100")
    if eval_items_count < BENCHMARK_MIN_EVAL_ITEMS:
        blockers.append("eval_items < 30")
    if len(source_paths) < BENCHMARK_MIN_SOURCE_PATHS:
        blockers.append("source_paths < 3")
    if manifest is not None and not freshness["fresh"]:
        blockers.append("dataset sources changed")

    class_requirements = {}
    for class_name, minimum in BENCHMARK_CLASS_MINIMUMS.items():
        count = int(class_counts.get(class_name) or 0)
        class_requirements[class_name] = {"count": count, "minimum": minimum}
        if count < minimum:
            blockers.append(f"class {class_name} < {minimum}")

    class_group_requirements = {}
    for group_name, config in BENCHMARK_CLASS_GROUPS.items():
        count = sum(int(class_counts.get(class_name) or 0) for class_name in config["classes"])
        minimum = int(config["minimum"])
        class_group_requirements[group_name] = {
            "classes": list(config["classes"]),
            "count": count,
            "minimum": minimum,
        }
        if count < minimum:
            blockers.append(f"class_group {group_name} < {minimum}")

    eval_requirements = {}
    for category, minimum in BENCHMARK_EVAL_CATEGORY_MINIMUMS.items():
        count = int(eval_category_counts.get(category) or 0)
        eval_requirements[category] = {"count": count, "minimum": minimum}
        if count < minimum:
            blockers.append(f"eval_category {category} < {minimum}")

    return {
        "allowed": not blockers,
        "blockers": blockers,
        "eval_category_counts": eval_category_counts,
        "requirements": {
            "class_counts": class_requirements,
            "class_groups": class_group_requirements,
            "eval_categories": eval_requirements,
            "route_counts": {
                "adapter_training": {
                    "count": int(route_counts.get("adapter_training") or 0),
                    "minimum": BENCHMARK_MIN_TRAINABLE,
                },
            },
            "source_paths": {
                "count": len(source_paths),
                "minimum": BENCHMARK_MIN_SOURCE_PATHS,
            },
        },
    }


def _eval_category_counts(eval_path: Path) -> dict[str, int]:
    if eval_path.is_symlink() or not eval_path.is_file():
        return {}
    reject_symlink_components(eval_path, "Learning eval seed")
    counts = Counter()
    for line in eval_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        category = item.get("category")
        if isinstance(category, str) and category:
            counts[category] += 1
    return dict(sorted(counts.items()))
