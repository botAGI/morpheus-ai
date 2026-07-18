"""Fail-closed provenance validation for Morpheus learning datasets."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterable

from morpheus.core.compiler import compute_sha256
from morpheus.core.learning.categories import (
    BENCHMARK_CATEGORY_SCHEMA,
    KNOWN_BENCHMARK_CATEGORIES,
)
from morpheus.core.learning.evals import (
    eval_items_for_candidate,
    heldout_eval_items_for_candidate,
    heldout_truth_gate_negative_eval_items,
    truth_gate_negative_eval_items,
    unsupported_claim_eval_item,
)
from morpheus.core.learning.examples import (
    chat_examples_from_instruction,
    instruction_examples_for_candidate,
    sharegpt_examples_from_instruction,
)
from morpheus.core.learning.safety import (
    contains_secret_like_text,
    load_morpheusignore,
    path_is_ignored,
)
from morpheus.core.learning.team import team_feedback_projection_error
from morpheus.core.provenance import compute_sha256_file
from morpheus.core.provenance import latest_receipt_file
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.semantic.routing import ROUTING_POLICY_VERSION, route_candidate
from morpheus.core.semantic.verifier import verify_candidate_span
from morpheus.core.state_authority import state_authority_transaction
from morpheus.core.verify import verify_receipt_chain


MANIFEST_FORMAT_VERSION = "morpheus-learning-manifest/3"
PROVENANCE_SCHEMA = "morpheus-dataset-provenance/1"
REVIEW_SNAPSHOT_SCHEMA = "morpheus-review-snapshot/1"
ACCEPTED_REVIEW_SCOPE = "accepted_review_live"
LAB_REVIEW_SCOPE = "lab_review_snapshot"
ACTIVE_STATE_SCOPE = "active_state_receipt"
SOURCE_SCOPES = {ACCEPTED_REVIEW_SCOPE, LAB_REVIEW_SCOPE, ACTIVE_STATE_SCOPE}
REQUIRED_ARTIFACTS = {
    "dataset.instruction.jsonl",
    "dataset.sharegpt.jsonl",
    "eval.heldout.jsonl",
    "eval.seed.jsonl",
    "skipped.jsonl",
    "test.jsonl",
    "train.jsonl",
    "valid.jsonl",
}
DATASET_FORMAT_BINDINGS = {
    "instruction": ("dataset.instruction.jsonl", "morpheus-instruction/1"),
    "sharegpt": ("dataset.sharegpt.jsonl", "morpheus-sharegpt/1"),
    "chat": ("train.jsonl", "morpheus-chat/1"),
}
POSITIVE_KINDS = {
    "current_state",
    "active_decision",
    "agent_rule",
    "source_reference",
    "open_task",
}
TRAIN_REQUIRED_EXAMPLE_TYPES = {
    "eval_aligned_recall",
    "outdated_correction",
}
TRAIN_EXAMPLE_REPEATS = {
    "eval_aligned_recall": 3,
    "outdated_correction": 24,
}
MANIFEST_COUNT_FIELDS = frozenset({
    "candidate_count",
    "trainable_candidate_count",
    "examples_count",
    "eval_items_count",
    "heldout_eval_items_count",
    "skipped_count",
})
MANIFEST_COUNT_MAP_FIELDS = frozenset({
    "split_counts",
    "class_counts",
    "trainability_counts",
    "route_counts",
})
ACTIVE_STATE_CONTEXT_PATHS = {
    ".morpheus/state.json": ("state.json", True),
    ".morpheus/evidence.jsonl": ("evidence.jsonl", True),
    ".morpheus/WAKE.md": ("WAKE.md", True),
}


def parse_registry_timestamp_identity(
    identity: object,
    *,
    prefix: str = "",
) -> datetime | None:
    """Parse one canonical registry identity without accepting loose timestamps."""
    if not isinstance(identity, str):
        return None
    if prefix and not identity.startswith(prefix):
        return None
    raw_timestamp = identity[len(prefix):] if prefix else identity
    formats = {
        16: "%Y%m%dT%H%M%SZ",
        22: "%Y%m%dT%H%M%S%fZ",
    }
    timestamp_format = formats.get(len(raw_timestamp))
    if timestamp_format is None:
        return None
    try:
        return datetime.strptime(raw_timestamp, timestamp_format).replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def canonical_review_snapshot(
    candidates: Iterable[SemanticCandidate],
    *,
    reject_duplicates: bool = True,
) -> dict:
    """Return an order-independent digest of the complete review store."""
    records = []
    ids = []
    for candidate in candidates:
        candidate_id = candidate.id
        ids.append(candidate_id)
        records.append({
            "id": candidate_id,
            "sha256": _canonical_sha256(candidate.model_dump(mode="json")),
        })
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(ids).items()
        if count > 1
    )
    if reject_duplicates and duplicate_ids:
        raise ValueError(
            "Review store contains duplicate candidate ids: "
            + ", ".join(duplicate_ids)
        )
    payload = {
        "schema": REVIEW_SNAPSHOT_SCHEMA,
        "candidate_count": len(records),
        "candidates": sorted(records, key=lambda item: (item["id"], item["sha256"])),
    }
    return {
        **payload,
        "sha256": _canonical_sha256(payload),
    }


def build_dataset_provenance(
    project_root: Path,
    *,
    source: str,
    review_candidates: Iterable[SemanticCandidate] | None,
    source_receipt_id: str | None,
    context_paths: Iterable[str],
    source_receipt_sha256: str | None = None,
) -> dict:
    """Build the source authority recorded by a v2 dataset manifest."""
    project_root = project_root.expanduser().resolve()
    if source == "active-state":
        return {
            "schema": PROVENANCE_SCHEMA,
            "source_scope": ACTIVE_STATE_SCOPE,
            "source_root": str(project_root),
            "source_receipt_id": source_receipt_id,
            "source_receipt_sha256": source_receipt_sha256,
            "routing_policy_version": ROUTING_POLICY_VERSION,
            "context_paths": sorted(set(context_paths)),
            "review_snapshot": None,
        }

    source_scope = ACCEPTED_REVIEW_SCOPE
    lab_id = None
    if _lab_workspace_id(project_root) is not None:
        source_scope = LAB_REVIEW_SCOPE
        lab_id = _lab_workspace_id(project_root)
    candidates = list(review_candidates or [])
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "source_scope": source_scope,
        "source_root": str(project_root),
        "source_receipt_id": source_receipt_id,
        "routing_policy_version": ROUTING_POLICY_VERSION,
        "context_paths": sorted(set(context_paths)),
        "review_snapshot": canonical_review_snapshot(candidates),
    }
    if lab_id is not None:
        provenance["lab_id"] = lab_id
    return provenance


def capture_active_state_authority(project_root: Path) -> dict:
    """Read one signed, stable active-state snapshot for dataset compilation."""
    project_root = project_root.expanduser().resolve()
    with state_authority_transaction(project_root):
        return _capture_active_state_authority_locked(project_root)


def _capture_active_state_authority_locked(project_root: Path) -> dict:
    """Capture active state while its project authority lock is held."""
    morpheus_dir = project_root / ".morpheus"
    before = _read_active_state_context(project_root)
    valid, errors = verify_receipt_chain(morpheus_dir)
    if not valid:
        raise ValueError("Active-state receipt chain invalid: " + "; ".join(errors))
    receipt_path = latest_receipt_file(morpheus_dir / "receipts")
    if receipt_path is None:
        raise ValueError("Active-state receipt missing")
    receipt_bytes = _safe_read_bytes(receipt_path, "Active-state receipt")
    try:
        receipt = json.loads(receipt_bytes)
        state = json.loads(before["bytes"][".morpheus/state.json"])
        evidence_rows = [
            json.loads(line)
            for line in before["bytes"][".morpheus/evidence.jsonl"].decode().splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Active-state artifacts invalid: {exc}") from exc
    if not isinstance(receipt, dict) or not isinstance(state, dict):
        raise ValueError("Active-state receipt and state must be JSON objects")
    if any(not isinstance(row, dict) for row in evidence_rows):
        raise ValueError("Active-state evidence must contain JSON objects")

    after = _read_active_state_context(project_root)
    current_receipt_path = latest_receipt_file(morpheus_dir / "receipts")
    if current_receipt_path is None:
        raise ValueError("Active-state receipt missing after snapshot")
    current_receipt_bytes = _safe_read_bytes(
        current_receipt_path,
        "Active-state receipt",
    )
    if (
        before["hashes"] != after["hashes"]
        or receipt_path != current_receipt_path
        or receipt_bytes != current_receipt_bytes
    ):
        raise ValueError("Active state changed while its authority was captured")

    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id:
        raise ValueError("Active-state receipt id invalid")
    if state.get("receipt_id") != receipt_id:
        raise ValueError("Active-state receipt id does not match state")
    if receipt.get("state_json_sha256") != before["hashes"][".morpheus/state.json"]:
        raise ValueError("Active-state receipt does not bind state.json")
    if (
        receipt.get("evidence_jsonl_sha256")
        != before["hashes"][".morpheus/evidence.jsonl"]
    ):
        raise ValueError("Active-state receipt does not bind evidence.jsonl")
    if receipt.get("wake_md_sha256") != before["hashes"][".morpheus/WAKE.md"]:
        raise ValueError("Active-state receipt does not bind WAKE.md")
    return {
        "state": state,
        "evidence_rows": evidence_rows,
        "context_hashes": before["hashes"],
        "receipt_id": receipt_id,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }


def artifact_manifest(dataset_dir: Path, paths: Iterable[Path]) -> dict:
    """Hash generated dataset artifacts using dataset-relative names."""
    artifacts = {}
    for path in paths:
        relative = path.relative_to(dataset_dir).as_posix()
        artifacts[relative] = {
            "sha256": compute_sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return dict(sorted(artifacts.items()))


def dataset_binding_sha256(manifest: dict) -> str:
    """Digest all manifest fields except the digest itself."""
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "dataset_binding_sha256"
    }
    return _canonical_sha256(payload)


def manifest_count(manifest: object, field: str) -> int:
    """Return one exact non-negative manifest count, or zero when invalid."""
    if not isinstance(manifest, dict):
        return 0
    value = manifest.get(field)
    return value if _valid_manifest_count(value) else 0


def validate_dataset(
    project_root: Path,
    dataset_dir: Path,
    manifest: dict | None = None,
) -> dict:
    """Validate a dataset against its current source authority and artifacts."""
    project_root = project_root.expanduser().resolve()
    raw_dataset_dir = dataset_dir.expanduser()
    dataset_dir_is_symlink = raw_dataset_dir.is_symlink()
    dataset_dir = raw_dataset_dir.resolve()
    blockers: list[str] = []
    result = {
        "available": False,
        "valid": False,
        "blockers": blockers,
        "dataset_binding_sha256": None,
        "source_scope": None,
        "eval_coverage": _empty_eval_coverage(),
        "source_freshness": _empty_source_freshness(False),
        "review_snapshot": {
            "required": False,
            "matches": None,
            "expected_sha256": None,
            "current_sha256": None,
            "duplicate_candidate_ids": [],
        },
        "artifacts": {
            "checked": 0,
            "changed_paths": [],
            "missing_paths": [],
            "invalid_paths": [],
            "size_mismatch_paths": [],
        },
    }
    try:
        reject_symlink_components(raw_dataset_dir, "Learning dataset")
        if not dataset_dir.is_dir() or dataset_dir_is_symlink:
            _add_blocker(blockers, "dataset_directory_invalid")
            return result
        if not _is_within(dataset_dir, project_root):
            _add_blocker(blockers, "dataset_directory_outside_project")
            return result
        if manifest is None:
            manifest = _read_manifest(dataset_dir / "manifest.json")
    except (OSError, ValueError, json.JSONDecodeError):
        _add_blocker(blockers, "dataset_manifest_invalid")
        return result
    result["available"] = True

    format_versions = manifest.get("format_versions")
    manifest_version = (
        format_versions.get("manifest")
        if isinstance(format_versions, dict)
        else None
    )
    if manifest_version != MANIFEST_FORMAT_VERSION:
        _add_blocker(blockers, "legacy_unbound_manifest")
    elif not _valid_v2_manifest_fields(manifest):
        _add_blocker(blockers, "dataset_manifest_fields_invalid")

    expected_binding = manifest.get("dataset_binding_sha256")
    result["dataset_binding_sha256"] = (
        expected_binding if _valid_sha256(expected_binding) else None
    )
    if not _valid_sha256(expected_binding):
        _add_blocker(blockers, "missing_dataset_binding")
    elif dataset_binding_sha256(manifest) != expected_binding:
        _add_blocker(blockers, "dataset_binding_mismatch")

    provenance = manifest.get("provenance")
    source_root = None
    if not isinstance(provenance, dict) or provenance.get("schema") != PROVENANCE_SCHEMA:
        _add_blocker(blockers, "dataset_provenance_invalid")
    else:
        source_scope = provenance.get("source_scope")
        result["source_scope"] = source_scope
        if source_scope not in SOURCE_SCOPES:
            _add_blocker(blockers, "dataset_source_scope_invalid")
        if provenance.get("routing_policy_version") != ROUTING_POLICY_VERSION:
            _add_blocker(blockers, "routing_policy_mismatch")
        source_root = _validated_source_root(
            project_root,
            dataset_dir,
            provenance,
            blockers,
        )
        if source_root is not None:
            _validate_scope_identity(
                source_root,
                dataset_dir,
                provenance,
                manifest,
                blockers,
            )

    result["source_freshness"] = _validate_source_freshness(
        source_root,
        manifest,
        provenance if isinstance(provenance, dict) else None,
    )
    freshness = result["source_freshness"]
    if not freshness["fresh"]:
        _add_blocker(blockers, "dataset_sources_changed")

    if isinstance(provenance, dict) and source_root is not None:
        _validate_review_snapshot(
            source_root,
            provenance,
            result["review_snapshot"],
            blockers,
        )

    artifact_rows = _validate_artifacts(
        dataset_dir,
        manifest,
        result["artifacts"],
        blockers,
    )
    semantics_valid, eval_coverage = _validate_manifest_semantics(
        manifest,
        artifact_rows,
    )
    result["eval_coverage"] = eval_coverage
    if not semantics_valid:
        _add_blocker(blockers, "dataset_manifest_semantics_invalid")
    authority_valid, generated_artifacts_valid = _validate_source_authority(
        source_root,
        provenance if isinstance(provenance, dict) else None,
        manifest,
        artifact_rows,
    )
    if not authority_valid:
        _add_blocker(blockers, "dataset_source_authority_mismatch")
    if not generated_artifacts_valid:
        _add_blocker(blockers, "dataset_generated_artifacts_mismatch")
    result["valid"] = not blockers
    return result


def require_valid_dataset(
    project_root: Path,
    dataset_dir: Path,
    manifest: dict | None = None,
) -> dict:
    validation = validate_dataset(project_root, dataset_dir, manifest)
    if not validation["valid"]:
        raise ValueError(
            "Dataset validation failed: " + ", ".join(validation["blockers"])
        )
    return validation


def validate_dataset_artifacts(dataset_dir: Path, manifest: dict) -> dict:
    """Validate only the manifest-bound files in a private dataset snapshot."""
    raw_dataset_dir = dataset_dir.expanduser()
    blockers: list[str] = []
    result = {
        "valid": False,
        "blockers": blockers,
        "eval_coverage": _empty_eval_coverage(),
        "artifacts": {
            "checked": 0,
            "changed_paths": [],
            "missing_paths": [],
            "invalid_paths": [],
            "size_mismatch_paths": [],
        },
    }
    try:
        reject_symlink_components(raw_dataset_dir, "Learning dataset snapshot")
    except (OSError, ValueError):
        _add_blocker(blockers, "dataset_directory_invalid")
        return result
    if raw_dataset_dir.is_symlink() or not raw_dataset_dir.is_dir():
        _add_blocker(blockers, "dataset_directory_invalid")
        return result
    artifact_rows = _validate_artifacts(
        raw_dataset_dir.resolve(),
        manifest,
        result["artifacts"],
        blockers,
    )
    semantics_valid, eval_coverage = _validate_manifest_semantics(
        manifest,
        artifact_rows,
    )
    result["eval_coverage"] = eval_coverage
    if not semantics_valid:
        _add_blocker(blockers, "dataset_manifest_semantics_invalid")
    source_root, provenance = _snapshot_source_authority(manifest)
    authority_valid, generated_artifacts_valid = _validate_source_authority(
        source_root,
        provenance,
        manifest,
        artifact_rows,
    )
    if not authority_valid:
        _add_blocker(blockers, "dataset_source_authority_mismatch")
    if not generated_artifacts_valid:
        _add_blocker(blockers, "dataset_generated_artifacts_mismatch")
    result["valid"] = not blockers
    return result


def validation_blocker_messages(validation: dict) -> list[str]:
    """Convert structured validator reasons into stable quality-gate messages."""
    blockers = set(validation.get("blockers") or [])
    messages = []
    if "review_snapshot_changed" in blockers or "review_snapshot_invalid" in blockers:
        messages.append("dataset review snapshot changed")
    if "dataset_sources_changed" in blockers:
        messages.append("dataset sources changed")
    if blockers & {
        "dataset_artifacts_invalid",
        "dataset_artifact_missing",
        "dataset_artifact_hash_mismatch",
        "dataset_artifact_size_mismatch",
        "dataset_selected_artifact_invalid",
    }:
        messages.append("dataset artifacts changed")
    explained = {
        "review_snapshot_changed",
        "review_snapshot_invalid",
        "dataset_sources_changed",
        "dataset_artifacts_invalid",
        "dataset_artifact_missing",
        "dataset_artifact_hash_mismatch",
        "dataset_artifact_size_mismatch",
        "dataset_selected_artifact_invalid",
    }
    if blockers - explained:
        messages.append("dataset provenance invalid")
    return messages


def _validated_source_root(
    project_root: Path,
    dataset_dir: Path,
    provenance: dict,
    blockers: list[str],
) -> Path | None:
    raw_root = provenance.get("source_root")
    if not isinstance(raw_root, str) or not raw_root:
        _add_blocker(blockers, "dataset_source_root_invalid")
        return None
    source_root = Path(raw_root).expanduser()
    if source_root.is_symlink():
        _add_blocker(blockers, "dataset_source_root_invalid")
        return None
    try:
        reject_symlink_components(source_root, "Dataset source root")
        source_root = source_root.resolve()
    except (OSError, ValueError):
        _add_blocker(blockers, "dataset_source_root_invalid")
        return None
    if not source_root.is_dir() or not _is_within(source_root, project_root):
        _add_blocker(blockers, "dataset_source_root_invalid")
        return None

    scope = provenance.get("source_scope")
    if scope in {ACCEPTED_REVIEW_SCOPE, ACTIVE_STATE_SCOPE}:
        expected_registry = project_root / ".morpheus" / "training" / "datasets"
        if source_root != project_root:
            _add_blocker(blockers, "dataset_source_root_mismatch")
            return None
        if dataset_dir.parent != expected_registry.resolve():
            _add_blocker(blockers, "dataset_registry_scope_mismatch")
            return None
    elif scope == LAB_REVIEW_SCOPE:
        lab_id = provenance.get("lab_id")
        expected_lab_dir = project_root / ".morpheus" / "lab" / str(lab_id or "")
        expected_workspace = expected_lab_dir / "workspace"
        expected_dataset = expected_lab_dir / "dataset"
        if (
            not isinstance(lab_id, str)
            or not lab_id.startswith("lab_")
            or source_root != expected_workspace.resolve()
            or dataset_dir != expected_dataset.resolve()
        ):
            _add_blocker(blockers, "dataset_lab_scope_mismatch")
            return None
    return source_root


def _validate_review_snapshot(
    source_root: Path,
    provenance: dict,
    review_result: dict,
    blockers: list[str],
) -> None:
    scope = provenance.get("source_scope")
    if scope not in {ACCEPTED_REVIEW_SCOPE, LAB_REVIEW_SCOPE}:
        if provenance.get("review_snapshot") is not None:
            _add_blocker(blockers, "review_snapshot_invalid")
        return
    review_result["required"] = True
    expected = provenance.get("review_snapshot")
    if not isinstance(expected, dict) or expected.get("schema") != REVIEW_SNAPSHOT_SCHEMA:
        _add_blocker(blockers, "review_snapshot_invalid")
        review_result["matches"] = False
        return
    expected_sha = expected.get("sha256")
    review_result["expected_sha256"] = expected_sha
    try:
        candidates = ReviewStore(source_root).load_candidates()
        current = canonical_review_snapshot(candidates, reject_duplicates=False)
    except (OSError, ValueError, json.JSONDecodeError):
        _add_blocker(blockers, "review_snapshot_invalid")
        review_result["matches"] = False
        return
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(item.id for item in candidates).items()
        if count > 1
    )
    review_result["duplicate_candidate_ids"] = duplicate_ids
    review_result["current_sha256"] = current["sha256"]
    review_result["matches"] = bool(
        not duplicate_ids
        and _valid_sha256(expected_sha)
        and expected_sha == current["sha256"]
        and expected == current
    )
    if duplicate_ids:
        _add_blocker(blockers, "duplicate_review_candidate_ids")
    if not review_result["matches"]:
        _add_blocker(blockers, "review_snapshot_changed")


def _validate_scope_identity(
    source_root: Path,
    dataset_dir: Path,
    provenance: dict,
    manifest: dict,
    blockers: list[str],
) -> None:
    scope = provenance.get("source_scope")
    declared_source = manifest.get("source")
    dataset_id = manifest.get("dataset_id")
    if parse_registry_timestamp_identity(dataset_id) is None:
        _add_blocker(blockers, "dataset_id_invalid")
    if scope == ACTIVE_STATE_SCOPE:
        if declared_source != "active-state":
            _add_blocker(blockers, "dataset_source_scope_mismatch")
        context_paths = provenance.get("context_paths")
        if context_paths != sorted(ACTIVE_STATE_CONTEXT_PATHS):
            _add_blocker(blockers, "active_state_context_invalid")
        try:
            authority = capture_active_state_authority(source_root)
        except (OSError, ValueError):
            _add_blocker(blockers, "active_state_receipt_invalid")
            return
        if (
            authority["receipt_id"] != provenance.get("source_receipt_id")
            or authority["receipt_sha256"] != provenance.get("source_receipt_sha256")
            or not isinstance(manifest.get("source_hashes"), dict)
            or any(
                manifest["source_hashes"].get(path) != sha256
                for path, sha256 in authority["context_hashes"].items()
            )
        ):
            _add_blocker(blockers, "active_state_receipt_mismatch")
    elif scope in {ACCEPTED_REVIEW_SCOPE, LAB_REVIEW_SCOPE}:
        if declared_source != "accepted":
            _add_blocker(blockers, "dataset_source_scope_mismatch")
    if (
        scope in {ACCEPTED_REVIEW_SCOPE, ACTIVE_STATE_SCOPE}
        and parse_registry_timestamp_identity(dataset_dir.name) is None
    ):
        _add_blocker(blockers, "dataset_registry_identity_invalid")
    if (
        scope == LAB_REVIEW_SCOPE
        and parse_registry_timestamp_identity(
            provenance.get("lab_id"),
            prefix="lab_",
        )
        is None
    ):
        _add_blocker(blockers, "dataset_registry_identity_invalid")
    if (
        scope in {ACCEPTED_REVIEW_SCOPE, ACTIVE_STATE_SCOPE}
        and dataset_id != dataset_dir.name
    ):
        _add_blocker(blockers, "dataset_id_directory_mismatch")


def _validate_source_freshness(
    source_root: Path | None,
    manifest: dict,
    provenance: dict | None,
) -> dict:
    result = _empty_source_freshness(True)
    source_paths = manifest.get("source_paths")
    if not isinstance(source_paths, list):
        result["invalid_paths"].append("source_paths")
        result["fresh"] = False
        return result
    context_paths = provenance.get("context_paths") if provenance else []
    if not isinstance(context_paths, list):
        result["invalid_paths"].append("provenance.context_paths")
        context_paths = []
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        source_hashes = {}
    for raw_path in [*source_paths, *context_paths]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            result["invalid_paths"].append(_invalid_path_label(raw_path))
            continue
        rel_path = Path(raw_path)
        if not rel_path.parts or rel_path.is_absolute() or ".." in rel_path.parts:
            result["invalid_paths"].append(raw_path)
            continue
        if source_root is None:
            result["invalid_paths"].append(raw_path)
            continue
        source_path = source_root / rel_path
        try:
            reject_symlink_paths([source_path], "Dataset source")
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
        result["checked_paths"] += 1
        if compute_sha256_file(source_path) != expected_sha:
            result["changed_paths"].append(raw_path)
    for key in ("changed_paths", "missing_paths", "missing_hash_paths", "invalid_paths"):
        result[key] = sorted(set(result[key]))
    result["fresh"] = not any(
        result[key]
        for key in ("changed_paths", "missing_paths", "missing_hash_paths", "invalid_paths")
    )
    return result


def _validate_artifacts(
    dataset_dir: Path,
    manifest: dict,
    artifact_result: dict,
    blockers: list[str],
) -> dict[str, list[dict] | None]:
    artifact_rows: dict[str, list[dict] | None] = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        _add_blocker(blockers, "dataset_artifacts_invalid")
        return artifact_rows
    for raw_path, metadata in sorted(artifacts.items()):
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or not isinstance(metadata, dict)
            or not _valid_sha256(metadata.get("sha256"))
            or not _valid_manifest_count(metadata.get("size_bytes"))
        ):
            artifact_result["invalid_paths"].append(_invalid_path_label(raw_path))
            continue
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            artifact_result["invalid_paths"].append(raw_path)
            continue
        path = dataset_dir / relative
        try:
            reject_symlink_paths([path], "Dataset artifact")
            reject_symlink_components(path, "Dataset artifact")
        except ValueError:
            artifact_result["invalid_paths"].append(raw_path)
            continue
        if not path.is_file():
            artifact_result["missing_paths"].append(raw_path)
            continue
        artifact_result["checked"] += 1
        try:
            artifact_bytes = _safe_read_bytes(path, "Dataset artifact")
        except (OSError, ValueError):
            artifact_result["invalid_paths"].append(raw_path)
            continue
        if len(artifact_bytes) != metadata["size_bytes"]:
            artifact_result["size_mismatch_paths"].append(raw_path)
        if hashlib.sha256(artifact_bytes).hexdigest() != metadata["sha256"]:
            artifact_result["changed_paths"].append(raw_path)
        if raw_path in REQUIRED_ARTIFACTS:
            artifact_rows[raw_path] = _strict_jsonl_objects(artifact_bytes)
    if artifact_result["invalid_paths"]:
        _add_blocker(blockers, "dataset_artifacts_invalid")
    if artifact_result["missing_paths"]:
        _add_blocker(blockers, "dataset_artifact_missing")
    if artifact_result["changed_paths"]:
        _add_blocker(blockers, "dataset_artifact_hash_mismatch")
    if artifact_result["size_mismatch_paths"]:
        _add_blocker(blockers, "dataset_artifact_size_mismatch")

    selected_file = manifest.get("selected_dataset_file")
    selected_meta = artifacts.get(selected_file) if isinstance(selected_file, str) else None
    if (
        not isinstance(selected_meta, dict)
        or manifest.get("dataset_sha256") != selected_meta.get("sha256")
    ):
        _add_blocker(blockers, "dataset_selected_artifact_invalid")
    return artifact_rows


def _snapshot_source_authority(manifest: dict) -> tuple[Path | None, dict | None]:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return None, None
    if provenance.get("source_scope") not in SOURCE_SCOPES:
        return None, provenance
    raw_source_root = provenance.get("source_root")
    if not isinstance(raw_source_root, str) or not raw_source_root:
        return None, provenance
    source_root = Path(raw_source_root).expanduser()
    try:
        if source_root.is_symlink():
            return None, provenance
        reject_symlink_components(source_root, "Dataset source authority")
        source_root = source_root.resolve()
    except (OSError, ValueError):
        return None, provenance
    return (source_root if source_root.is_dir() else None), provenance


def _validate_source_authority(
    source_root: Path | None,
    provenance: dict | None,
    manifest: dict,
    artifact_rows: dict[str, list[dict] | None],
) -> tuple[bool, bool]:
    source_scope = provenance.get("source_scope") if provenance else None
    if source_scope not in SOURCE_SCOPES:
        return True, True
    if source_root is None:
        return False, False
    include_corrections = manifest.get("include_corrections")
    include_refusals = manifest.get("include_refusals")
    if type(include_corrections) is not bool or type(include_refusals) is not bool:
        return False, False
    try:
        authority_binding_valid = True
        if source_scope in {ACCEPTED_REVIEW_SCOPE, LAB_REVIEW_SCOPE}:
            authority_candidates = ReviewStore(source_root).load_candidates()
        else:
            active_authority = capture_active_state_authority(source_root)
            authority_binding_valid = bool(
                provenance.get("source_receipt_id")
                == active_authority["receipt_id"]
                and provenance.get("source_receipt_sha256")
                == active_authority["receipt_sha256"]
                and manifest.get("source_receipt_id")
                == active_authority["receipt_id"]
            )
            authority_candidates = _active_authority_candidates(active_authority)
        candidate_ids = [candidate.id for candidate in authority_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            return False, False
        expected = _expected_authority_dataset(
            source_root,
            authority_candidates,
            include_corrections=include_corrections,
            include_refusals=include_refusals,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False, False

    eval_rows = artifact_rows.get("eval.seed.jsonl")
    skipped_rows = artifact_rows.get("skipped.jsonl")
    if eval_rows is None or skipped_rows is None:
        return False, False
    actual_candidate_ids = [
        row.get("source_candidate_id")
        for row in eval_rows
        if row.get("source_candidate_id") is not None
    ]
    actual_skipped_ids = [row.get("candidate_id") for row in skipped_rows]
    if (
        any(not isinstance(candidate_id, str) for candidate_id in actual_candidate_ids)
        or any(not isinstance(candidate_id, str) for candidate_id in actual_skipped_ids)
    ):
        return False, False
    actual_candidate_rows = {
        row.get("source_candidate_id"): row
        for row in eval_rows
        if row.get("source_candidate_id") is not None
    }
    expected_candidate_rows = {
        row["source_candidate_id"]: row
        for row in expected["eval.seed.jsonl"]
        if row.get("source_candidate_id") is not None
    }
    metadata_fields = (
        "source_path",
        "kind",
        "semantic_class",
        "trainability_status",
        "memory_route",
    )
    actual_skipped_ids = sorted(actual_skipped_ids)
    expected_skipped_ids = sorted(
        row["candidate_id"] for row in expected["skipped.jsonl"]
    )
    eligible_candidates = expected["eligible_candidates"]
    expected_ids = sorted(candidate.id for candidate in eligible_candidates)
    expected_paths = sorted({candidate.source_path for candidate in eligible_candidates})
    authority_valid = bool(
        authority_binding_valid
        and manifest.get("candidate_count") == len(authority_candidates)
        and sorted(actual_candidate_rows) == expected_ids
        and actual_skipped_ids == expected_skipped_ids
        and manifest.get("source_candidate_ids") == expected_ids
        and manifest.get("source_paths") == expected_paths
        and manifest.get("class_counts")
        == dict(sorted(Counter(
            candidate.semantic_class for candidate in eligible_candidates
        ).items()))
        and manifest.get("trainability_counts")
        == dict(sorted(Counter(
            candidate.trainability_status for candidate in eligible_candidates
        ).items()))
        and manifest.get("route_counts")
        == dict(sorted(Counter(
            candidate.memory_route for candidate in eligible_candidates
        ).items()))
        and manifest.get("trainable_candidate_count")
        == sum(
            candidate.memory_route == "adapter_training"
            for candidate in eligible_candidates
        )
        and all(
            candidate_id in expected_candidate_rows
            and all(
                actual_row.get(field) == expected_candidate_rows[candidate_id].get(field)
                for field in metadata_fields
            )
            for candidate_id, actual_row in actual_candidate_rows.items()
        )
    )
    generated_valid = _generated_artifacts_match(expected, artifact_rows)
    return authority_valid, generated_valid


def _active_authority_candidates(authority: dict) -> list[SemanticCandidate]:
    state = authority["state"]
    evidence_by_claim = {
        str(item.get("claim_id")): item
        for item in authority["evidence_rows"]
        if isinstance(item, dict)
    }
    candidates = []
    timestamp = datetime.now(timezone.utc)
    claims = state.get("claims", []) if isinstance(state, dict) else []
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("status", "active") != "active":
            continue
        evidence = evidence_by_claim.get(str(claim.get("id")))
        if not evidence:
            continue
        excerpt = str(
            evidence.get("excerpt") or claim.get("excerpt") or ""
        ).strip()
        source_path = str(evidence.get("path") or "")
        source_sha256 = str(evidence.get("source_sha256") or "")
        if not excerpt or not source_path or not source_sha256:
            continue
        evidence_sha256 = str(evidence.get("excerpt_sha256") or "")
        if len(evidence_sha256) != 64:
            evidence_sha256 = hashlib.sha256(excerpt.encode()).hexdigest()
        candidates.append(route_candidate(SemanticCandidate(
            id=f"active_{claim.get('id')}",
            run_id=str(state.get("receipt_id") or "active_state"),
            kind=_active_claim_kind(str(claim.get("category") or "")),
            claim=str(claim.get("excerpt") or excerpt),
            source_path=source_path,
            source_sha256=source_sha256,
            source_mtime=timestamp,
            source_revision=f"state:{state.get('receipt_id') or 'unknown'}",
            line_start=int(
                evidence.get("line_start") or claim.get("line_start") or 1
            ),
            line_end=int(
                evidence.get("line_end")
                or claim.get("line_end")
                or evidence.get("line_start")
                or 1
            ),
            evidence_excerpt=excerpt,
            evidence_sha256=evidence_sha256,
            confidence=1.0,
            label="source_backed",
            status="accepted",
            created_at=timestamp,
            provider={"name": "active-state", "model": "local"},
            prompt_sha256="0" * 64,
        )))
    return candidates


def _active_claim_kind(category: str) -> str:
    return {
        "decision": "active_decision",
        "task": "open_task",
        "agent_rule": "agent_rule",
        "source_reference": "source_reference",
        "outdated": "outdated_claim",
    }.get(category, "current_state")


def _expected_authority_dataset(
    source_root: Path,
    review_candidates: list[SemanticCandidate],
    *,
    include_corrections: bool,
    include_refusals: bool,
) -> dict:
    ignore_patterns = load_morpheusignore(source_root)
    routed_candidates = [route_candidate(candidate) for candidate in review_candidates]
    eligible_candidates = []
    skipped_rows = []
    for candidate in routed_candidates:
        eligible, reason = _authority_eligible_candidate(
            source_root,
            candidate,
            ignore_patterns=ignore_patterns,
            include_corrections=include_corrections,
        )
        if eligible is None:
            skipped_rows.append(_authority_skip_record(candidate, reason))
        else:
            eligible_candidates.append(eligible)

    instruction_rows = []
    eval_rows = []
    heldout_rows = []
    for candidate in eligible_candidates:
        if candidate.memory_route in {"adapter_training", "negative_example"}:
            instruction_rows.extend(instruction_examples_for_candidate(candidate))
        eval_rows.extend(eval_items_for_candidate(candidate))
        heldout_rows.extend(heldout_eval_items_for_candidate(candidate))
    if include_refusals:
        eval_rows.append(unsupported_claim_eval_item())
        eval_rows.extend(truth_gate_negative_eval_items())
        heldout_rows.extend(heldout_truth_gate_negative_eval_items())

    chat_rows = chat_examples_from_instruction(instruction_rows)
    split_rows = _expected_chat_splits(chat_rows)
    return {
        "dataset.instruction.jsonl": instruction_rows,
        "dataset.sharegpt.jsonl": sharegpt_examples_from_instruction(instruction_rows),
        "eval.seed.jsonl": eval_rows,
        "eval.heldout.jsonl": heldout_rows,
        "skipped.jsonl": skipped_rows,
        "train.jsonl": split_rows["train"],
        "valid.jsonl": split_rows["valid"],
        "test.jsonl": split_rows["test"],
        "eligible_candidates": eligible_candidates,
    }


def _authority_eligible_candidate(
    source_root: Path,
    candidate: SemanticCandidate,
    *,
    ignore_patterns: set[str],
    include_corrections: bool,
) -> tuple[SemanticCandidate | None, str]:
    if candidate.status != "accepted":
        return None, f"status_{candidate.status}"
    if candidate.label != "source_backed":
        return None, f"label_{candidate.label}"
    if candidate.kind == "outdated_claim" and not include_corrections:
        return None, "corrections_disabled"
    if candidate.kind not in POSITIVE_KINDS and candidate.kind != "outdated_claim":
        return None, f"kind_{candidate.kind}"

    if not canonical_source_path(candidate.source_path):
        return None, "invalid_source_path"
    rel_path = Path(candidate.source_path)
    if path_is_ignored(rel_path, ignore_patterns):
        return None, "ignored_path"
    source_path = source_root / rel_path
    try:
        reject_symlink_paths([source_path], "Learning source authority")
        reject_symlink_components(source_path, "Learning source authority")
    except ValueError:
        return None, "unsafe_source_path"
    if not source_path.is_file():
        return None, "missing_source_path"
    try:
        current_sha = compute_sha256(source_path)
    except (OSError, ValueError):
        return None, "unreadable_source_path"
    if current_sha != candidate.source_sha256:
        return None, "source_sha256_mismatch"

    verified = verify_candidate_span(source_root, candidate)
    if verified.label != "source_backed":
        return None, "invalid_source_span"
    if (
        contains_secret_like_text(candidate.claim)
        or contains_secret_like_text(candidate.evidence_excerpt)
        or contains_secret_like_text(candidate.correction_text or "")
    ):
        return None, "secret_like"
    projection_error = team_feedback_projection_error(verified)
    if projection_error:
        return None, projection_error
    return route_candidate(verified), ""


def _authority_skip_record(candidate: SemanticCandidate, reason: str) -> dict:
    return {
        "candidate_id": candidate.id,
        "reason": reason,
        "kind": candidate.kind,
        "semantic_class": candidate.semantic_class,
        "trainability_status": candidate.trainability_status,
        "memory_route": candidate.memory_route,
        "source_path": candidate.source_path,
        "line_start": candidate.line_start,
        "line_end": candidate.line_end,
    }


def _generated_artifacts_match(
    expected: dict,
    artifact_rows: dict[str, list[dict] | None],
) -> bool:
    instruction_rows = artifact_rows.get("dataset.instruction.jsonl")
    if instruction_rows is None or not _rows_match_unordered(
        instruction_rows,
        expected["dataset.instruction.jsonl"],
    ):
        return False
    expected_sharegpt = sharegpt_examples_from_instruction(instruction_rows)
    expected_splits = _expected_chat_splits(
        chat_examples_from_instruction(instruction_rows)
    )
    return bool(
        artifact_rows.get("dataset.sharegpt.jsonl") == expected_sharegpt
        and artifact_rows.get("train.jsonl") == expected_splits["train"]
        and artifact_rows.get("valid.jsonl") == expected_splits["valid"]
        and artifact_rows.get("test.jsonl") == expected_splits["test"]
        and _rows_match_unordered(
            artifact_rows.get("eval.seed.jsonl"),
            expected["eval.seed.jsonl"],
        )
        and _rows_match_unordered(
            artifact_rows.get("eval.heldout.jsonl"),
            expected["eval.heldout.jsonl"],
        )
        and _rows_match_unordered(
            artifact_rows.get("skipped.jsonl"),
            expected["skipped.jsonl"],
        )
    )


def _rows_match_unordered(actual: object, expected: list[dict]) -> bool:
    if not isinstance(actual, list):
        return False
    return Counter(_canonical_row(row) for row in actual) == Counter(
        _canonical_row(row) for row in expected
    )


def _canonical_row(row: dict) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _expected_chat_splits(rows: list[dict]) -> dict[str, list[dict]]:
    if not rows:
        return {"train": [], "valid": [], "test": []}
    if len(rows) == 1:
        return {"train": rows, "valid": rows, "test": rows}
    if len(rows) == 2:
        return {"train": rows[:1], "valid": rows[1:], "test": rows[1:]}
    required = _expand_required_rows(
        row for row in rows if _train_required_row(row)
    )
    remaining = [row for row in rows if not _train_required_row(row)]
    if len(rows) < 20:
        return {
            "train": [*required, *rows[:-2]],
            "valid": rows[-2:-1],
            "test": rows[-1:],
        }

    train_end = max(1, int(len(rows) * 0.8))
    valid_target = max(1, int(len(rows) * 0.1))
    train = list(required)
    valid = []
    test = []
    for row in remaining:
        if len(train) < train_end:
            train.append(row)
        elif len(valid) < valid_target:
            valid.append(row)
        else:
            test.append(row)
    if not valid:
        valid = rows[-2:-1]
    if not test:
        test = rows[-1:]
    return {"train": train, "valid": valid, "test": test}


def _train_required_row(row: dict) -> bool:
    metadata = row.get("metadata") if isinstance(row, dict) else None
    return bool(
        isinstance(metadata, dict)
        and str(metadata.get("example_type") or "") in TRAIN_REQUIRED_EXAMPLE_TYPES
    )


def _source_bound_training_metadata(row: object) -> dict | None:
    if not isinstance(row, dict):
        return None
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    candidate_id = metadata.get("source_candidate_id")
    source_path = metadata.get("source_path")
    line_start = metadata.get("line_start")
    line_end = metadata.get("line_end")
    evidence_sha256 = metadata.get("evidence_sha256")
    kind = metadata.get("kind")
    semantic_class = metadata.get("semantic_class")
    trainability_status = metadata.get("trainability_status")
    memory_route = metadata.get("memory_route")
    if not (
        isinstance(candidate_id, str)
        and candidate_id.strip()
        and isinstance(source_path, str)
        and canonical_source_path(source_path)
        and type(line_start) is int
        and line_start >= 1
        and type(line_end) is int
        and line_end >= line_start
        and _valid_sha256(evidence_sha256)
        and isinstance(kind, str)
        and kind.strip()
        and isinstance(semantic_class, str)
        and semantic_class.strip()
        and isinstance(trainability_status, str)
        and trainability_status.strip()
        and memory_route in {"adapter_training", "negative_example"}
    ):
        return None
    if (
        memory_route == "adapter_training"
        and trainability_status != "trainable"
    ) or (
        memory_route == "negative_example"
        and trainability_status != "negative_example"
    ):
        return None
    return metadata


def _expand_required_rows(rows: Iterable[dict]) -> list[dict]:
    expanded = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row, dict) else {}
        example_type = str(metadata.get("example_type") or "")
        expanded.extend(
            row for _ in range(TRAIN_EXAMPLE_REPEATS.get(example_type, 1))
        )
    return expanded


def _validate_manifest_semantics(
    manifest: dict,
    artifact_rows: dict[str, list[dict] | None],
) -> tuple[bool, dict]:
    rows = {
        path: artifact_rows.get(path)
        for path in REQUIRED_ARTIFACTS
    }
    if any(value is None for value in rows.values()):
        return False, _empty_eval_coverage()

    instruction_rows = rows["dataset.instruction.jsonl"] or []
    sharegpt_rows = rows["dataset.sharegpt.jsonl"] or []
    eval_rows = rows["eval.seed.jsonl"] or []
    heldout_rows = rows["eval.heldout.jsonl"] or []
    skipped_rows = rows["skipped.jsonl"] or []
    split_rows = {
        split: rows[f"{split}.jsonl"] or []
        for split in ("train", "valid", "test")
    }
    valid = True
    training_metadata = []
    for item in [
        *instruction_rows,
        *sharegpt_rows,
        *split_rows["train"],
        *split_rows["valid"],
        *split_rows["test"],
    ]:
        metadata = _source_bound_training_metadata(item)
        if metadata is None:
            valid = False
        else:
            training_metadata.append(metadata)

    valid = bool(
        valid
        and manifest.get("examples_count") == len(instruction_rows)
        and manifest.get("examples_count") == len(sharegpt_rows)
        and manifest.get("eval_items_count") == len(eval_rows)
        and manifest.get("heldout_eval_items_count") == len(heldout_rows)
        and manifest.get("skipped_count") == len(skipped_rows)
        and manifest.get("split_counts")
        == {split: len(items) for split, items in split_rows.items()}
    )

    selected_format = manifest.get("selected_format")
    format_binding = DATASET_FORMAT_BINDINGS.get(selected_format)
    format_versions = manifest.get("format_versions")
    if (
        format_binding is None
        or not isinstance(format_versions, dict)
        or manifest.get("selected_dataset_file") != format_binding[0]
        or manifest.get("format_version") != format_binding[1]
        or format_versions.get(selected_format) != format_binding[1]
        or format_versions.get("eval_seed") != "morpheus-eval-seed/2"
        or format_versions.get("heldout_eval") != "morpheus-heldout-eval/2"
        or format_versions.get("benchmark_categories") != BENCHMARK_CATEGORY_SCHEMA
    ):
        valid = False

    eval_categories = Counter()
    candidates: dict[str, dict] = {}
    for item in eval_rows:
        category = item.get("category")
        if category not in KNOWN_BENCHMARK_CATEGORIES:
            valid = False
            continue
        eval_categories[category] += 1
        candidate_id = item.get("source_candidate_id")
        if candidate_id is None:
            continue
        metadata = {
            "source_path": item.get("source_path"),
            "line_start": item.get("line_start"),
            "line_end": item.get("line_end"),
            "evidence_sha256": item.get("evidence_sha256"),
            "kind": item.get("kind"),
            "semantic_class": item.get("semantic_class"),
            "trainability_status": item.get("trainability_status"),
            "memory_route": item.get("memory_route"),
        }
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or candidate_id in candidates
            or not isinstance(metadata["source_path"], str)
            or not canonical_source_path(metadata["source_path"])
            or type(metadata["line_start"]) is not int
            or metadata["line_start"] < 1
            or type(metadata["line_end"]) is not int
            or metadata["line_end"] < metadata["line_start"]
            or not _valid_sha256(metadata["evidence_sha256"])
            or any(
                not isinstance(value, str) or not value.strip()
                for field, value in metadata.items()
                if field not in {
                    "source_path",
                    "line_start",
                    "line_end",
                    "evidence_sha256",
                }
            )
        ):
            valid = False
            continue
        candidates[candidate_id] = metadata

    for metadata in training_metadata:
        candidate = candidates.get(metadata["source_candidate_id"])
        if candidate is None or any(
            metadata.get(field) != candidate.get(field)
            for field in (
                "source_path",
                "line_start",
                "line_end",
                "evidence_sha256",
                "kind",
                "semantic_class",
                "trainability_status",
                "memory_route",
            )
        ):
            valid = False

    skipped_candidate_ids = []
    for item in skipped_rows:
        candidate_id = item.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or candidate_id in candidates
            or candidate_id in skipped_candidate_ids
        ):
            valid = False
            continue
        skipped_candidate_ids.append(candidate_id)

    candidate_ids = sorted(candidates)
    source_paths = sorted({item["source_path"] for item in candidates.values()})
    if any(not canonical_source_path(path) for path in source_paths):
        valid = False
    class_counts = dict(sorted(Counter(
        item["semantic_class"] for item in candidates.values()
    ).items()))
    trainability_counts = dict(sorted(Counter(
        item["trainability_status"] for item in candidates.values()
    ).items()))
    route_counts = dict(sorted(Counter(
        item["memory_route"] for item in candidates.values()
    ).items()))
    trainable_candidate_count = sum(
        item["memory_route"] == "adapter_training"
        for item in candidates.values()
    )
    if not (
        manifest.get("candidate_count")
        == len(candidate_ids) + len(skipped_candidate_ids)
        and manifest.get("source_candidate_ids") == candidate_ids
        and manifest.get("source_paths") == source_paths
        and manifest.get("class_counts") == class_counts
        and manifest.get("trainability_counts") == trainability_counts
        and manifest.get("route_counts") == route_counts
        and manifest.get("trainable_candidate_count")
        == trainable_candidate_count
    ):
        valid = False
    return valid, {
        "total_items": len(eval_rows),
        "by_category": dict(sorted(eval_categories.items())),
    }


def _strict_jsonl_objects(data: bytes) -> list[dict] | None:
    try:
        text = data.decode("utf-8")
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(item, dict) or not _finite_json_value(item):
                return None
            rows.append(item)
        return rows
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def canonical_source_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        value
        and "\\" not in value
        and not path.is_absolute()
        and ".." not in path.parts
        and path.parts
        and path.as_posix() == value
    )


def _empty_source_freshness(available: bool) -> dict:
    return {
        "available": available,
        "fresh": available,
        "checked_paths": 0,
        "changed_paths": [],
        "missing_paths": [],
        "missing_hash_paths": [],
        "invalid_paths": [],
    }


def _empty_eval_coverage() -> dict:
    return {"total_items": 0, "by_category": {}}


def _valid_v2_manifest_fields(manifest: dict) -> bool:
    if not _finite_json_value(manifest):
        return False
    if any(
        not _valid_manifest_count(manifest.get(field))
        for field in MANIFEST_COUNT_FIELDS
    ):
        return False
    for field in MANIFEST_COUNT_MAP_FIELDS:
        counts = manifest.get(field)
        if not isinstance(counts, dict) or any(
            not isinstance(key, str)
            or not key
            or not _valid_manifest_count(value)
            for key, value in counts.items()
        ):
            return False
    source_paths = manifest.get("source_paths")
    if not (
        isinstance(source_paths, list)
        and all(
            isinstance(source_path, str) and bool(source_path.strip())
            for source_path in source_paths
        )
    ):
        return False
    artifacts = manifest.get("artifacts")
    return bool(
        isinstance(artifacts, dict)
        and all(
            isinstance(metadata, dict)
            and _valid_manifest_count(metadata.get("size_bytes"))
            for metadata in artifacts.values()
        )
    )


def _valid_manifest_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _finite_json_value(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite_json_value(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_finite_json_value(item) for item in value)
    return value is None or isinstance(value, (str, int, bool))


def _read_manifest(path: Path) -> dict:
    reject_symlink_paths([path], "Dataset manifest")
    reject_symlink_components(path, "Dataset manifest")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Dataset manifest must be a JSON object")
    return data


def _read_active_state_context(project_root: Path) -> dict:
    contents = {}
    hashes = {}
    for manifest_path, (morpheus_name, required) in ACTIVE_STATE_CONTEXT_PATHS.items():
        path = (
            project_root / ".morpheus" / str(morpheus_name)
            if morpheus_name is not None
            else project_root / manifest_path
        )
        if not path.is_file():
            if required:
                raise ValueError(f"Active-state artifact missing: {manifest_path}")
            continue
        data = _safe_read_bytes(path, "Active-state artifact")
        contents[manifest_path] = data
        hashes[manifest_path] = hashlib.sha256(data).hexdigest()
    return {"bytes": contents, "hashes": dict(sorted(hashes.items()))}


def _safe_read_bytes(path: Path, label: str) -> bytes:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    return path.read_bytes()


def _lab_workspace_id(project_root: Path) -> str | None:
    if (
        project_root.name == "workspace"
        and project_root.parent.name.startswith("lab_")
        and project_root.parent.parent.name == "lab"
        and project_root.parent.parent.parent.name == ".morpheus"
    ):
        return project_root.parent.name
    return None


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _add_blocker(blockers: list[str], blocker: str) -> None:
    if blocker not in blockers:
        blockers.append(blocker)


def _invalid_path_label(value: object) -> str:
    if value == "":
        return "<empty>"
    return json.dumps(value, sort_keys=True, default=str)
