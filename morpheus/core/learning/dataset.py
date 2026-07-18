"""Compile reviewed semantic candidates into local training datasets."""
from collections import Counter
from collections.abc import Callable
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from morpheus.core.compiler import compute_sha256
from morpheus.core.learning.authority import learning_authority_transaction
from morpheus.core.learning.categories import BENCHMARK_CATEGORY_SCHEMA
from morpheus.core.learning.evals import (
    eval_items_for_candidate,
    heldout_eval_items_for_candidate,
    heldout_truth_gate_negative_eval_items,
    truth_gate_negative_eval_items,
    unsupported_claim_eval_item,
)
from morpheus.core.learning.dataset_validation import (
    MANIFEST_FORMAT_VERSION,
    artifact_manifest,
    build_dataset_provenance,
    canonical_source_path,
    canonical_review_snapshot,
    capture_active_state_authority,
    dataset_binding_sha256,
)
from morpheus.core.learning.examples import (
    CHAT_FORMAT_VERSION,
    INSTRUCTION_FORMAT_VERSION,
    SHAREGPT_FORMAT_VERSION,
    chat_examples_from_instruction,
    instruction_examples_for_candidate,
    sharegpt_examples_from_instruction,
)
from morpheus.core.learning.registry import datasets_root
from morpheus.core.learning.safety import (
    contains_secret_like_text,
    load_morpheusignore,
    path_is_ignored,
)
from morpheus.core.learning.team import team_feedback_projection_error
from morpheus.core.provenance import compute_sha256_file, latest_receipt_file
from morpheus.core.portable_lock import portable_file_lock
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.semantic.routing import route_candidate
from morpheus.core.semantic.verifier import verify_candidate_span
from morpheus.core.state_authority import state_authority_transaction


DATASET_FORMATS = {"instruction", "sharegpt", "chat"}
DATASET_SOURCES = {"accepted", "active-state"}
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


@dataclass
class Eligibility:
    candidate: SemanticCandidate
    trainable_positive: bool


def build_learning_dataset(
    project_root: Path,
    *,
    dataset_format: str = "instruction",
    source: str = "accepted",
    include_corrections: bool = True,
    include_refusals: bool = True,
    output: Path | None = None,
) -> dict:
    """Build a reviewed learning dataset without reading raw files for examples."""
    if dataset_format not in DATASET_FORMATS:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    if source not in DATASET_SOURCES:
        raise ValueError(f"Unsupported dataset source: {source}")
    project_root = _safe_project_root(project_root)
    review_store = ReviewStore(project_root) if source == "accepted" else None
    review_candidates = None
    active_authority = None
    if review_store is not None:
        with review_store.transaction():
            review_candidates = review_store.load_candidates()
            canonical_review_snapshot(review_candidates)
        candidates = [route_candidate(candidate) for candidate in review_candidates]
    else:
        active_authority = capture_active_state_authority(project_root)
        candidates = _active_state_candidates(project_root, active_authority)
    ignore_patterns = load_morpheusignore(project_root)

    eligible: list[Eligibility] = []
    skipped: list[dict] = []
    source_hashes: dict[str, str] = {}
    prompt_sha256_values = sorted({
        candidate.prompt_sha256
        for candidate in candidates
        if getattr(candidate, "prompt_sha256", None)
    })

    for candidate in candidates:
        eligibility, reason, current_sha = _eligible_candidate(
            project_root,
            candidate,
            ignore_patterns=ignore_patterns,
            include_corrections=include_corrections,
        )
        if current_sha:
            source_hashes[candidate.source_path] = current_sha
        if eligibility is None:
            skipped.append(_skip_record(candidate, reason))
            continue
        eligible.append(eligibility)

    instruction_examples: list[dict] = []
    eval_items: list[dict] = []
    heldout_items: list[dict] = []
    for item in eligible:
        if item.candidate.memory_route in {"adapter_training", "negative_example"}:
            instruction_examples.extend(instruction_examples_for_candidate(item.candidate))
        eval_items.extend(eval_items_for_candidate(item.candidate))
        heldout_items.extend(heldout_eval_items_for_candidate(item.candidate))
    if include_refusals:
        eval_items.append(unsupported_claim_eval_item())
        eval_items.extend(truth_gate_negative_eval_items())
        heldout_items.extend(heldout_truth_gate_negative_eval_items())

    sharegpt_examples = sharegpt_examples_from_instruction(instruction_examples)
    chat_examples = chat_examples_from_instruction(instruction_examples)
    split_rows = _split_chat_rows(chat_examples)
    dataset_id = _dataset_id()
    registry_root = _validated_datasets_root(project_root)
    out_dir = registry_root / dataset_id
    staging_dir, staging_identity = _create_private_staging_dir(
        registry_root,
        dataset_id,
    )

    instruction_path = staging_dir / "dataset.instruction.jsonl"
    sharegpt_path = staging_dir / "dataset.sharegpt.jsonl"
    skipped_path = staging_dir / "skipped.jsonl"
    eval_path = staging_dir / "eval.seed.jsonl"
    heldout_eval_path = staging_dir / "eval.heldout.jsonl"
    manifest_path = staging_dir / "manifest.json"
    train_path = staging_dir / "train.jsonl"
    valid_path = staging_dir / "valid.jsonl"
    test_path = staging_dir / "test.jsonl"
    _write_jsonl(instruction_path, instruction_examples)
    _write_jsonl(sharegpt_path, sharegpt_examples)
    _write_jsonl(train_path, split_rows["train"])
    _write_jsonl(valid_path, split_rows["valid"])
    _write_jsonl(test_path, split_rows["test"])
    _write_jsonl(skipped_path, skipped)
    _write_jsonl(eval_path, eval_items)
    _write_jsonl(heldout_eval_path, heldout_items)

    canonical_selected_path = _selected_dataset_path(
        dataset_format,
        instruction_path=instruction_path,
        sharegpt_path=sharegpt_path,
        train_path=train_path,
    )
    context_hashes = (
        dict(active_authority["context_hashes"])
        if active_authority is not None
        else {}
    )
    source_hashes.update(context_hashes)
    source_receipt_id = (
        str(active_authority["receipt_id"])
        if active_authority is not None
        else _source_receipt_id(project_root)
    )
    source_receipt_sha256 = (
        str(active_authority["receipt_sha256"])
        if active_authority is not None
        else None
    )
    provenance = build_dataset_provenance(
        project_root,
        source=source,
        review_candidates=review_candidates,
        source_receipt_id=source_receipt_id,
        source_receipt_sha256=source_receipt_sha256,
        context_paths=context_hashes,
    )
    artifacts = artifact_manifest(staging_dir, [
        instruction_path,
        sharegpt_path,
        train_path,
        valid_path,
        test_path,
        skipped_path,
        eval_path,
        heldout_eval_path,
    ])
    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "source_receipt_id": source_receipt_id,
        "candidate_count": len(candidates),
        "trainable_candidate_count": sum(1 for item in eligible if item.trainable_positive),
        "examples_count": len(instruction_examples),
        "eval_items_count": len(eval_items),
        "heldout_eval_items_count": len(heldout_items),
        "skipped_count": len(skipped),
        "split_counts": {key: len(value) for key, value in split_rows.items()},
        "class_counts": dict(Counter(item.candidate.semantic_class for item in eligible)),
        "trainability_counts": dict(Counter(item.candidate.trainability_status for item in eligible)),
        "route_counts": dict(Counter(item.candidate.memory_route for item in eligible)),
        "smoke_mode": len(instruction_examples) < 20,
        "source_candidate_ids": sorted({
            item.candidate.id
            for item in eligible
            if item.candidate.kind != "outdated_claim" or include_corrections
        }),
        "source_paths": sorted({
            item.candidate.source_path
            for item in eligible
            if item.candidate.kind != "outdated_claim" or include_corrections
        }),
        "source_hashes": dict(sorted(source_hashes.items())),
        "prompt_sha256_values": prompt_sha256_values,
        "dataset_sha256": compute_sha256_file(canonical_selected_path),
        "selected_dataset_file": canonical_selected_path.name,
        "artifacts": artifacts,
        "provenance": provenance,
        "selected_format": dataset_format,
        "format_version": _format_version(dataset_format),
        "source": source,
        "include_corrections": include_corrections,
        "include_refusals": include_refusals,
        "format_versions": {
            "instruction": INSTRUCTION_FORMAT_VERSION,
            "sharegpt": SHAREGPT_FORMAT_VERSION,
            "chat": CHAT_FORMAT_VERSION,
            "eval_seed": "morpheus-eval-seed/2",
            "heldout_eval": "morpheus-heldout-eval/2",
            "manifest": MANIFEST_FORMAT_VERSION,
            "benchmark_categories": BENCHMARK_CATEGORY_SCHEMA,
        },
    }
    manifest["dataset_binding_sha256"] = dataset_binding_sha256(manifest)
    source_paths = manifest["source_paths"]
    if review_store is not None:
        with review_store.transaction():
            _ensure_review_authority_current(
                project_root,
                source_paths,
                source_hashes,
                review_store,
                provenance["review_snapshot"],
            )
            _write_private_text(manifest_path, _manifest_text(manifest))
            _publish_staged_dataset(
                staging_dir,
                out_dir,
                staging_identity=staging_identity,
                expected_manifest=manifest,
                authority_check=lambda: _ensure_review_authority_current(
                    project_root,
                    source_paths,
                    source_hashes,
                    review_store,
                    provenance["review_snapshot"],
                ),
            )
    else:
        with state_authority_transaction(project_root):
            _ensure_active_authority_current(
                project_root,
                active_authority,
                source_paths,
                source_hashes,
            )
            _write_private_text(manifest_path, _manifest_text(manifest))
            _publish_staged_dataset(
                staging_dir,
                out_dir,
                staging_identity=staging_identity,
                expected_manifest=manifest,
                authority_check=lambda: _ensure_active_authority_current(
                    project_root,
                    active_authority,
                    source_paths,
                    source_hashes,
                ),
            )
    published_selected_path = out_dir / canonical_selected_path.name
    selected_path = published_selected_path
    if output is not None:
        selected_path = _write_selected_output(
            project_root,
            output,
            published_selected_path.read_text(),
        )
    published_manifest_path = out_dir / "manifest.json"
    return {
        "dataset_id": dataset_id,
        "dataset_dir": str(out_dir),
        "selected_dataset_path": str(selected_path),
        "manifest_path": str(published_manifest_path),
        "examples_count": len(instruction_examples),
        "skipped_count": len(skipped),
    }


def _eligible_candidate(
    project_root: Path,
    candidate: SemanticCandidate,
    *,
    ignore_patterns: set[str],
    include_corrections: bool,
) -> tuple[Eligibility | None, str, str | None]:
    if candidate.status != "accepted":
        return None, f"status_{candidate.status}", None
    if candidate.label != "source_backed":
        return None, f"label_{candidate.label}", None
    if candidate.kind == "outdated_claim" and not include_corrections:
        return None, "corrections_disabled", None
    if candidate.kind not in POSITIVE_KINDS and candidate.kind != "outdated_claim":
        return None, f"kind_{candidate.kind}", None

    if not canonical_source_path(candidate.source_path):
        return None, "invalid_source_path", None
    rel_path = Path(candidate.source_path)
    if path_is_ignored(rel_path, ignore_patterns):
        return None, "ignored_path", None

    source_path = project_root / rel_path
    try:
        reject_symlink_paths([source_path], "Learning source path")
        reject_symlink_components(source_path, "Learning source path")
    except ValueError:
        return None, "unsafe_source_path", None
    if not source_path.is_file():
        return None, "missing_source_path", None

    try:
        current_sha = compute_sha256(source_path)
    except (OSError, ValueError):
        return None, "unreadable_source_path", None
    if current_sha != candidate.source_sha256:
        return None, "source_sha256_mismatch", current_sha

    verified = verify_candidate_span(project_root, candidate)
    if verified.label != "source_backed":
        return None, "invalid_source_span", current_sha
    if (
        contains_secret_like_text(candidate.claim)
        or contains_secret_like_text(candidate.evidence_excerpt)
        or contains_secret_like_text(candidate.correction_text or "")
    ):
        return None, "secret_like", current_sha
    projection_error = team_feedback_projection_error(verified)
    if projection_error:
        return None, projection_error, current_sha

    verified = route_candidate(verified)
    return Eligibility(
        candidate=verified,
        trainable_positive=verified.memory_route == "adapter_training",
    ), "", current_sha


def _active_state_candidates(
    project_root: Path,
    authority: dict,
) -> list[SemanticCandidate]:
    state = authority["state"]
    evidence_rows = authority["evidence_rows"]
    evidence_by_claim = {
        str(item.get("claim_id")): item
        for item in evidence_rows
        if isinstance(item, dict)
    }
    candidates = []
    timestamp = datetime.now(timezone.utc)
    for claim in state.get("claims", []):
        if not isinstance(claim, dict) or claim.get("status", "active") != "active":
            continue
        evidence = evidence_by_claim.get(str(claim.get("id")))
        if not evidence:
            continue
        excerpt = str(evidence.get("excerpt") or claim.get("excerpt") or "").strip()
        source_path = str(evidence.get("path") or "")
        source_sha = str(evidence.get("source_sha256") or "")
        if not excerpt or not source_path or not source_sha:
            continue
        evidence_sha = str(evidence.get("excerpt_sha256") or "")
        if len(evidence_sha) != 64:
            evidence_sha = hashlib.sha256(excerpt.encode()).hexdigest()
        candidates.append(route_candidate(SemanticCandidate(
            id=f"active_{claim.get('id')}",
            run_id=str(state.get("receipt_id") or "active_state"),
            kind=_kind_from_claim_category(str(claim.get("category") or "")),
            claim=str(claim.get("excerpt") or excerpt),
            source_path=source_path,
            source_sha256=source_sha,
            source_mtime=timestamp,
            source_revision=f"state:{state.get('receipt_id') or 'unknown'}",
            line_start=int(evidence.get("line_start") or claim.get("line_start") or 1),
            line_end=int(evidence.get("line_end") or claim.get("line_end") or evidence.get("line_start") or 1),
            evidence_excerpt=excerpt,
            evidence_sha256=evidence_sha,
            confidence=1.0,
            label="source_backed",
            status="accepted",
            created_at=timestamp,
            provider={"name": "active-state", "model": "local"},
            prompt_sha256="0" * 64,
        )))
    return candidates


def _kind_from_claim_category(category: str) -> str:
    return {
        "decision": "active_decision",
        "task": "open_task",
        "agent_rule": "agent_rule",
        "source_reference": "source_reference",
        "outdated": "outdated_claim",
    }.get(category, "current_state")


def _skip_record(candidate: SemanticCandidate, reason: str) -> dict:
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


def _selected_dataset_path(
    dataset_format: str,
    *,
    instruction_path: Path,
    sharegpt_path: Path,
    train_path: Path,
) -> Path:
    if dataset_format == "instruction":
        return instruction_path
    if dataset_format == "sharegpt":
        return sharegpt_path
    return train_path


def _format_version(dataset_format: str) -> str:
    return {
        "instruction": INSTRUCTION_FORMAT_VERSION,
        "sharegpt": SHAREGPT_FORMAT_VERSION,
        "chat": CHAT_FORMAT_VERSION,
    }[dataset_format]


def _split_chat_rows(rows: list[dict]) -> dict[str, list[dict]]:
    if not rows:
        return {"train": [], "valid": [], "test": []}
    if len(rows) == 1:
        return {"train": rows, "valid": rows, "test": rows}
    if len(rows) == 2:
        return {"train": rows[:1], "valid": rows[1:], "test": rows[1:]}
    required = _expand_required_rows(row for row in rows if _train_required_row(row))
    remaining = [row for row in rows if not _train_required_row(row)]
    if len(rows) < 20:
        train = [*required, *rows[:-2]]
        return {"train": train, "valid": rows[-2:-1], "test": rows[-1:]}

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
    if not isinstance(metadata, dict):
        return False
    return str(metadata.get("example_type") or "") in TRAIN_REQUIRED_EXAMPLE_TYPES


def _expand_required_rows(rows) -> list[dict]:
    expanded = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row, dict) else {}
        example_type = str(metadata.get("example_type") or "")
        repeats = TRAIN_EXAMPLE_REPEATS.get(example_type, 1)
        expanded.extend(row for _ in range(repeats))
    return expanded


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        key = json.dumps(row, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _safe_project_root(project_root: Path) -> Path:
    project_root = project_root.expanduser()
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")
    return project_root


def _validated_datasets_root(project_root: Path) -> Path:
    root = datasets_root(project_root)
    if root.is_symlink():
        raise ValueError(f"Dataset registry must not be a symlink: {root}")
    reject_symlink_components(root.parent, "Dataset registry")
    root.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(root, "Dataset registry")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Dataset registry must be a directory: {root}")
    return root.resolve()


def _create_private_staging_dir(
    registry_root: Path,
    dataset_id: str,
) -> tuple[Path, tuple[int, int]]:
    if (
        not dataset_id
        or dataset_id.startswith(".")
        or Path(dataset_id).name != dataset_id
        or "/" in dataset_id
        or "\\" in dataset_id
    ):
        raise ValueError(f"Dataset identity is unsafe: {dataset_id!r}")
    reject_symlink_components(registry_root, "Dataset registry")
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f".{dataset_id}.",
        suffix=".staging",
        dir=registry_root,
    ))
    os.chmod(staging_dir, 0o700)
    staging_stat = staging_dir.stat(follow_symlinks=False)
    if not stat.S_ISDIR(staging_stat.st_mode):
        raise ValueError(f"Staged dataset must be a directory: {staging_dir}")
    return staging_dir, (staging_stat.st_dev, staging_stat.st_ino)


def _publish_staged_dataset(
    staging_dir: Path,
    out_dir: Path,
    *,
    staging_identity: tuple[int, int],
    expected_manifest: dict,
    authority_check: Callable[[], None],
) -> None:
    registry_root = out_dir.parent
    if staging_dir.parent != registry_root:
        raise ValueError("Staged dataset is outside its dataset registry")
    reject_symlink_components(registry_root, "Dataset registry")
    lock_path = registry_root / ".registry.lock"
    reject_symlink_paths([lock_path], "Dataset registry lock")
    reject_symlink_components(lock_path, "Dataset registry lock")
    authority_root = registry_root.parents[2]
    with learning_authority_transaction(authority_root):
        with portable_file_lock(lock_path):
            if _descriptor_publish_supported():
                _publish_staged_dataset_with_descriptors(
                    registry_root,
                    staging_dir,
                    out_dir,
                    staging_identity=staging_identity,
                    expected_manifest=expected_manifest,
                    authority_check=authority_check,
                )
            else:  # pragma: no cover - descriptor APIs are available on POSIX.
                _publish_staged_dataset_with_paths(
                    staging_dir,
                    out_dir,
                    staging_identity=staging_identity,
                    expected_manifest=expected_manifest,
                    authority_check=authority_check,
                )


def _publish_staged_dataset_with_descriptors(
    registry_root: Path,
    staging_dir: Path,
    out_dir: Path,
    *,
    staging_identity: tuple[int, int],
    expected_manifest: dict,
    authority_check: Callable[[], None],
) -> None:
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    registry_descriptor = os.open(registry_root, directory_flags)
    staging_descriptor = -1
    try:
        try:
            staging_descriptor = os.open(
                staging_dir.name,
                directory_flags,
                dir_fd=registry_descriptor,
            )
        except OSError as exc:
            raise ValueError(
                "Dataset staging identity changed before publication"
            ) from exc
        _verify_staged_dataset_descriptor(
            staging_descriptor,
            staging_identity=staging_identity,
            expected_manifest=expected_manifest,
        )
        authority_check()
        _verify_staged_dataset_descriptor(
            staging_descriptor,
            staging_identity=staging_identity,
            expected_manifest=expected_manifest,
        )
        current = os.stat(
            staging_dir.name,
            dir_fd=registry_descriptor,
            follow_symlinks=False,
        )
        if (current.st_dev, current.st_ino) != staging_identity:
            raise ValueError("Dataset staging identity changed before publication")
        try:
            os.stat(
                out_dir.name,
                dir_fd=registry_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"Dataset output already exists: {out_dir}")
        _fsync_descriptor(staging_descriptor)
        os.rename(
            staging_dir.name,
            out_dir.name,
            src_dir_fd=registry_descriptor,
            dst_dir_fd=registry_descriptor,
        )
        published = os.stat(
            out_dir.name,
            dir_fd=registry_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(published.st_mode)
            or (published.st_dev, published.st_ino) != staging_identity
        ):
            raise ValueError("Published dataset identity changed during publication")
        _fsync_descriptor(registry_descriptor)
    finally:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        os.close(registry_descriptor)


def _publish_staged_dataset_with_paths(
    staging_dir: Path,
    out_dir: Path,
    *,
    staging_identity: tuple[int, int],
    expected_manifest: dict,
    authority_check: Callable[[], None],
) -> None:
    _verify_staged_dataset_path(
        staging_dir,
        staging_identity=staging_identity,
        expected_manifest=expected_manifest,
    )
    authority_check()
    _verify_staged_dataset_path(
        staging_dir,
        staging_identity=staging_identity,
        expected_manifest=expected_manifest,
    )
    reject_symlink_paths([out_dir], "Dataset output")
    if out_dir.exists() or out_dir.is_symlink():
        raise ValueError(f"Dataset output already exists: {out_dir}")
    current = staging_dir.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino) != staging_identity:
        raise ValueError("Dataset staging identity changed before publication")
    _fsync_directory_path(staging_dir)
    staging_dir.rename(out_dir)
    published = out_dir.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(published.st_mode)
        or (published.st_dev, published.st_ino) != staging_identity
    ):
        raise ValueError("Published dataset identity changed during publication")
    _fsync_directory_path(out_dir.parent)


def _verify_staged_dataset_descriptor(
    staging_descriptor: int,
    *,
    staging_identity: tuple[int, int],
    expected_manifest: dict,
) -> None:
    staging_stat = os.fstat(staging_descriptor)
    if (
        not stat.S_ISDIR(staging_stat.st_mode)
        or (staging_stat.st_dev, staging_stat.st_ino) != staging_identity
    ):
        raise ValueError("Dataset staging identity changed before publication")
    if os.name != "nt" and stat.S_IMODE(staging_stat.st_mode) != 0o700:
        raise ValueError("Staged dataset directory permissions changed")
    expected_names = _expected_staging_names(expected_manifest)
    actual_names = set(os.listdir(staging_descriptor))
    if actual_names != expected_names:
        raise ValueError("Staged dataset has unexpected entries; publication refused")
    contents = {
        name: _read_private_regular_file(name, dir_fd=staging_descriptor)
        for name in sorted(actual_names)
    }
    _verify_staged_dataset_contents(contents, expected_manifest)


def _verify_staged_dataset_path(
    staging_dir: Path,
    *,
    staging_identity: tuple[int, int],
    expected_manifest: dict,
) -> None:
    try:
        reject_symlink_components(staging_dir, "Staged dataset")
        staging_stat = staging_dir.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(
            "Dataset staging identity changed before publication"
        ) from exc
    if (
        not stat.S_ISDIR(staging_stat.st_mode)
        or (staging_stat.st_dev, staging_stat.st_ino) != staging_identity
    ):
        raise ValueError("Dataset staging identity changed before publication")
    expected_names = _expected_staging_names(expected_manifest)
    actual_names = {entry.name for entry in staging_dir.iterdir()}
    if actual_names != expected_names:
        raise ValueError("Staged dataset has unexpected entries; publication refused")
    contents = {
        name: _read_private_regular_file(staging_dir / name)
        for name in sorted(actual_names)
    }
    _verify_staged_dataset_contents(contents, expected_manifest)


def _expected_staging_names(manifest: dict) -> set[str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Staged dataset manifest artifacts changed")
    names = set(artifacts)
    if any(
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name.startswith(".")
        for name in names
    ):
        raise ValueError("Staged dataset manifest artifact names changed")
    return names | {"manifest.json"}


def _read_private_regular_file(
    path: str | Path,
    *,
    dir_fd: int | None = None,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise ValueError(
            f"Staged dataset entry must be a regular file: {path}"
        ) from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"Staged dataset entry must be a regular file: {path}")
        if os.name != "nt" and stat.S_IMODE(file_stat.st_mode) != 0o600:
            raise ValueError(f"Staged dataset file permissions changed: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_staged_dataset_contents(contents: dict[str, bytes], manifest: dict) -> None:
    if contents.get("manifest.json") != _manifest_text(manifest).encode():
        raise ValueError("Staged dataset manifest changed before publication")
    if manifest.get("dataset_binding_sha256") != dataset_binding_sha256(manifest):
        raise ValueError("Staged dataset manifest binding changed before publication")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Staged dataset manifest artifacts changed before publication")
    for name, metadata in artifacts.items():
        payload = contents.get(name)
        if (
            payload is None
            or not isinstance(metadata, dict)
            or metadata.get("size_bytes") != len(payload)
            or metadata.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError(f"Staged dataset artifact changed before publication: {name}")
    selected_name = manifest.get("selected_dataset_file")
    selected_metadata = artifacts.get(selected_name)
    if (
        not isinstance(selected_name, str)
        or not isinstance(selected_metadata, dict)
        or manifest.get("dataset_sha256") != selected_metadata.get("sha256")
    ):
        raise ValueError("Staged dataset selected artifact changed before publication")


def _descriptor_publish_supported() -> bool:
    return bool(
        os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
        and os.listdir in os.supports_fd
    )


def _fsync_descriptor(descriptor: int) -> None:
    if os.name != "nt":
        os.fsync(descriptor)


def _fsync_directory_path(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_review_authority_current(
    project_root: Path,
    source_paths: list[str],
    source_hashes: dict[str, str],
    review_store: ReviewStore,
    expected_review_snapshot: dict,
) -> None:
    if not _source_hashes_are_current(project_root, source_paths, source_hashes):
        raise ValueError(
            "Learning sources changed while compiling the dataset; rebuild required."
        )
    current_snapshot = canonical_review_snapshot(review_store.load_candidates())
    if current_snapshot != expected_review_snapshot:
        raise ValueError(
            "Review state changed while compiling the learning dataset; rebuild required."
        )


def _ensure_active_authority_current(
    project_root: Path,
    expected_authority: dict,
    source_paths: list[str],
    source_hashes: dict[str, str],
) -> None:
    current_authority = capture_active_state_authority(project_root)
    if (
        current_authority["context_hashes"] != expected_authority["context_hashes"]
        or current_authority["receipt_id"] != expected_authority["receipt_id"]
        or current_authority["receipt_sha256"] != expected_authority["receipt_sha256"]
    ):
        raise ValueError(
            "Active state changed while compiling the learning dataset; rebuild required."
        )
    if not _source_hashes_are_current(project_root, source_paths, source_hashes):
        raise ValueError(
            "Learning sources changed while compiling the dataset; rebuild required."
        )


def _source_hashes_are_current(
    project_root: Path,
    source_paths: list[str],
    source_hashes: dict[str, str],
) -> bool:
    for raw_path in source_paths:
        rel_path = Path(raw_path)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            return False
        source_path = project_root / rel_path
        try:
            reject_symlink_paths([source_path], "Learning source path")
            reject_symlink_components(source_path, "Learning source path")
            if (
                not source_path.is_file()
                or compute_sha256_file(source_path) != source_hashes.get(raw_path)
            ):
                return False
        except (OSError, ValueError):
            return False
    return True


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _write_private_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _write_private_text(path: Path, content: str) -> None:
    reject_symlink_components(path.parent, "Dataset artifact root")
    reject_symlink_paths([path], "Dataset artifact")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _write_selected_output(project_root: Path, output: Path, content: str) -> Path:
    output = output.expanduser()
    if not output.is_absolute():
        output = project_root / output
    output = Path(os.path.abspath(os.fspath(output)))
    reject_symlink_components(output.parent, "Dataset output")
    output.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(output.parent, "Dataset output")
    reject_symlink_paths([output], "Dataset output")
    if hasattr(os, "O_NOFOLLOW"):
        _write_selected_output_nofollow(output, content)
    else:  # pragma: no cover - O_NOFOLLOW is available on supported POSIX hosts.
        _replace_selected_output_without_following(output, content)
    return output


def _write_selected_output_nofollow(output: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(output, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Dataset output must be a regular file: {output}")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _replace_selected_output_without_following(output: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        reject_symlink_paths([output], "Dataset output")
        os.replace(temporary_path, output)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _dataset_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _source_receipt_id(project_root: Path) -> str | None:
    state_path = project_root / ".morpheus" / "state.json"
    try:
        if state_path.is_file() and not state_path.is_symlink():
            state = json.loads(state_path.read_text())
            receipt_id = state.get("receipt_id")
            if isinstance(receipt_id, str) and receipt_id:
                return receipt_id
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    receipts_dir = project_root / ".morpheus" / "receipts"
    try:
        latest = latest_receipt_file(receipts_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if latest is None:
        return None
    try:
        receipt = json.loads(latest.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    receipt_id = receipt.get("receipt_id")
    return receipt_id if isinstance(receipt_id, str) else None
