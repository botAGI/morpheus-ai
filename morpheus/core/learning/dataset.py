"""Compile reviewed semantic candidates into local training datasets."""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from morpheus.core.compiler import compute_sha256
from morpheus.core.learning.evals import (
    eval_items_for_candidate,
    truth_gate_negative_eval_items,
    unsupported_claim_eval_item,
)
from morpheus.core.learning.examples import (
    CHAT_FORMAT_VERSION,
    INSTRUCTION_FORMAT_VERSION,
    SHAREGPT_FORMAT_VERSION,
    chat_examples_from_instruction,
    instruction_examples_for_candidate,
    sharegpt_examples_from_instruction,
    truth_gate_negative_instruction_examples,
)
from morpheus.core.learning.registry import datasets_root
from morpheus.core.learning.safety import (
    contains_secret_like_text,
    load_morpheusignore,
    path_is_ignored,
)
from morpheus.core.provenance import compute_sha256_file, latest_receipt_file
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.semantic.verifier import verify_candidate_span


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
    "outdated_claim_correction",
    "unsupported_claim_refusal",
    "agent_rule_adherence",
    "command_cli_capability_claims",
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
    candidates = _load_dataset_candidates(project_root, source)
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
    for item in eligible:
        instruction_examples.extend(instruction_examples_for_candidate(item.candidate))
        eval_items.extend(eval_items_for_candidate(item.candidate))
    if eligible and include_refusals:
        instruction_examples.extend(truth_gate_negative_instruction_examples())
        eval_items.append(unsupported_claim_eval_item())
        eval_items.extend(truth_gate_negative_eval_items())
    elif include_refusals:
        eval_items.append(unsupported_claim_eval_item())
        eval_items.extend(truth_gate_negative_eval_items())

    sharegpt_examples = sharegpt_examples_from_instruction(instruction_examples)
    chat_examples = chat_examples_from_instruction(instruction_examples)
    split_rows = _split_chat_rows(chat_examples)
    dataset_id = _dataset_id()
    out_dir = datasets_root(project_root) / dataset_id
    _ensure_output_dir(out_dir)

    instruction_path = out_dir / "dataset.instruction.jsonl"
    sharegpt_path = out_dir / "dataset.sharegpt.jsonl"
    skipped_path = out_dir / "skipped.jsonl"
    eval_path = out_dir / "eval.seed.jsonl"
    manifest_path = out_dir / "manifest.json"
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    test_path = out_dir / "test.jsonl"
    _write_jsonl(instruction_path, instruction_examples)
    _write_jsonl(sharegpt_path, sharegpt_examples)
    _write_jsonl(train_path, split_rows["train"])
    _write_jsonl(valid_path, split_rows["valid"])
    _write_jsonl(test_path, split_rows["test"])
    _write_jsonl(skipped_path, skipped)
    _write_jsonl(eval_path, eval_items)

    selected_path = _selected_dataset_path(
        dataset_format,
        instruction_path=instruction_path,
        sharegpt_path=sharegpt_path,
        train_path=train_path,
    )
    if output is not None:
        selected_path = _write_selected_output(project_root, output, selected_path.read_text())

    source_hashes.update(_state_context_hashes(project_root))
    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "source_receipt_id": _source_receipt_id(project_root),
        "candidate_count": len(candidates),
        "trainable_candidate_count": sum(1 for item in eligible if item.trainable_positive),
        "examples_count": len(instruction_examples),
        "skipped_count": len(skipped),
        "split_counts": {key: len(value) for key, value in split_rows.items()},
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
        "dataset_sha256": compute_sha256_file(selected_path),
        "selected_format": dataset_format,
        "format_version": _format_version(dataset_format),
        "source": source,
        "include_corrections": include_corrections,
        "include_refusals": include_refusals,
        "format_versions": {
            "instruction": INSTRUCTION_FORMAT_VERSION,
            "sharegpt": SHAREGPT_FORMAT_VERSION,
            "chat": CHAT_FORMAT_VERSION,
            "eval_seed": "morpheus-eval-seed/1",
            "manifest": "morpheus-learning-manifest/1",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "dataset_id": dataset_id,
        "dataset_dir": str(out_dir),
        "selected_dataset_path": str(selected_path),
        "manifest_path": str(manifest_path),
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

    rel_path = Path(candidate.source_path)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        return None, "invalid_source_path", None
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
    if contains_secret_like_text(candidate.claim) or contains_secret_like_text(candidate.evidence_excerpt):
        return None, "secret_like", current_sha

    return Eligibility(
        candidate=verified,
        trainable_positive=verified.kind in POSITIVE_KINDS,
    ), "", current_sha


def _load_dataset_candidates(project_root: Path, source: str) -> list[SemanticCandidate]:
    if source == "accepted":
        return ReviewStore(project_root).load_candidates()
    return _active_state_candidates(project_root)


def _active_state_candidates(project_root: Path) -> list[SemanticCandidate]:
    state_path = project_root / ".morpheus" / "state.json"
    evidence_path = project_root / ".morpheus" / "evidence.jsonl"
    if not state_path.is_file() or not evidence_path.is_file():
        return []
    state = _read_json_object(state_path, "state.json")
    evidence_rows = _read_jsonl(evidence_path, "evidence.jsonl")
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
            evidence_sha = compute_sha256_file(project_root / source_path)
        candidates.append(SemanticCandidate(
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
        ))
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
    required = [row for row in rows if _train_required_row(row)]
    remaining = [row for row in rows if not _train_required_row(row)]
    if len(rows) < 20:
        train = _dedupe_rows([*required, *rows[:-2]])
        return {"train": train, "valid": rows[-2:-1], "test": rows[-1:]}

    train_end = max(1, int(len(rows) * 0.8))
    valid_target = max(1, int(len(rows) * 0.1))
    train = _dedupe_rows(required)
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


def _ensure_output_dir(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Dataset output must not be a symlink: {path}")
    reject_symlink_components(path.parent, "Dataset output root")
    path.mkdir(parents=True, exist_ok=False)
    reject_symlink_components(path, "Dataset output")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    reject_symlink_paths([path], "Dataset artifact")
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _read_json_object(path: Path, label: str) -> dict:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} invalid: expected JSON object")
    return data


def _read_jsonl(path: Path, label: str) -> list[dict]:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number} invalid JSON: {exc.msg}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_selected_output(project_root: Path, output: Path, content: str) -> Path:
    output = output.expanduser()
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve()
    reject_symlink_components(output.parent, "Dataset output")
    output.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_paths([output], "Dataset output")
    output.write_text(content)
    return output


def _dataset_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _writeable_hash(path: Path) -> str | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        return compute_sha256_file(path)
    except (OSError, ValueError):
        return None


def _state_context_hashes(project_root: Path) -> dict[str, str]:
    candidates = {
        ".morpheus/state.json": project_root / ".morpheus" / "state.json",
        ".morpheus/evidence.jsonl": project_root / ".morpheus" / "evidence.jsonl",
        "WAKE.md": project_root / "WAKE.md",
        ".morpheus/WAKE.md": project_root / ".morpheus" / "WAKE.md",
    }
    hashes = {}
    for rel, path in candidates.items():
        value = _writeable_hash(path)
        if value:
            hashes[rel] = value
    return hashes


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
