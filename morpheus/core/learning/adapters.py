"""Adapter registry, activation, and rollback for Morpheus learning."""
import base64
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
import json
import os
import secrets
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from morpheus.core.learning.adapter_artifacts import (
    validate_adapter_artifact_manifest,
    validate_registered_adapter_artifact,
)
from morpheus.core.learning.authority import learning_authority_transaction
from morpheus.core.learning.dataset_validation import validate_dataset
from morpheus.core.learning.eval import (
    check_activation_gate,
    latest_adapter_eval_status,
)
from morpheus.core.portable_lock import portable_file_lock
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.state_authority import state_authority_transaction

_ACTIVATION_JOURNAL_SCHEMA = "morpheus-activation-transaction/1"


def adapters_root(project_root: Path) -> Path:
    return project_root / ".morpheus" / "training" / "adapters"


def active_adapter_path(project_root: Path) -> Path:
    return project_root / ".morpheus" / "training" / "active_adapter.json"


def rollback_log_path(project_root: Path) -> Path:
    return project_root / ".morpheus" / "training" / "rollback_log.jsonl"


def _activation_gate_failure(gate: dict) -> str:
    reason = str(gate.get("reason") or "blocked")
    blockers = gate.get("benchmark_blockers")
    if not isinstance(blockers, list) or not blockers:
        return reason
    return f"{reason}: {', '.join(str(blocker) for blocker in blockers)}"


def list_adapters(project_root: Path) -> list[dict]:
    project_root = _safe_project_root(project_root)
    with _activation_state_transaction(project_root):
        return _list_adapters_locked(project_root)


def _list_adapters_locked(project_root: Path) -> list[dict]:
    root = adapters_root(project_root)
    if root.is_symlink():
        raise ValueError(f"Adapter registry must not be a symlink: {root}")
    reject_symlink_components(root, "Adapter registry")
    if not root.is_dir():
        return []
    active_status = _active_adapter_status_locked(project_root)
    active_id = active_status.get("adapter_id") if active_status else None
    adapters = []
    for manifest_path in sorted(root.glob("*/adapter_manifest.json"), key=lambda item: item.as_posix()):
        manifest = _read_json(manifest_path, "Adapter manifest")
        adapter_id = _safe_adapter_id(
            manifest.get("adapter_id") or manifest_path.parent.name,
            "Adapter manifest",
        )
        eval_status = latest_adapter_eval_status(project_root, adapter_id)
        metrics = eval_status["metrics"]
        manifest_status = str(manifest.get("status") or "planned")
        if manifest_status == "active" and adapter_id != active_id:
            manifest_status = "inactive"
        adapters.append({
            "adapter_id": adapter_id,
            "status": (
                active_status["status"]
                if adapter_id == active_id and active_status is not None
                else manifest_status
            ),
            "backend": manifest.get("backend"),
            "method": manifest.get("method"),
            "base_model": manifest.get("base_model"),
            "created_at": manifest.get("created_at"),
            "eval_id": eval_status["eval_id"],
            "eval_score": metrics.get("pass_rate"),
            "hallucination_rate": metrics.get("hallucination_rate"),
            "eval_status": eval_status["status"],
            "eval_blocker": eval_status["blocker"],
            "adapter_manifest_path": str(manifest_path),
        })
    return adapters


def activate_adapter(
    project_root: Path,
    adapter_id: str,
    *,
    force: bool = False,
    confirm_force: bool = False,
) -> dict:
    project_root = _safe_project_root(project_root)
    if force and not confirm_force:
        raise ValueError("--force requires --yes-i-know-this-can-degrade")
    with _activation_state_transaction(project_root):
        with state_authority_transaction(project_root):
            with _review_authority_transaction(project_root):
                with learning_authority_transaction(project_root):
                    return _activate_adapter_locked(
                        project_root,
                        adapter_id,
                        force=force,
                    )


def _activate_adapter_locked(
    project_root: Path,
    adapter_id: str,
    *,
    force: bool,
) -> dict:
    adapter_dir = _adapter_dir_or_error(project_root, adapter_id)
    gate = check_activation_gate(project_root, adapter_id)
    if not gate["allowed"]:
        gate_failure = _activation_gate_failure(gate)
        if force:
            raise ValueError(
                f"Cannot activate adapter {adapter_id}: force cannot bypass the eval gate "
                f"({gate_failure})"
            )
        raise ValueError(f"Cannot activate adapter {adapter_id}: {gate_failure}")

    previous = _read_active_adapter(project_root)
    previous_id = previous.get("adapter_id") if previous else None
    if previous_id == adapter_id:
        previous_id = previous.get("previous_adapter_id") if previous else None
    initial_identity = _activation_authority_identity(
        project_root,
        adapter_dir,
        adapter_id,
        gate,
    )
    if not _gate_matches_identity(gate, initial_identity):
        raise ValueError(
            f"Cannot activate adapter {adapter_id}: activation authority changed "
            "before final gate"
        )
    final_gate = check_activation_gate(project_root, adapter_id)
    if not final_gate["allowed"] or final_gate.get("eval_id") != gate.get("eval_id"):
        raise ValueError(
            f"Cannot activate adapter {adapter_id}: activation authority changed "
            f"({_activation_gate_failure(final_gate)})"
        )
    final_identity = _activation_authority_identity(
        project_root,
        adapter_dir,
        adapter_id,
        final_gate,
    )
    if (
        final_identity != initial_identity
        or not _gate_matches_identity(final_gate, final_identity)
    ):
        raise ValueError(
            f"Cannot activate adapter {adapter_id}: activation authority bytes changed"
        )

    activated_at = datetime.now(timezone.utc).isoformat()
    transaction_id = _timestamp_id("activation_txn")
    metrics = final_identity["metrics"]
    receipt = {
        "receipt_id": _timestamp_id("activate"),
        "created_at": activated_at,
        "adapter_id": adapter_id,
        "previous_adapter_id": previous_id,
        "forced": force,
        "transaction_id": transaction_id,
        "gate": final_gate,
        "eval_id": final_identity["eval_id"],
        "metrics": metrics,
        "adapter_manifest_sha256": final_identity["adapter_manifest_sha256"],
        "eval_config_sha256": final_identity["eval_config_sha256"],
        "eval_results_sha256": final_identity["eval_results_sha256"],
        "eval_activation_receipt_sha256": final_identity[
            "eval_activation_receipt_sha256"
        ],
        "base_eval_config_sha256": final_identity["base_eval_config_sha256"],
        "base_eval_results_sha256": final_identity["base_eval_results_sha256"],
        "base_eval_activation_receipt_sha256": final_identity[
            "base_eval_activation_receipt_sha256"
        ],
        "dataset_manifest_sha256": final_identity["dataset_manifest_sha256"],
        "dataset_authority_sha256": final_identity["dataset_authority_sha256"],
        "weight_artifact_path": final_identity["weight_artifact_path"],
        "weight_artifact_sha256": final_identity["weight_artifact_sha256"],
        "weight_artifact_size": final_identity["weight_artifact_size"],
    }
    active_payload = {
        "adapter_id": adapter_id,
        "activated_at": activated_at,
        "previous_adapter_id": previous_id,
        "forced": force,
        "transaction_id": transaction_id,
        "eval_id": final_identity["eval_id"],
        "eval_score": metrics.get("pass_rate"),
        "eval_activation_receipt_sha256": final_identity[
            "eval_activation_receipt_sha256"
        ],
        "base_eval_activation_receipt_sha256": final_identity[
            "base_eval_activation_receipt_sha256"
        ],
        "dataset_id": final_identity["dataset_id"],
        "dataset_binding_sha256": final_identity["dataset_binding_sha256"],
        "weight_artifact_path": final_identity["weight_artifact_path"],
        "weight_artifact_sha256": final_identity["weight_artifact_sha256"],
        "weight_artifact_size": final_identity["weight_artifact_size"],
        "adapter_manifest_path": str(adapter_dir / "adapter_manifest.json"),
    }
    receipt_path = adapter_dir / "activate_receipt.json"
    activation_event = {
        "event": "activate",
        "created_at": activated_at,
        "transaction_id": transaction_id,
        "adapter_id": adapter_id,
        "previous_adapter_id": previous_id,
        "forced": force,
    }
    writes = {
        receipt_path: _json_bytes(receipt),
        adapter_dir / "adapter_manifest.json": _adapter_status_bytes(
            project_root,
            adapter_id,
            "active",
            activated_at=activated_at,
            transaction_id=transaction_id,
        ),
        rollback_log_path(project_root): _rollback_log_bytes(
            project_root,
            activation_event,
        ),
        active_adapter_path(project_root): _json_bytes(active_payload),
    }
    if previous_id and previous_id != adapter_id:
        previous_manifest = _adapter_status_bytes(
            project_root,
            previous_id,
            "inactive",
            transaction_id=transaction_id,
        )
        if previous_manifest is not None:
            writes[
                adapters_root(project_root)
                / previous_id
                / "adapter_manifest.json"
            ] = previous_manifest
    def require_precommit_authority() -> None:
        _require_matching_adapter_artifact(
            adapter_dir,
            adapter_id,
            final_identity,
            action="activate",
        )
        _require_matching_activation_gate(
            project_root,
            adapter_id,
            final_identity,
            action="activate",
            authority="activation",
        )

    def require_postwrite_authority() -> None:
        _require_matching_adapter_artifact(
            adapter_dir,
            adapter_id,
            final_identity,
            action="activate",
            require_manifest_identity=False,
        )
        _require_matching_activation_gate(
            project_root,
            adapter_id,
            final_identity,
            action="activate",
            authority="activation",
        )

    _commit_activation_transaction(
        project_root,
        transaction_id=transaction_id,
        writes=writes,
        pointer_path=active_adapter_path(project_root),
        precommit_check=require_precommit_authority,
        postwrite_check=require_postwrite_authority,
    )
    return {
        "activated": True,
        "adapter_id": adapter_id,
        "previous_adapter_id": previous_id,
        "active_adapter_path": str(active_adapter_path(project_root)),
        "activate_receipt_path": str(receipt_path),
        "forced": force,
    }


def rollback_adapter(project_root: Path) -> dict:
    project_root = _safe_project_root(project_root)
    with _activation_state_transaction(project_root):
        with state_authority_transaction(project_root):
            with _review_authority_transaction(project_root):
                with learning_authority_transaction(project_root):
                    return _rollback_adapter_locked(project_root)


def _rollback_adapter_locked(project_root: Path) -> dict:
    active = _read_active_adapter(project_root)
    if active is None:
        raise ValueError("No active adapter to rollback.")
    current_id = active.get("adapter_id")
    previous_id = active.get("previous_adapter_id")
    if previous_id == current_id:
        previous_id = None
    transaction_id = _timestamp_id("activation_txn")
    active_payload = None
    if previous_id:
        previous_id = str(previous_id)
        previous_dir = _adapter_dir_or_error(project_root, previous_id)
        gate = check_activation_gate(project_root, previous_id)
        if not gate["allowed"]:
            raise ValueError(
                f"Cannot rollback to adapter {previous_id}: "
                f"{_activation_gate_failure(gate)}"
            )
        initial_identity = _activation_authority_identity(
            project_root,
            previous_dir,
            previous_id,
            gate,
        )
        if not _gate_matches_identity(gate, initial_identity):
            raise ValueError(
                f"Cannot rollback to adapter {previous_id}: "
                "rollback authority changed before final gate"
            )
        final_gate = check_activation_gate(project_root, previous_id)
        if not final_gate["allowed"] or final_gate.get("eval_id") != gate.get("eval_id"):
            raise ValueError(
                f"Cannot rollback to adapter {previous_id}: rollback authority changed "
                f"({_activation_gate_failure(final_gate)})"
            )
        final_identity = _activation_authority_identity(
            project_root,
            previous_dir,
            previous_id,
            final_gate,
        )
        if (
            final_identity != initial_identity
            or not _gate_matches_identity(final_gate, final_identity)
        ):
            raise ValueError(
                f"Cannot rollback to adapter {previous_id}: "
                "rollback authority bytes changed"
            )
        rolled_back_at = datetime.now(timezone.utc).isoformat()
        metrics = final_identity["metrics"]
        active_payload = {
            "adapter_id": previous_id,
            "activated_at": rolled_back_at,
            "previous_adapter_id": None,
            "forced": False,
            "transaction_id": transaction_id,
            "eval_id": final_identity["eval_id"],
            "eval_score": metrics.get("pass_rate"),
            "eval_activation_receipt_sha256": final_identity[
                "eval_activation_receipt_sha256"
            ],
            "base_eval_activation_receipt_sha256": final_identity[
                "base_eval_activation_receipt_sha256"
            ],
            "dataset_id": final_identity["dataset_id"],
            "dataset_binding_sha256": final_identity["dataset_binding_sha256"],
            "weight_artifact_path": final_identity["weight_artifact_path"],
            "weight_artifact_sha256": final_identity["weight_artifact_sha256"],
            "weight_artifact_size": final_identity["weight_artifact_size"],
            "adapter_manifest_path": str(previous_dir / "adapter_manifest.json"),
        }
    else:
        rolled_back_at = datetime.now(timezone.utc).isoformat()
    path = active_adapter_path(project_root)
    rollback_event = {
        "event": "rollback",
        "created_at": rolled_back_at,
        "transaction_id": transaction_id,
        "adapter_id": previous_id,
        "rolled_back_from": current_id,
    }
    writes = {
        rollback_log_path(project_root): _rollback_log_bytes(
            project_root,
            rollback_event,
        ),
        path: _json_bytes(active_payload) if active_payload is not None else None,
    }
    if previous_id:
        previous_manifest = _adapter_status_bytes(
            project_root,
            previous_id,
            "active",
            activated_at=rolled_back_at,
            transaction_id=transaction_id,
        )
        if previous_manifest is not None:
            writes[
                adapters_root(project_root)
                / previous_id
                / "adapter_manifest.json"
            ] = previous_manifest
    if current_id:
        current_manifest = _adapter_status_bytes(
            project_root,
            current_id,
            "inactive",
            transaction_id=transaction_id,
        )
        if current_manifest is not None:
            writes[
                adapters_root(project_root)
                / current_id
                / "adapter_manifest.json"
            ] = current_manifest
    precommit_check = None
    postwrite_check = None
    if previous_id:
        def precommit_check() -> None:
            _require_matching_adapter_artifact(
                previous_dir,
                previous_id,
                final_identity,
                action="rollback to",
            )
            _require_matching_activation_gate(
                project_root,
                previous_id,
                final_identity,
                action="rollback to",
                authority="rollback",
            )

        def postwrite_check() -> None:
            _require_matching_adapter_artifact(
                previous_dir,
                previous_id,
                final_identity,
                action="rollback to",
                require_manifest_identity=False,
            )
            _require_matching_activation_gate(
                project_root,
                previous_id,
                final_identity,
                action="rollback to",
                authority="rollback",
            )
    _commit_activation_transaction(
        project_root,
        transaction_id=transaction_id,
        writes=writes,
        pointer_path=path,
        precommit_check=precommit_check,
        postwrite_check=postwrite_check,
    )
    return {
        "rolled_back": True,
        "previous_adapter_id": current_id,
        "active_adapter_id": previous_id,
    }


def active_adapter_status(project_root: Path) -> dict | None:
    project_root = _safe_project_root(project_root)
    with _activation_state_transaction(project_root):
        return _active_adapter_status_locked(project_root)


def _active_adapter_status_locked(project_root: Path) -> dict | None:
    active = _read_active_adapter(project_root)
    if active is None:
        return None
    adapter_id = str(active.get("adapter_id") or "")
    adapter_dir = adapters_root(project_root) / adapter_id
    manifest = {}
    manifest_path = adapter_dir / "adapter_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, "Adapter manifest")
    artifact_validation = validate_registered_adapter_artifact(
        adapter_dir,
        expected_adapter_id=adapter_id,
    )
    expected_artifact = {
        "path": active.get("weight_artifact_path"),
        "sha256": active.get("weight_artifact_sha256"),
        "size": active.get("weight_artifact_size"),
    }
    artifact_valid = bool(
        artifact_validation["valid"]
        and artifact_validation.get("artifact") == expected_artifact
    )
    artifact_blockers = list(artifact_validation.get("blockers") or [])
    if artifact_validation.get("artifact") != expected_artifact:
        artifact_blockers.append("active_weight_artifact_identity_mismatch")
    return {
        "adapter_id": adapter_id,
        "status": "active" if artifact_valid else "invalid",
        "artifact_valid": artifact_valid,
        "artifact_blockers": artifact_blockers,
        "created_at": manifest.get("created_at"),
        "activated_at": active.get("activated_at"),
        "eval_id": active.get("eval_id"),
        "eval_score": active.get("eval_score"),
        "backend": manifest.get("backend"),
        "method": manifest.get("method"),
        "base_model": manifest.get("base_model"),
    }


def _adapter_dir_or_error(project_root: Path, adapter_id: str) -> Path:
    adapter_id = _safe_adapter_id(adapter_id, "Adapter")
    adapter_dir = adapters_root(project_root) / adapter_id
    reject_symlink_components(adapter_dir, "Adapter path")
    manifest_path = adapter_dir / "adapter_manifest.json"
    reject_symlink_paths([manifest_path], "Adapter manifest")
    if not manifest_path.is_file():
        raise ValueError(f"Adapter not found: {adapter_id}")
    return adapter_dir


def _activation_authority_identity(
    project_root: Path,
    adapter_dir: Path,
    adapter_id: str,
    gate: dict,
) -> dict:
    eval_id = gate.get("eval_id")
    base_eval_id = gate.get("base_eval_id")
    dataset_id = gate.get("dataset_id")
    dataset_binding = gate.get("dataset_binding_sha256")
    dataset_dir_value = gate.get("dataset_dir")
    if (
        not isinstance(eval_id, str)
        or not eval_id
        or Path(eval_id).name != eval_id
        or eval_id in {".", ".."}
    ):
        raise ValueError(f"Activation eval identity invalid: {eval_id!r}")
    eval_dir = project_root / ".morpheus" / "training" / "evals" / eval_id
    reject_symlink_components(eval_dir, "Activation eval")
    if eval_dir.is_symlink() or not eval_dir.is_dir():
        raise ValueError(f"Activation eval not found: {eval_id}")
    if (
        not isinstance(base_eval_id, str)
        or not base_eval_id
        or Path(base_eval_id).name != base_eval_id
        or base_eval_id in {".", ".."}
    ):
        raise ValueError(f"Activation base eval identity invalid: {base_eval_id!r}")
    base_eval_dir = (
        project_root / ".morpheus" / "training" / "evals" / base_eval_id
    )
    reject_symlink_components(base_eval_dir, "Activation base eval")
    if base_eval_dir.is_symlink() or not base_eval_dir.is_dir():
        raise ValueError(f"Activation base eval not found: {base_eval_id}")
    if not isinstance(dataset_dir_value, str) or not dataset_dir_value:
        raise ValueError("Activation dataset path missing")
    dataset_dir = Path(dataset_dir_value)
    if not dataset_dir.is_absolute():
        raise ValueError("Activation dataset path must be absolute")
    reject_symlink_components(dataset_dir, "Activation dataset")

    manifest, manifest_sha = _read_stable_json_identity(
        adapter_dir / "adapter_manifest.json",
        "Adapter manifest",
    )
    adapter_artifact = validate_adapter_artifact_manifest(
        adapter_dir,
        manifest,
        expected_adapter_id=adapter_id,
    )
    if not adapter_artifact["valid"]:
        raise ValueError(
            f"Activation adapter artifact invalid for {adapter_id}: "
            + ", ".join(adapter_artifact["blockers"])
        )
    config, config_sha = _read_stable_json_identity(
        eval_dir / "eval_config.json",
        "Eval config",
    )
    results, results_sha = _read_stable_json_identity(
        eval_dir / "eval_results.json",
        "Eval results",
    )
    eval_receipt, eval_receipt_sha = _read_stable_json_identity(
        eval_dir / "activation_eval_receipt.json",
        "Activation eval receipt",
    )
    base_config, base_config_sha = _read_stable_json_identity(
        base_eval_dir / "eval_config.json",
        "Base eval config",
    )
    base_results, base_results_sha = _read_stable_json_identity(
        base_eval_dir / "eval_results.json",
        "Base eval results",
    )
    base_eval_receipt, base_eval_receipt_sha = _read_stable_json_identity(
        base_eval_dir / "activation_eval_receipt.json",
        "Base activation eval receipt",
    )
    dataset_manifest, dataset_manifest_sha = _read_stable_json_identity(
        dataset_dir / "manifest.json",
        "Dataset manifest",
    )
    dataset_validation = validate_dataset(
        project_root,
        dataset_dir,
        dataset_manifest,
    )
    final_dataset_manifest, final_dataset_manifest_sha = _read_stable_json_identity(
        dataset_dir / "manifest.json",
        "Dataset manifest",
    )
    final_dataset_validation = validate_dataset(
        project_root,
        dataset_dir,
        final_dataset_manifest,
    )
    metrics = results.get("metrics")
    if (
        manifest.get("adapter_id") != adapter_id
        or manifest.get("dataset_id") != dataset_id
        or manifest.get("dataset_binding_sha256") != dataset_binding
        or config.get("adapter_id") != adapter_id
        or results.get("adapter_id") != adapter_id
        or config.get("base_only") is not False
        or results.get("base_only") is not False
        or config.get("eval_id") != eval_id
        or results.get("eval_id") != eval_id
        or config.get("dataset_id") != dataset_id
        or results.get("dataset_id") != dataset_id
        or config.get("dataset_binding_sha256") != dataset_binding
        or results.get("dataset_binding_sha256") != dataset_binding
        or eval_receipt.get("eval_id") != eval_id
        or base_config.get("adapter_id") is not None
        or base_results.get("adapter_id") is not None
        or base_config.get("base_only") is not True
        or base_results.get("base_only") is not True
        or base_config.get("eval_id") != base_eval_id
        or base_results.get("eval_id") != base_eval_id
        or base_config.get("dataset_id") != dataset_id
        or base_results.get("dataset_id") != dataset_id
        or base_config.get("dataset_binding_sha256") != dataset_binding
        or base_results.get("dataset_binding_sha256") != dataset_binding
        or base_eval_receipt.get("eval_id") != base_eval_id
        or not isinstance(metrics, dict)
    ):
        raise ValueError(
            f"Activation authority identity mismatch for adapter {adapter_id}"
        )
    if (
        not dataset_validation["valid"]
        or not final_dataset_validation["valid"]
        or dataset_validation.get("dataset_binding_sha256") != dataset_binding
        or final_dataset_validation.get("dataset_binding_sha256") != dataset_binding
        or dataset_manifest_sha != final_dataset_manifest_sha
        or dataset_manifest != final_dataset_manifest
        or dataset_validation != final_dataset_validation
    ):
        blockers = sorted({
            *dataset_validation.get("blockers", []),
            *final_dataset_validation.get("blockers", []),
        })
        raise ValueError(
            "Activation dataset authority invalid"
            + (": " + ", ".join(blockers) if blockers else "")
        )
    identity_files = {
        adapter_dir / "adapter_manifest.json": manifest_sha,
        eval_dir / "eval_config.json": config_sha,
        eval_dir / "eval_results.json": results_sha,
        eval_dir / "activation_eval_receipt.json": eval_receipt_sha,
        base_eval_dir / "eval_config.json": base_config_sha,
        base_eval_dir / "eval_results.json": base_results_sha,
        base_eval_dir / "activation_eval_receipt.json": base_eval_receipt_sha,
        dataset_dir / "manifest.json": final_dataset_manifest_sha,
    }
    for path, expected_sha in identity_files.items():
        _, current_sha = _read_stable_json_identity(path, "Activation authority")
        if current_sha != expected_sha:
            raise ValueError("Activation authority changed while it was captured")
    closing_dataset_validation = validate_dataset(
        project_root,
        dataset_dir,
        final_dataset_manifest,
    )
    if closing_dataset_validation != final_dataset_validation:
        raise ValueError("Activation dataset authority changed while it was captured")
    closing_adapter_artifact = validate_adapter_artifact_manifest(
        adapter_dir,
        manifest,
        expected_adapter_id=adapter_id,
    )
    if closing_adapter_artifact != adapter_artifact:
        raise ValueError("Activation adapter artifact changed while it was captured")
    dataset_authority_sha = _canonical_json_sha256({
        "dataset_manifest_sha256": dataset_manifest_sha,
        "validation": closing_dataset_validation,
    })
    return {
        "adapter_manifest_sha256": manifest_sha,
        "eval_config_sha256": config_sha,
        "eval_results_sha256": results_sha,
        "eval_activation_receipt_sha256": eval_receipt_sha,
        "base_eval_config_sha256": base_config_sha,
        "base_eval_results_sha256": base_results_sha,
        "base_eval_activation_receipt_sha256": base_eval_receipt_sha,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "dataset_authority_sha256": dataset_authority_sha,
        "eval_id": eval_id,
        "base_eval_id": base_eval_id,
        "dataset_id": dataset_id,
        "dataset_binding_sha256": dataset_binding,
        "weight_artifact_path": adapter_artifact["artifact"]["path"],
        "weight_artifact_sha256": adapter_artifact["artifact"]["sha256"],
        "weight_artifact_size": adapter_artifact["artifact"]["size"],
        "metrics": metrics,
    }


def _require_matching_adapter_artifact(
    adapter_dir: Path,
    adapter_id: str,
    identity: dict,
    *,
    action: str,
    require_manifest_identity: bool = True,
) -> None:
    validation = validate_registered_adapter_artifact(
        adapter_dir,
        expected_adapter_id=adapter_id,
    )
    expected_artifact = {
        "path": identity["weight_artifact_path"],
        "sha256": identity["weight_artifact_sha256"],
        "size": identity["weight_artifact_size"],
    }
    if (
        not validation["valid"]
        or (
            require_manifest_identity
            and validation.get("manifest_sha256")
            != identity["adapter_manifest_sha256"]
        )
        or validation.get("artifact") != expected_artifact
    ):
        blockers = validation.get("blockers") or ["artifact_identity_changed"]
        raise ValueError(
            f"Cannot {action} adapter {adapter_id}: adapter artifact changed "
            "before commit (" + ", ".join(blockers) + ")"
        )


def _require_matching_activation_gate(
    project_root: Path,
    adapter_id: str,
    identity: dict,
    *,
    action: str,
    authority: str,
) -> None:
    gate = check_activation_gate(project_root, adapter_id)
    if not _gate_matches_identity(gate, identity):
        raise ValueError(
            f"Cannot {action} adapter {adapter_id}: {authority} authority changed "
            f"({_activation_gate_failure(gate)})"
        )


def _read_stable_json_identity(path: Path, label: str) -> tuple[dict, str]:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    try:
        first = path.read_bytes()
        second = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} invalid: {exc}") from exc
    if first != second:
        raise ValueError(f"{label} changed while reading")
    try:
        data = json.loads(first)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} invalid: expected JSON object")
    return data, sha256(first).hexdigest()


def _gate_matches_identity(gate: dict, identity: dict) -> bool:
    return bool(
        gate.get("allowed") is True
        and gate.get("eval_id") == identity["eval_id"]
        and gate.get("base_eval_id") == identity["base_eval_id"]
        and gate.get("dataset_id") == identity["dataset_id"]
        and gate.get("dataset_binding_sha256")
        == identity["dataset_binding_sha256"]
        and gate.get("metrics") == identity["metrics"]
        and gate.get("eval_activation_receipt_sha256")
        == identity["eval_activation_receipt_sha256"]
        and gate.get("base_eval_activation_receipt_sha256")
        == identity["base_eval_activation_receipt_sha256"]
        and gate.get("weight_artifact") == {
            "path": identity["weight_artifact_path"],
            "sha256": identity["weight_artifact_sha256"],
            "size": identity["weight_artifact_size"],
        }
    )


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(payload).hexdigest()


@contextmanager
def _activation_state_transaction(project_root: Path):
    """Serialize activation commits across threads and local worker processes."""
    training_root = project_root / ".morpheus" / "training"
    reject_symlink_components(training_root, "Activation state directory")
    if training_root.is_symlink():
        raise ValueError(f"Activation state directory must not be a symlink: {training_root}")
    training_root.mkdir(parents=True, exist_ok=True)
    lock_path = training_root / ".activation.lock"
    reject_symlink_paths([lock_path], "Activation state lock")
    reject_symlink_components(lock_path, "Activation state lock")
    with portable_file_lock(lock_path):
        _recover_activation_transaction(project_root)
        yield


@contextmanager
def _review_authority_transaction(project_root: Path):
    """Hold every reachable review authority stable through the pointer commit."""
    review_roots = [project_root]
    lab_root = project_root / ".morpheus" / "lab"
    if lab_root.exists():
        if lab_root.is_symlink():
            raise ValueError(f"Lab registry must not be a symlink: {lab_root}")
        reject_symlink_components(lab_root, "Lab registry")
        for workspace in sorted(
            lab_root.glob("lab_*/workspace"),
            key=lambda path: path.as_posix(),
        ):
            review_path = workspace / ".morpheus/review/semantic_candidates.jsonl"
            if not review_path.exists():
                continue
            reject_symlink_components(workspace, "Lab review workspace")
            if workspace.is_symlink():
                raise ValueError(f"Lab review workspace must not be a symlink: {workspace}")
            review_roots.append(workspace)
    with ExitStack() as stack:
        for review_root in review_roots:
            stack.enter_context(ReviewStore(review_root).transaction())
        yield


def _activation_journal_path(project_root: Path) -> Path:
    return project_root / ".morpheus" / "training" / ".activation-transaction.json"


def _commit_activation_transaction(
    project_root: Path,
    *,
    transaction_id: str,
    writes: dict[Path, bytes | None],
    pointer_path: Path,
    precommit_check: Callable[[], None] | None = None,
    postwrite_check: Callable[[], None] | None = None,
) -> None:
    """Durably journal a multi-file activation update before its pointer commit."""
    if pointer_path not in writes:
        raise ValueError("Activation transaction is missing its canonical pointer")
    entries = []
    for path, after in writes.items():
        relative = _activation_artifact_relative(project_root, path)
        reject_symlink_paths([path], "Activation transaction artifact")
        reject_symlink_components(path, "Activation transaction artifact")
        before = path.read_bytes() if path.is_file() else None
        entries.append({
            "path": relative,
            "before": _encode_optional_bytes(before),
            "after": _encode_optional_bytes(after),
        })
    journal = {
        "schema": _ACTIVATION_JOURNAL_SCHEMA,
        "transaction_id": transaction_id,
        "pointer": _activation_artifact_relative(project_root, pointer_path),
        "entries": entries,
    }
    journal_path = _activation_journal_path(project_root)
    _atomic_write_bytes(
        journal_path,
        _json_bytes(journal),
        "Activation transaction journal",
    )
    try:
        if precommit_check is not None:
            precommit_check()
        for path, payload in writes.items():
            if path != pointer_path:
                _apply_activation_artifact(path, payload)
        if postwrite_check is not None:
            postwrite_check()
        pointer_payload = writes[pointer_path]
        if pointer_payload is None:
            _apply_activation_artifact(pointer_path, None)
        else:
            pointer_data = json.loads(pointer_payload)
            if not isinstance(pointer_data, dict):
                raise ValueError("Activation pointer payload must be a JSON object")
            _write_json(pointer_path, pointer_data)
    except Exception:
        committed = _recover_activation_transaction(
            project_root,
            remove_journal=False,
        )
        if committed is True:
            if postwrite_check is not None:
                try:
                    postwrite_check()
                except Exception:
                    _rollback_activation_transaction(project_root)
                    raise
            _remove_activation_journal(project_root)
            return
        _remove_activation_journal(project_root)
        raise
    if postwrite_check is not None:
        try:
            postwrite_check()
        except Exception:
            _rollback_activation_transaction(project_root)
            raise
    try:
        _remove_activation_journal(project_root)
    except Exception:
        committed = _recover_activation_transaction(project_root)
        if committed is True:
            return
        raise


def _rollback_activation_transaction(project_root: Path) -> None:
    """Force a journaled pointer commit back to its exact before-state."""
    journal_path = _activation_journal_path(project_root)
    journal = _read_json(journal_path, "Activation transaction journal")
    pointer_relative = journal.get("pointer")
    entries = journal.get("entries")
    if not isinstance(pointer_relative, str) or not isinstance(entries, list):
        raise ValueError("Activation transaction journal structure invalid")
    pointer_entry = next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("path") == pointer_relative
        ),
        None,
    )
    if not isinstance(pointer_entry, dict) or "before" not in pointer_entry:
        raise ValueError("Activation transaction journal pointer is missing")
    pointer_path = _activation_artifact_from_relative(
        project_root,
        pointer_relative,
    )
    _apply_activation_artifact(
        pointer_path,
        _decode_optional_bytes(pointer_entry["before"]),
    )
    committed = _recover_activation_transaction(project_root)
    if committed is not False:
        raise ValueError("Activation transaction rollback did not reach before-state")


def _recover_activation_transaction(
    project_root: Path,
    *,
    remove_journal: bool = True,
) -> bool | None:
    """Roll a durable activation journal backward or forward from pointer state."""
    journal_path = _activation_journal_path(project_root)
    reject_symlink_paths([journal_path], "Activation transaction journal")
    reject_symlink_components(journal_path, "Activation transaction journal")
    if not journal_path.exists():
        return None
    journal = _read_json(journal_path, "Activation transaction journal")
    if journal.get("schema") != _ACTIVATION_JOURNAL_SCHEMA:
        raise ValueError("Activation transaction journal schema invalid")
    transaction_id = journal.get("transaction_id")
    pointer_relative = journal.get("pointer")
    raw_entries = journal.get("entries")
    if (
        not isinstance(transaction_id, str)
        or not transaction_id
        or not isinstance(pointer_relative, str)
        or not isinstance(raw_entries, list)
        or not raw_entries
    ):
        raise ValueError("Activation transaction journal structure invalid")
    expected_pointer = _activation_artifact_relative(
        project_root,
        active_adapter_path(project_root),
    )
    if pointer_relative != expected_pointer:
        raise ValueError("Activation transaction journal pointer invalid")
    decoded: dict[str, tuple[Path, bytes | None, bytes | None]] = {}
    for item in raw_entries:
        if (
            not isinstance(item, dict)
            or not {"path", "before", "after"}.issubset(item)
            or not isinstance(item["path"], str)
        ):
            raise ValueError("Activation transaction journal entry invalid")
        relative = item["path"]
        if relative in decoded:
            raise ValueError("Activation transaction journal contains duplicate paths")
        path = _activation_artifact_from_relative(project_root, relative)
        if path.exists() and not path.is_file():
            raise ValueError("Activation transaction artifact type invalid")
        decoded[relative] = (
            path,
            _decode_optional_bytes(item["before"]),
            _decode_optional_bytes(item["after"]),
        )
    if pointer_relative not in decoded:
        raise ValueError("Activation transaction journal pointer is missing")
    pointer_path, pointer_before, pointer_after = decoded[pointer_relative]
    if pointer_before == pointer_after:
        raise ValueError("Activation transaction journal pointer transition invalid")
    pointer_current = pointer_path.read_bytes() if pointer_path.is_file() else None
    if pointer_current == pointer_after:
        use_after = True
    elif pointer_current == pointer_before:
        use_after = False
    else:
        raise ValueError("Activation transaction pointer does not match its journal")
    for path, before, after in decoded.values():
        current = path.read_bytes() if path.is_file() else None
        if current not in {before, after}:
            raise ValueError(
                "Activation transaction artifact does not match its journal"
            )
    for relative, (path, before, after) in decoded.items():
        if relative != pointer_relative:
            _apply_activation_artifact(path, after if use_after else before)
    _apply_activation_artifact(
        pointer_path,
        pointer_after if use_after else pointer_before,
    )
    if remove_journal:
        _remove_activation_journal(project_root)
    return use_after


def _remove_activation_journal(project_root: Path) -> None:
    path = _activation_journal_path(project_root)
    reject_symlink_paths([path], "Activation transaction journal")
    if path.exists():
        _unlink_and_fsync(path)


def _apply_activation_artifact(path: Path, payload: bytes | None) -> None:
    if payload is None:
        reject_symlink_paths([path], "Activation transaction artifact")
        reject_symlink_components(path, "Activation transaction artifact")
        if path.exists():
            _unlink_and_fsync(path)
        return
    _atomic_write_bytes(path, payload, "Activation transaction artifact")


def _activation_artifact_relative(project_root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError("Activation transaction artifact is outside the project") from exc
    _activation_artifact_from_relative(project_root, relative)
    return relative


def _activation_artifact_from_relative(project_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Activation transaction artifact path invalid")
    parts = candidate.parts
    fixed = (".morpheus", "training")
    allowed_fixed = {
        (*fixed, "active_adapter.json"),
        (*fixed, "rollback_log.jsonl"),
    }
    allowed_adapter = bool(
        len(parts) == 5
        and parts[:3] == (*fixed, "adapters")
        and parts[4] in {"adapter_manifest.json", "activate_receipt.json"}
    )
    if tuple(parts) not in allowed_fixed and not allowed_adapter:
        raise ValueError("Activation transaction artifact path is not allowed")
    if allowed_adapter:
        _safe_adapter_id(parts[3], "Activation transaction adapter")
    path = project_root / candidate
    reject_symlink_components(path, "Activation transaction artifact")
    return path


def _encode_optional_bytes(payload: bytes | None) -> str | None:
    return None if payload is None else base64.b64encode(payload).decode("ascii")


def _decode_optional_bytes(payload: object) -> bytes | None:
    if payload is None:
        return None
    if not isinstance(payload, str):
        raise ValueError("Activation transaction journal bytes invalid")
    try:
        return base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise ValueError("Activation transaction journal bytes invalid") from exc


def _read_active_adapter(project_root: Path) -> dict | None:
    path = active_adapter_path(project_root)
    reject_symlink_paths([path], "Active adapter")
    reject_symlink_components(path, "Active adapter")
    if not path.is_file():
        return None
    active = _read_json(path, "Active adapter")
    active["adapter_id"] = _safe_adapter_id(
        active.get("adapter_id"),
        "Active adapter",
    )
    previous_id = active.get("previous_adapter_id")
    if previous_id is not None:
        active["previous_adapter_id"] = _safe_adapter_id(
            previous_id,
            "Active adapter previous",
        )
    return active


def _active_adapter_id(project_root: Path) -> str | None:
    active = _read_active_adapter(project_root)
    if not active:
        return None
    adapter_id = active.get("adapter_id")
    return adapter_id if isinstance(adapter_id, str) else None


def _adapter_status_bytes(
    project_root: Path,
    adapter_id: str,
    status: str,
    *,
    activated_at: str | None = None,
    transaction_id: str,
) -> bytes | None:
    adapter_id = _safe_adapter_id(adapter_id, "Adapter status")
    manifest_path = adapters_root(project_root) / adapter_id / "adapter_manifest.json"
    reject_symlink_paths([manifest_path], "Adapter manifest")
    reject_symlink_components(manifest_path, "Adapter manifest")
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path, "Adapter manifest")
    manifest["status"] = status
    manifest["activated"] = status == "active"
    manifest["activation_transaction_id"] = transaction_id
    if activated_at:
        manifest["activated_at"] = activated_at
    return _json_bytes(manifest)


def _rollback_log_bytes(project_root: Path, event: dict) -> bytes:
    path = rollback_log_path(project_root)
    reject_symlink_components(path.parent, "Rollback log")
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_paths([path], "Rollback log")
    existing = path.read_bytes() if path.is_file() else b""
    line = (json.dumps(event, sort_keys=True) + "\n").encode()
    return existing + line


def _safe_adapter_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).name != value
    ):
        raise ValueError(f"{label} identity invalid: {value!r}")
    return value


def _safe_project_root(project_root: Path) -> Path:
    project_root = project_root.expanduser()
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")
    return project_root


def _read_json(path: Path, label: str) -> dict:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} invalid: expected JSON object")
    return data


def _write_json(path: Path, data: dict) -> None:
    _atomic_write_bytes(path, _json_bytes(data), "JSON output")


def _json_bytes(data: dict) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write_bytes(path: Path, payload: bytes, label: str) -> None:
    reject_symlink_components(path.parent, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    temporary_path = path.parent / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    reject_symlink_paths([temporary_path], label)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        reject_symlink_paths([path], label)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _unlink_and_fsync(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    """Persist directory entries where the platform exposes directory fsync."""
    if os.name == "nt":  # pragma: no cover - Windows rejects directory os.open/fsync.
        return
    directory_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
