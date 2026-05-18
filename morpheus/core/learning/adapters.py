"""Adapter registry, activation, and rollback for Morpheus learning."""
import json
from datetime import datetime, timezone
from pathlib import Path

from morpheus.core.learning.eval import check_activation_gate
from morpheus.core.provenance import compute_sha256_file
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths


def adapters_root(project_root: Path) -> Path:
    return project_root / ".morpheus" / "training" / "adapters"


def active_adapter_path(project_root: Path) -> Path:
    return project_root / ".morpheus" / "training" / "active_adapter.json"


def rollback_log_path(project_root: Path) -> Path:
    return project_root / ".morpheus" / "training" / "rollback_log.jsonl"


def list_adapters(project_root: Path) -> list[dict]:
    project_root = _safe_project_root(project_root)
    root = adapters_root(project_root)
    if root.is_symlink():
        raise ValueError(f"Adapter registry must not be a symlink: {root}")
    reject_symlink_components(root, "Adapter registry")
    if not root.is_dir():
        return []
    active_id = _active_adapter_id(project_root)
    adapters = []
    for manifest_path in sorted(root.glob("*/adapter_manifest.json"), key=lambda item: item.as_posix()):
        manifest = _read_json(manifest_path, "Adapter manifest")
        adapter_id = str(manifest.get("adapter_id") or manifest_path.parent.name)
        latest_eval = _latest_adapter_eval(project_root, adapter_id)
        metrics = _eval_metrics(latest_eval) if latest_eval else {}
        adapters.append({
            "adapter_id": adapter_id,
            "status": "active" if adapter_id == active_id else str(manifest.get("status") or "planned"),
            "backend": manifest.get("backend"),
            "method": manifest.get("method"),
            "base_model": manifest.get("base_model"),
            "created_at": manifest.get("created_at"),
            "eval_id": _eval_id(latest_eval),
            "eval_score": metrics.get("pass_rate"),
            "hallucination_rate": metrics.get("hallucination_rate"),
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
    adapter_dir = _adapter_dir_or_error(project_root, adapter_id)
    gate = check_activation_gate(project_root, adapter_id)
    if not gate["allowed"] and not force:
        raise ValueError(f"Cannot activate adapter {adapter_id}: {gate['reason']}")

    previous = _read_active_adapter(project_root)
    previous_id = previous.get("adapter_id") if previous else None
    activated_at = datetime.now(timezone.utc).isoformat()
    latest_eval = _latest_adapter_eval(project_root, adapter_id)
    metrics = _eval_metrics(latest_eval) if latest_eval else {}
    receipt = {
        "receipt_id": _timestamp_id("activate"),
        "created_at": activated_at,
        "adapter_id": adapter_id,
        "previous_adapter_id": previous_id,
        "forced": force,
        "gate": gate,
        "adapter_manifest_sha256": compute_sha256_file(adapter_dir / "adapter_manifest.json"),
        "eval_results_sha256": compute_sha256_file(latest_eval) if latest_eval else None,
    }
    active_payload = {
        "adapter_id": adapter_id,
        "activated_at": activated_at,
        "previous_adapter_id": previous_id,
        "forced": force,
        "eval_id": gate.get("eval_id") or _eval_id(latest_eval),
        "eval_score": metrics.get("pass_rate"),
        "adapter_manifest_path": str(adapter_dir / "adapter_manifest.json"),
    }
    _write_json(adapter_dir / "activate_receipt.json", receipt)
    _write_json(active_adapter_path(project_root), active_payload)
    _append_rollback_log(project_root, {
        "event": "activate",
        "created_at": activated_at,
        "adapter_id": adapter_id,
        "previous_adapter_id": previous_id,
        "forced": force,
    })
    _set_adapter_status(project_root, adapter_id, "active", activated_at=activated_at)
    if previous_id and previous_id != adapter_id:
        _set_adapter_status(project_root, previous_id, "inactive")
    return {
        "activated": True,
        "adapter_id": adapter_id,
        "previous_adapter_id": previous_id,
        "active_adapter_path": str(active_adapter_path(project_root)),
        "activate_receipt_path": str(adapter_dir / "activate_receipt.json"),
        "forced": force,
    }


def rollback_adapter(project_root: Path) -> dict:
    project_root = _safe_project_root(project_root)
    active = _read_active_adapter(project_root)
    if active is None:
        raise ValueError("No active adapter to rollback.")
    current_id = active.get("adapter_id")
    previous_id = active.get("previous_adapter_id")
    rolled_back_at = datetime.now(timezone.utc).isoformat()
    if previous_id:
        previous_dir = _adapter_dir_or_error(project_root, str(previous_id))
        latest_eval = _latest_adapter_eval(project_root, str(previous_id))
        metrics = _eval_metrics(latest_eval) if latest_eval else {}
        active_payload = {
            "adapter_id": previous_id,
            "activated_at": rolled_back_at,
            "previous_adapter_id": None,
            "forced": False,
            "eval_id": _eval_id(latest_eval),
            "eval_score": metrics.get("pass_rate"),
            "adapter_manifest_path": str(previous_dir / "adapter_manifest.json"),
        }
        _write_json(active_adapter_path(project_root), active_payload)
        _set_adapter_status(project_root, str(previous_id), "active", activated_at=rolled_back_at)
    else:
        path = active_adapter_path(project_root)
        reject_symlink_paths([path], "Active adapter")
        if path.exists():
            path.unlink()
    if current_id:
        _set_adapter_status(project_root, str(current_id), "inactive")
    _append_rollback_log(project_root, {
        "event": "rollback",
        "created_at": rolled_back_at,
        "adapter_id": previous_id,
        "rolled_back_from": current_id,
    })
    return {
        "rolled_back": True,
        "previous_adapter_id": current_id,
        "active_adapter_id": previous_id,
    }


def active_adapter_status(project_root: Path) -> dict | None:
    project_root = _safe_project_root(project_root)
    active = _read_active_adapter(project_root)
    if active is None:
        return None
    adapter_id = str(active.get("adapter_id") or "")
    adapter_dir = adapters_root(project_root) / adapter_id
    manifest = {}
    manifest_path = adapter_dir / "adapter_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, "Adapter manifest")
    return {
        "adapter_id": adapter_id,
        "status": manifest.get("status") or "active",
        "created_at": manifest.get("created_at"),
        "activated_at": active.get("activated_at"),
        "eval_id": active.get("eval_id"),
        "eval_score": active.get("eval_score"),
        "backend": manifest.get("backend"),
        "method": manifest.get("method"),
        "base_model": manifest.get("base_model"),
    }


def _adapter_dir_or_error(project_root: Path, adapter_id: str) -> Path:
    adapter_dir = adapters_root(project_root) / adapter_id
    reject_symlink_components(adapter_dir, "Adapter path")
    manifest_path = adapter_dir / "adapter_manifest.json"
    reject_symlink_paths([manifest_path], "Adapter manifest")
    if not manifest_path.is_file():
        raise ValueError(f"Adapter not found: {adapter_id}")
    return adapter_dir


def _latest_adapter_eval(project_root: Path, adapter_id: str) -> Path | None:
    evals_root = project_root / ".morpheus" / "training" / "evals"
    if evals_root.is_symlink():
        raise ValueError(f"Eval registry must not be a symlink: {evals_root}")
    reject_symlink_components(evals_root, "Eval registry")
    if not evals_root.is_dir():
        return None
    matches = []
    for config_path in sorted(evals_root.glob("*/eval_config.json"), key=lambda item: item.as_posix()):
        config = _read_json(config_path, "Eval config")
        if config.get("adapter_id") == adapter_id and not config.get("base_only"):
            results_path = config_path.parent / "eval_results.json"
            if results_path.is_file():
                matches.append(results_path)
    return matches[-1] if matches else None


def _eval_metrics(eval_results_path: Path | None) -> dict:
    if eval_results_path is None:
        return {}
    results = _read_json(eval_results_path, "Eval results")
    metrics = results.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _eval_id(eval_results_path: Path | None) -> str | None:
    if eval_results_path is None:
        return None
    results = _read_json(eval_results_path, "Eval results")
    eval_id = results.get("eval_id")
    return eval_id if isinstance(eval_id, str) else None


def _read_active_adapter(project_root: Path) -> dict | None:
    path = active_adapter_path(project_root)
    reject_symlink_paths([path], "Active adapter")
    reject_symlink_components(path, "Active adapter")
    if not path.is_file():
        return None
    return _read_json(path, "Active adapter")


def _active_adapter_id(project_root: Path) -> str | None:
    active = _read_active_adapter(project_root)
    if not active:
        return None
    adapter_id = active.get("adapter_id")
    return adapter_id if isinstance(adapter_id, str) else None


def _set_adapter_status(
    project_root: Path,
    adapter_id: str,
    status: str,
    *,
    activated_at: str | None = None,
) -> None:
    adapter_dir = adapters_root(project_root) / adapter_id
    manifest_path = adapter_dir / "adapter_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path, "Adapter manifest")
    manifest["status"] = status
    manifest["activated"] = status == "active"
    if activated_at:
        manifest["activated_at"] = activated_at
    _write_json(manifest_path, manifest)


def _append_rollback_log(project_root: Path, event: dict) -> None:
    path = rollback_log_path(project_root)
    reject_symlink_components(path.parent, "Rollback log")
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_paths([path], "Rollback log")
    with path.open("a") as file:
        file.write(json.dumps(event, sort_keys=True) + "\n")


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
    reject_symlink_components(path.parent, "JSON output")
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_paths([path], "JSON output")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
