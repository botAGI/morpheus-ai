"""Local registry helpers for Morpheus learning artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path

from morpheus.core.learning.dataset_validation import (
    manifest_count,
    parse_registry_timestamp_identity,
    validate_dataset,
)
from morpheus.core.safe_io import reject_symlink_components


def datasets_root(project_root: Path) -> Path:
    return project_root / ".morpheus" / "training" / "datasets"


def latest_dataset_dir(project_root: Path) -> Path | None:
    root = datasets_root(project_root)
    if root.is_symlink():
        raise ValueError(f"Dataset registry must not be a symlink: {root}")
    reject_symlink_components(root, "Dataset registry")
    if not root.is_dir():
        return None
    candidates = [path for path in root.iterdir() if not path.name.startswith(".")]
    return (
        max(candidates, key=lambda item: _registry_order_key(item.name))
        if candidates
        else None
    )


def dataset_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    reject_symlink_components(manifest_path, "Dataset manifest")
    data = json.loads(manifest_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Dataset manifest invalid: {manifest_path}")
    return data


def dataset_summary(project_root: Path, dataset_dir: Path, *, source: str) -> dict:
    try:
        manifest = dataset_manifest(dataset_dir)
        validation_manifest = manifest
    except (OSError, ValueError, json.JSONDecodeError):
        manifest = {}
        validation_manifest = None
    examples_count = manifest_count(manifest, "examples_count")
    validation = validate_dataset(project_root, dataset_dir, validation_manifest)
    return {
        "source": source,
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(dataset_dir / "manifest.json"),
        "dataset_id": _manifest_string(manifest, "dataset_id"),
        "dataset_sha256": _manifest_string(manifest, "dataset_sha256"),
        "examples_count": examples_count,
        "eval_items_count": manifest_count(manifest, "eval_items_count"),
        "skipped_count": manifest_count(manifest, "skipped_count"),
        "created_at": _manifest_string(manifest, "created_at"),
        "trainable": examples_count > 0 and validation["valid"],
        "validation": validation,
    }


def labs_root(project_root: Path) -> Path:
    return project_root / ".morpheus" / "lab"


def latest_lab_dir(project_root: Path) -> Path | None:
    root = labs_root(project_root)
    if root.is_symlink():
        raise ValueError(f"Lab registry must not be a symlink: {root}")
    reject_symlink_components(root, "Lab registry")
    if not root.is_dir():
        return None
    candidates = [path for path in root.iterdir() if path.name.startswith("lab_")]
    return (
        max(
            candidates,
            key=lambda item: _registry_order_key(item.name, prefix="lab_"),
        )
        if candidates
        else None
    )


def latest_lab_status(project_root: Path) -> dict | None:
    latest = latest_lab_dir(project_root)
    if latest is None:
        return None
    if latest.is_symlink() or not latest.is_dir():
        return _invalid_lab_status(latest, "lab_registry_entry_invalid")
    summary_path = latest / "lab_summary.json"
    try:
        reject_symlink_components(summary_path, "Lab summary")
        if summary_path.is_symlink() or not summary_path.is_file():
            return _invalid_lab_status(latest, "lab_summary_missing")
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        summary = None
    if (
        not isinstance(summary, dict)
        or not isinstance(summary.get("lab_id"), str)
        or summary.get("lab_id") != latest.name
    ):
        return _invalid_lab_status(latest, "lab_summary_invalid")
    try:
        json.dumps(summary, allow_nan=False)
    except (TypeError, ValueError):
        return _invalid_lab_status(latest, "lab_summary_invalid")
    dataset_dir = latest / "dataset"
    try:
        if dataset_dir.is_symlink() or not dataset_dir.is_dir():
            return _invalid_lab_status(latest, "lab_dataset_invalid")
        reject_symlink_components(dataset_dir, "Lab dataset")
    except (OSError, ValueError):
        return _invalid_lab_status(latest, "lab_dataset_invalid")
    return summary


def latest_lab_dataset_dir(project_root: Path) -> Path | None:
    latest = latest_lab_dir(project_root)
    if latest is None:
        return None
    return latest / "dataset"


def latest_effective_dataset(project_root: Path) -> dict | None:
    candidates: list[tuple[tuple[int, int, str], str, dict]] = []
    standalone = latest_dataset_dir(project_root)
    if standalone is not None:
        candidates.append((
            _registry_order_key(standalone.name),
            standalone.as_posix(),
            dataset_summary(project_root, standalone, source="standalone"),
        ))
    lab_dataset = latest_lab_dataset_dir(project_root)
    if lab_dataset is not None:
        candidates.append((
            _registry_order_key(lab_dataset.parent.name, prefix="lab_"),
            lab_dataset.as_posix(),
            dataset_summary(project_root, lab_dataset, source="lab"),
        ))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _registry_order_key(identity: str, *, prefix: str = "") -> tuple[int, int, str]:
    parsed = parse_registry_timestamp_identity(identity, prefix=prefix)
    if parsed is not None:
        delta = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
        sequence = (
            (delta.days * 86_400 + delta.seconds) * 1_000_000
            + delta.microseconds
        )
        return (0, sequence, identity)
    # A malformed registry identity is conservatively newer so it blocks fallback.
    return (1, 0, identity)


def latest_usable_dataset_dir(project_root: Path) -> Path | None:
    effective = latest_effective_dataset(project_root)
    if effective is None or not effective["trainable"]:
        return None
    return Path(str(effective["dataset_dir"]))


def learning_status(project_root: Path) -> dict:
    project_root = project_root.expanduser().resolve()
    from morpheus.core.learning.adapters import active_adapter_status

    latest = latest_dataset_dir(project_root)
    active_adapter = active_adapter_status(project_root)
    latest_lab = latest_lab_status(project_root)
    effective_dataset = latest_effective_dataset(project_root)
    if latest is None:
        return {
            "has_datasets": False,
            "latest_dataset_dir": None,
            "latest_manifest": None,
            "latest_standalone_dataset": None,
            "effective_dataset": effective_dataset,
            "has_labs": latest_lab is not None,
            "latest_lab": latest_lab,
            "active_adapter": active_adapter,
        }
    summary = dataset_summary(project_root, latest, source="standalone")
    try:
        manifest = dataset_manifest(latest)
    except (OSError, ValueError, json.JSONDecodeError):
        manifest = None
    if not summary["validation"]["valid"]:
        manifest = None
    return {
        "has_datasets": True,
        "latest_dataset_dir": str(latest),
        "latest_manifest": manifest,
        "latest_standalone_dataset": summary,
        "effective_dataset": effective_dataset,
        "has_labs": latest_lab is not None,
        "latest_lab": latest_lab,
        "active_adapter": active_adapter,
    }


def _manifest_string(manifest: object, field: str) -> str | None:
    if not isinstance(manifest, dict):
        return None
    value = manifest.get(field)
    return value if isinstance(value, str) else None


def _invalid_lab_status(lab_dir: Path, reason: str) -> dict:
    return {
        "lab_id": lab_dir.name,
        "lab_dir": str(lab_dir),
        "invalid": True,
        "validation_error": reason,
    }
