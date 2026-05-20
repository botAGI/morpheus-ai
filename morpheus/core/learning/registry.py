"""Local registry helpers for Morpheus learning artifacts."""
import json
from pathlib import Path

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
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and not path.is_symlink() and (path / "manifest.json").is_file()
    ]
    return sorted(candidates, key=lambda item: item.name)[-1] if candidates else None


def dataset_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    reject_symlink_components(manifest_path, "Dataset manifest")
    data = json.loads(manifest_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Dataset manifest invalid: {manifest_path}")
    return data


def dataset_summary(dataset_dir: Path, *, source: str) -> dict:
    manifest = dataset_manifest(dataset_dir)
    examples_count = int(manifest.get("examples_count") or 0)
    return {
        "source": source,
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(dataset_dir / "manifest.json"),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "examples_count": examples_count,
        "eval_items_count": int(manifest.get("eval_items_count") or 0),
        "skipped_count": int(manifest.get("skipped_count") or 0),
        "created_at": manifest.get("created_at"),
        "trainable": examples_count > 0,
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
    candidates = [
        path for path in root.iterdir()
        if (
            path.is_dir()
            and not path.is_symlink()
            and path.name.startswith("lab_")
            and (path / "lab_summary.json").is_file()
        )
    ]
    return sorted(candidates, key=lambda item: item.name)[-1] if candidates else None


def latest_lab_status(project_root: Path) -> dict | None:
    latest = latest_lab_dir(project_root)
    if latest is None:
        return None
    return json.loads((latest / "lab_summary.json").read_text())


def latest_lab_dataset_dir(project_root: Path) -> Path | None:
    latest = latest_lab_dir(project_root)
    if latest is None:
        return None
    dataset_dir = latest / "dataset"
    if dataset_dir.is_symlink():
        raise ValueError(f"Lab dataset path must not be a symlink: {dataset_dir}")
    reject_symlink_components(dataset_dir, "Lab dataset")
    if (dataset_dir / "manifest.json").is_file():
        return dataset_dir
    return None


def latest_effective_dataset(project_root: Path) -> dict | None:
    candidates = []
    standalone = latest_dataset_dir(project_root)
    if standalone is not None:
        candidates.append(dataset_summary(standalone, source="standalone"))
    lab_dataset = latest_lab_dataset_dir(project_root)
    if lab_dataset is not None:
        candidates.append(dataset_summary(lab_dataset, source="lab"))
    usable = [item for item in candidates if item["trainable"]]
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda item: str(item.get("created_at") or item.get("dataset_id") or ""),
    )[-1]


def latest_usable_dataset_dir(project_root: Path) -> Path | None:
    effective = latest_effective_dataset(project_root)
    if effective is None:
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
    manifest = dataset_manifest(latest)
    return {
        "has_datasets": True,
        "latest_dataset_dir": str(latest),
        "latest_manifest": manifest,
        "latest_standalone_dataset": dataset_summary(latest, source="standalone"),
        "effective_dataset": effective_dataset,
        "has_labs": latest_lab is not None,
        "latest_lab": latest_lab,
        "active_adapter": active_adapter,
    }
