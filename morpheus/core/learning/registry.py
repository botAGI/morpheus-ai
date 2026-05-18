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


def learning_status(project_root: Path) -> dict:
    project_root = project_root.expanduser().resolve()
    from morpheus.core.learning.adapters import active_adapter_status

    latest = latest_dataset_dir(project_root)
    active_adapter = active_adapter_status(project_root)
    if latest is None:
        return {
            "has_datasets": False,
            "latest_dataset_dir": None,
            "latest_manifest": None,
            "active_adapter": active_adapter,
        }
    manifest = json.loads((latest / "manifest.json").read_text())
    return {
        "has_datasets": True,
        "latest_dataset_dir": str(latest),
        "latest_manifest": manifest,
        "active_adapter": active_adapter,
    }
