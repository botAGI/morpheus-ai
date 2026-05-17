"""
Filesystem safety helpers for local Morpheus state writes.
"""
from pathlib import Path


def reject_symlink_paths(paths: list[Path], label: str = "Path") -> None:
    """Reject paths that would follow a symlink during a read or write."""
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink: {path}")


def reject_symlink_components(path: Path, label: str = "Path") -> None:
    """Reject a path when any component in its chain is a symlink."""
    components = (path, *path.parents)
    for component in components:
        anchor = Path(component.anchor)
        if (
            component == component.parent
            or component == anchor
            or component == Path(".")
            or (path.is_absolute() and component.parent == anchor)
        ):
            continue
        if component.is_symlink():
            raise ValueError(f"{label} must not contain a symlink: {component}")
