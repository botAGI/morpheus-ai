"""Cross-process authority lock for active state and receipt operations."""

from contextlib import contextmanager
from pathlib import Path

from morpheus.core.portable_lock import portable_file_lock
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths


def _safe_project_root(project_root: Path) -> Path:
    project_root = project_root.expanduser()
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")
    return project_root


def _state_lock_path(project_root: Path) -> Path:
    morpheus_dir = project_root / ".morpheus"
    reject_symlink_components(morpheus_dir, "State authority directory")
    if morpheus_dir.is_symlink():
        raise ValueError(
            f"State authority directory must not be a symlink: {morpheus_dir}"
        )
    if not morpheus_dir.is_dir():
        raise ValueError(f"State authority directory not found: {morpheus_dir}")
    lock_path = morpheus_dir / ".state-authority.lock"
    reject_symlink_paths([lock_path], "State authority lock")
    reject_symlink_components(lock_path, "State authority lock")
    return lock_path


@contextmanager
def state_authority_transaction(project_root: Path):
    """Hold active state and its receipt authority stable through one operation.

    This lock is process- and thread-safe and reentrant for one thread. Callers
    that also need a ``ReviewStore`` transaction must acquire this lock first.
    """
    project_root = _safe_project_root(project_root)
    lock_path = _state_lock_path(project_root)
    with portable_file_lock(lock_path):
        yield
