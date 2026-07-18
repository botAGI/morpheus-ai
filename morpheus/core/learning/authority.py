"""Cross-registry authority lock for learning publication and activation."""

from contextlib import contextmanager
from pathlib import Path

from morpheus.core.portable_lock import portable_file_lock
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths


@contextmanager
def learning_authority_transaction(project_root: Path):
    """Keep dataset, lab, and eval authority stable for one local operation.

    Callers that also need state or review authority acquire those locks first.
    Registry publishers hold this lock across their atomic publication step;
    activation and rollback hold it through the pointer transaction.
    """
    project_root = project_root.expanduser()
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")

    morpheus_dir = project_root / ".morpheus"
    reject_symlink_components(morpheus_dir, "Learning authority directory")
    morpheus_dir.mkdir(mode=0o700, exist_ok=True)
    reject_symlink_components(morpheus_dir, "Learning authority directory")
    if morpheus_dir.is_symlink() or not morpheus_dir.is_dir():
        raise ValueError(
            f"Learning authority directory not found: {morpheus_dir}"
        )
    lock_path = morpheus_dir / ".learning-authority.lock"
    reject_symlink_paths([lock_path], "Learning authority lock")
    reject_symlink_components(lock_path, "Learning authority lock")
    with portable_file_lock(lock_path):
        yield
