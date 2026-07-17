"""Fail-closed execution guard for planned learning runs."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import stat
import sys
from typing import Iterator

from morpheus.core.learning.dataset_validation import dataset_binding_sha256
from morpheus.core.learning.dataset_validation import parse_registry_timestamp_identity
from morpheus.core.learning.dataset_validation import require_valid_dataset
from morpheus.core.learning.dataset_validation import validate_dataset_artifacts
from morpheus.core.learning.training_runtime import pin_dataset_snapshot
from morpheus.core.learning.training_runtime import pin_training_output_directory
from morpheus.core.learning.training_runtime import MLX_PINNED_LOADER_CONTRACT
from morpheus.core.learning.training_runtime import supervise_training_backend
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.state_authority import state_authority_transaction


def validate_training_run_guard(
    project_root: Path,
    source_dataset_dir: Path,
    snapshot_dir: Path,
    expected_binding_sha256: str,
) -> dict:
    """Recheck live authority and the private snapshot immediately before training."""
    project_root = _safe_directory(project_root, "Project root")
    source_dataset_dir = _safe_directory(source_dataset_dir, "Source dataset")
    snapshot_dir = _safe_directory(snapshot_dir, "Training dataset snapshot")
    _validate_snapshot_location(project_root, snapshot_dir)
    _require_sealed_snapshot(snapshot_dir)

    source_manifest = _read_json(
        source_dataset_dir / "manifest.json",
        "Source dataset manifest",
    )
    source_validation = require_valid_dataset(
        project_root,
        source_dataset_dir,
        source_manifest,
    )
    if source_validation["dataset_binding_sha256"] != expected_binding_sha256:
        raise ValueError("Source dataset binding changed after training was planned")

    snapshot_manifest = _read_json(
        snapshot_dir / "manifest.json",
        "Training dataset snapshot manifest",
    )
    if (
        snapshot_manifest.get("dataset_binding_sha256") != expected_binding_sha256
        or dataset_binding_sha256(snapshot_manifest) != expected_binding_sha256
    ):
        raise ValueError("Training dataset snapshot binding mismatch")
    snapshot_validation = validate_dataset_artifacts(snapshot_dir, snapshot_manifest)
    if not snapshot_validation["valid"]:
        raise ValueError(
            "Training dataset snapshot validation failed: "
            + ", ".join(snapshot_validation["blockers"])
        )

    final_source_manifest = _read_json(
        source_dataset_dir / "manifest.json",
        "Source dataset manifest",
    )
    final_source_validation = require_valid_dataset(
        project_root,
        source_dataset_dir,
        final_source_manifest,
    )
    if (
        final_source_manifest != source_manifest
        or final_source_validation["dataset_binding_sha256"] != expected_binding_sha256
    ):
        raise ValueError("Source dataset changed while the training guard was running")

    final_snapshot_manifest = _read_json(
        snapshot_dir / "manifest.json",
        "Training dataset snapshot manifest",
    )
    final_snapshot_validation = validate_dataset_artifacts(
        snapshot_dir,
        final_snapshot_manifest,
    )
    if (
        final_snapshot_manifest != snapshot_manifest
        or dataset_binding_sha256(final_snapshot_manifest) != expected_binding_sha256
        or not final_snapshot_validation["valid"]
    ):
        blockers = final_snapshot_validation["blockers"] or [
            "dataset_snapshot_changed"
        ]
        raise ValueError(
            "Training dataset snapshot validation failed: " + ", ".join(blockers)
        )
    return {
        "valid": True,
        "dataset_binding_sha256": expected_binding_sha256,
        "source_validation": final_source_validation,
        "snapshot_validation": final_snapshot_validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--expected-binding", required=True)
    parser.add_argument("--backend-command")
    parser.add_argument("--trusted-loader")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-output-device", type=int)
    parser.add_argument("--expected-output-inode", type=int)
    args = parser.parse_args(argv)
    try:
        with training_authority_lease(
            args.project_root,
            args.source_dataset_dir,
        ):
            if args.backend_command is not None:
                validate_training_run_guard(
                    args.project_root,
                    args.source_dataset_dir,
                    args.snapshot_dir,
                    args.expected_binding,
                )
                if args.trusted_loader != MLX_PINNED_LOADER_CONTRACT:
                    raise ValueError(
                        "Executable training requires the trusted pinned-FD MLX loader"
                    )
                return _run_backend_under_authority_lease(args)
            if args.trusted_loader is not None:
                raise ValueError("Trusted training loader has no backend command")
            validate_training_run_guard(
                args.project_root,
                args.source_dataset_dir,
                args.snapshot_dir,
                args.expected_binding,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Training guard failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _run_backend_under_authority_lease(args: argparse.Namespace) -> int:
    output_arguments = (
        args.output_dir,
        args.expected_output_device,
        args.expected_output_inode,
    )
    if any(value is None for value in output_arguments):
        raise ValueError("Trusted MLX training output identity is incomplete")
    _validate_output_location(
        args.project_root,
        args.snapshot_dir,
        args.output_dir,
    )
    output_identity = (
        args.expected_output_device,
        args.expected_output_inode,
    )
    with pin_training_output_directory(
        args.output_dir,
        output_identity,
    ) as output_descriptor:
        with pin_dataset_snapshot(
            args.snapshot_dir,
            args.expected_binding,
            view_parent_descriptor=output_descriptor,
            view_parent_path=args.output_dir,
            expected_view_parent_identity=output_identity,
        ) as snapshot:
            validate_training_run_guard(
                args.project_root,
                args.source_dataset_dir,
                args.snapshot_dir,
                args.expected_binding,
            )
            returncode = supervise_training_backend(
                args.backend_command,
                snapshot,
                trusted_loader=args.trusted_loader,
                output_descriptor=output_descriptor,
            )
            validate_training_run_guard(
                args.project_root,
                args.source_dataset_dir,
                args.snapshot_dir,
                args.expected_binding,
            )
            return returncode


@contextmanager
def training_authority_lease(
    project_root: Path,
    source_dataset_dir: Path,
) -> Iterator[None]:
    """Keep state and review authority stable for the complete backend run."""
    project_root = _safe_directory(project_root, "Project root")
    source_dataset_dir = _safe_directory(source_dataset_dir, "Source dataset")
    review_root = _review_authority_root(project_root, source_dataset_dir)
    with state_authority_transaction(project_root):
        with ReviewStore(review_root).transaction():
            yield


def _review_authority_root(project_root: Path, source_dataset_dir: Path) -> Path:
    registry_root = project_root / ".morpheus" / "training" / "datasets"
    try:
        if (
            source_dataset_dir.parent == registry_root.resolve()
            and parse_registry_timestamp_identity(source_dataset_dir.name) is not None
        ):
            return project_root
    except OSError:
        pass

    lab_root = project_root / ".morpheus" / "lab"
    try:
        relative = source_dataset_dir.relative_to(lab_root.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError("Source dataset is outside an allowed authority scope") from exc
    if (
        len(relative.parts) != 2
        or relative.parts[1] != "dataset"
        or parse_registry_timestamp_identity(relative.parts[0], prefix="lab_") is None
    ):
        raise ValueError("Source dataset authority scope is invalid")
    return _safe_directory(
        lab_root / relative.parts[0] / "workspace",
        "Lab review workspace",
    )


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


def _safe_directory(path: Path, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    reject_symlink_components(path, label)
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return path


def _validate_snapshot_location(project_root: Path, snapshot_dir: Path) -> None:
    lab_root = project_root / ".morpheus" / "lab"
    try:
        relative_lab = snapshot_dir.relative_to(lab_root)
    except ValueError:
        relative_lab = None
    if (
        relative_lab is not None
        and len(relative_lab.parts) == 2
        and relative_lab.parts[1] == "dataset"
        and parse_registry_timestamp_identity(
            relative_lab.parts[0],
            prefix="lab_",
        )
        is not None
    ):
        reject_symlink_components(lab_root, "Lab registry")
        return

    runs_root = _safe_directory(
        project_root / ".morpheus" / "training" / "runs",
        "Training runs registry",
    )
    try:
        relative_run = snapshot_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("Training dataset snapshot is outside an allowed registry") from exc
    if (
        len(relative_run.parts) != 2
        or relative_run.parts[1] != "dataset"
        or parse_registry_timestamp_identity(
            relative_run.parts[0],
            prefix="train_",
        )
        is None
    ):
        raise ValueError("Training dataset snapshot identity is invalid")


def _validate_output_location(
    project_root: Path,
    snapshot_dir: Path,
    output_dir: Path,
) -> None:
    project_root = project_root.expanduser().resolve()
    snapshot_dir = snapshot_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().absolute()
    lab_root = project_root / ".morpheus" / "lab"
    try:
        relative_lab = snapshot_dir.relative_to(lab_root)
    except ValueError as exc:
        raise ValueError("Pinned training output is only supported for MLX labs") from exc
    if (
        len(relative_lab.parts) != 2
        or relative_lab.parts[1] != "dataset"
        or parse_registry_timestamp_identity(
            relative_lab.parts[0],
            prefix="lab_",
        )
        is None
        or output_dir != snapshot_dir.parent / "training" / "adapter"
    ):
        raise ValueError("Training output directory is outside the guarded MLX lab")


def _require_sealed_snapshot(snapshot_dir: Path) -> None:
    writable = []
    for path in [snapshot_dir, *snapshot_dir.rglob("*")]:
        reject_symlink_paths([path], "Training dataset snapshot")
        reject_symlink_components(path, "Training dataset snapshot")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            writable.append(path.relative_to(snapshot_dir).as_posix() or ".")
    if writable:
        raise ValueError(
            "Training dataset snapshot is not sealed: " + ", ".join(sorted(writable))
        )


if __name__ == "__main__":
    raise SystemExit(main())
