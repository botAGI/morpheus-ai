"""Dry-run training run planner for reviewed Morpheus datasets."""
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from morpheus.core.learning.adapter_artifacts import ADAPTER_ARTIFACT_SCHEMA
from morpheus.core.learning.backends import get_backend
from morpheus.core.learning.dataset_validation import dataset_binding_sha256
from morpheus.core.learning.dataset_validation import manifest_count
from morpheus.core.learning.dataset_validation import require_valid_dataset
from morpheus.core.learning.dataset_validation import validate_dataset_artifacts
from morpheus.core.learning.registry import latest_effective_dataset
from morpheus.core.learning.training_runtime import (
    RUNTIME_DATASET_DIR_PLACEHOLDER,
    RUNTIME_DATASET_PATH_PLACEHOLDER,
    render_guarded_training_command,
    seal_dataset_snapshot,
)
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_METHOD = "qlora"
DEFAULT_BACKEND = "llamafactory"
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_DROPOUT = 0.05
DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = "2e-4"
DEFAULT_MAX_SEQ_LENGTH = 4096
SMALL_DATASET_THRESHOLD = 20


@dataclass(frozen=True)
class TrainingConfig:
    run_id: str
    adapter_id: str
    backend: str
    method: str
    base_model: str
    rank: int
    alpha: int
    dropout: float
    epochs: int
    learning_rate: str
    max_seq_length: int
    dataset_manifest_path: str
    dataset_path: str
    dataset_dir: str
    dataset_name: str
    output_dir: str


def plan_training_run(
    project_root: Path,
    *,
    backend: str = DEFAULT_BACKEND,
    method: str = DEFAULT_METHOD,
    base_model: str = DEFAULT_BASE_MODEL,
    rank: int = DEFAULT_RANK,
    alpha: int = DEFAULT_ALPHA,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: str = DEFAULT_LEARNING_RATE,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    dry_run: bool = True,
    execute: bool = False,
    confirm_execute: bool = False,
) -> dict:
    project_root = _safe_project_root(project_root)
    if execute and not confirm_execute:
        raise ValueError("--execute requires --yes-i-know-this-will-train")
    if execute:
        raise ValueError(
            "Direct `morpheus learn train --execute` is unsupported; "
            "use `morpheus learn lab . --backend mlx` for guarded local execution."
        )
    if not dry_run:
        raise ValueError("`morpheus learn train` supports dry-run planning only")

    effective_dataset = latest_effective_dataset(project_root)
    if effective_dataset is None:
        raise ValueError(
            "No learning dataset manifest found. Run `morpheus learn dataset .` "
            "or `morpheus learn lab . --no-train` first."
        )
    dataset_dir = Path(str(effective_dataset["dataset_dir"]))
    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = _read_json(dataset_manifest_path, "Dataset manifest")
    examples_count = manifest_count(dataset_manifest, "examples_count")
    if examples_count <= 0:
        raise ValueError("Refusing to train: dataset has zero examples.")
    validation = require_valid_dataset(project_root, dataset_dir, dataset_manifest)

    selected_dataset_path = dataset_dir / str(dataset_manifest["selected_dataset_file"])
    reject_symlink_paths([selected_dataset_path], "Selected dataset")
    reject_symlink_components(selected_dataset_path, "Selected dataset")

    backend_impl = get_backend(backend)
    backend_impl.validate_method(method)
    run_id = _timestamp_id("train")
    adapter_id = _timestamp_id("adapter")
    output_dir = project_root / ".morpheus" / "training" / "adapters" / adapter_id
    run_dir = project_root / ".morpheus" / "training" / "runs" / run_id
    current_manifest = _read_json(dataset_manifest_path, "Dataset manifest")
    current_validation = require_valid_dataset(
        project_root,
        dataset_dir,
        current_manifest,
    )
    if (
        current_validation["dataset_binding_sha256"]
        != validation["dataset_binding_sha256"]
    ):
        raise ValueError("Dataset binding changed while planning training.")
    _ensure_run_dir(run_dir)
    snapshot_dir = run_dir / "dataset"
    snapshot_manifest_path = _copy_dataset_snapshot(
        dataset_dir,
        snapshot_dir,
        current_manifest,
        current_validation["dataset_binding_sha256"],
    )
    _ensure_adapter_dir(output_dir)
    snapshot_selected_dataset_path = snapshot_dir / str(
        current_manifest["selected_dataset_file"]
    )

    config = TrainingConfig(
        run_id=run_id,
        adapter_id=adapter_id,
        backend=backend,
        method=method,
        base_model=base_model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        dataset_manifest_path=str(snapshot_manifest_path),
        dataset_path=str(snapshot_selected_dataset_path),
        dataset_dir=str(snapshot_dir),
        dataset_name=snapshot_selected_dataset_path.stem,
        output_dir=str(output_dir),
    )
    config_dict = asdict(config)
    runtime_config = {
        **config_dict,
        "dataset_path": RUNTIME_DATASET_PATH_PLACEHOLDER,
        "dataset_dir": RUNTIME_DATASET_DIR_PLACEHOLDER,
    }
    rendered = backend_impl.render_command(runtime_config, dry_run=dry_run)
    guarded_command = render_guarded_training_command(
        rendered.command,
        project_root=project_root,
        source_dataset_dir=dataset_dir,
        snapshot_dir=snapshot_dir,
        expected_binding_sha256=current_validation["dataset_binding_sha256"],
    )

    train_config_path = run_dir / "train_config.yaml"
    command_path = run_dir / "command.sh"
    dataset_manifest_copy = run_dir / "dataset_manifest.json"
    run_manifest_path = run_dir / "run_manifest.json"
    adapter_manifest_path = run_dir / "adapter_manifest.json"
    registry_adapter_manifest_path = output_dir / "adapter_manifest.json"

    train_config_path.write_text(_render_yaml(config_dict))
    command_path.write_text(_render_preview_only_command(guarded_command))
    command_path.chmod(0o644)
    dataset_manifest_copy.write_text(
        json.dumps(current_manifest, indent=2, sort_keys=True) + "\n"
    )

    warnings = []
    if examples_count < SMALL_DATASET_THRESHOLD:
        warnings.append(
            f"Dataset has {examples_count} examples; recommended minimum is {SMALL_DATASET_THRESHOLD}."
        )

    adapter_manifest = {
        "adapter_id": adapter_id,
        "artifact_schema": ADAPTER_ARTIFACT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "training_status": "planned",
        "weight_artifact": None,
        "backend": backend,
        "method": method,
        "base_model": base_model,
        "output_dir": str(output_dir.resolve()),
        "run_id": run_id,
        "dataset_id": dataset_manifest.get("dataset_id"),
        "dataset_binding_sha256": validation["dataset_binding_sha256"],
        "activated": False,
    }
    run_manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "method": method,
        "base_model": base_model,
        "dry_run": dry_run,
        "execute": execute,
        "dataset_id": dataset_manifest.get("dataset_id"),
        "dataset_source": effective_dataset.get("source"),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_snapshot_dir": str(snapshot_dir),
        "dataset_snapshot_manifest_path": str(snapshot_manifest_path),
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
        "dataset_binding_sha256": validation["dataset_binding_sha256"],
        "dataset_examples_count": examples_count,
        "adapter_id": adapter_id,
        "adapter_manifest_path": str(adapter_manifest_path),
        "warnings": warnings,
        "backend_notes": rendered.backend_notes,
    }
    adapter_manifest_path.write_text(json.dumps(adapter_manifest, indent=2, sort_keys=True) + "\n")
    registry_adapter_manifest_path.write_text(json.dumps(adapter_manifest, indent=2, sort_keys=True) + "\n")
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    return {
        "run_id": run_id,
        "adapter_id": adapter_id,
        "run_dir": str(run_dir),
        "train_config_path": str(train_config_path),
        "command_path": str(command_path),
        "run_manifest_path": str(run_manifest_path),
        "adapter_manifest_path": str(adapter_manifest_path),
        "dataset_snapshot_dir": str(snapshot_dir),
        "dry_run": dry_run,
        "execute": execute,
        "warnings": warnings,
    }


def _safe_project_root(project_root: Path) -> Path:
    project_root = project_root.expanduser()
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")
    return project_root


def _ensure_run_dir(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Training run path must not be a symlink: {path}")
    reject_symlink_components(path.parent, "Training runs path")
    path.mkdir(parents=True, exist_ok=False)
    reject_symlink_components(path, "Training run path")


def _ensure_adapter_dir(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Adapter output must not be a symlink: {path}")
    reject_symlink_components(path.parent, "Adapter output root")
    path.mkdir(parents=True, exist_ok=False)
    reject_symlink_components(path, "Adapter output")


def _copy_dataset_snapshot(
    source_dir: Path,
    snapshot_dir: Path,
    manifest: dict,
    expected_binding_sha256: str,
) -> Path:
    if snapshot_dir.is_symlink():
        raise ValueError(f"Training dataset snapshot must not be a symlink: {snapshot_dir}")
    snapshot_dir.mkdir(parents=False, exist_ok=False)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Dataset artifacts invalid while creating training snapshot")
    for raw_path in sorted(artifacts):
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Dataset artifact path invalid: {raw_path}")
        source_path = source_dir / relative
        destination_path = snapshot_dir / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path, follow_symlinks=False)
    snapshot_manifest_path = snapshot_dir / "manifest.json"
    snapshot_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if (
        manifest.get("dataset_binding_sha256") != expected_binding_sha256
        or dataset_binding_sha256(manifest) != expected_binding_sha256
    ):
        raise ValueError("Dataset binding changed while creating training snapshot")
    snapshot_validation = validate_dataset_artifacts(snapshot_dir, manifest)
    if not snapshot_validation["valid"]:
        raise ValueError(
            "Training dataset snapshot validation failed: "
            + ", ".join(snapshot_validation["blockers"])
        )
    seal_dataset_snapshot(snapshot_dir)
    return snapshot_manifest_path


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


def _timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def _render_yaml(data: dict) -> str:
    lines = []
    for key, value in data.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"


def _render_preview_only_command(guarded_command: str) -> str:
    preview = [f"# {line}" if line else "#" for line in guarded_command.splitlines()]
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'echo "Training preview only: direct learn train execution is unsupported." >&2',
        "exit 2",
        "",
        "# Guarded backend preview (comments only):",
        *preview,
        "",
    ])
