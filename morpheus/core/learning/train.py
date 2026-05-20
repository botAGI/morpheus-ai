"""Dry-run training run planner for reviewed Morpheus datasets."""
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from morpheus.core.learning.backends import get_backend
from morpheus.core.learning.registry import (
    latest_dataset_dir,
    latest_effective_dataset,
    latest_usable_dataset_dir,
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
    if execute and dry_run:
        raise ValueError("Use --execute without --dry-run")

    dataset_dir = latest_usable_dataset_dir(project_root)
    if dataset_dir is None:
        if latest_dataset_dir(project_root) is not None:
            raise ValueError("Refusing to train: dataset has zero examples.")
        raise ValueError(
            "No learning dataset manifest found. Run `morpheus learn dataset .` "
            "or `morpheus learn lab . --no-train` first."
        )
    effective_dataset = latest_effective_dataset(project_root) or {}
    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = _read_json(dataset_manifest_path, "Dataset manifest")
    examples_count = int(dataset_manifest.get("examples_count") or 0)
    if examples_count <= 0:
        raise ValueError("Refusing to train: dataset has zero examples.")

    selected_dataset_path = Path(str(dataset_manifest.get("selected_dataset_path") or ""))
    if not selected_dataset_path.is_absolute():
        selected_dataset_path = dataset_dir / selected_dataset_path
    if not selected_dataset_path.is_file():
        selected_dataset_path = _fallback_dataset_path(dataset_dir, dataset_manifest)
    reject_symlink_paths([selected_dataset_path], "Selected dataset")
    reject_symlink_components(selected_dataset_path, "Selected dataset")

    backend_impl = get_backend(backend)
    backend_impl.validate_method(method)
    run_id = _timestamp_id("train")
    adapter_id = _timestamp_id("adapter")
    output_dir = project_root / ".morpheus" / "training" / "adapters" / adapter_id
    run_dir = project_root / ".morpheus" / "training" / "runs" / run_id
    _ensure_run_dir(run_dir)
    _ensure_adapter_dir(output_dir)

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
        dataset_manifest_path=str(dataset_manifest_path),
        dataset_path=str(selected_dataset_path),
        dataset_dir=str(selected_dataset_path.parent),
        dataset_name=selected_dataset_path.stem,
        output_dir=str(output_dir),
    )
    config_dict = asdict(config)
    rendered = backend_impl.render_command(config_dict, dry_run=dry_run)

    train_config_path = run_dir / "train_config.yaml"
    command_path = run_dir / "command.sh"
    dataset_manifest_copy = run_dir / "dataset_manifest.json"
    run_manifest_path = run_dir / "run_manifest.json"
    adapter_manifest_path = run_dir / "adapter_manifest.json"
    registry_adapter_manifest_path = output_dir / "adapter_manifest.json"

    train_config_path.write_text(_render_yaml(config_dict))
    command_path.write_text(rendered.command)
    command_path.chmod(0o755)
    dataset_manifest_copy.write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n")

    warnings = []
    if examples_count < SMALL_DATASET_THRESHOLD:
        warnings.append(
            f"Dataset has {examples_count} examples; recommended minimum is {SMALL_DATASET_THRESHOLD}."
        )

    adapter_manifest = {
        "adapter_id": adapter_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "backend": backend,
        "method": method,
        "base_model": base_model,
        "output_dir": str(output_dir.resolve()),
        "run_id": run_id,
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
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
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


def _fallback_dataset_path(dataset_dir: Path, manifest: dict) -> Path:
    selected = str(manifest.get("selected_format") or "instruction")
    candidates = [dataset_dir / f"dataset.{selected}.jsonl"]
    if selected == "chat":
        candidates.extend([dataset_dir / "train.jsonl", dataset_dir / "dataset.sharegpt.jsonl"])
    candidates.extend(
        [
            dataset_dir / "train.jsonl",
            dataset_dir / "dataset.instruction.jsonl",
            dataset_dir / "dataset.sharegpt.jsonl",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError("Selected dataset file not found for latest manifest.")


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
