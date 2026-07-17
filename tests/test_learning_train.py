import json
import os
from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.train import plan_training_run
from tests.test_learning_dataset import copy_learning_project


def test_train_dry_run_creates_config_command_and_manifests(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = plan_training_run(project_root, backend="llamafactory", method="qlora", dry_run=True)

    run_dir = Path(result["run_dir"])
    train_config = (run_dir / "train_config.yaml").read_text()
    command = (run_dir / "command.sh").read_text()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    adapter_manifest = json.loads((run_dir / "adapter_manifest.json").read_text())
    dataset_manifest = json.loads((run_dir / "dataset_manifest.json").read_text())

    assert result["dry_run"] is True
    assert "base_model: Qwen/Qwen2.5-7B-Instruct" in train_config
    assert "method: qlora" in train_config
    assert "llamafactory-cli train" in command
    assert '--dataset_dir "${MORPHEUS_DATASET_DIR}"' in command
    assert run_manifest["dataset_sha256"] == dataset_manifest["dataset_sha256"]
    assert run_manifest["backend"] == "llamafactory"
    assert adapter_manifest["status"] == "planned"
    assert adapter_manifest["artifact_schema"] == "morpheus-adapter-artifact/1"
    assert adapter_manifest["training_status"] == "planned"
    assert adapter_manifest["weight_artifact"] is None
    assert Path(adapter_manifest["output_dir"]).is_absolute()


def test_train_peft_lora_dry_run_uses_peft_backend_without_download(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = plan_training_run(project_root, backend="peft", method="lora", dry_run=True)

    run_dir = Path(result["run_dir"])
    train_config = (run_dir / "train_config.yaml").read_text()
    command = (run_dir / "command.sh").read_text()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())

    assert "backend: peft" in train_config
    assert "method: lora" in train_config
    assert "--dry-run" in command
    assert "python -m morpheus.learning_peft_train" in command
    assert run_manifest["execute"] is False


def test_user_model_value_cannot_expand_guard_runtime_variables(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    user_model = "__MORPHEUS_RUNTIME_DATASET_DIR__"

    result = plan_training_run(
        project_root,
        backend="peft",
        method="lora",
        base_model=user_model,
        dry_run=True,
    )

    command = (Path(result["run_dir"]) / "command.sh").read_text()
    assert f"--base-model {user_model}" in command
    assert "${MORPHEUS_DATASET_DIR}" not in command


def test_train_refuses_missing_dataset_manifest(tmp_path):
    project_root = tmp_path / "empty_project"
    project_root.mkdir()

    result = CliRunner().invoke(app, ["learn", "train", str(project_root), "--dry-run"])

    assert result.exit_code == 2
    assert "No learning dataset manifest found" in result.output


def test_train_refuses_zero_example_dataset(tmp_path):
    project_root = tmp_path / "zero_project"
    project_root.mkdir()
    review_dir = project_root / ".morpheus" / "review"
    review_dir.mkdir(parents=True)
    (review_dir / "semantic_candidates.jsonl").write_text("")
    build_learning_dataset(project_root)

    result = CliRunner().invoke(app, ["learn", "train", str(project_root), "--dry-run"])

    assert result.exit_code == 2
    assert "zero examples" in result.output


def test_cli_train_dry_run_warns_for_small_dataset_and_writes_artifacts(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = CliRunner().invoke(
        app,
        ["learn", "train", str(project_root), "--backend", "peft", "--method", "lora", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "warning" in result.output.casefold()
    payload = json.loads(result.output[result.output.index("{"):])
    run_dir = Path(payload["run_dir"])
    assert (run_dir / "train_config.yaml").is_file()
    assert (run_dir / "command.sh").is_file()


def test_cli_train_confirmed_execute_is_unsupported_without_registry_writes(
    tmp_path,
):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    runs_root = project_root / ".morpheus/training/runs"
    adapters_root = project_root / ".morpheus/training/adapters"

    result = CliRunner().invoke(
        app,
        [
            "learn",
            "train",
            str(project_root),
            "--backend",
            "peft",
            "--execute",
            "--yes-i-know-this-will-train",
        ],
    )

    assert result.exit_code == 2
    assert "Direct `morpheus learn train --execute` is unsupported" in result.output
    assert not runs_root.exists()
    assert not adapters_root.exists()


def test_cli_train_no_dry_run_is_rejected_without_registry_writes(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = CliRunner().invoke(
        app,
        ["learn", "train", str(project_root), "--no-dry-run"],
    )

    assert result.exit_code == 2
    assert "supports dry-run planning only" in result.output
    assert not (project_root / ".morpheus/training/runs").exists()
    assert not (project_root / ".morpheus/training/adapters").exists()
    with pytest.raises(ValueError, match="supports dry-run planning only"):
        plan_training_run(project_root, dry_run=False)


@pytest.mark.parametrize(
    ("backend", "fake_binary"),
    [("llamafactory", "llamafactory-cli"), ("peft", "python")],
)
def test_dry_run_command_is_nonexecutable_preview_and_never_invokes_backend(
    tmp_path,
    backend,
    fake_binary,
):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    marker = tmp_path / "backend-ran.txt"
    fake_bin_dir = tmp_path / "bin"
    fake_bin_dir.mkdir()
    fake_backend = fake_bin_dir / fake_binary
    fake_backend.write_text(
        "#!/bin/sh\n"
        f"printf ran > {marker}\n"
        "exit 0\n"
    )
    fake_backend.chmod(0o755)
    training = plan_training_run(
        project_root,
        backend=backend,
        method="lora",
        dry_run=True,
    )
    command_path = Path(training["command_path"])
    environment = os.environ.copy()
    environment["PATH"] = str(fake_bin_dir) + os.pathsep + environment.get("PATH", "")

    completed = subprocess.run(
        ["/bin/bash", str(command_path)],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert command_path.stat().st_mode & 0o111 == 0
    assert completed.returncode == 2
    assert "Training preview only" in completed.stderr
    assert not marker.exists()
