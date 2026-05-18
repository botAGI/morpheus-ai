import json
from pathlib import Path

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
    assert run_manifest["dataset_sha256"] == dataset_manifest["dataset_sha256"]
    assert run_manifest["backend"] == "llamafactory"
    assert adapter_manifest["status"] == "planned"
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
