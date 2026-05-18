import json
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.eval import check_activation_gate, run_learning_eval
from morpheus.core.learning.train import plan_training_run
from tests.test_learning_dataset import copy_learning_project, read_jsonl


def test_eval_reads_seed_and_writes_results_and_report(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)

    result = run_learning_eval(project_root, dry_run=True, base_only=True)

    eval_dir = Path(result["eval_dir"])
    config = json.loads((eval_dir / "eval_config.json").read_text())
    results = json.loads((eval_dir / "eval_results.json").read_text())
    report = (eval_dir / "eval_report.md").read_text()
    seed_items = read_jsonl(Path(dataset["dataset_dir"]) / "eval.seed.jsonl")

    assert config["base_only"] is True
    assert results["metrics"]["pass_rate"] >= 0
    assert len(results["items"]) == len(seed_items)
    assert "unsupported_claim_refusal_rate" in results["metrics"]
    assert "# Morpheus Learning Eval" in report


def test_eval_adapter_dry_run_uses_adapter_fake_provider(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)

    result = run_learning_eval(project_root, adapter_id=train["adapter_id"], dry_run=True)

    eval_dir = Path(result["eval_dir"])
    config = json.loads((eval_dir / "eval_config.json").read_text())
    results = json.loads((eval_dir / "eval_results.json").read_text())

    assert config["adapter_id"] == train["adapter_id"]
    assert config["provider"]["name"] == "fake-adapter"
    assert results["metrics"]["pass_rate"] == 1.0


def test_cli_eval_dry_run_writes_artifacts(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = CliRunner().invoke(app, ["learn", "eval", str(project_root), "--dry-run"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    eval_dir = Path(payload["eval_dir"])
    assert (eval_dir / "eval_config.json").is_file()
    assert (eval_dir / "eval_results.json").is_file()
    assert (eval_dir / "eval_report.md").is_file()


def test_activation_refused_without_eval(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "missing_eval"


def test_activation_refused_if_eval_below_threshold(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    run_learning_eval(
        project_root,
        adapter_id=train["adapter_id"],
        dry_run=True,
        fake_quality="failing",
    )

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] in {"pass_rate_below_threshold", "critical_outdated_claim_failure"}


def test_activation_allowed_if_eval_passes(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=train["adapter_id"], dry_run=True)

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is True
    assert gate["reason"] == "passed"
