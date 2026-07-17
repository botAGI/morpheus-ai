import json
from pathlib import Path

import pytest

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
    assert "by_category" in results["metrics"]
    assert results["metrics"]["by_category"]["unsupported_claim_refusal"]["total_items"] >= 1
    assert "## Category Metrics" in report
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
    run_learning_eval(project_root, base_only=True, dry_run=True)
    run_learning_eval(project_root, adapter_id=train["adapter_id"], dry_run=True)

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is True
    assert gate["reason"] == "passed"


def test_activation_refused_without_matching_base_eval(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=train["adapter_id"], dry_run=True)

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "missing_base_eval"
    assert gate["dataset_id"] == dataset["dataset_id"]


@pytest.mark.parametrize(
    "category",
    [
        "outdated_claim_correction",
        "agent_rule_adherence",
        "unsupported_claim_refusal",
    ],
)
def test_activation_refused_on_critical_category_regression(tmp_path, category):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_regression"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000002Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=category,
    )

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "critical_category_regression"
    assert gate["critical_regressions"][0]["category"] == category


def _write_gate_eval(
    project_root: Path,
    *,
    eval_id: str,
    dataset_id: str,
    adapter_id: str | None,
    base_only: bool,
    regressed_category: str | None,
) -> None:
    eval_dir = project_root / ".morpheus/training/evals" / eval_id
    eval_dir.mkdir(parents=True)
    categories = {}
    for category in (
        "outdated_claim_correction",
        "agent_rule_adherence",
        "unsupported_claim_refusal",
    ):
        passed = category != regressed_category
        categories[category] = {
            "total_items": 1,
            "passed_items": int(passed),
            "pass_rate": 1.0 if passed else 0.0,
            "hallucinated_items": 0,
            "hallucination_rate": 0.0,
            "critical_failures": 0,
        }
    config = {
        "eval_id": eval_id,
        "dataset_id": dataset_id,
        "adapter_id": adapter_id,
        "base_only": base_only,
    }
    results = {
        **config,
        "metrics": {
            "pass_rate": 1.0,
            "hallucination_rate": 0.0,
            "critical_outdated_claim_failures": 0,
            "by_category": categories,
        },
    }
    (eval_dir / "eval_config.json").write_text(json.dumps(config))
    (eval_dir / "eval_results.json").write_text(json.dumps(results))
