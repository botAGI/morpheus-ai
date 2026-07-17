import json
from pathlib import Path

import pytest

from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.eval import check_activation_gate, run_learning_eval
from morpheus.core.learning.train import plan_training_run
from tests.test_learning_dataset import copy_learning_project, read_jsonl


def mark_eval_activation_eligible(eval_dir: Path) -> None:
    config_path = eval_dir / "eval_config.json"
    config = json.loads(config_path.read_text())
    config.update({
        "activation_eligible": True,
        "dry_run": False,
        "evaluation_mode": "heldout_external",
        "provider": {"name": "external-heldout"},
    })
    config_path.write_text(json.dumps(config))
    results_path = eval_dir / "eval_results.json"
    results = json.loads(results_path.read_text())
    results.update({
        "activation_eligible": True,
        "evaluation_mode": "heldout_external",
    })
    results_path.write_text(json.dumps(results))


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
    assert results["metrics"]["total_items"] == len(seed_items)
    assert results["metrics"]["passed_items"] <= results["metrics"]["total_items"]
    assert results["metrics"]["hallucinated_items"] <= results["metrics"]["total_items"]
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
    assert config["evaluation_mode"] == "diagnostic_fake"
    assert config["activation_eligible"] is False
    assert results["evaluation_mode"] == "diagnostic_fake"
    assert results["activation_eligible"] is False
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
    evaluation = run_learning_eval(
        project_root,
        adapter_id=train["adapter_id"],
        dry_run=True,
        fake_quality="failing",
    )
    mark_eval_activation_eligible(Path(evaluation["eval_dir"]))

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] in {"pass_rate_below_threshold", "critical_outdated_claim_failure"}


def test_activation_refuses_diagnostic_eval_even_if_metrics_pass(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, base_only=True, dry_run=True)
    run_learning_eval(
        project_root,
        adapter_id=train["adapter_id"],
        dry_run=True,
    )

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "diagnostic_eval_not_activation_eligible"
    assert gate["evaluation_mode"] == "diagnostic_fake"
    assert gate["provider"] == "fake-adapter"


def test_activation_refuses_when_only_eval_config_claims_eligibility(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    base = run_learning_eval(project_root, base_only=True, dry_run=True)
    adapter = run_learning_eval(
        project_root,
        adapter_id=train["adapter_id"],
        dry_run=True,
    )
    mark_eval_activation_eligible(Path(base["eval_dir"]))
    config_path = Path(adapter["eval_dir"]) / "eval_config.json"
    config = json.loads(config_path.read_text())
    config.update({
        "activation_eligible": True,
        "dry_run": False,
        "evaluation_mode": "heldout_external",
        "provider": {"name": "external-heldout"},
    })
    config_path.write_text(json.dumps(config))

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "diagnostic_eval_not_activation_eligible"


def test_activation_refuses_diagnostic_base_for_eligible_adapter(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, base_only=True, dry_run=True)
    adapter = run_learning_eval(
        project_root,
        adapter_id=train["adapter_id"],
        dry_run=True,
    )
    mark_eval_activation_eligible(Path(adapter["eval_dir"]))

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "diagnostic_base_eval_not_activation_eligible"


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("eval_id", "eval_other"),
        ("adapter_id", "adapter_other"),
        ("dataset_id", "dataset_other"),
        ("base_only", True),
    ],
)
def test_activation_refuses_mismatched_adapter_eval_artifacts(
    tmp_path,
    field,
    mismatched_value,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_identity"
    base_eval_id = "eval_20260522T000000000001Z"
    adapter_eval_id = "eval_20260522T000000000002Z"
    _write_gate_eval(
        project_root,
        eval_id=base_eval_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    _write_gate_eval(
        project_root,
        eval_id=adapter_eval_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    results_path = (
        project_root
        / ".morpheus/training/evals"
        / adapter_eval_id
        / "eval_results.json"
    )
    results = json.loads(results_path.read_text())
    results[field] = mismatched_value
    results_path.write_text(json.dumps(results))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "eval_artifact_identity_mismatch"


def test_activation_refuses_eval_id_that_does_not_match_directory(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_directory_identity"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    adapter_eval_id = "eval_20260522T000000000002Z"
    _write_gate_eval(
        project_root,
        eval_id=adapter_eval_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    eval_dir = project_root / ".morpheus/training/evals" / adapter_eval_id
    for filename in ("eval_config.json", "eval_results.json"):
        path = eval_dir / filename
        payload = json.loads(path.read_text())
        payload["eval_id"] = "eval_payload_identity"
        path.write_text(json.dumps(payload))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "eval_artifact_identity_mismatch"


def test_activation_refuses_mismatched_base_eval_artifacts(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_base_identity"
    base_eval_id = "eval_20260522T000000000001Z"
    _write_gate_eval(
        project_root,
        eval_id=base_eval_id,
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
        regressed_category=None,
    )
    results_path = (
        project_root
        / ".morpheus/training/evals"
        / base_eval_id
        / "eval_results.json"
    )
    results = json.loads(results_path.read_text())
    results["eval_id"] = "eval_other"
    results_path.write_text(json.dumps(results))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "base_eval_artifact_identity_mismatch"


@pytest.mark.parametrize("malformed_field", ["provider_name", "evaluation_mode"])
def test_activation_refuses_non_string_eligibility_metadata(
    tmp_path,
    malformed_field,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_malformed_eligibility"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    adapter_eval_id = "eval_20260522T000000000002Z"
    _write_gate_eval(
        project_root,
        eval_id=adapter_eval_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    eval_dir = project_root / ".morpheus/training/evals" / adapter_eval_id
    config_path = eval_dir / "eval_config.json"
    results_path = eval_dir / "eval_results.json"
    config = json.loads(config_path.read_text())
    results = json.loads(results_path.read_text())
    if malformed_field == "provider_name":
        config["provider"]["name"] = ["external-heldout"]
    else:
        config["evaluation_mode"] = True
        results["evaluation_mode"] = "True"
    config_path.write_text(json.dumps(config))
    results_path.write_text(json.dumps(results))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "diagnostic_eval_not_activation_eligible"


def test_activation_refused_without_matching_base_eval(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    evaluation = run_learning_eval(
        project_root,
        adapter_id=train["adapter_id"],
        dry_run=True,
    )
    mark_eval_activation_eligible(Path(evaluation["eval_dir"]))

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "missing_base_eval"
    assert gate["dataset_id"] == dataset["dataset_id"]


@pytest.mark.parametrize("unpaired_role", ["adapter", "base"])
def test_activation_comparison_ignores_later_unpaired_results(
    tmp_path,
    unpaired_role,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_exact_comparison"
    base_eval_id = "eval_20260522T000000000001Z"
    adapter_eval_id = "eval_20260522T000000000002Z"
    regressed_category = "unsupported_claim_refusal"
    _write_gate_eval(
        project_root,
        eval_id=base_eval_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    _write_gate_eval(
        project_root,
        eval_id=adapter_eval_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=regressed_category,
    )
    source_eval_id = adapter_eval_id if unpaired_role == "adapter" else base_eval_id
    source_path = (
        project_root
        / ".morpheus/training/evals"
        / source_eval_id
        / "eval_results.json"
    )
    unpaired = json.loads(source_path.read_text())
    unpaired_eval_id = "eval_20260522T000000000003Z"
    unpaired["eval_id"] = unpaired_eval_id
    category_metrics = unpaired["metrics"]["by_category"][regressed_category]
    if unpaired_role == "adapter":
        category_metrics.update({"passed_items": 1, "pass_rate": 1.0})
    else:
        category_metrics.update({"passed_items": 0, "pass_rate": 0.0})
    unpaired_dir = project_root / ".morpheus/training/evals" / unpaired_eval_id
    unpaired_dir.mkdir()
    (unpaired_dir / "eval_results.json").write_text(json.dumps(unpaired))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "critical_category_regression"
    assert gate["critical_regressions"][0]["category"] == regressed_category


@pytest.mark.parametrize(
    "invalid_rate",
    [float("nan"), float("inf"), -0.1, 1.1],
    ids=["nan", "infinity", "below-zero", "above-one"],
)
@pytest.mark.parametrize("invalid_role", ["adapter", "base"])
def test_activation_refuses_invalid_eval_rates(tmp_path, invalid_rate, invalid_role):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_invalid_rate"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    adapter_eval_id = "eval_20260522T000000000002Z"
    _write_gate_eval(
        project_root,
        eval_id=adapter_eval_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    invalid_eval_id = (
        adapter_eval_id
        if invalid_role == "adapter"
        else "eval_20260522T000000000001Z"
    )
    results_path = (
        project_root
        / ".morpheus/training/evals"
        / invalid_eval_id
        / "eval_results.json"
    )
    results = json.loads(results_path.read_text())
    results["metrics"]["pass_rate"] = invalid_rate
    results_path.write_text(json.dumps(results))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    expected_reason = (
        "invalid_eval_metrics"
        if invalid_role == "adapter"
        else "invalid_base_eval_metrics"
    )
    assert gate["reason"] == expected_reason


@pytest.mark.parametrize("invalid_summary", ["zero_items", "inconsistent_rate"])
def test_activation_refuses_empty_or_inconsistent_eval_summary(
    tmp_path,
    invalid_summary,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_invalid_summary"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    adapter_eval_id = "eval_20260522T000000000002Z"
    _write_gate_eval(
        project_root,
        eval_id=adapter_eval_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    results_path = (
        project_root
        / ".morpheus/training/evals"
        / adapter_eval_id
        / "eval_results.json"
    )
    results = json.loads(results_path.read_text())
    metrics = results["metrics"]
    if invalid_summary == "zero_items":
        metrics.update({
            "total_items": 0,
            "passed_items": 0,
            "hallucinated_items": 0,
        })
    else:
        metrics.update({
            "total_items": 15,
            "passed_items": 0,
            "hallucinated_items": 0,
            "pass_rate": 1.0,
        })
    results_path.write_text(json.dumps(results))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "invalid_eval_metrics"


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
    items_per_category = 5
    for category in (
        "outdated_claim_correction",
        "agent_rule_adherence",
        "unsupported_claim_refusal",
    ):
        passed_items = (
            items_per_category
            if category != regressed_category
            else items_per_category - 1
        )
        categories[category] = {
            "total_items": items_per_category,
            "passed_items": passed_items,
            "pass_rate": round(passed_items / items_per_category, 4),
            "hallucinated_items": 0,
            "hallucination_rate": 0.0,
            "critical_failures": 0,
        }
    total_items = sum(item["total_items"] for item in categories.values())
    passed_items = sum(item["passed_items"] for item in categories.values())
    config = {
        "eval_id": eval_id,
        "dataset_id": dataset_id,
        "adapter_id": adapter_id,
        "base_only": base_only,
        "activation_eligible": True,
        "dry_run": False,
        "evaluation_mode": "heldout_external",
        "provider": {"name": "external-heldout"},
    }
    results = {
        **config,
        "metrics": {
            "pass_rate": round(passed_items / total_items, 4),
            "hallucination_rate": 0.0,
            "critical_outdated_claim_failures": 0,
            "total_items": total_items,
            "passed_items": passed_items,
            "hallucinated_items": 0,
            "by_category": categories,
        },
    }
    (eval_dir / "eval_config.json").write_text(json.dumps(config))
    (eval_dir / "eval_results.json").write_text(json.dumps(results))
