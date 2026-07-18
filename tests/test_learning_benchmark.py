import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.benchmark import write_benchmark_report
from morpheus.core.learning.categories import (
    BENCHMARK_CATEGORY_SCHEMA,
    CANONICAL_BENCHMARK_CATEGORIES,
)
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.eval import run_learning_eval
from morpheus.core.learning.lab import run_autonomous_lab
from morpheus.core.learning.train import plan_training_run
from morpheus.core.semantic.review import ReviewStore
from tests.test_learning_dataset import copy_learning_project
from tests.test_learning_eval import (
    _write_gate_eval,
    mark_eval_activation_eligible,
    register_test_adapter_weights,
)
from tests.test_learning_lab import copy_autonomous_repo


def test_benchmark_report_blocks_unbalanced_fixture_dataset(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["benchmark_allowed"] is False
    assert "trainable_candidate_count < 20" in result["benchmark_blockers"]
    assert Path(result["benchmark_report_path"]).is_file()
    assert Path(result["benchmark_report_md_path"]).is_file()
    assert "Benchmark blocked" in Path(result["benchmark_report_md_path"]).read_text()


def create_balanced_benchmark_project(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)
    lab = run_autonomous_lab(
        project_root,
        backend="fake",
        no_train=True,
        fixture_only=True,
    )
    lab_store = ReviewStore(Path(lab["lab_dir"]) / "workspace")
    ReviewStore(project_root).save_candidates(lab_store.load_candidates())
    build_learning_dataset(project_root)
    return project_root


def test_benchmark_report_allows_balanced_manifest(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["benchmark_allowed"] is True
    assert result["benchmark_blockers"] == []
    assert result["latest_base_eval"]["dataset_id"] == result["dataset_id"]
    assert set(result["benchmark_gate"]["eval_category_counts"]) == (
        CANONICAL_BENCHMARK_CATEGORIES
    )
    assert result["next_command"] == "morpheus learn lab . --backend mlx --max-iters 50"


def test_benchmark_report_creates_matching_base_eval_for_adapter_comparison(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)
    adapter_id = "adapter_balanced"
    _write_adapter_manifest(project_root, adapter_id)
    run_learning_eval(project_root, adapter_id=adapter_id, dry_run=True)

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["latest_base_eval"]["dataset_id"] == result["dataset_id"]
    assert result["latest_adapter_eval"]["dataset_id"] == result["dataset_id"]
    assert result["category_deltas"]
    assert result["activation_ready"] is False
    assert result["activation_gate"]["reason"] == (
        "diagnostic_eval_not_activation_eligible"
    )


def test_benchmark_creates_base_eval_for_the_compared_adapter_model(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)
    compared_adapter = "adapter_a_compared"
    _write_adapter_manifest(
        project_root,
        compared_adapter,
        base_model="Qwen/Compared@revision-1",
    )
    run_learning_eval(
        project_root,
        adapter_id=compared_adapter,
        dry_run=True,
    )
    _write_adapter_manifest(
        project_root,
        "adapter_z_unrelated_newer",
        base_model="Qwen/Unrelated@revision-2",
    )

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["latest_adapter_eval"]["adapter_id"] == compared_adapter
    assert (
        result["latest_base_eval"]["eval_pair_config_sha256"]
        == result["latest_adapter_eval"]["eval_pair_config_sha256"]
    )
    assert result["category_deltas"]


def test_benchmark_report_uses_the_same_paired_eval_as_activation_gate(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)
    adapter_id = "adapter_paired"
    original_manifest_path = next(
        project_root.glob(".morpheus/training/datasets/*/manifest.json")
    )
    original_dataset_id = json.loads(
        original_manifest_path.read_text()
    )["dataset_id"]
    _write_gate_eval(
        project_root,
        eval_id="eval_zzzz_unrelated_adapter",
        dataset_id=original_dataset_id,
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    current_dataset = build_learning_dataset(project_root)
    _write_adapter_manifest(
        project_root,
        adapter_id,
        dataset_id=current_dataset["dataset_id"],
    )
    base_eval = run_learning_eval(project_root, base_only=True, dry_run=True)
    adapter_eval = run_learning_eval(
        project_root,
        adapter_id=adapter_id,
        dry_run=True,
    )
    _mark_eval_activation_eligible(Path(base_eval["eval_dir"]))
    _mark_eval_activation_eligible(Path(adapter_eval["eval_dir"]))

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["latest_base_eval"]["eval_id"] == base_eval["eval_id"]
    assert result["latest_adapter_eval"]["eval_id"] == adapter_eval["eval_id"]
    assert result["activation_gate"]["eval_id"] == adapter_eval["eval_id"]
    assert result["activation_ready"] is True


def test_benchmark_report_keeps_legacy_paired_category_deltas(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)
    adapter_id = "adapter_legacy_diagnostic"
    _write_adapter_manifest(project_root, adapter_id)
    base_eval = run_learning_eval(project_root, base_only=True, dry_run=True)
    adapter_eval = run_learning_eval(
        project_root,
        adapter_id=adapter_id,
        dry_run=True,
    )
    for evaluation in (base_eval, adapter_eval):
        results_path = Path(evaluation["eval_results_path"])
        results = json.loads(results_path.read_text())
        results["metrics"].pop("hallucinated_items")
        results_path.write_text(json.dumps(results))

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["latest_base_eval"]["eval_id"] == base_eval["eval_id"]
    assert result["latest_adapter_eval"]["eval_id"] == adapter_eval["eval_id"]
    assert result["category_deltas"]
    assert result["activation_ready"] is False


def test_benchmark_report_ignores_signed_newer_evals_from_another_dataset(
    tmp_path,
):
    project_root = create_balanced_benchmark_project(tmp_path)
    original_manifest_path = next(
        project_root.glob(".morpheus/training/datasets/*/manifest.json")
    )
    original_dataset_id = json.loads(
        original_manifest_path.read_text()
    )["dataset_id"]
    _write_gate_eval(
        project_root,
        eval_id="eval_zzzz_unrelated_base",
        dataset_id=original_dataset_id,
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    _write_gate_eval(
        project_root,
        eval_id="eval_zzzz_unrelated_adapter",
        dataset_id=original_dataset_id,
        adapter_id="adapter_other",
        base_only=False,
        regressed_category=None,
    )
    current_dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_balanced"
    _write_adapter_manifest(
        project_root,
        adapter_id,
        dataset_id=current_dataset["dataset_id"],
    )
    base_eval = run_learning_eval(project_root, base_only=True, dry_run=True)
    adapter_eval = run_learning_eval(
        project_root,
        adapter_id=adapter_id,
        dry_run=True,
    )
    _mark_eval_activation_eligible(Path(base_eval["eval_dir"]))
    _mark_eval_activation_eligible(Path(adapter_eval["eval_dir"]))

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["latest_base_eval"]["eval_id"] == base_eval["eval_id"]
    assert result["latest_base_eval"]["dataset_id"] == result["dataset_id"]
    assert result["latest_adapter_eval"]["eval_id"] == adapter_eval["eval_id"]
    assert result["latest_adapter_eval"]["dataset_id"] == result["dataset_id"]
    assert result["category_deltas"]
    assert result["activation_ready"] is True


@pytest.mark.parametrize("invalid_role", ["adapter", "base"])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_config",
        "missing_results",
        "corrupt_config",
        "corrupt_results",
        "mismatched_results",
    ],
)
def test_benchmark_does_not_report_older_eval_past_newer_invalid_entry(
    tmp_path,
    invalid_role,
    mutation,
):
    project_root = create_balanced_benchmark_project(tmp_path)
    adapter_id = "adapter_invalid_latest"
    _write_adapter_manifest(project_root, adapter_id)
    run_learning_eval(project_root, base_only=True, dry_run=True)
    run_learning_eval(project_root, adapter_id=adapter_id, dry_run=True)
    newest = run_learning_eval(
        project_root,
        base_only=invalid_role == "base",
        adapter_id=adapter_id if invalid_role == "adapter" else None,
        dry_run=True,
    )
    newest_dir = Path(newest["eval_dir"])
    _invalidate_eval(newest_dir, mutation)

    result = write_benchmark_report(project_root, dry_run=True)

    latest = result[f"latest_{invalid_role}_eval"]
    assert latest["eval_id"] == newest["eval_id"]
    assert latest["valid"] is False
    assert result["activation_ready"] is False
    assert result["activation_gate"]["reason"] != "passed"


def test_benchmark_blocks_newer_invalid_lab_instead_of_falling_back(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)
    lab_dataset_dir = project_root / ".morpheus/lab/lab_zzzz/dataset"
    lab_dataset_dir.mkdir(parents=True)
    lab_manifest = {
        "dataset_id": "newer-lab-dataset",
        "created_at": "2027-01-01T00:00:00+00:00",
        "examples_count": 1,
    }
    (lab_dataset_dir / "manifest.json").write_text(json.dumps(lab_manifest))
    (lab_dataset_dir / "eval.seed.jsonl").write_text(json.dumps({
        "category": "project_recall",
        "question": "Lab question",
        "expected_answer": "Lab answer",
    }) + "\n")
    (lab_dataset_dir.parent / "lab_summary.json").write_text(json.dumps({"status": "ready"}))

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["benchmark_allowed"] is False
    assert result["latest_base_eval"] is None
    assert "dataset provenance invalid" in result["benchmark_blockers"]
    assert result["quality_report"]["dataset"]["effective_dataset"]["source"] == "lab"


def test_benchmark_report_includes_category_level_base_adapter_deltas(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, base_only=True, dry_run=True, fake_quality="failing")
    run_learning_eval(project_root, adapter_id=train["adapter_id"], dry_run=True)

    result = write_benchmark_report(project_root, dry_run=True)

    deltas = result["category_deltas"]
    assert deltas["unsupported_claim_refusal"]["base_pass_rate"] == 0.0
    assert deltas["unsupported_claim_refusal"]["adapter_pass_rate"] == 1.0
    assert deltas["unsupported_claim_refusal"]["pass_rate_delta"] == 1.0
    assert "base_hallucination_rate" in deltas["unsupported_claim_refusal"]
    assert "adapter_hallucination_rate" in deltas["unsupported_claim_refusal"]
    assert "hallucination_rate_delta" in deltas["unsupported_claim_refusal"]
    assert result["benchmark_category_schema"] == BENCHMARK_CATEGORY_SCHEMA
    assert result["category_regression_count"] == len(
        result["category_regressions"]
    )
    assert result["critical_regression_count"] == len(
        result["critical_regressions"]
    )
    assert result["activation_reason"] == result["activation_gate"]["reason"]
    markdown = Path(result["benchmark_report_md_path"]).read_text()
    assert "Category schema: `morpheus-benchmark-categories/1`" in markdown
    assert "Category regression count:" in markdown
    assert "Critical regression count:" in markdown
    assert "Activation reason:" in markdown
    assert "hallucination delta" in markdown


def test_benchmark_markdown_renders_all_and_critical_regression_sections(
    tmp_path,
):
    project_root = create_balanced_benchmark_project(tmp_path)
    manifest_path = next(
        project_root.glob(".morpheus/training/datasets/*/manifest.json")
    )
    dataset_id = json.loads(manifest_path.read_text())["dataset_id"]
    adapter_id = "adapter_critical_markdown"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset_id,
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000002Z",
        dataset_id=dataset_id,
        adapter_id=adapter_id,
        base_only=False,
        regressed_category="safety_rules",
    )

    result = write_benchmark_report(project_root, dry_run=True)
    markdown = Path(result["benchmark_report_md_path"]).read_text()

    assert result["critical_regressions"][0]["category"] == "safety_rules"
    assert "## Category Regressions" in markdown
    assert "## Critical Regressions" in markdown
    assert markdown.count("`safety_rules`") >= 2


def test_cli_learn_benchmark_outputs_json(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = CliRunner().invoke(app, ["learn", "benchmark", str(project_root), "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["benchmark_allowed"] is False
    assert payload["activation_ready"] is False
    assert "critical_regressions" in payload
    assert payload["paths"]["benchmark_report_path"].endswith("benchmark_report.json")


def test_cli_learn_benchmark_prints_activation_readiness(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = CliRunner().invoke(app, ["learn", "benchmark", str(project_root), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "activation_ready=False" in result.output
    assert "critical_regressions=0" in result.output


def _write_eval_result(
    project_root: Path,
    *,
    eval_id: str,
    dataset_id: str,
    adapter_id: str | None,
    base_only: bool,
) -> None:
    eval_dir = project_root / ".morpheus/training/evals" / eval_id
    eval_dir.mkdir(parents=True)
    dataset_binding_sha256 = "b" * 64
    config = {
        "eval_id": eval_id,
        "dataset_id": dataset_id,
        "dataset_binding_sha256": dataset_binding_sha256,
        "adapter_id": adapter_id,
        "base_only": base_only,
        "activation_eligible": False,
        "dry_run": True,
        "evaluation_mode": "diagnostic_fake",
        "provider": {"name": "fake-unrelated"},
    }
    result = {
        **config,
        "metrics": {
            "pass_rate": 1.0,
            "hallucination_rate": 0.0,
            "critical_outdated_claim_failures": 0,
            "total_items": 1,
            "passed_items": 1,
            "hallucinated_items": 0,
            "by_category": {
                "safety_rules": {
                    "total_items": 1,
                    "passed_items": 1,
                    "pass_rate": 1.0,
                    "hallucinated_items": 0,
                    "hallucination_rate": 0.0,
                    "critical_failures": 0,
                }
            },
        },
        "items": [{"category": "safety_rules"}],
    }
    (eval_dir / "eval_config.json").write_text(json.dumps(config))
    (eval_dir / "eval_results.json").write_text(json.dumps(result))


def _write_adapter_manifest(
    project_root: Path,
    adapter_id: str,
    *,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    dataset_id: str | None = None,
) -> None:
    manifest_paths = list(
        project_root.glob(".morpheus/training/datasets/*/manifest.json")
    )
    if dataset_id is not None:
        manifest_paths = [
            path
            for path in manifest_paths
            if json.loads(path.read_text()).get("dataset_id") == dataset_id
        ]
    assert len(manifest_paths) == 1
    dataset_manifest = json.loads(manifest_paths[0].read_text())
    adapter_dir = project_root / ".morpheus/training/adapters" / adapter_id
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_manifest.json").write_text(json.dumps({
        "adapter_id": adapter_id,
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_binding_sha256": dataset_manifest["dataset_binding_sha256"],
        "base_model": base_model,
        "status": "planned",
    }))
    register_test_adapter_weights(project_root, adapter_id)


def _mark_eval_activation_eligible(eval_dir: Path) -> None:
    mark_eval_activation_eligible(eval_dir)


def _invalidate_eval(eval_dir: Path, mutation: str) -> None:
    config_path = eval_dir / "eval_config.json"
    results_path = eval_dir / "eval_results.json"
    if mutation == "missing_config":
        config_path.unlink()
    elif mutation == "missing_results":
        results_path.unlink()
    elif mutation == "corrupt_config":
        config_path.write_text("{not-json")
    elif mutation == "corrupt_results":
        results_path.write_text("{not-json")
    elif mutation == "mismatched_results":
        results = json.loads(results_path.read_text())
        results["dataset_id"] = "mismatched-dataset"
        results_path.write_text(json.dumps(results))
    else:  # pragma: no cover - test helper guard.
        raise AssertionError(f"Unknown eval mutation: {mutation}")
