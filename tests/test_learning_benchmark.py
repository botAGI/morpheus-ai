import json
import hashlib
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.benchmark import write_benchmark_report
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.eval import run_learning_eval
from morpheus.core.learning.train import plan_training_run
from tests.test_learning_dataset import copy_learning_project


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
    project_root = tmp_path / "balanced"
    dataset_dir = project_root / ".morpheus/training/datasets/20260522T000000Z"
    dataset_dir.mkdir(parents=True)
    source_paths = ["README.md", "SPEC.md", "AGENTS.md"]
    source_hashes = {}
    for source_path in source_paths:
        source_text = f"Source-backed benchmark fixture: {source_path}\n"
        (project_root / source_path).write_text(source_text)
        source_hashes[source_path] = hashlib.sha256(source_text.encode()).hexdigest()
    manifest = {
        "dataset_id": "20260522T000000Z",
        "created_at": "2026-05-22T00:00:00+00:00",
        "examples_count": 120,
        "eval_items_count": 35,
        "skipped_count": 0,
        "trainable_candidate_count": 22,
        "dataset_sha256": "a" * 64,
        "class_counts": {
            "architecture": 2,
            "command": 4,
            "convention": 2,
            "product": 3,
        },
        "route_counts": {"adapter_training": 22},
        "source_paths": source_paths,
        "source_hashes": source_hashes,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    (dataset_dir / "eval.seed.jsonl").write_text(
        "\n".join([
            json.dumps({
                "category": "unsupported_claim_refusal",
                "question": "Can an unsupported claim be confirmed?",
                "expected_answer": "No",
            }),
            json.dumps({
                "category": "outdated_claim_correction",
                "question": "Is this stale claim current?",
                "expected_answer": "No. This claim is outdated.",
            }),
            json.dumps({
                "category": "agent_rule_adherence",
                "question": "What rule applies?",
                "expected_answer": "Follow AGENTS.md.",
            }),
        ])
        + "\n"
    )
    return project_root


def test_benchmark_report_allows_balanced_manifest(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["benchmark_allowed"] is True
    assert result["benchmark_blockers"] == []
    assert result["latest_base_eval"]["dataset_id"] == "20260522T000000Z"
    assert result["next_command"] == "morpheus learn lab . --backend mlx --max-iters 50"


def test_benchmark_report_creates_matching_base_eval_for_adapter_comparison(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)
    adapter_id = "adapter_balanced"
    (project_root / ".morpheus/training/adapters" / adapter_id).mkdir(parents=True)
    run_learning_eval(project_root, adapter_id=adapter_id, dry_run=True)

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["latest_base_eval"]["dataset_id"] == result["dataset_id"]
    assert result["latest_adapter_eval"]["dataset_id"] == result["dataset_id"]
    assert result["category_deltas"]
    assert result["activation_ready"] is False
    assert result["activation_gate"]["reason"] == (
        "diagnostic_eval_not_activation_eligible"
    )


def test_benchmark_report_uses_the_same_paired_eval_as_activation_gate(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)
    adapter_id = "adapter_paired"
    (project_root / ".morpheus/training/adapters" / adapter_id).mkdir(parents=True)
    base_eval = run_learning_eval(project_root, base_only=True, dry_run=True)
    adapter_eval = run_learning_eval(
        project_root,
        adapter_id=adapter_id,
        dry_run=True,
    )
    _mark_eval_activation_eligible(Path(base_eval["eval_dir"]))
    _mark_eval_activation_eligible(Path(adapter_eval["eval_dir"]))
    _write_eval_result(
        project_root,
        eval_id="eval_zzzz_unpaired_adapter",
        dataset_id="20260522T000000Z",
        adapter_id=adapter_id,
        base_only=False,
    )

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["latest_adapter_eval"]["eval_id"] == adapter_eval["eval_id"]
    assert result["activation_gate"]["eval_id"] == adapter_eval["eval_id"]
    assert result["activation_ready"] is True


def test_benchmark_report_keeps_legacy_paired_category_deltas(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)
    adapter_id = "adapter_legacy_diagnostic"
    (project_root / ".morpheus/training/adapters" / adapter_id).mkdir(parents=True)
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


def test_benchmark_report_ignores_newer_evals_from_another_dataset(tmp_path):
    project_root = create_balanced_benchmark_project(tmp_path)
    adapter_id = "adapter_balanced"
    (project_root / ".morpheus/training/adapters" / adapter_id).mkdir(parents=True)
    run_learning_eval(project_root, base_only=True, dry_run=True)
    run_learning_eval(project_root, adapter_id=adapter_id, dry_run=True)
    _write_eval_result(
        project_root,
        eval_id="eval_zzzz_unrelated_base",
        dataset_id="other-dataset",
        adapter_id=None,
        base_only=True,
    )
    _write_eval_result(
        project_root,
        eval_id="eval_zzzz_unrelated_adapter",
        dataset_id="other-dataset",
        adapter_id="adapter_other",
        base_only=False,
    )

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["latest_base_eval"]["dataset_id"] == "20260522T000000Z"
    assert result["latest_adapter_eval"]["dataset_id"] == "20260522T000000Z"


def test_benchmark_base_eval_uses_report_dataset_instead_of_newer_lab_dataset(tmp_path):
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

    assert result["latest_base_eval"]["dataset_id"] == "20260522T000000Z"


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
    assert "Category Deltas" in Path(result["benchmark_report_md_path"]).read_text()


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
    result = {
        "eval_id": eval_id,
        "dataset_id": dataset_id,
        "adapter_id": adapter_id,
        "base_only": base_only,
        "metrics": {
            "pass_rate": 1.0,
            "hallucination_rate": 0.0,
            "critical_outdated_claim_failures": 0,
            "by_category": {
                "agent_rule_adherence": {
                    "total_items": 1,
                    "passed_items": 1,
                    "pass_rate": 1.0,
                    "critical_failures": 0,
                }
            },
        },
    }
    (eval_dir / "eval_results.json").write_text(json.dumps(result))


def _mark_eval_activation_eligible(eval_dir: Path) -> None:
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
