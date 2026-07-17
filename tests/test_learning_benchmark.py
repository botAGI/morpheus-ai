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


def test_benchmark_report_allows_balanced_manifest(tmp_path):
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
            json.dumps({"category": "unsupported_claim_refusal"}),
            json.dumps({"category": "outdated_claim_correction"}),
            json.dumps({"category": "project_recall"}),
        ])
        + "\n"
    )

    result = write_benchmark_report(project_root, dry_run=True)

    assert result["benchmark_allowed"] is True
    assert result["benchmark_blockers"] == []
    assert result["next_command"] == "morpheus learn lab . --backend mlx --max-iters 50"


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
    assert payload["paths"]["benchmark_report_path"].endswith("benchmark_report.json")
