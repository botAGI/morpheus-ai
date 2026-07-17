import json
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.quality import build_quality_report, write_quality_report
from tests.test_learning_dataset import copy_learning_project


def test_quality_report_counts_review_routes_dataset_and_blockers(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    report = build_quality_report(project_root)

    assert report["review"]["candidates_total"] == 11
    assert report["review"]["by_trainability"]["trainable"] == 4
    assert report["review"]["by_trainability"]["negative_example"] == 1
    assert report["review"]["by_trainability"]["eval_only"] == 1
    assert report["review"]["by_trainability"]["excluded"] >= 3
    assert report["review"]["top_blockers"]["status_rejected"] == 1
    assert report["review"]["top_blockers"]["label_inferred"] == 1
    assert report["dataset"]["latest_manifest"]["examples_count"] > 0
    assert report["dataset"]["latest_manifest"]["class_counts"]["product"] >= 1
    assert report["dataset"]["latest_manifest"]["route_counts"]["adapter_training"] >= 1
    assert report["train_allowed"] is False
    assert "accepted candidates < 20" in report["train_blockers"]
    assert report["benchmark_allowed"] is False
    assert "trainable_candidate_count < 20" in report["benchmark_blockers"]
    assert "examples < 100" in report["benchmark_blockers"]
    assert "eval_items < 30" in report["benchmark_blockers"]
    assert "class command < 2" in report["benchmark_blockers"]
    assert report["benchmark_gate"]["eval_category_counts"]["unsupported_claim_refusal"] >= 1
    assert report["benchmark_gate"]["requirements"]["class_counts"]["product"]["count"] >= 1


def test_write_quality_report_creates_json_and_markdown(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = write_quality_report(project_root)

    json_path = Path(result["json_path"])
    md_path = Path(result["markdown_path"])
    payload = json.loads(json_path.read_text())

    assert json_path.is_file()
    assert md_path.is_file()
    assert payload["review"]["candidates_total"] == 11
    assert "Dataset Quality" in md_path.read_text()
    assert "Benchmark Gate" in md_path.read_text()


def test_cli_learn_quality_outputs_json_and_writes_report(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = CliRunner().invoke(app, ["learn", "quality", str(project_root), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["review"]["candidates_total"] == 11
    assert payload["benchmark_allowed"] is False
    assert payload["paths"]["json_path"].endswith("quality_report.json")
    assert Path(payload["paths"]["json_path"]).is_file()
