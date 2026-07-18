import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import morpheus.core.learning.quality as quality_module
from morpheus.cli import app
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.categories import CANONICAL_BENCHMARK_CATEGORIES
from morpheus.core.learning.quality import build_quality_report, write_quality_report
from morpheus.core.learning.readiness import benchmark_readiness_gate
from tests.test_learning_dataset import copy_learning_project


def benchmark_ready_manifest() -> dict:
    return {
        "trainable_candidate_count": 20,
        "examples_count": 100,
        "eval_items_count": 30,
        "source_paths": ["README.md", "SPEC.md", "docs/ROADMAP.md"],
        "class_counts": {
            "product": 1,
            "command": 2,
            "architecture": 1,
            "security": 1,
            "convention": 1,
        },
        "route_counts": {"adapter_training": 20},
    }


def benchmark_ready_validation() -> dict:
    return {
        "valid": True,
        "blockers": [],
        "eval_coverage": {
            "total_items": len(CANONICAL_BENCHMARK_CATEGORIES) + 100,
            "by_category": {
                **{
                    category: 1 for category in CANONICAL_BENCHMARK_CATEGORIES
                },
                "project_recall": 100,
            },
        },
    }


@pytest.mark.parametrize("missing_category", sorted(CANONICAL_BENCHMARK_CATEGORIES))
def test_benchmark_readiness_requires_every_canonical_category(missing_category):
    manifest = benchmark_ready_manifest()
    validation = benchmark_ready_validation()
    validation["eval_coverage"]["by_category"].pop(missing_category)

    gate = benchmark_readiness_gate(manifest, validation)

    assert gate["allowed"] is False
    assert f"eval_category {missing_category} < 1" in gate["blockers"]
    assert gate["blockers"] == sorted(gate["blockers"])


@pytest.mark.parametrize("missing_class", ["security", "convention"])
def test_benchmark_readiness_requires_security_and_convention_independently(
    missing_class,
):
    manifest = benchmark_ready_manifest()
    manifest["class_counts"][missing_class] = 0

    gate = benchmark_readiness_gate(manifest, benchmark_ready_validation())

    assert gate["allowed"] is False
    assert f"class {missing_class} < 1" in gate["blockers"]


@pytest.mark.parametrize(
    "validation",
    [
        {"valid": False, "blockers": []},
        {},
    ],
    ids=["false-without-blockers", "missing-validation-status"],
)
def test_benchmark_readiness_fails_closed_without_validation_details(validation):
    validation = {
        **validation,
        "eval_coverage": benchmark_ready_validation()["eval_coverage"],
    }

    gate = benchmark_readiness_gate(benchmark_ready_manifest(), validation)

    assert gate["allowed"] is False
    assert gate["blockers"] == ["dataset provenance invalid"]


def test_quality_report_counts_review_routes_dataset_and_blockers(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    report = build_quality_report(project_root)

    assert report["review"]["candidates_total"] == 11
    assert report["review"]["by_trainability"]["trainable"] == 3
    assert report["review"]["by_trainability"]["negative_example"] == 1
    assert report["review"]["by_trainability"]["retrievable"] == 2
    assert report["review"]["by_trainability"]["excluded"] >= 3
    assert report["review"]["top_blockers"]["status_rejected"] == 1
    assert report["review"]["top_blockers"]["label_inferred"] == 1
    assert report["dataset"]["latest_manifest"]["examples_count"] > 0
    assert report["dataset"]["latest_manifest"]["class_counts"]["product"] >= 1
    assert report["dataset"]["latest_manifest"]["route_counts"]["adapter_training"] >= 1
    assert report["dataset"]["freshness"]["fresh"] is True
    assert report["dataset"]["freshness"]["changed_paths"] == []
    assert report["dataset"]["freshness"]["missing_paths"] == []
    assert report["dataset"]["freshness"]["missing_hash_paths"] == []
    assert report["dataset"]["freshness"]["invalid_paths"] == []
    assert report["train_allowed"] is False
    assert "accepted candidates < 20" in report["train_blockers"]
    assert report["benchmark_allowed"] is False
    assert "trainable_candidate_count < 20" in report["benchmark_blockers"]
    assert "examples < 100" in report["benchmark_blockers"]
    assert "eval_items < 30" in report["benchmark_blockers"]
    assert "class command < 2" in report["benchmark_blockers"]
    assert report["benchmark_gate"]["eval_category_counts"]["unsupported_claim_refusal"] >= 1
    assert report["benchmark_gate"]["requirements"]["class_counts"]["product"]["count"] >= 1
    routing = report["routing"]
    assert routing["policy_version"] == "morpheus-memory-routing/1"
    assert len(routing["decisions"]) == 11
    assert routing["by_route"] == report["review"]["by_route"]
    assert {item["id"] for item in routing["prompt_context"]} == {"c_task"}


def test_quality_routing_audit_downgrades_stale_and_invalid_source_evidence(tmp_path):
    project_root = copy_learning_project(tmp_path)
    candidates_path = project_root / ".morpheus/review/semantic_candidates.jsonl"
    candidates = [
        json.loads(line)
        for line in candidates_path.read_text().splitlines()
        if line.strip()
    ]
    by_id = {item["id"]: item for item in candidates}
    by_id["c_decision"]["source_sha256"] = "0" * 64
    by_id["c_rule"]["line_start"] = 999
    by_id["c_rule"]["line_end"] = 999
    candidates_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in candidates) + "\n"
    )

    report = build_quality_report(project_root)

    decisions = {item["id"]: item for item in report["routing"]["decisions"]}
    assert decisions["c_current"]["memory_route"] == "adapter_training"
    assert decisions["c_current"]["trainability_status"] == "trainable"
    assert decisions["c_decision"]["memory_route"] == "human_review"
    assert decisions["c_decision"]["trainability_reason"] == "source_sha256_mismatch"
    assert decisions["c_rule"]["memory_route"] == "human_review"
    assert decisions["c_rule"]["trainability_reason"] == "invalid_source_span"
    assert decisions["c_ignored"]["memory_route"] == "excluded"
    assert decisions["c_ignored"]["trainability_reason"] == "ignored_path"
    assert set(decisions["c_current"]) == {
        "id",
        "claim",
        "status",
        "label",
        "kind",
        "semantic_class",
        "trainability_status",
        "trainability_reason",
        "memory_route",
        "source_path",
        "line_start",
        "line_end",
    }


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
    assert payload["dataset"]["freshness"]["fresh"] is True
    assert payload["routing"]["policy_version"] == "morpheus-memory-routing/1"
    assert len(payload["routing"]["decisions"]) == 11
    assert payload["benchmark_allowed"] is False
    assert payload["paths"]["json_path"].endswith("quality_report.json")
    assert Path(payload["paths"]["json_path"]).is_file()


def test_cli_learn_quality_prints_routing_audit_summary(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)

    result = CliRunner().invoke(app, ["learn", "quality", str(project_root)])

    assert result.exit_code == 0, result.output
    assert "routing_policy=morpheus-memory-routing/1" in result.output
    assert "audited_decisions=11" in result.output


def test_quality_report_blocks_dataset_after_source_changes(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    changed_path = "README.md"
    (project_root / changed_path).write_text("Morpheus changed after dataset compilation.\n")
    monkeypatch.setattr(quality_module, "TRAIN_MIN_ACCEPTED", 0)
    monkeypatch.setattr(quality_module, "TRAIN_MIN_EXAMPLES", 0)

    report = build_quality_report(project_root)

    freshness = report["dataset"]["freshness"]
    assert freshness["fresh"] is False
    assert freshness["changed_paths"] == [changed_path]
    assert report["train_allowed"] is False
    assert report["benchmark_allowed"] is False
    assert "dataset sources changed" in report["train_blockers"]
    assert "dataset sources changed" in report["benchmark_blockers"]
    assert report["next_actions"] == [
        "morpheus learn dataset . --from accepted --format instruction"
    ]


@pytest.mark.parametrize(
    ("case", "expected_bucket", "expected_path"),
    [
        ("missing_hash", "missing_hash_paths", "README.md"),
        ("missing_file", "missing_paths", "README.md"),
        ("parent_traversal", "invalid_paths", "../outside.md"),
    ],
)
def test_quality_report_blocks_invalid_dataset_source_state(
    tmp_path,
    case,
    expected_bucket,
    expected_path,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    manifest_path = Path(dataset["dataset_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    if case == "missing_hash":
        manifest["source_hashes"].pop(expected_path)
        manifest_path.write_text(json.dumps(manifest))
    elif case == "missing_file":
        (project_root / expected_path).unlink()
    else:
        manifest["source_paths"] = [expected_path]
        manifest["source_hashes"][expected_path] = "a" * 64
        manifest_path.write_text(json.dumps(manifest))

    report = build_quality_report(project_root)

    freshness = report["dataset"]["freshness"]
    assert freshness["fresh"] is False
    assert freshness[expected_bucket] == [expected_path]
    assert report["train_allowed"] is False
    assert report["benchmark_allowed"] is False


def test_quality_markdown_names_changed_dataset_sources(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    (project_root / "README.md").write_text("Changed after dataset compilation.\n")

    result = write_quality_report(project_root)

    markdown = Path(result["markdown_path"]).read_text()
    assert "Policy: `morpheus-memory-routing/1`" in markdown
    assert "Audited decisions: 11" in markdown
    assert "## Dataset Freshness" in markdown
    assert "Fresh: False" in markdown
    assert "Changed: `README.md`" in markdown


def test_quality_report_blocks_manifest_without_source_path_list(tmp_path):
    project_root = tmp_path / "malformed-manifest"
    dataset_dir = project_root / ".morpheus/training/datasets/20260717T000000Z"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text("{}\n")

    report = build_quality_report(project_root)

    assert report["dataset"]["freshness"]["fresh"] is False
    assert report["dataset"]["freshness"]["invalid_paths"] == ["source_paths"]
    assert "dataset sources changed" in report["train_blockers"]
    assert "dataset sources changed" in report["benchmark_blockers"]
