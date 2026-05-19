import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.lab import run_autonomous_lab


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "autonomous_learning_repo"


def copy_autonomous_repo(tmp_path: Path) -> Path:
    project_root = tmp_path / "autonomous_learning_repo"
    shutil.copytree(FIXTURE_ROOT, project_root)
    return project_root


def latest_lab_dir(project_root: Path) -> Path:
    latest = project_root / ".morpheus" / "lab" / "LATEST_REPORT.md"
    assert latest.is_file()
    lab_dirs = [
        path
        for path in (project_root / ".morpheus" / "lab").iterdir()
        if path.is_dir()
    ]
    assert lab_dirs
    return max(lab_dirs, key=lambda path: path.stat().st_mtime)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_learn_lab_fixture_no_train_builds_strict_dataset_and_report(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--fixture-only", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["verdict"] == "ML_CORE_PARTIAL"
    assert payload["training_ran"] is False
    assert payload["strict_accepted_candidates"] >= 20
    assert payload["examples_count"] >= 100
    assert payload["eval_items_count"] >= 30
    assert payload["production_ready"] is False
    assert "source_mode_fixture_not_real_project_data" in payload["production_blockers"]
    assert "training_not_run" in payload["production_blockers"]

    lab_dir = Path(payload["lab_dir"])
    assert (lab_dir / "REPORT.md").is_file()
    assert (lab_dir / "accepted_candidates.jsonl").is_file()
    assert (lab_dir / "dataset" / "train.jsonl").is_file()
    assert (lab_dir / "dataset" / "valid.jsonl").is_file()
    assert (lab_dir / "dataset" / "test.jsonl").is_file()
    assert (lab_dir / "dataset" / "manifest.json").is_file()
    assert (lab_dir / "dataset" / "eval.seed.jsonl").is_file()
    assert (lab_dir / "training" / "train_command.sh").is_file()
    assert (lab_dir / "training" / "adapter_manifest.json").is_file()
    assert not (project_root / ".morpheus" / "training" / "active_adapter.json").exists()


def test_learn_lab_dataset_excludes_raw_markdown_secrets_and_pending_state(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--fixture-only", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    lab_dir = Path(payload["lab_dir"])
    dataset_text = (lab_dir / "dataset" / "dataset.instruction.jsonl").read_text()
    accepted = read_jsonl(lab_dir / "accepted_candidates.jsonl")

    assert "MORPHEUS_API_KEY" not in dataset_text
    assert "sk-test-" not in dataset_text
    assert "Ignore previous instructions" not in dataset_text
    assert all(item["status"] == "accepted" for item in accepted)
    assert all(item["label"] == "source_backed" for item in accepted)
    assert "raw markdown" not in dataset_text.lower() or "never train on raw markdown" in dataset_text.lower()


def test_learn_lab_default_uses_fixture_when_dogfood_is_blocked(tmp_path):
    project_root = tmp_path / "empty_repo"
    project_root.mkdir()
    (project_root / "README.md").write_text("# Empty\n")

    result = CliRunner().invoke(app, ["learn", "lab", str(project_root), "--no-train"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["source"] == "fixture"
    assert payload["dogfood_blocked_reason"]
    assert payload["dogfood"]["strict_accepted_candidates"] == 0
    assert payload["dogfood"]["train_allowed"] is False
    assert payload["production_ready"] is False
    assert "source_mode_fixture_not_real_project_data" in payload["production_blockers"]
    assert payload["verdict"] == "ML_CORE_PARTIAL"
    lab_dir = Path(payload["lab_dir"])
    assert (lab_dir / "dogfood_inventory.json").is_file()
    report = (lab_dir / "REPORT.md").read_text()
    assert "Fixture benchmark is not production data." in report


def test_learn_lab_dogfood_no_train_reports_real_data_metrics(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--dogfood", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["source"] == "dogfood"
    assert payload["dogfood"]["strict_accepted_candidates"] >= 20
    assert payload["dogfood"]["train_allowed"] is True
    assert payload["train_allowed"] is True
    assert payload["production_ready"] is False
    assert payload["production_blockers"] == ["training_not_run"]


def test_learn_lab_trained_fake_adapter_writes_base_vs_adapter_eval(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)

    def fake_training(lab_dir, *, backend, model, max_iters, no_train, train_allowed):
        return {
            "training_ran": True,
            "adapter_path": str(lab_dir / "training" / "adapter"),
            "status": "trained_smoke",
            "reason": None,
            "returncode": 0,
            "backend": "fake",
            "model": model,
        }

    monkeypatch.setattr("morpheus.core.learning.lab._run_or_plan_training", fake_training)

    result = run_autonomous_lab(project_root, fixture_only=True)

    eval_dir = Path(result["eval"]["eval_dir"])
    base = json.loads((eval_dir / "base_results.json").read_text())
    adapter = json.loads((eval_dir / "adapter_results.json").read_text())
    report = (eval_dir / "eval_report.md").read_text()

    assert result["verdict"] == "ML_CORE_PASS"
    assert result["production_ready"] is False
    assert "source_mode_fixture_not_real_project_data" in result["production_blockers"]
    assert result["eval"]["comparison"]["adapter_delta"] >= 0
    assert base["items"]
    assert adapter["items"]
    assert adapter["pass_rate"] >= base["pass_rate"]
    assert "Base vs Adapter" in report
    assert "pending_manual_eval" not in report


def test_learn_lab_marks_trained_adapter_regression_as_fail(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)

    def fake_training(lab_dir, *, backend, model, max_iters, no_train, train_allowed):
        return {
            "training_ran": True,
            "adapter_path": str(lab_dir / "training" / "adapter"),
            "status": "trained_smoke",
            "reason": None,
            "returncode": 0,
            "backend": "fake",
            "model": model,
        }

    def degrading_eval(lab_dir, *, eval_items_count, training_result, model):
        eval_dir = lab_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        base = {"mode": "base", "items": [], "pass_rate": 0.8, "critical_failures": 0}
        adapter = {"mode": "adapter", "items": [], "pass_rate": 0.2, "critical_failures": 2}
        (eval_dir / "base_results.json").write_text(json.dumps(base))
        (eval_dir / "adapter_results.json").write_text(json.dumps(adapter))
        (eval_dir / "eval_report.md").write_text("# Base vs Adapter\n")
        return {
            "eval_dir": str(eval_dir),
            "base": base,
            "adapter": adapter,
            "comparison": {
                "adapter_delta": -0.6,
                "regression_count": 2,
                "critical_regression": True,
            },
        }

    monkeypatch.setattr("morpheus.core.learning.lab._run_or_plan_training", fake_training)
    monkeypatch.setattr("morpheus.core.learning.lab._write_lab_eval", degrading_eval)

    result = run_autonomous_lab(project_root, fixture_only=True)

    assert result["verdict"] == "ML_CORE_FAIL"
    assert result["eval"]["comparison"]["critical_regression"] is True


def test_readme_learning_lab_marketing_is_precise():
    readme = Path("README.md").read_text()

    assert "First verify. Then learn." in readme
    assert "morpheus check" in readme
    assert "morpheus learn lab" in readme
    assert "adapter is the source of truth" not in readme
    assert "train on raw markdown" not in readme
