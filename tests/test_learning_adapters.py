import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.adapters import activate_adapter, list_adapters, rollback_adapter
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.eval import run_learning_eval
from morpheus.core.learning.train import plan_training_run
from tests.test_learning_dataset import copy_learning_project


def mark_all_evals_activation_eligible(project_root: Path) -> None:
    for config_path in project_root.glob(
        ".morpheus/training/evals/*/eval_config.json"
    ):
        config = json.loads(config_path.read_text())
        config.update({
            "activation_eligible": True,
            "dry_run": False,
            "evaluation_mode": "heldout_external",
            "provider": {"name": "external-heldout"},
        })
        config_path.write_text(json.dumps(config))
        results_path = config_path.parent / "eval_results.json"
        results = json.loads(results_path.read_text())
        results.update({
            "activation_eligible": True,
            "evaluation_mode": "heldout_external",
        })
        results_path.write_text(json.dumps(results))


def planned_adapter(
    project_root: Path,
    *,
    passing_eval: bool = True,
    activation_eligible: bool = False,
) -> dict:
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, base_only=True, dry_run=True)
    run_learning_eval(
        project_root,
        adapter_id=train["adapter_id"],
        dry_run=True,
        fake_quality="passing" if passing_eval else "failing",
    )
    if activation_eligible:
        mark_all_evals_activation_eligible(project_root)
    return train


def active_adapter_id(project_root: Path) -> str | None:
    path = project_root / ".morpheus" / "training" / "active_adapter.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("adapter_id")


def test_list_adapters_includes_planned_adapter_and_eval_score(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root)

    adapters = list_adapters(project_root)

    assert [adapter["adapter_id"] for adapter in adapters] == [train["adapter_id"]]
    assert adapters[0]["status"] == "planned"
    assert adapters[0]["eval_score"] == 1.0


def test_activate_refuses_passing_diagnostic_adapter_without_writes(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root)

    with pytest.raises(ValueError, match="diagnostic_eval_not_activation_eligible"):
        activate_adapter(project_root, train["adapter_id"])

    adapter_dir = project_root / ".morpheus" / "training" / "adapters" / train["adapter_id"]
    assert active_adapter_id(project_root) is None
    assert not (adapter_dir / "activate_receipt.json").exists()
    assert not (project_root / ".morpheus/training/rollback_log.jsonl").exists()


def test_activate_refuses_failing_adapter_without_force(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(
        project_root,
        passing_eval=False,
        activation_eligible=True,
    )

    with pytest.raises(ValueError, match="Cannot activate adapter"):
        activate_adapter(project_root, train["adapter_id"])

    assert active_adapter_id(project_root) is None


def test_force_activation_requires_confirmation(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(
        project_root,
        passing_eval=False,
        activation_eligible=True,
    )

    with pytest.raises(ValueError, match="--force requires --yes-i-know-this-can-degrade"):
        activate_adapter(project_root, train["adapter_id"], force=True)


def test_confirmed_force_cannot_bypass_failed_eval_gate(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(
        project_root,
        passing_eval=False,
        activation_eligible=True,
    )

    with pytest.raises(ValueError, match="force cannot bypass the eval gate"):
        activate_adapter(
            project_root,
            train["adapter_id"],
            force=True,
            confirm_force=True,
        )

    adapter_dir = project_root / ".morpheus/training/adapters" / train["adapter_id"]
    assert active_adapter_id(project_root) is None
    assert not (adapter_dir / "activate_receipt.json").exists()
    assert not (project_root / ".morpheus/training/rollback_log.jsonl").exists()


def test_cli_confirmed_force_cannot_bypass_failed_eval_gate(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(
        project_root,
        passing_eval=False,
        activation_eligible=True,
    )

    result = CliRunner().invoke(
        app,
        [
            "learn",
            "activate",
            train["adapter_id"],
            "--project",
            str(project_root),
            "--force",
            "--yes-i-know-this-can-degrade",
        ],
    )

    assert result.exit_code == 2
    assert "force cannot bypass the eval gate" in result.output
    assert active_adapter_id(project_root) is None


def test_rollback_restores_previous_active_adapter(tmp_path):
    project_root = copy_learning_project(tmp_path)
    first = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, first["adapter_id"])
    second = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=second["adapter_id"], dry_run=True)
    mark_all_evals_activation_eligible(project_root)
    activate_adapter(project_root, second["adapter_id"])

    result = rollback_adapter(project_root)

    assert result["active_adapter_id"] == first["adapter_id"]
    assert active_adapter_id(project_root) == first["adapter_id"]
    rollback_log = project_root / ".morpheus" / "training" / "rollback_log.jsonl"
    assert "rollback" in rollback_log.read_text()


def test_status_json_includes_active_adapter_dataset_and_eval_score(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, train["adapter_id"])

    result = CliRunner().invoke(app, ["learn", "status", str(project_root), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["active_adapter"]["adapter_id"] == train["adapter_id"]
    assert payload["latest_manifest"]["dataset_id"]
    assert payload["active_adapter"]["eval_score"] == 1.0


def test_cli_list_activate_and_rollback(tmp_path):
    project_root = copy_learning_project(tmp_path)
    first = planned_adapter(project_root, activation_eligible=True)
    second = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=second["adapter_id"], dry_run=True)
    mark_all_evals_activation_eligible(project_root)
    runner = CliRunner()

    listed = runner.invoke(app, ["learn", "list-adapters", str(project_root), "--json"])
    activated_first = runner.invoke(
        app,
        ["learn", "activate", first["adapter_id"], "--project", str(project_root)],
    )
    activated_second = runner.invoke(
        app,
        ["learn", "activate", second["adapter_id"], "--project", str(project_root)],
    )
    rolled_back = runner.invoke(app, ["learn", "rollback", "--project", str(project_root)])

    assert listed.exit_code == 0, listed.output
    assert first["adapter_id"] in listed.output
    assert activated_first.exit_code == 0, activated_first.output
    assert activated_second.exit_code == 0, activated_second.output
    assert rolled_back.exit_code == 0, rolled_back.output
    assert active_adapter_id(project_root) == first["adapter_id"]
