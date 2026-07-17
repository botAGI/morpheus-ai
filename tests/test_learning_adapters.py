import json
from hashlib import sha256
from pathlib import Path

import pytest
from typer.testing import CliRunner

import morpheus.core.learning.adapters as adapters_module
from morpheus.cli import app
from morpheus.core.learning.adapters import (
    activate_adapter,
    active_adapter_status,
    list_adapters,
    rollback_adapter,
)
from morpheus.core.learning.adapter_artifacts import validate_registered_adapter_artifact
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.eval import check_activation_gate, run_learning_eval
from morpheus.core.learning.train import plan_training_run
from morpheus.core.semantic.review import ReviewStore
from tests.test_learning_dataset import copy_learning_project
from tests.test_learning_eval import mark_eval_activation_eligible


def register_test_adapter_weights(project_root: Path, adapter_id: str) -> Path:
    adapter_dir = project_root / ".morpheus/training/adapters" / adapter_id
    weight_path = adapter_dir / "adapter.safetensors"
    weight_bytes = f"test weights for {adapter_id}\n".encode()
    weight_path.write_bytes(weight_bytes)
    manifest_path = adapter_dir / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "artifact_schema": "morpheus-adapter-artifact/1",
        "training_status": "trained",
        "weight_artifact": {
            "path": weight_path.name,
            "sha256": sha256(weight_bytes).hexdigest(),
            "size": len(weight_bytes),
        },
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return weight_path


def mark_all_evals_activation_eligible(
    project_root: Path,
) -> None:
    for manifest_path in project_root.glob(
        ".morpheus/training/adapters/*/adapter_manifest.json"
    ):
        manifest = json.loads(manifest_path.read_text())
        register_test_adapter_weights(project_root, manifest["adapter_id"])
    for config_path in project_root.glob(
        ".morpheus/training/evals/*/eval_config.json"
    ):
        mark_eval_activation_eligible(config_path.parent)


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


def adapter_eval_dir(project_root: Path, adapter_id: str) -> Path:
    for config_path in project_root.glob(
        ".morpheus/training/evals/*/eval_config.json"
    ):
        config = json.loads(config_path.read_text())
        if config.get("adapter_id") == adapter_id and config.get("base_only") is False:
            return config_path.parent
    raise AssertionError(f"Missing eval for adapter {adapter_id}")


def base_eval_dir(project_root: Path) -> Path:
    for config_path in project_root.glob(
        ".morpheus/training/evals/*/eval_config.json"
    ):
        config = json.loads(config_path.read_text())
        if config.get("base_only") is True:
            return config_path.parent
    raise AssertionError("Missing base eval")


def activation_state_bytes(project_root: Path, *adapter_ids: str) -> dict[str, bytes | None]:
    paths = {
        "active": project_root / ".morpheus/training/active_adapter.json",
        "log": project_root / ".morpheus/training/rollback_log.jsonl",
    }
    for adapter_id in adapter_ids:
        paths[f"manifest:{adapter_id}"] = (
            project_root
            / ".morpheus/training/adapters"
            / adapter_id
            / "adapter_manifest.json"
        )
    return {
        label: path.read_bytes() if path.is_file() else None
        for label, path in paths.items()
    }


def test_list_adapters_includes_planned_adapter_and_eval_score(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root)

    adapters = list_adapters(project_root)

    assert [adapter["adapter_id"] for adapter in adapters] == [train["adapter_id"]]
    assert adapters[0]["status"] == "planned"
    assert adapters[0]["eval_score"] == 1.0


def test_list_adapters_surfaces_newest_invalid_eval_without_older_fallback(
    tmp_path,
):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root)
    old_eval = adapter_eval_dir(project_root, train["adapter_id"])
    newest_eval = old_eval.with_name("eval_99991231T235959999999Z")
    newest_eval.mkdir()
    old_config = json.loads((old_eval / "eval_config.json").read_text())
    old_config["eval_id"] = newest_eval.name
    (newest_eval / "eval_config.json").write_text(json.dumps(old_config))
    (newest_eval / "eval_results.json").write_text("{not-json")

    adapters = list_adapters(project_root)

    assert adapters[0]["eval_id"] == newest_eval.name
    assert adapters[0]["eval_score"] is None
    assert adapters[0]["hallucination_rate"] is None
    assert adapters[0]["eval_status"] == "invalid"
    assert adapters[0]["eval_blocker"]


def test_list_adapters_rejects_metrics_that_disagree_with_result_items(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root)
    eval_dir = adapter_eval_dir(project_root, train["adapter_id"])
    results_path = eval_dir / "eval_results.json"
    results = json.loads(results_path.read_text())
    results["items"][0]["passed"] = not results["items"][0]["passed"]
    results_path.write_text(json.dumps(results))

    adapters = list_adapters(project_root)

    assert adapters[0]["eval_id"] == eval_dir.name
    assert adapters[0]["eval_status"] == "invalid"
    assert adapters[0]["eval_score"] is None
    assert adapters[0]["eval_blocker"] == "eval metrics invalid"


def test_artifact_validation_rejects_unsafe_adapter_id_before_registry_read(tmp_path):
    escaped_dir = tmp_path / "adapters" / ".." / "escaped"
    escaped_dir.mkdir(parents=True)
    weight_bytes = b"unregistered weights"
    (escaped_dir / "adapter.safetensors").write_bytes(weight_bytes)
    (escaped_dir / "adapter_manifest.json").write_text(json.dumps({
        "adapter_id": "../escaped",
        "artifact_schema": "morpheus-adapter-artifact/1",
        "training_status": "trained",
        "weight_artifact": {
            "path": "adapter.safetensors",
            "sha256": sha256(weight_bytes).hexdigest(),
            "size": len(weight_bytes),
        },
    }))

    validation = validate_registered_adapter_artifact(
        escaped_dir,
        expected_adapter_id="../escaped",
    )

    assert validation["valid"] is False
    assert validation["blockers"] == ["adapter_id_invalid"]


def test_activate_refuses_passing_diagnostic_adapter_without_writes(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root)
    register_test_adapter_weights(project_root, train["adapter_id"])

    with pytest.raises(ValueError, match="diagnostic_eval_not_activation_eligible"):
        activate_adapter(project_root, train["adapter_id"])

    adapter_dir = project_root / ".morpheus" / "training" / "adapters" / train["adapter_id"]
    assert active_adapter_id(project_root) is None
    assert not (adapter_dir / "activate_receipt.json").exists()
    assert not (project_root / ".morpheus/training/rollback_log.jsonl").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "planned",
        "missing",
        "symlink",
        "tampered",
        "zero",
        "unsafe_path",
        "wrong_extension",
    ],
)
def test_activation_refuses_adapter_without_exact_registered_weights(
    tmp_path,
    mutation,
):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    adapter_dir = (
        project_root / ".morpheus/training/adapters" / train["adapter_id"]
    )
    manifest_path = adapter_dir / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    weight_path = adapter_dir / manifest["weight_artifact"]["path"]
    if mutation == "planned":
        weight_path.unlink()
        manifest.update({
            "training_status": "planned",
            "weight_artifact": None,
        })
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "missing":
        weight_path.unlink()
    elif mutation == "symlink":
        target = tmp_path / "unregistered.safetensors"
        target.write_bytes(weight_path.read_bytes())
        weight_path.unlink()
        weight_path.symlink_to(target)
    elif mutation == "tampered":
        weight_path.write_bytes(weight_path.read_bytes() + b"tampered")
    elif mutation == "zero":
        weight_path.write_bytes(b"")
        manifest["weight_artifact"].update({
            "sha256": sha256(b"").hexdigest(),
            "size": 0,
        })
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "unsafe_path":
        manifest["weight_artifact"]["path"] = "../adapter.safetensors"
        manifest_path.write_text(json.dumps(manifest))
    else:
        renamed = weight_path.with_suffix(".bin")
        weight_path.rename(renamed)
        manifest["weight_artifact"]["path"] = renamed.name
        manifest_path.write_text(json.dumps(manifest))

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "adapter_artifact_invalid"
    assert gate["adapter_artifact_blockers"]
    with pytest.raises(ValueError, match="adapter_artifact_invalid"):
        activate_adapter(project_root, train["adapter_id"])
    assert active_adapter_id(project_root) is None


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


def test_rollback_refuses_ineligible_previous_adapter_without_state_changes(tmp_path):
    project_root = copy_learning_project(tmp_path)
    first = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, first["adapter_id"])
    second = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=second["adapter_id"], dry_run=True)
    mark_all_evals_activation_eligible(project_root)
    activate_adapter(project_root, second["adapter_id"])
    first_eval_config = adapter_eval_dir(
        project_root,
        first["adapter_id"],
    ) / "eval_config.json"
    config = json.loads(first_eval_config.read_text())
    config["activation_eligible"] = False
    first_eval_config.write_text(json.dumps(config))
    state_before = activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    )

    with pytest.raises(
        ValueError,
        match="Cannot rollback to adapter .*diagnostic_eval_not_activation_eligible",
    ):
        rollback_adapter(project_root)

    assert activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    ) == state_before


@pytest.mark.parametrize("authority_loss", ["review_revocation", "artifact_tamper"])
def test_rollback_refuses_previous_adapter_after_dataset_authority_loss(
    tmp_path,
    authority_loss,
):
    project_root = copy_learning_project(tmp_path)
    first = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, first["adapter_id"])
    second = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=second["adapter_id"], dry_run=True)
    mark_all_evals_activation_eligible(project_root)
    activate_adapter(project_root, second["adapter_id"])

    if authority_loss == "review_revocation":
        ReviewStore(project_root).reject(
            "c_current",
            reason="revoked before rollback",
        )
    else:
        dataset_artifact = next(
            project_root.glob(
                ".morpheus/training/datasets/*/dataset.instruction.jsonl"
            )
        )
        dataset_artifact.write_bytes(dataset_artifact.read_bytes() + b"{}\n")
    state_before = activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    )

    with pytest.raises(ValueError, match="dataset_not_current"):
        rollback_adapter(project_root)

    assert active_adapter_id(project_root) == second["adapter_id"]
    assert activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    ) == state_before


def test_rollback_refuses_previous_adapter_after_weight_tamper(tmp_path):
    project_root = copy_learning_project(tmp_path)
    first = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, first["adapter_id"])
    second = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=second["adapter_id"], dry_run=True)
    mark_all_evals_activation_eligible(project_root)
    activate_adapter(project_root, second["adapter_id"])
    first_weight = (
        project_root
        / ".morpheus/training/adapters"
        / first["adapter_id"]
        / "adapter.safetensors"
    )
    first_weight.write_bytes(first_weight.read_bytes() + b"tampered")
    state_before = activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    )

    with pytest.raises(
        ValueError,
        match="Cannot rollback to adapter .*adapter_artifact_invalid",
    ):
        rollback_adapter(project_root)

    assert active_adapter_id(project_root) == second["adapter_id"]
    assert activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    ) == state_before


def test_rollback_rechecks_previous_adapter_gate_immediately_before_writes(
    tmp_path,
    monkeypatch,
):
    project_root = copy_learning_project(tmp_path)
    first = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, first["adapter_id"])
    second = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=second["adapter_id"], dry_run=True)
    mark_all_evals_activation_eligible(project_root)
    activate_adapter(project_root, second["adapter_id"])
    state_before = activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    )
    real_gate = adapters_module.check_activation_gate
    calls = 0

    def changing_gate(project, adapter_id, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return {
                "allowed": False,
                "reason": "dataset_not_current",
                "adapter_id": adapter_id,
            }
        return real_gate(project, adapter_id, **kwargs)

    monkeypatch.setattr(adapters_module, "check_activation_gate", changing_gate)

    with pytest.raises(ValueError, match="rollback authority changed.*dataset_not_current"):
        rollback_adapter(project_root)

    assert calls == 2
    assert activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    ) == state_before


@pytest.mark.parametrize(
    "artifact_name",
    [
        "adapter_manifest.json",
        "eval_config.json",
        "eval_results.json",
        "eval_activation_receipt.json",
        "base_eval_config.json",
        "base_eval_results.json",
        "base_eval_activation_receipt.json",
        "dataset_artifact",
        "review_snapshot",
    ],
)
def test_activate_aborts_when_authority_bytes_change_around_final_gate(
    tmp_path,
    monkeypatch,
    artifact_name,
):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    adapter_dir = (
        project_root / ".morpheus/training/adapters" / train["adapter_id"]
    )
    eval_dir = adapter_eval_dir(project_root, train["adapter_id"])
    base_dir = base_eval_dir(project_root)
    dataset_artifact = next(
        project_root.glob(
            ".morpheus/training/datasets/*/dataset.instruction.jsonl"
        )
    )
    artifact_paths = {
        "adapter_manifest.json": adapter_dir / "adapter_manifest.json",
        "eval_config.json": eval_dir / "eval_config.json",
        "eval_results.json": eval_dir / "eval_results.json",
        "eval_activation_receipt.json": eval_dir / "activation_eval_receipt.json",
        "base_eval_config.json": base_dir / "eval_config.json",
        "base_eval_results.json": base_dir / "eval_results.json",
        "base_eval_activation_receipt.json": (
            base_dir / "activation_eval_receipt.json"
        ),
        "dataset_artifact": dataset_artifact,
    }
    manifest_before = (adapter_dir / "adapter_manifest.json").read_bytes()
    real_gate = adapters_module.check_activation_gate
    calls = 0

    def mutate_after_final_gate(project, adapter_id, **kwargs):
        nonlocal calls
        calls += 1
        result = real_gate(project, adapter_id, **kwargs)
        if calls == 2:
            if artifact_name == "review_snapshot":
                store = ReviewStore(project_root)
                candidates = store.load_candidates()
                candidates[0] = candidates[0].model_copy(update={"status": "pending"})
                store.save_candidates(candidates)
            else:
                artifact_path = artifact_paths[artifact_name]
                artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(
        adapters_module,
        "check_activation_gate",
        mutate_after_final_gate,
    )

    with pytest.raises(ValueError, match="[Aa]ctivation (authority|dataset authority)"):
        activate_adapter(project_root, train["adapter_id"])

    assert calls == 2
    assert active_adapter_id(project_root) is None
    assert not (adapter_dir / "activate_receipt.json").exists()
    assert not (project_root / ".morpheus/training/rollback_log.jsonl").exists()
    if artifact_name != "adapter_manifest.json":
        assert (adapter_dir / "adapter_manifest.json").read_bytes() == manifest_before


def test_activate_revalidates_weight_immediately_before_commit(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    adapter_dir = (
        project_root / ".morpheus/training/adapters" / train["adapter_id"]
    )
    weight_path = adapter_dir / "adapter.safetensors"
    state_before = activation_state_bytes(project_root, train["adapter_id"])
    real_validate = adapters_module.validate_registered_adapter_artifact

    def tamper_before_closing_validation(*args, **kwargs):
        weight_path.write_bytes(weight_path.read_bytes() + b"changed before commit")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(
        adapters_module,
        "validate_registered_adapter_artifact",
        tamper_before_closing_validation,
    )

    with pytest.raises(ValueError, match="adapter artifact changed before commit"):
        activate_adapter(project_root, train["adapter_id"])

    assert activation_state_bytes(project_root, train["adapter_id"]) == state_before
    assert not (adapter_dir / "activate_receipt.json").exists()


@pytest.mark.parametrize("mutation_point", ["commit_entry", "pointer_write"])
def test_activation_rolls_back_when_weight_changes_during_commit(
    tmp_path,
    monkeypatch,
    mutation_point,
):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    adapter_dir = (
        project_root / ".morpheus/training/adapters" / train["adapter_id"]
    )
    weight_path = adapter_dir / "adapter.safetensors"
    pointer_path = adapters_module.active_adapter_path(project_root)
    state_before = activation_state_bytes(project_root, train["adapter_id"])

    if mutation_point == "commit_entry":
        real_commit = adapters_module._commit_activation_transaction

        def tamper_then_commit(*args, **kwargs):
            weight_path.write_bytes(weight_path.read_bytes() + b"changed at commit")
            return real_commit(*args, **kwargs)

        monkeypatch.setattr(
            adapters_module,
            "_commit_activation_transaction",
            tamper_then_commit,
        )
    else:
        real_write = adapters_module._write_json

        def tamper_during_pointer_write(path, data):
            if path == pointer_path:
                weight_path.write_bytes(
                    weight_path.read_bytes() + b"changed at pointer"
                )
            return real_write(path, data)

        monkeypatch.setattr(adapters_module, "_write_json", tamper_during_pointer_write)

    with pytest.raises(ValueError, match="adapter artifact changed before commit"):
        activate_adapter(project_root, train["adapter_id"])

    assert activation_state_bytes(project_root, train["adapter_id"]) == state_before
    assert not (adapter_dir / "activate_receipt.json").exists()


def test_rollback_rechecks_target_weight_inside_commit(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    first = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, first["adapter_id"])
    second = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=second["adapter_id"], dry_run=True)
    mark_all_evals_activation_eligible(project_root)
    activate_adapter(project_root, second["adapter_id"])
    target_weight = (
        project_root
        / ".morpheus/training/adapters"
        / first["adapter_id"]
        / "adapter.safetensors"
    )
    state_before = activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    )
    real_commit = adapters_module._commit_activation_transaction

    def tamper_then_commit(*args, **kwargs):
        target_weight.write_bytes(target_weight.read_bytes() + b"changed at rollback")
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(
        adapters_module,
        "_commit_activation_transaction",
        tamper_then_commit,
    )

    with pytest.raises(ValueError, match="adapter artifact changed before commit"):
        rollback_adapter(project_root)

    assert activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    ) == state_before


def test_active_status_fails_closed_after_weight_tamper(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, train["adapter_id"])
    weight_path = (
        project_root
        / ".morpheus/training/adapters"
        / train["adapter_id"]
        / "adapter.safetensors"
    )
    weight_path.write_bytes(weight_path.read_bytes() + b"changed after activation")

    status = active_adapter_status(project_root)
    listed = list_adapters(project_root)

    assert status is not None
    assert status["status"] == "invalid"
    assert status["artifact_valid"] is False
    assert status["artifact_blockers"]
    assert listed[0]["status"] == "invalid"


def test_activation_receipt_and_active_payload_use_exact_eval_identity(tmp_path):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    adapter_dir = (
        project_root / ".morpheus/training/adapters" / train["adapter_id"]
    )
    eval_dir = adapter_eval_dir(project_root, train["adapter_id"])
    base_dir = base_eval_dir(project_root)
    manifest_sha = sha256((adapter_dir / "adapter_manifest.json").read_bytes()).hexdigest()
    config_sha = sha256((eval_dir / "eval_config.json").read_bytes()).hexdigest()
    results_bytes = (eval_dir / "eval_results.json").read_bytes()
    results_sha = sha256(results_bytes).hexdigest()
    eval_receipt_sha = sha256(
        (eval_dir / "activation_eval_receipt.json").read_bytes()
    ).hexdigest()
    base_config_sha = sha256((base_dir / "eval_config.json").read_bytes()).hexdigest()
    base_results_sha = sha256((base_dir / "eval_results.json").read_bytes()).hexdigest()
    base_eval_receipt_sha = sha256(
        (base_dir / "activation_eval_receipt.json").read_bytes()
    ).hexdigest()
    results = json.loads(results_bytes)
    weight_path = adapter_dir / "adapter.safetensors"
    weight_sha = sha256(weight_path.read_bytes()).hexdigest()
    weight_size = weight_path.stat().st_size

    activate_adapter(project_root, train["adapter_id"])

    receipt = json.loads((adapter_dir / "activate_receipt.json").read_text())
    active = json.loads(
        (project_root / ".morpheus/training/active_adapter.json").read_text()
    )
    assert receipt["adapter_manifest_sha256"] == manifest_sha
    assert receipt["eval_config_sha256"] == config_sha
    assert receipt["eval_results_sha256"] == results_sha
    assert receipt["eval_activation_receipt_sha256"] == eval_receipt_sha
    assert receipt["base_eval_config_sha256"] == base_config_sha
    assert receipt["base_eval_results_sha256"] == base_results_sha
    assert receipt["base_eval_activation_receipt_sha256"] == base_eval_receipt_sha
    assert len(receipt["dataset_manifest_sha256"]) == 64
    assert len(receipt["dataset_authority_sha256"]) == 64
    assert receipt["weight_artifact_path"] == "adapter.safetensors"
    assert receipt["weight_artifact_sha256"] == weight_sha
    assert receipt["weight_artifact_size"] == weight_size
    assert receipt["eval_id"] == results["eval_id"]
    assert receipt["metrics"] == results["metrics"]
    assert active["eval_id"] == results["eval_id"]
    assert active["eval_score"] == results["metrics"]["pass_rate"]
    assert active["eval_activation_receipt_sha256"] == eval_receipt_sha
    assert active["base_eval_activation_receipt_sha256"] == base_eval_receipt_sha
    assert active["weight_artifact_path"] == "adapter.safetensors"
    assert active["weight_artifact_sha256"] == weight_sha


def test_activation_pointer_write_failure_restores_every_state_view(
    tmp_path,
    monkeypatch,
):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    adapter_dir = (
        project_root / ".morpheus/training/adapters" / train["adapter_id"]
    )
    state_before = activation_state_bytes(project_root, train["adapter_id"])
    real_write = adapters_module._write_json

    def fail_pointer(path, data):
        if path == adapters_module.active_adapter_path(project_root):
            raise OSError("simulated pointer failure")
        return real_write(path, data)

    monkeypatch.setattr(adapters_module, "_write_json", fail_pointer)

    with pytest.raises(OSError, match="simulated pointer failure"):
        activate_adapter(project_root, train["adapter_id"])

    assert activation_state_bytes(project_root, train["adapter_id"]) == state_before
    assert not (adapter_dir / "activate_receipt.json").exists()


def test_activation_reports_success_when_pointer_write_failed_after_replace(
    tmp_path,
    monkeypatch,
):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    pointer_path = adapters_module.active_adapter_path(project_root)
    real_write = adapters_module._write_json

    def write_then_fail(path, data):
        real_write(path, data)
        if path == pointer_path:
            raise OSError("simulated post-replace durability failure")

    monkeypatch.setattr(adapters_module, "_write_json", write_then_fail)

    result = activate_adapter(project_root, train["adapter_id"])

    assert result["activated"] is True
    assert active_adapter_id(project_root) == train["adapter_id"]
    assert not adapters_module._activation_journal_path(project_root).exists()


def test_rollback_pointer_write_failure_restores_every_state_view(
    tmp_path,
    monkeypatch,
):
    project_root = copy_learning_project(tmp_path)
    first = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, first["adapter_id"])
    second = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, adapter_id=second["adapter_id"], dry_run=True)
    mark_all_evals_activation_eligible(project_root)
    activate_adapter(project_root, second["adapter_id"])
    state_before = activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    )
    real_write = adapters_module._write_json

    def fail_pointer(path, data):
        if path == adapters_module.active_adapter_path(project_root):
            raise OSError("simulated rollback pointer failure")
        return real_write(path, data)

    monkeypatch.setattr(adapters_module, "_write_json", fail_pointer)

    with pytest.raises(OSError, match="simulated rollback pointer failure"):
        rollback_adapter(project_root)

    assert activation_state_bytes(
        project_root,
        first["adapter_id"],
        second["adapter_id"],
    ) == state_before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adapter_id", "../escaped"),
        ("adapter_id", "/tmp/escaped"),
        ("previous_adapter_id", "..\\escaped"),
    ],
)
def test_active_pointer_rejects_unsafe_adapter_ids(tmp_path, field, value):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, train["adapter_id"])
    pointer_path = adapters_module.active_adapter_path(project_root)
    pointer = json.loads(pointer_path.read_text())
    pointer[field] = value
    pointer_path.write_text(json.dumps(pointer))

    with pytest.raises(ValueError, match="Active adapter .*identity invalid"):
        rollback_adapter(project_root)


def test_activation_recovers_uncommitted_crash_journal(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    state_before = activation_state_bytes(project_root, train["adapter_id"])
    real_apply = adapters_module._apply_activation_artifact
    calls = 0

    def crash_after_first_artifact(path, payload):
        nonlocal calls
        real_apply(path, payload)
        calls += 1
        if calls == 1:
            raise SystemExit("simulated process crash before pointer commit")

    monkeypatch.setattr(
        adapters_module,
        "_apply_activation_artifact",
        crash_after_first_artifact,
    )

    with pytest.raises(SystemExit, match="before pointer commit"):
        activate_adapter(project_root, train["adapter_id"])

    assert adapters_module._activation_journal_path(project_root).is_file()
    monkeypatch.setattr(
        adapters_module,
        "_apply_activation_artifact",
        real_apply,
    )
    list_adapters(project_root)

    assert activation_state_bytes(project_root, train["adapter_id"]) == state_before
    assert not adapters_module._activation_journal_path(project_root).exists()


def test_activation_recovers_committed_crash_journal(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    train = planned_adapter(project_root, activation_eligible=True)
    real_remove = adapters_module._remove_activation_journal
    calls = 0

    def crash_before_journal_cleanup(project):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SystemExit("simulated process crash after pointer commit")
        return real_remove(project)

    monkeypatch.setattr(
        adapters_module,
        "_remove_activation_journal",
        crash_before_journal_cleanup,
    )

    with pytest.raises(SystemExit, match="after pointer commit"):
        activate_adapter(project_root, train["adapter_id"])

    assert active_adapter_id(project_root) == train["adapter_id"]
    assert adapters_module._activation_journal_path(project_root).is_file()
    monkeypatch.setattr(
        adapters_module,
        "_remove_activation_journal",
        real_remove,
    )
    listed = list_adapters(project_root)

    assert listed[0]["status"] == "active"
    assert not adapters_module._activation_journal_path(project_root).exists()


def test_activation_recovery_rejects_incomplete_journal_before_writes(tmp_path):
    project_root = copy_learning_project(tmp_path)
    training_root = project_root / ".morpheus/training"
    adapter_dir = training_root / "adapters/adapter_safe"
    adapter_dir.mkdir(parents=True)
    receipt_path = adapter_dir / "activate_receipt.json"
    pointer_path = adapters_module.active_adapter_path(project_root)
    journal_path = adapters_module._activation_journal_path(project_root)
    journal_path.write_text(json.dumps({
        "schema": "morpheus-activation-transaction/1",
        "transaction_id": "activation_txn_incomplete",
        "pointer": ".morpheus/training/active_adapter.json",
        "entries": [
            {
                "path": ".morpheus/training/adapters/adapter_safe/activate_receipt.json",
                "before": None,
                "after": "e30K",
            },
            {
                "path": ".morpheus/training/active_adapter.json",
                "before": None,
            },
        ],
    }))

    with pytest.raises(ValueError, match="journal entry invalid"):
        list_adapters(project_root)

    assert not receipt_path.exists()
    assert not pointer_path.exists()
    assert journal_path.is_file()


def test_directory_fsync_is_platform_safe_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters_module.os, "name", "nt")

    def unexpected_open(*args, **kwargs):
        raise AssertionError("Windows directory fsync must not call os.open")

    monkeypatch.setattr(adapters_module.os, "open", unexpected_open)

    adapters_module._fsync_directory(tmp_path)


def test_atomic_write_does_not_unlink_replaced_temporary_file(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "state.json"
    replacement = b"unowned replacement"
    observed = {}

    def replace_then_fail(source, destination):
        source = Path(source)
        observed["temporary"] = source
        source.rename(tmp_path / "original-temporary")
        source.write_bytes(replacement)
        raise OSError("simulated replacement race")

    monkeypatch.setattr(adapters_module.os, "replace", replace_then_fail)

    with pytest.raises(OSError, match="replacement race"):
        adapters_module._atomic_write_bytes(target, b"trusted", "test")

    assert observed["temporary"].read_bytes() == replacement


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
