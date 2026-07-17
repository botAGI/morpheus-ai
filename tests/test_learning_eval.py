from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from threading import Event

import pytest

from typer.testing import CliRunner

import morpheus.core.learning.eval as eval_module
from morpheus.cli import app
from morpheus.core.config import MorpheusConfig
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.eval import check_activation_gate, run_learning_eval
from morpheus.core.learning.train import plan_training_run
from morpheus.core.semantic.review import ReviewStore
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
    project_root = eval_dir.parents[3]
    if config.get("adapter_id") is not None:
        register_test_adapter_weights(project_root, config["adapter_id"])
    write_trusted_activation_eval_receipt(eval_dir)


def write_trusted_activation_eval_receipt(eval_dir: Path) -> None:
    project_root = eval_dir.parents[3]
    MorpheusConfig(project_root=project_root).init_default()
    receipt_path = eval_dir / "activation_eval_receipt.json"
    receipt_path.write_bytes(
        eval_module._build_activation_eval_receipt_bytes(project_root, eval_dir)
    )
    if os.name != "nt":
        receipt_path.chmod(0o600)


def register_test_adapter_weights(project_root: Path, adapter_id: str) -> None:
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


def test_eval_publication_is_atomic_and_private(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    original_publish = eval_module._publish_staged_eval
    observed = {}

    def inspect_then_publish(staging_dir, eval_dir, **kwargs):
        observed["staging_name"] = staging_dir.name
        observed["canonical_absent"] = not eval_dir.exists()
        observed["entries"] = sorted(path.name for path in staging_dir.iterdir())
        if os.name != "nt":
            observed["directory_mode"] = stat.S_IMODE(
                staging_dir.stat(follow_symlinks=False).st_mode
            )
            observed["file_modes"] = {
                path.name: stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
                for path in staging_dir.iterdir()
            }
        return original_publish(staging_dir, eval_dir, **kwargs)

    monkeypatch.setattr(eval_module, "_publish_staged_eval", inspect_then_publish)

    result = run_learning_eval(project_root, dry_run=True, base_only=True)

    eval_dir = Path(result["eval_dir"])
    assert observed["staging_name"].startswith(f".{result['eval_id']}.")
    assert observed["staging_name"].endswith(".staging")
    assert observed["canonical_absent"] is True
    assert observed["entries"] == [
        "eval_config.json",
        "eval_report.md",
        "eval_results.json",
    ]
    assert sorted(path.name for path in eval_dir.iterdir()) == observed["entries"]
    if os.name != "nt":
        assert observed["directory_mode"] == 0o700
        assert set(observed["file_modes"].values()) == {0o600}


def test_trusted_eval_receipt_is_added_before_atomic_publication(
    tmp_path,
    monkeypatch,
):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    diagnostic = run_learning_eval(project_root, base_only=True, dry_run=True)
    source_dir = Path(diagnostic["eval_dir"])
    eval_id = "eval_99991231T235959999999Z"
    config = json.loads((source_dir / "eval_config.json").read_text())
    config.update({
        "eval_id": eval_id,
        "activation_eligible": True,
        "dry_run": False,
        "evaluation_mode": "heldout_external",
        "provider": {"name": "external-heldout"},
    })
    results = json.loads((source_dir / "eval_results.json").read_text())
    results.update({
        "eval_id": eval_id,
        "activation_eligible": True,
        "evaluation_mode": "heldout_external",
    })
    expected_contents = {
        "eval_config.json": json.dumps(config, indent=2, sort_keys=True) + "\n",
        "eval_results.json": json.dumps(results, indent=2, sort_keys=True) + "\n",
        "eval_report.md": "# trusted local held-out eval\n",
    }
    evals_root = source_dir.parent
    staging_dir, staging_identity = eval_module._create_private_eval_staging(
        evals_root,
        eval_id,
    )
    for name, content in expected_contents.items():
        eval_module._write_private_text(staging_dir / name, content)
    MorpheusConfig(project_root=project_root).init_default()
    observed = {}
    original_publish = eval_module._publish_staged_eval

    def inspect_then_publish(staging, destination, **kwargs):
        observed["entries"] = sorted(path.name for path in staging.iterdir())
        observed["expected"] = sorted(kwargs["expected_contents"])
        return original_publish(staging, destination, **kwargs)

    monkeypatch.setattr(eval_module, "_publish_staged_eval", inspect_then_publish)

    eval_module._publish_staged_activation_eval(
        project_root,
        staging_dir,
        evals_root / eval_id,
        staging_identity=staging_identity,
        expected_contents=expected_contents,
    )

    assert observed["entries"] == [
        "activation_eval_receipt.json",
        "eval_config.json",
        "eval_report.md",
        "eval_results.json",
    ]
    assert observed["expected"] == observed["entries"]
    assert sorted(path.name for path in (evals_root / eval_id).iterdir()) == observed[
        "entries"
    ]


def test_eval_publication_refuses_replaced_staging_without_deleting_it(
    tmp_path,
    monkeypatch,
):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    original_publish = eval_module._publish_staged_eval
    sentinel = {}

    def replace_then_publish(staging_dir, eval_dir, **kwargs):
        moved = staging_dir.with_name(staging_dir.name + ".original")
        staging_dir.rename(moved)
        staging_dir.mkdir(mode=0o700)
        sentinel_path = staging_dir / "do-not-delete.txt"
        sentinel_path.write_text("preserve unknown replacement")
        sentinel["path"] = sentinel_path
        return original_publish(staging_dir, eval_dir, **kwargs)

    monkeypatch.setattr(eval_module, "_publish_staged_eval", replace_then_publish)

    with pytest.raises(ValueError, match="staging identity changed"):
        run_learning_eval(project_root, dry_run=True, base_only=True)

    assert sentinel["path"].read_text() == "preserve unknown replacement"
    canonical = [
        path
        for path in (project_root / ".morpheus/training/evals").iterdir()
        if path.name.startswith("eval_")
    ]
    assert canonical == []


@pytest.mark.parametrize("mutation", ["changed_content", "unknown_entry"])
def test_eval_publication_validates_exact_staging_product(
    tmp_path,
    monkeypatch,
    mutation,
):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    original_publish = eval_module._publish_staged_eval
    observed = {}

    def mutate_then_publish(staging_dir, eval_dir, **kwargs):
        observed["staging_dir"] = staging_dir
        if mutation == "changed_content":
            (staging_dir / "eval_results.json").write_text("{}\n")
        else:
            (staging_dir / "unexpected.txt").write_text("unexpected")
        return original_publish(staging_dir, eval_dir, **kwargs)

    monkeypatch.setattr(eval_module, "_publish_staged_eval", mutate_then_publish)

    with pytest.raises(ValueError, match="Eval staging"):
        run_learning_eval(project_root, dry_run=True, base_only=True)

    assert observed["staging_dir"].is_dir()
    canonical = [
        path
        for path in (project_root / ".morpheus/training/evals").iterdir()
        if path.name.startswith("eval_")
    ]
    assert canonical == []


def test_eval_holds_review_authority_lease_through_atomic_publication(
    tmp_path,
    monkeypatch,
):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    publication_entered = Event()
    allow_publication = Event()
    original_publish = eval_module._publish_staged_eval

    def blocking_publish(*args, **kwargs):
        publication_entered.set()
        assert allow_publication.wait(timeout=5)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(eval_module, "_publish_staged_eval", blocking_publish)

    with ThreadPoolExecutor(max_workers=2) as pool:
        eval_future = pool.submit(
            run_learning_eval,
            project_root,
            base_only=True,
            dry_run=True,
        )
        assert publication_entered.wait(timeout=5)
        reject_future = pool.submit(
            ReviewStore(project_root).reject,
            "c_current",
            reason="revoked during eval publication",
        )
        try:
            with pytest.raises(FuturesTimeoutError):
                reject_future.result(timeout=0.2)
        finally:
            allow_publication.set()
        evaluation = eval_future.result(timeout=5)
        rejected = reject_future.result(timeout=5)

    assert Path(evaluation["eval_dir"]).is_dir()
    assert rejected.status == "rejected"


def test_eval_refuses_dataset_selection_change_before_authority_lease(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    selected = Path(dataset["dataset_dir"])
    replacement = selected.with_name("dataset_replacement")
    replacement.mkdir()
    calls = 0

    def changing_selector(project, dataset_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            return selected
        return replacement

    monkeypatch.setattr(eval_module, "_dataset_dir_for_eval", changing_selector)

    with pytest.raises(ValueError, match="Dataset selection changed"):
        run_learning_eval(project_root, base_only=True, dry_run=True)

    assert calls == 2
    evals_root = project_root / ".morpheus/training/evals"
    assert not evals_root.exists() or not any(
        path.name.startswith("eval_") for path in evals_root.iterdir()
    )


def test_eval_config_categories_match_bound_seed(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)

    result = run_learning_eval(project_root, dry_run=True, base_only=True)

    seed_items = read_jsonl(Path(dataset["dataset_dir"]) / "eval.seed.jsonl")
    config = json.loads(Path(result["eval_config_path"]).read_text())
    assert config["categories"] == sorted({item["category"] for item in seed_items})


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
    register_test_adapter_weights(project_root, train["adapter_id"])

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "diagnostic_eval_not_activation_eligible"
    assert gate["evaluation_mode"] == "diagnostic_fake"
    assert gate["provider"] == "fake-adapter"


def test_activation_refuses_relabelled_diagnostic_eval_without_trusted_receipt(
    tmp_path,
):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    train = plan_training_run(project_root, dry_run=True)
    base = run_learning_eval(project_root, base_only=True, dry_run=True)
    adapter = run_learning_eval(
        project_root,
        adapter_id=train["adapter_id"],
        dry_run=True,
    )
    register_test_adapter_weights(project_root, train["adapter_id"])
    for evaluation in (base, adapter):
        eval_dir = Path(evaluation["eval_dir"])
        config_path = eval_dir / "eval_config.json"
        results_path = eval_dir / "eval_results.json"
        config = json.loads(config_path.read_text())
        config.update({
            "activation_eligible": True,
            "dry_run": False,
            "evaluation_mode": "heldout_external",
            "provider": {"name": "external-heldout"},
        })
        results = json.loads(results_path.read_text())
        results.update({
            "activation_eligible": True,
            "evaluation_mode": "heldout_external",
        })
        config_path.write_text(json.dumps(config))
        results_path.write_text(json.dumps(results))

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "eval_activation_receipt_invalid"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_activation_receipt_refuses_non_private_signing_key(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    evaluation = run_learning_eval(project_root, base_only=True, dry_run=True)
    MorpheusConfig(project_root=project_root).init_default()
    (project_root / ".morpheus/keys/local.key").chmod(0o644)

    with pytest.raises(ValueError, match="permissions must be 0600"):
        mark_eval_activation_eligible(Path(evaluation["eval_dir"]))


@pytest.mark.parametrize("mutation", ["config", "results", "signature"])
def test_activation_receipt_binds_exact_eval_artifacts(tmp_path, mutation):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_receipt_binding"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    eval_id = "eval_20260522T000000000002Z"
    _write_gate_eval(
        project_root,
        eval_id=eval_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    eval_dir = project_root / ".morpheus/training/evals" / eval_id
    if mutation in {"config", "results"}:
        path = eval_dir / f"eval_{mutation}.json"
        payload = json.loads(path.read_text())
        payload["unreceipted_field"] = "changed"
        path.write_text(json.dumps(payload))
    else:
        path = eval_dir / "activation_eval_receipt.json"
        payload = json.loads(path.read_text())
        payload["signature"]["signature_b64"] = "AA=="
        path.write_text(json.dumps(payload))
        if os.name != "nt":
            path.chmod(0o600)

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "eval_activation_receipt_invalid"


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
    register_test_adapter_weights(project_root, train["adapter_id"])

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
def test_activation_comparison_ignores_unpublished_hidden_results(
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
    unpaired_dir = (
        project_root
        / ".morpheus/training/evals"
        / f".{unpaired_eval_id}.interrupted.staging"
    )
    unpaired_dir.mkdir()
    (unpaired_dir / "eval_results.json").write_text(json.dumps(unpaired))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "critical_category_regression"
    assert gate["critical_regressions"][0]["category"] == regressed_category


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_config",
        "missing_results",
        "corrupt_config",
        "corrupt_results",
        "mismatched_results",
        "mismatched_binding",
    ],
)
def test_activation_does_not_fall_back_past_newer_invalid_adapter_eval(
    tmp_path,
    mutation,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_newest_invalid"
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
        regressed_category=None,
    )
    newest_id = "eval_20260522T000000000003Z"
    _write_gate_eval(
        project_root,
        eval_id=newest_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    _invalidate_eval(
        project_root / ".morpheus/training/evals" / newest_id,
        mutation,
    )

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["eval_id"] == newest_id
    expected_reason = {
        "mismatched_results": "eval_artifact_identity_mismatch",
        "mismatched_binding": "dataset_not_current",
    }.get(mutation, "eval_artifacts_invalid")
    assert gate["reason"] == expected_reason


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_config",
        "missing_results",
        "corrupt_config",
        "corrupt_results",
        "mismatched_results",
        "mismatched_binding",
    ],
)
def test_activation_does_not_fall_back_past_newer_invalid_base_eval(
    tmp_path,
    mutation,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_newest_invalid_base"
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
        regressed_category=None,
    )
    newest_id = "eval_20260522T000000000003Z"
    _write_gate_eval(
        project_root,
        eval_id=newest_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    _invalidate_eval(
        project_root / ".morpheus/training/evals" / newest_id,
        mutation,
    )

    gate = check_activation_gate(
        project_root,
        adapter_id,
        eval_id="eval_20260522T000000000002Z",
    )

    assert gate["allowed"] is False
    assert gate["base_eval_id"] == newest_id
    expected_reason = {
        "mismatched_results": "base_eval_artifact_identity_mismatch",
        "mismatched_binding": "base_eval_dataset_binding_mismatch",
    }.get(mutation, "base_eval_artifacts_invalid")
    assert gate["reason"] == expected_reason


def test_explicit_eval_id_selects_exact_invalid_entry(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_explicit_invalid"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    explicit_id = "eval_20260522T000000000002Z"
    _write_gate_eval(
        project_root,
        eval_id=explicit_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    (project_root / ".morpheus/training/evals" / explicit_id / "eval_config.json").unlink()

    gate = check_activation_gate(project_root, adapter_id, eval_id=explicit_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "eval_artifacts_invalid"
    assert gate["eval_id"] == explicit_id


def test_activation_ignores_newer_valid_eval_for_another_adapter(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_selected"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    selected_id = "eval_20260522T000000000002Z"
    _write_gate_eval(
        project_root,
        eval_id=selected_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000003Z",
        dataset_id=dataset["dataset_id"],
        adapter_id="adapter_unrelated",
        base_only=False,
        regressed_category=None,
    )

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is True
    assert gate["eval_id"] == selected_id


def test_activation_ignores_newer_provably_unrelated_corrupt_eval(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_selected_corrupt_unrelated"
    _write_gate_eval(
        project_root,
        eval_id="eval_20260522T000000000001Z",
        dataset_id=dataset["dataset_id"],
        adapter_id=None,
        base_only=True,
        regressed_category=None,
    )
    selected_id = "eval_20260522T000000000002Z"
    _write_gate_eval(
        project_root,
        eval_id=selected_id,
        dataset_id=dataset["dataset_id"],
        adapter_id=adapter_id,
        base_only=False,
        regressed_category=None,
    )
    unrelated_dir = (
        project_root
        / ".morpheus/training/evals/eval_20260522T000000000003Z"
    )
    unrelated_dir.mkdir(parents=True)
    (unrelated_dir / "eval_config.json").write_text(json.dumps({
        "eval_id": unrelated_dir.name,
        "dataset_id": "dataset_unrelated",
        "dataset_binding_sha256": "f" * 64,
        "adapter_id": "adapter_unrelated",
        "base_only": False,
    }))
    (unrelated_dir / "eval_results.json").write_text("{not-json")

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is True
    assert gate["eval_id"] == selected_id


def test_category_comparison_ignores_provably_unrelated_corrupt_eval(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_comparison_selected"
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
        regressed_category=None,
    )
    unrelated_dir = (
        project_root
        / ".morpheus/training/evals/eval_20260522T000000000003Z"
    )
    unrelated_dir.mkdir(parents=True)
    (unrelated_dir / "eval_config.json").write_text(json.dumps({
        "eval_id": unrelated_dir.name,
        "dataset_id": "dataset_unrelated",
        "dataset_binding_sha256": "f" * 64,
        "adapter_id": "adapter_unrelated",
        "base_only": False,
    }))
    (unrelated_dir / "eval_results.json").write_text("{not-json")

    comparison = eval_module.latest_eval_category_comparison(
        project_root,
        dataset_id=dataset["dataset_id"],
        dataset_binding_sha256=json.loads(
            (Path(dataset["dataset_dir"]) / "manifest.json").read_text()
        )["dataset_binding_sha256"],
        adapter_id=adapter_id,
    )

    assert comparison["base_eval"].get("valid", True) is True
    assert comparison["adapter_eval"].get("valid", True) is True
    assert comparison["adapter_eval"]["adapter_id"] == adapter_id


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


@pytest.mark.parametrize("invalid_role", ["adapter", "base"])
@pytest.mark.parametrize(
    ("field", "mutated_value"),
    [
        ("passed", False),
        ("hallucinated", True),
        ("critical_outdated_claim_failure", True),
    ],
)
def test_activation_refuses_metrics_not_recomputed_from_result_items(
    tmp_path,
    invalid_role,
    field,
    mutated_value,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_stale_item_metrics"
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
    invalid_eval_id = adapter_eval_id if invalid_role == "adapter" else base_eval_id
    results_path = (
        project_root
        / ".morpheus/training/evals"
        / invalid_eval_id
        / "eval_results.json"
    )
    results = json.loads(results_path.read_text())
    results["items"][0][field] = mutated_value
    results_path.write_text(json.dumps(results))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    expected_reason = (
        "invalid_eval_metrics"
        if invalid_role == "adapter"
        else "invalid_base_eval_metrics"
    )
    assert gate["reason"] == expected_reason


@pytest.mark.parametrize(
    ("field", "non_boolean_value"),
    [
        ("passed", 1),
        ("hallucinated", 0),
        ("critical_outdated_claim_failure", 0),
    ],
)
def test_activation_refuses_non_boolean_result_item_flags(
    tmp_path,
    field,
    non_boolean_value,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    adapter_id = "adapter_non_boolean_item_flag"
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
    results["items"][0][field] = non_boolean_value
    results_path.write_text(json.dumps(results))

    gate = check_activation_gate(project_root, adapter_id)

    assert gate["allowed"] is False
    assert gate["reason"] == "invalid_eval_metrics"


@pytest.mark.parametrize("invalid_role", ["adapter", "base"])
@pytest.mark.parametrize(
    "coverage_mutation",
    ["missing_category", "total_mismatch", "items_mismatch"],
)
def test_activation_refuses_eval_coverage_that_does_not_match_bound_seed(
    tmp_path,
    invalid_role,
    coverage_mutation,
):
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
    mark_eval_activation_eligible(Path(adapter["eval_dir"]))
    invalid_eval = adapter if invalid_role == "adapter" else base
    results_path = Path(invalid_eval["eval_results_path"])
    results = json.loads(results_path.read_text())
    items = list(results["items"])
    if coverage_mutation == "missing_category":
        removed_category = items[0]["category"]
        items = [item for item in items if item["category"] != removed_category]
        results["items"] = items
        results["metrics"] = eval_module._metrics(items)
    elif coverage_mutation == "total_mismatch":
        duplicate_category = next(
            category
            for category in {item["category"] for item in items}
            if sum(item["category"] == category for item in items) > 1
        )
        remove_index = next(
            index
            for index, item in enumerate(items)
            if item["category"] == duplicate_category
        )
        items.pop(remove_index)
        results["items"] = items
        results["metrics"] = eval_module._metrics(items)
    else:
        results["items"] = items[:-1]
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    gate = check_activation_gate(project_root, train["adapter_id"])

    assert gate["allowed"] is False
    expected_reason = (
        "eval_dataset_coverage_mismatch"
        if invalid_role == "adapter"
        else "base_eval_dataset_coverage_mismatch"
    )
    assert gate["reason"] == expected_reason


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
    if category == "outdated_claim_correction":
        assert gate["reason"] == "critical_outdated_claim_failure"
    else:
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
    dataset_manifests = [
        path
        for path in project_root.glob(".morpheus/training/datasets/*/manifest.json")
        if json.loads(path.read_text()).get("dataset_id") == dataset_id
    ]
    assert len(dataset_manifests) == 1
    dataset_manifest_path = dataset_manifests[0]
    dataset_binding_sha256 = json.loads(
        dataset_manifest_path.read_text()
    )["dataset_binding_sha256"]
    if adapter_id is not None:
        adapter_dir = project_root / ".morpheus/training/adapters" / adapter_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_manifest_path = adapter_dir / "adapter_manifest.json"
        if not adapter_manifest_path.exists():
            adapter_manifest_path.write_text(json.dumps({
                "adapter_id": adapter_id,
                "dataset_id": dataset_id,
                "dataset_binding_sha256": dataset_binding_sha256,
                "status": "planned",
            }))
    eval_dir = project_root / ".morpheus/training/evals" / eval_id
    eval_dir.mkdir(parents=True)
    seed_items = read_jsonl(dataset_manifest_path.parent / "eval.seed.jsonl")
    regression_written = False
    result_items = []
    for item in seed_items:
        category = item["category"]
        passed = not (
            category == regressed_category
            and not regression_written
        )
        if not passed:
            regression_written = True
        result_items.append({
            "category": category,
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "source_candidate_id": item.get("source_candidate_id"),
            "source_path": item.get("source_path"),
            "kind": item["kind"],
            "passed": passed,
            "hallucinated": False,
            "critical_outdated_claim_failure": (
                category == "outdated_claim_correction" and not passed
            ),
        })
    assert regressed_category is None or regression_written
    config = {
        "eval_id": eval_id,
        "dataset_id": dataset_id,
        "dataset_binding_sha256": dataset_binding_sha256,
        "adapter_id": adapter_id,
        "base_only": base_only,
        "activation_eligible": True,
        "dry_run": False,
        "evaluation_mode": "heldout_external",
        "provider": {"name": "external-heldout"},
        "evaluator": dict(eval_module._ACTIVATION_EVALUATOR),
    }
    results = {
        **config,
        "metrics": eval_module._metrics(result_items),
        "items": result_items,
    }
    (eval_dir / "eval_config.json").write_text(json.dumps(config))
    (eval_dir / "eval_results.json").write_text(json.dumps(results))
    if adapter_id is not None:
        register_test_adapter_weights(project_root, adapter_id)
    write_trusted_activation_eval_receipt(eval_dir)


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
        results["dataset_binding_sha256"] = "f" * 64
        results_path.write_text(json.dumps(results))
    elif mutation == "mismatched_binding":
        for path in (config_path, results_path):
            payload = json.loads(path.read_text())
            payload["dataset_binding_sha256"] = "f" * 64
            path.write_text(json.dumps(payload))
    else:  # pragma: no cover - test helper guard.
        raise AssertionError(f"Unknown eval mutation: {mutation}")
