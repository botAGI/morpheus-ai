import json
from pathlib import Path
import shutil

import pytest
from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.benchmark import write_benchmark_report
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.dataset_validation import dataset_binding_sha256
from morpheus.core.learning.eval import run_learning_eval
from morpheus.core.learning.lab import run_autonomous_lab
from morpheus.core.learning.quality import build_quality_report, write_quality_report
from morpheus.core.learning.registry import latest_effective_dataset, learning_status
from morpheus.core.learning.train import plan_training_run
from morpheus.core.semantic.review import ReviewStore
from tests.test_learning_dataset import copy_learning_project
from tests.test_learning_lab import copy_autonomous_repo


def _fixture_lab(project_root: Path) -> dict:
    return run_autonomous_lab(
        project_root,
        backend="fake",
        no_train=True,
        fixture_only=True,
    )


def _strict_json(text: str):
    def reject_constant(value: str):
        raise ValueError(f"non-RFC JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def test_quality_uses_effective_lab_authority_for_training_gate(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)
    lab = _fixture_lab(project_root)
    ReviewStore(project_root).save_candidates([])

    report = build_quality_report(project_root)

    assert report["review"]["candidates_total"] == 0
    assert report["dataset"]["effective_dataset"]["source"] == "lab"
    assert report["dataset"]["latest_dataset_dir"] == str(
        Path(lab["lab_dir"]) / "dataset"
    )
    assert report["dataset"]["validation"]["valid"] is True
    assert report["dataset"]["trainable_candidate_count"] >= 20
    assert report["train_allowed"] is True
    assert "accepted candidates < 20" not in report["train_blockers"]


def test_touching_older_dataset_cannot_override_newer_invalid_lab(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)
    first_lab = _fixture_lab(project_root)
    root_store = ReviewStore(project_root)
    root_store.save_candidates(
        ReviewStore(Path(first_lab["lab_dir"]) / "workspace").load_candidates()
    )
    standalone = build_learning_dataset(project_root)
    newer_lab_dir = (
        project_root / ".morpheus/lab/lab_29990101T000000000000Z"
    )
    shutil.copytree(Path(first_lab["lab_dir"]), newer_lab_dir)
    manifest_path = newer_lab_dir / "dataset/manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["lab_id"] = newer_lab_dir.name
    manifest["provenance"]["source_root"] = str(newer_lab_dir / "workspace")
    manifest["dataset_binding_sha256"] = dataset_binding_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    newer_store = ReviewStore(newer_lab_dir / "workspace")
    candidates = newer_store.load_candidates()
    candidates[0] = candidates[0].model_copy(update={"status": "pending"})
    newer_store.save_candidates(candidates)

    standalone_dir = Path(standalone["dataset_dir"])
    (standalone_dir / "unrelated.tmp").write_text("changes only directory mtime\n")
    effective = latest_effective_dataset(project_root)

    assert effective["source"] == "lab"
    assert effective["dataset_dir"] == str(newer_lab_dir / "dataset")
    assert effective["trainable"] is False
    assert "review_snapshot_changed" in effective["validation"]["blockers"]


def test_malformed_standalone_registry_identity_is_selected_but_not_trainable(
    tmp_path,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    original_dir = Path(dataset["dataset_dir"])
    imported_dir = original_dir.parent / "imported_dataset"
    shutil.copytree(original_dir, imported_dir)
    manifest_path = imported_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_id"] = imported_dir.name
    manifest["dataset_binding_sha256"] = dataset_binding_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    effective = latest_effective_dataset(project_root)

    assert effective["dataset_dir"] == str(imported_dir)
    assert effective["trainable"] is False
    assert "dataset_id_invalid" in effective["validation"]["blockers"]
    assert "dataset_registry_identity_invalid" in effective["validation"]["blockers"]
    with pytest.raises(ValueError, match="dataset_id_invalid"):
        plan_training_run(project_root, dry_run=True)


def test_malformed_lab_registry_identity_is_selected_but_not_trainable(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)
    lab = _fixture_lab(project_root)
    imported_lab_dir = project_root / ".morpheus/lab/lab_imported"
    shutil.copytree(Path(lab["lab_dir"]), imported_lab_dir)
    manifest_path = imported_lab_dir / "dataset/manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["lab_id"] = imported_lab_dir.name
    manifest["provenance"]["source_root"] = str(imported_lab_dir / "workspace")
    manifest["dataset_binding_sha256"] = dataset_binding_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    effective = latest_effective_dataset(project_root)

    assert effective["source"] == "lab"
    assert effective["dataset_dir"] == str(imported_lab_dir / "dataset")
    assert effective["trainable"] is False
    assert "dataset_registry_identity_invalid" in effective["validation"]["blockers"]


def test_missing_newest_standalone_manifest_blocks_older_dataset(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    corrupt_dir = (
        project_root
        / ".morpheus/training/datasets/29990101T000000000000Z"
    )
    corrupt_dir.mkdir(parents=True)

    effective = latest_effective_dataset(project_root)

    assert effective["dataset_dir"] == str(corrupt_dir)
    assert effective["trainable"] is False
    assert "dataset_manifest_invalid" in effective["validation"]["blockers"]


def test_missing_latest_lab_manifest_blocks_older_standalone(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)
    lab = _fixture_lab(project_root)
    lab_dir = Path(lab["lab_dir"])
    root_store = ReviewStore(project_root)
    root_store.save_candidates(ReviewStore(lab_dir / "workspace").load_candidates())
    standalone = build_learning_dataset(project_root)
    newer_lab_dir = project_root / ".morpheus/lab/lab_29990101T000000000000Z"
    shutil.copytree(lab_dir, newer_lab_dir)
    newer_dataset_dir = newer_lab_dir / "dataset"
    newer_dataset_dir.chmod(0o755)
    (newer_dataset_dir / "manifest.json").unlink()

    effective = latest_effective_dataset(project_root)

    assert effective["source"] == "lab"
    assert effective["dataset_dir"] == str(newer_lab_dir / "dataset")
    assert effective["dataset_dir"] != standalone["dataset_dir"]
    assert effective["trainable"] is False
    assert "dataset_manifest_invalid" in effective["validation"]["blockers"]


def test_newest_lab_without_summary_blocks_older_standalone(tmp_path):
    project_root = copy_learning_project(tmp_path)
    standalone = build_learning_dataset(project_root)
    newest_lab = project_root / ".morpheus/lab/lab_29990101T000000000000Z"
    newest_lab.mkdir(parents=True)

    effective = latest_effective_dataset(project_root)
    status = learning_status(project_root)
    quality = build_quality_report(project_root)
    human_status = CliRunner().invoke(app, ["learn", "status", str(project_root)])

    assert effective["source"] == "lab"
    assert effective["dataset_dir"] == str(newest_lab / "dataset")
    assert effective["dataset_dir"] != standalone["dataset_dir"]
    assert effective["trainable"] is False
    assert status["latest_lab"]["validation_error"] == "lab_summary_missing"
    assert quality["dataset"]["effective_dataset"]["dataset_dir"] == str(
        newest_lab / "dataset"
    )
    assert quality["train_allowed"] is False
    assert human_status.exit_code == 0
    assert "latest lab: invalid" in human_status.stdout


def test_deleted_newest_lab_summary_blocks_older_standalone(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)
    first_lab = _fixture_lab(project_root)
    ReviewStore(project_root).save_candidates(
        ReviewStore(Path(first_lab["lab_dir"]) / "workspace").load_candidates()
    )
    standalone = build_learning_dataset(project_root)
    newest_lab = project_root / ".morpheus/lab/lab_29990101T000000000000Z"
    shutil.copytree(Path(first_lab["lab_dir"]), newest_lab)
    (newest_lab / "lab_summary.json").unlink()

    effective = latest_effective_dataset(project_root)
    status = learning_status(project_root)

    assert effective["source"] == "lab"
    assert effective["dataset_dir"] == str(newest_lab / "dataset")
    assert effective["dataset_dir"] != standalone["dataset_dir"]
    assert effective["trainable"] is False
    assert status["latest_lab"]["validation_error"] == "lab_summary_missing"


@pytest.mark.parametrize("entry_kind", ["symlink", "file"])
def test_unsafe_newest_lab_entry_blocks_without_status_or_quality_crash(
    tmp_path,
    entry_kind,
):
    project_root = copy_learning_project(tmp_path)
    standalone = build_learning_dataset(project_root)
    newest_lab = project_root / ".morpheus/lab/lab_29990101T000000000000Z"
    newest_lab.parent.mkdir(parents=True, exist_ok=True)
    if entry_kind == "symlink":
        external = tmp_path / "external-lab"
        external.mkdir()
        newest_lab.symlink_to(external, target_is_directory=True)
    else:
        newest_lab.write_text("not a lab directory\n")

    effective = latest_effective_dataset(project_root)
    status = learning_status(project_root)
    quality = build_quality_report(project_root)
    human_status = CliRunner().invoke(app, ["learn", "status", str(project_root)])

    assert effective["source"] == "lab"
    assert effective["dataset_dir"] == str(newest_lab / "dataset")
    assert effective["dataset_dir"] != standalone["dataset_dir"]
    assert effective["trainable"] is False
    assert status["latest_lab"]["validation_error"] == "lab_registry_entry_invalid"
    assert quality["train_allowed"] is False
    assert human_status.exit_code == 0
    assert "latest lab: invalid" in human_status.stdout


@pytest.mark.parametrize(
    "invalid_count",
    [
        pytest.param({}, id="object"),
        pytest.param("many", id="string"),
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(True, id="bool"),
    ],
)
def test_corrupt_v2_manifest_count_is_selected_but_diagnosable(
    tmp_path,
    invalid_count,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    dataset_dir = Path(dataset["dataset_dir"])
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["examples_count"] = invalid_count
    manifest["dataset_binding_sha256"] = dataset_binding_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    effective = latest_effective_dataset(project_root)
    quality_result = write_quality_report(project_root)
    quality = quality_result["report"]
    status = learning_status(project_root)
    benchmark = write_benchmark_report(project_root)
    human_status = CliRunner().invoke(app, ["learn", "status", str(project_root)])
    machine_status = CliRunner().invoke(
        app,
        ["learn", "status", str(project_root), "--json"],
    )

    assert effective["dataset_dir"] == str(dataset_dir)
    assert effective["examples_count"] == 0
    assert effective["trainable"] is False
    assert "dataset_manifest_fields_invalid" in effective["validation"]["blockers"]
    assert quality["dataset"]["effective_dataset"]["dataset_dir"] == str(dataset_dir)
    assert quality["dataset"]["latest_manifest"] is None
    assert quality["dataset"]["trainable_candidate_count"] >= 0
    assert quality["train_allowed"] is False
    assert quality["benchmark_allowed"] is False
    assert status["latest_standalone_dataset"]["examples_count"] == 0
    assert status["latest_standalone_dataset"]["validation"]["valid"] is False
    assert status["latest_manifest"] is None
    assert benchmark["examples_count"] == 0
    assert benchmark["benchmark_allowed"] is False
    assert human_status.exit_code == 0
    assert "latest standalone dataset: invalid" in human_status.stdout
    assert machine_status.exit_code == 0
    assert _strict_json(machine_status.stdout)["latest_manifest"] is None
    assert _strict_json(Path(quality_result["json_path"]).read_text())["dataset"][
        "latest_manifest"
    ] is None
    assert _strict_json(Path(benchmark["benchmark_report_path"]).read_text())[
        "quality_report"
    ]["dataset"]["latest_manifest"] is None
    with pytest.raises(ValueError):
        plan_training_run(project_root, dry_run=True)
    with pytest.raises(ValueError):
        run_learning_eval(project_root, base_only=True)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        pytest.param("class_counts", [], id="class-counts-list"),
        pytest.param(
            "route_counts",
            {"adapter_training": "many"},
            id="route-count-string",
        ),
        pytest.param("source_paths", {}, id="source-paths-object"),
    ],
)
def test_corrupt_v2_manifest_collection_shape_is_fail_closed(
    tmp_path,
    field,
    invalid_value,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    dataset_dir = Path(dataset["dataset_dir"])
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = invalid_value
    manifest["dataset_binding_sha256"] = dataset_binding_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    effective = latest_effective_dataset(project_root)
    quality = build_quality_report(project_root)
    benchmark = write_benchmark_report(project_root)
    human_status = CliRunner().invoke(app, ["learn", "status", str(project_root)])

    assert effective["dataset_dir"] == str(dataset_dir)
    assert effective["trainable"] is False
    assert "dataset_manifest_fields_invalid" in effective["validation"]["blockers"]
    assert quality["train_allowed"] is False
    assert quality["benchmark_allowed"] is False
    assert benchmark["benchmark_allowed"] is False
    assert human_status.exit_code == 0
    assert "latest standalone dataset: invalid" in human_status.stdout
    with pytest.raises(ValueError, match="dataset_manifest_fields_invalid"):
        plan_training_run(project_root, dry_run=True)
    with pytest.raises(ValueError, match="dataset_manifest_fields_invalid"):
        run_learning_eval(project_root, base_only=True)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_value_anywhere_in_v2_manifest_is_not_public_json(
    tmp_path,
    nonfinite,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    manifest_path = Path(dataset["dataset_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["extension_metadata"] = {"score": nonfinite}
    manifest["dataset_binding_sha256"] = dataset_binding_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    effective = latest_effective_dataset(project_root)
    quality_result = write_quality_report(project_root)
    status_result = CliRunner().invoke(
        app,
        ["learn", "status", str(project_root), "--json"],
    )

    assert "dataset_manifest_fields_invalid" in effective["validation"]["blockers"]
    assert quality_result["report"]["dataset"]["latest_manifest"] is None
    assert _strict_json(Path(quality_result["json_path"]).read_text())["dataset"][
        "latest_manifest"
    ] is None
    assert status_result.exit_code == 0
    assert _strict_json(status_result.stdout)["latest_manifest"] is None


@pytest.mark.parametrize("invalid_size", [True, -1, "1"])
def test_v2_manifest_requires_exact_nonnegative_artifact_sizes(
    tmp_path,
    invalid_size,
):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    manifest_path = Path(dataset["dataset_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected_file = manifest["selected_dataset_file"]
    manifest["artifacts"][selected_file]["size_bytes"] = invalid_size
    manifest["dataset_binding_sha256"] = dataset_binding_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    effective = latest_effective_dataset(project_root)
    quality = build_quality_report(project_root)

    assert effective["trainable"] is False
    assert "dataset_manifest_fields_invalid" in effective["validation"]["blockers"]
    assert quality["dataset"]["latest_manifest"] is None
    assert quality["train_allowed"] is False


def test_human_status_marks_empty_object_manifest_invalid(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    manifest_path = Path(dataset["dataset_dir"]) / "manifest.json"
    manifest_path.write_text("{}\n")

    result = CliRunner().invoke(app, ["learn", "status", str(project_root)])

    assert result.exit_code == 0
    assert "latest standalone dataset: invalid" in result.stdout


@pytest.mark.parametrize("summary_text", ["[]\n", "{}\n", "not json\n"])
def test_human_status_marks_invalid_latest_lab_summary_invalid(
    tmp_path,
    summary_text,
):
    project_root = copy_autonomous_repo(tmp_path)
    lab = _fixture_lab(project_root)
    summary_path = Path(lab["lab_dir"]) / "lab_summary.json"
    summary_path.write_text(summary_text)

    status = learning_status(project_root)
    result = CliRunner().invoke(app, ["learn", "status", str(project_root)])

    assert status["has_labs"] is True
    assert status["latest_lab"]["invalid"] is True
    assert result.exit_code == 0
    assert "latest lab: invalid" in result.stdout
