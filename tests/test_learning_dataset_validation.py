import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

import morpheus.core.learning.dataset as dataset_module
import morpheus.core.learning.quality as quality_module
from morpheus.core.config import MorpheusConfig
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.dataset_validation import (
    _validate_manifest_semantics,
    dataset_binding_sha256,
    validate_dataset,
)
from morpheus.core.learning.examples import (
    chat_examples_from_instruction,
    sharegpt_examples_from_instruction,
)
from morpheus.core.learning.eval import check_activation_gate, run_learning_eval
from morpheus.core.learning.lab import run_autonomous_lab
from morpheus.core.learning.quality import build_quality_report
from morpheus.core.learning.registry import latest_effective_dataset
from morpheus.core.learning.team import run_team_learning_loop
from morpheus.core.learning.train import plan_training_run
from morpheus.core.semantic.review import ReviewStore, apply_accepted_candidates
from morpheus.core.semantic.routing import route_candidate
from tests.test_learning_dataset import copy_learning_project
from tests.test_learning_eval import mark_eval_activation_eligible
from tests.test_learning_lab import copy_autonomous_repo


def _rewrite_manifest(dataset_dir: Path, manifest: dict) -> None:
    manifest["dataset_binding_sha256"] = dataset_binding_sha256(manifest)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _increment_count_map(manifest: dict, field: str) -> None:
    counts = manifest[field]
    key = next(iter(counts), "invented")
    counts[key] = counts.get(key, 0) + 1


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rewrite_bound_jsonl(dataset_dir: Path, manifest: dict, name: str, rows: list[dict]) -> None:
    path = dataset_dir / name
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    data = path.read_bytes()
    manifest["artifacts"][name] = {
        "sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    if manifest["selected_dataset_file"] == name:
        manifest["dataset_sha256"] = manifest["artifacts"][name]["sha256"]


def _forge_candidate_promotion(
    project_root: Path,
    dataset_dir: Path,
    manifest: dict,
    promoted,
) -> None:
    promoted = route_candidate(promoted)
    eval_rows = _read_jsonl(dataset_dir / "eval.seed.jsonl")
    replaced_id = eval_rows[0]["source_candidate_id"]
    eval_rows[0].update({
        "source_candidate_id": promoted.id,
        "source_path": promoted.source_path,
        "semantic_class": promoted.semantic_class,
        "trainability_status": promoted.trainability_status,
        "memory_route": promoted.memory_route,
    })
    skipped_rows = _read_jsonl(dataset_dir / "skipped.jsonl")
    promoted_skip = next(
        row for row in skipped_rows if row["candidate_id"] == promoted.id
    )
    promoted_skip["candidate_id"] = replaced_id
    _rewrite_bound_jsonl(dataset_dir, manifest, "eval.seed.jsonl", eval_rows)
    _rewrite_bound_jsonl(dataset_dir, manifest, "skipped.jsonl", skipped_rows)

    eligible_rows = [
        row for row in eval_rows if row.get("source_candidate_id") is not None
    ]
    source_path = project_root / promoted.source_path
    if source_path.is_file():
        manifest["source_hashes"][promoted.source_path] = sha256(
            source_path.read_bytes()
        ).hexdigest()
    manifest.update({
        "source_candidate_ids": sorted(
            row["source_candidate_id"] for row in eligible_rows
        ),
        "source_paths": sorted({row["source_path"] for row in eligible_rows}),
        "class_counts": dict(Counter(
            row["semantic_class"] for row in eligible_rows
        )),
        "trainability_counts": dict(Counter(
            row["trainability_status"] for row in eligible_rows
        )),
        "route_counts": dict(Counter(
            row["memory_route"] for row in eligible_rows
        )),
        "trainable_candidate_count": sum(
            row["memory_route"] == "adapter_training"
            for row in eligible_rows
        ),
    })
    _rewrite_manifest(dataset_dir, manifest)


@pytest.mark.parametrize(
    "candidate_id",
    ["c_rejected", "c_pending", "c_inferred", "c_secret", "c_ignored"],
)
def test_dataset_cannot_promote_ineligible_review_candidate(
    tmp_path,
    candidate_id,
):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    candidates = {
        candidate.id: candidate
        for candidate in ReviewStore(project_root).load_candidates()
    }
    _forge_candidate_promotion(
        project_root,
        dataset_dir,
        manifest,
        candidates[candidate_id],
    )

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "dataset_source_authority_mismatch" in validation["blockers"]
    with pytest.raises(ValueError, match="dataset_source_authority_mismatch"):
        plan_training_run(project_root, dry_run=True)


def test_dataset_cannot_promote_candidate_with_stale_source_hash(tmp_path):
    project_root = copy_learning_project(tmp_path)
    store = ReviewStore(project_root)
    candidates = store.load_candidates()
    stale = next(candidate for candidate in candidates if candidate.id == "c_current")
    stale = stale.model_copy(update={"source_sha256": "0" * 64})
    store.save_candidates([
        stale if candidate.id == stale.id else candidate
        for candidate in candidates
    ])
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    _forge_candidate_promotion(project_root, dataset_dir, manifest, stale)

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "dataset_source_authority_mismatch" in validation["blockers"]


def test_dataset_cannot_promote_team_projection_mismatch(tmp_path):
    project_root = copy_learning_project(tmp_path)
    run_team_learning_loop(project_root, [{
        "source_type": "human_correction",
        "external_id": "validator-red-team",
        "claim": "Morpheus trains raw project Markdown directly.",
        "correction": "Morpheus trains only accepted source-backed candidates.",
    }])
    store = ReviewStore(project_root)
    team_candidate = next(
        candidate
        for candidate in store.load_candidates()
        if candidate.provider.get("name") == "morpheus-team-loop"
    )
    team_candidate = store.accept(team_candidate.id, reviewed_by="tester")
    team_candidate = team_candidate.model_copy(update={
        "correction_text": "A replacement absent from signed team evidence.",
    })
    candidates = store.load_candidates()
    store.save_candidates([
        team_candidate if candidate.id == team_candidate.id else candidate
        for candidate in candidates
    ])
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    _forge_candidate_promotion(
        project_root,
        dataset_dir,
        manifest,
        team_candidate,
    )

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "dataset_source_authority_mismatch" in validation["blockers"]


@pytest.mark.parametrize(
    "artifact_name",
    ["dataset.instruction.jsonl", "dataset.sharegpt.jsonl", "train.jsonl"],
)
def test_dataset_cannot_inject_arbitrary_generated_row(tmp_path, artifact_name):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    rows = _read_jsonl(dataset_dir / artifact_name)
    rows[0] = {
        **rows[0],
        "injected_raw_markdown": "Unreviewed raw text must never become memory.",
    }
    _rewrite_bound_jsonl(dataset_dir, manifest, artifact_name, rows)
    _rewrite_manifest(dataset_dir, manifest)

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "dataset_generated_artifacts_mismatch" in validation["blockers"]
    with pytest.raises(ValueError, match="dataset_generated_artifacts_mismatch"):
        plan_training_run(project_root, dry_run=True)


def test_active_state_dataset_cannot_inject_arbitrary_generated_row(tmp_path):
    project_root = copy_learning_project(tmp_path)
    MorpheusConfig(project_root=project_root).init_default()
    apply_accepted_candidates(project_root)
    result = build_learning_dataset(project_root, source="active-state")
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    rows = _read_jsonl(dataset_dir / "dataset.instruction.jsonl")
    rows[0] = {**rows[0], "injected_raw_markdown": "unsigned active data"}
    _rewrite_bound_jsonl(
        dataset_dir,
        manifest,
        "dataset.instruction.jsonl",
        rows,
    )
    _rewrite_manifest(dataset_dir, manifest)

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "dataset_generated_artifacts_mismatch" in validation["blockers"]
    with pytest.raises(ValueError, match="dataset_generated_artifacts_mismatch"):
        plan_training_run(project_root, dry_run=True)


def test_active_state_dataset_cannot_forge_candidate_partition(tmp_path):
    project_root = copy_learning_project(tmp_path)
    MorpheusConfig(project_root=project_root).init_default()
    apply_accepted_candidates(project_root)
    result = build_learning_dataset(project_root, source="active-state")
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    rows = _read_jsonl(dataset_dir / "eval.seed.jsonl")
    replaced_id = rows[0]["source_candidate_id"]
    rows[0]["source_candidate_id"] = "active_forged"
    manifest["source_candidate_ids"] = sorted(
        "active_forged" if candidate_id == replaced_id else candidate_id
        for candidate_id in manifest["source_candidate_ids"]
    )
    _rewrite_bound_jsonl(dataset_dir, manifest, "eval.seed.jsonl", rows)
    _rewrite_manifest(dataset_dir, manifest)

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "dataset_source_authority_mismatch" in validation["blockers"]
    with pytest.raises(ValueError, match="dataset_source_authority_mismatch"):
        plan_training_run(project_root, dry_run=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate_count",
        "trainable_candidate_count",
        "examples_count",
        "eval_items_count",
        "heldout_eval_items_count",
        "skipped_count",
        "split_counts",
        "class_counts",
        "trainability_counts",
        "route_counts",
        "source_candidate_ids",
        "source_paths",
        "selected_format",
        "selected_dataset_file",
        "format_version",
    ],
)
def test_recomputed_manifest_semantic_lie_is_invalid(tmp_path, mutation):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())

    if mutation in {
        "candidate_count",
        "trainable_candidate_count",
        "examples_count",
        "eval_items_count",
        "heldout_eval_items_count",
        "skipped_count",
    }:
        manifest[mutation] += 1
    elif mutation in {
        "split_counts",
        "class_counts",
        "trainability_counts",
        "route_counts",
    }:
        _increment_count_map(manifest, mutation)
    elif mutation == "source_candidate_ids":
        manifest[mutation] = [*manifest[mutation], "candidate_invented"]
    elif mutation == "source_paths":
        manifest[mutation] = [manifest[mutation][0], manifest[mutation][0]]
    elif mutation == "selected_format":
        manifest[mutation] = "sharegpt"
    elif mutation == "selected_dataset_file":
        manifest[mutation] = "dataset.sharegpt.jsonl"
        manifest["dataset_sha256"] = manifest["artifacts"][
            "dataset.sharegpt.jsonl"
        ]["sha256"]
    else:
        manifest[mutation] = "morpheus-sharegpt/1"
    _rewrite_manifest(dataset_dir, manifest)

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "dataset_manifest_semantics_invalid" in validation["blockers"]


@pytest.mark.parametrize(
    "artifact_text",
    [
        "[]\n",
        "not json\n",
        '{"instruction": NaN}\n',
    ],
    ids=["non-object", "malformed", "non-finite"],
)
def test_manifest_bound_jsonl_must_contain_strict_objects(tmp_path, artifact_text):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    selected = dataset_dir / manifest["selected_dataset_file"]
    selected.write_text(artifact_text)
    selected_bytes = selected.read_bytes()
    selected_metadata = manifest["artifacts"][manifest["selected_dataset_file"]]
    selected_metadata["sha256"] = sha256(selected_bytes).hexdigest()
    selected_metadata["size_bytes"] = len(selected_bytes)
    manifest["dataset_sha256"] = selected_metadata["sha256"]
    _rewrite_manifest(dataset_dir, manifest)

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "dataset_manifest_semantics_invalid" in validation["blockers"]


def test_manifest_semantics_rejects_source_unbound_training_row(tmp_path):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    artifact_rows = {
        name: _read_jsonl(dataset_dir / name)
        for name in {
            "dataset.instruction.jsonl",
            "dataset.sharegpt.jsonl",
            "eval.heldout.jsonl",
            "eval.seed.jsonl",
            "skipped.jsonl",
            "test.jsonl",
            "train.jsonl",
            "valid.jsonl",
        }
    }
    unbound = {
        "instruction": "Apply a synthetic project rule.",
        "input": "Can this unreviewed claim enter training?",
        "output": "No.",
        "metadata": {
            "source_candidate_id": None,
            "source_path": None,
            "line_start": None,
            "line_end": None,
            "evidence_sha256": None,
            "memory_route": None,
            "example_type": "unsupported_claim_refusal",
        },
    }
    instruction_rows = [*artifact_rows["dataset.instruction.jsonl"], unbound]
    chat_rows = chat_examples_from_instruction(instruction_rows)
    split_rows = dataset_module._split_chat_rows(chat_rows)
    artifact_rows.update({
        "dataset.instruction.jsonl": instruction_rows,
        "dataset.sharegpt.jsonl": sharegpt_examples_from_instruction(
            instruction_rows
        ),
        "train.jsonl": split_rows["train"],
        "valid.jsonl": split_rows["valid"],
        "test.jsonl": split_rows["test"],
    })
    manifest["examples_count"] = len(instruction_rows)
    manifest["split_counts"] = {
        split: len(rows)
        for split, rows in split_rows.items()
    }

    semantics_valid, _ = _validate_manifest_semantics(manifest, artifact_rows)

    assert semantics_valid is False


def test_manifest_semantics_rejects_spoofed_training_span_and_evidence(tmp_path):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    artifact_rows = {
        name: _read_jsonl(dataset_dir / name)
        for name in {
            "dataset.instruction.jsonl",
            "dataset.sharegpt.jsonl",
            "eval.heldout.jsonl",
            "eval.seed.jsonl",
            "skipped.jsonl",
            "test.jsonl",
            "train.jsonl",
            "valid.jsonl",
        }
    }
    candidate_id = artifact_rows["dataset.instruction.jsonl"][0]["metadata"][
        "source_candidate_id"
    ]
    for artifact_name in (
        "dataset.instruction.jsonl",
        "dataset.sharegpt.jsonl",
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
    ):
        for row in artifact_rows[artifact_name]:
            metadata = row.get("metadata")
            if metadata.get("source_candidate_id") == candidate_id:
                metadata.update({
                    "line_start": 999_999,
                    "line_end": 999_999,
                    "evidence_sha256": "f" * 64,
                })

    semantics_valid, _ = _validate_manifest_semantics(manifest, artifact_rows)

    assert semantics_valid is False


def test_semantic_manifest_lie_blocks_all_dataset_consumers(tmp_path):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    manifest["examples_count"] += 1
    _rewrite_manifest(dataset_dir, manifest)

    effective = latest_effective_dataset(project_root)
    quality = build_quality_report(project_root)

    assert effective["trainable"] is False
    assert "dataset_manifest_semantics_invalid" in effective["validation"]["blockers"]
    assert quality["train_allowed"] is False
    assert quality["benchmark_allowed"] is False
    assert quality["dataset"]["latest_manifest"] is None
    with pytest.raises(ValueError, match="dataset_manifest_semantics_invalid"):
        plan_training_run(project_root, dry_run=True)
    with pytest.raises(ValueError, match="dataset_manifest_semantics_invalid"):
        run_learning_eval(project_root, base_only=True, dry_run=True)


def test_active_state_projection_hashes_excerpt_without_reading_unsafe_path(
    tmp_path,
    monkeypatch,
):
    authority = {
        "state": {
            "receipt_id": "rcpt_projection",
            "claims": [
                {
                    "id": "clm_unsafe",
                    "category": "decision",
                    "excerpt": "DECISION: keep active-state paths scoped",
                    "status": "active",
                }
            ],
        },
        "evidence_rows": [
            {
                "claim_id": "clm_unsafe",
                "path": "../../outside-project.md",
                "source_sha256": "a" * 64,
                "excerpt": "DECISION: keep active-state paths scoped",
                "excerpt_sha256": "missing",
                "line_start": 1,
                "line_end": 1,
            }
        ],
    }

    def unexpected_file_hash(_path):
        raise AssertionError("active-state projection must not read an unscoped path")

    monkeypatch.setattr(dataset_module, "compute_sha256_file", unexpected_file_hash)

    candidates = dataset_module._active_state_candidates(tmp_path, authority)

    assert len(candidates) == 1
    assert candidates[0].source_path == "../../outside-project.md"
    assert candidates[0].evidence_sha256 == sha256(
        b"DECISION: keep active-state paths scoped"
    ).hexdigest()


def test_dataset_manifest_binds_review_state_and_generated_artifacts(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root)

    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    provenance = manifest["provenance"]
    assert manifest["format_versions"]["manifest"] == "morpheus-learning-manifest/2"
    assert provenance["schema"] == "morpheus-dataset-provenance/1"
    assert provenance["source_scope"] == "accepted_review_live"
    assert provenance["routing_policy_version"] == "morpheus-memory-routing/1"
    assert provenance["review_snapshot"]["schema"] == "morpheus-review-snapshot/1"
    assert provenance["review_snapshot"]["candidate_count"] == 11
    assert len(provenance["review_snapshot"]["sha256"]) == 64
    assert set(manifest["artifacts"]) == {
        "dataset.instruction.jsonl",
        "dataset.sharegpt.jsonl",
        "eval.heldout.jsonl",
        "eval.seed.jsonl",
        "skipped.jsonl",
        "test.jsonl",
        "train.jsonl",
        "valid.jsonl",
    }
    assert manifest["selected_dataset_file"] == "dataset.instruction.jsonl"
    assert len(manifest["dataset_binding_sha256"]) == 64
    validation = validate_dataset(project_root, dataset_dir)
    assert validation["valid"] is True
    assert validation["blockers"] == []
    assert validation["dataset_binding_sha256"] == manifest["dataset_binding_sha256"]


def test_review_snapshot_is_order_independent(tmp_path):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    candidates_path = project_root / ".morpheus/review/semantic_candidates.jsonl"
    rows = candidates_path.read_text().splitlines()
    candidates_path.write_text("\n".join(reversed(rows)) + "\n")

    validation = validate_dataset(project_root, Path(result["dataset_dir"]))

    assert validation["valid"] is True
    assert validation["review_snapshot"]["matches"] is True


@pytest.mark.parametrize("transition", ["rejected", "pending", "tampered"])
def test_review_transition_invalidates_dataset_before_train_or_eval(
    tmp_path,
    monkeypatch,
    transition,
):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    store = ReviewStore(project_root)
    if transition == "rejected":
        store.reject("c_current", reason="revoked after dataset build")
    else:
        candidates = store.load_candidates()
        index = next(index for index, item in enumerate(candidates) if item.id == "c_current")
        if transition == "pending":
            candidates[index] = candidates[index].model_copy(update={"status": "pending"})
        else:
            candidates[index] = candidates[index].model_copy(update={
                "claim": "Tampered claim that was not part of the compiled review snapshot."
            })
        store.save_candidates(candidates)

    dataset_dir = Path(result["dataset_dir"])
    validation = validate_dataset(project_root, dataset_dir)
    assert validation["valid"] is False
    assert "review_snapshot_changed" in validation["blockers"]

    monkeypatch.setattr(quality_module, "TRAIN_MIN_ACCEPTED", 0)
    monkeypatch.setattr(quality_module, "TRAIN_MIN_EXAMPLES", 0)
    report = build_quality_report(project_root)
    assert report["train_allowed"] is False
    assert report["benchmark_allowed"] is False
    assert "dataset review snapshot changed" in report["train_blockers"]
    assert "dataset review snapshot changed" in report["benchmark_blockers"]

    runs_root = project_root / ".morpheus/training/runs"
    adapters_root = project_root / ".morpheus/training/adapters"
    evals_root = project_root / ".morpheus/training/evals"
    with pytest.raises(ValueError, match="review_snapshot_changed"):
        plan_training_run(project_root, dry_run=True)
    with pytest.raises(ValueError, match="review_snapshot_changed"):
        run_learning_eval(
            project_root,
            base_only=True,
            dry_run=True,
            dataset_id=result["dataset_id"],
        )
    assert not runs_root.exists()
    assert not adapters_root.exists()
    assert not evals_root.exists()


def test_dataset_artifact_tamper_fails_closed_before_training_writes(tmp_path):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    (dataset_dir / "dataset.instruction.jsonl").write_text("{}\n")

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "dataset_artifact_hash_mismatch" in validation["blockers"]
    with pytest.raises(ValueError, match="dataset_artifact_hash_mismatch"):
        plan_training_run(project_root, dry_run=True)
    assert not (project_root / ".morpheus/training/runs").exists()
    assert not (project_root / ".morpheus/training/adapters").exists()


def test_legacy_unbound_manifest_is_visible_but_not_usable(tmp_path):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root)
    dataset_dir = Path(result["dataset_dir"])
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for field in ("provenance", "artifacts", "dataset_binding_sha256"):
        manifest.pop(field)
    manifest["format_versions"]["manifest"] = "morpheus-learning-manifest/1"
    manifest_path.write_text(json.dumps(manifest))

    validation = validate_dataset(project_root, dataset_dir)

    assert validation["valid"] is False
    assert "legacy_unbound_manifest" in validation["blockers"]
    with pytest.raises(ValueError, match="legacy_unbound_manifest"):
        plan_training_run(project_root, dry_run=True)


def test_zero_example_dataset_cannot_be_evaluated(tmp_path):
    project_root = tmp_path / "zero_eval_project"
    review_dir = project_root / ".morpheus/review"
    review_dir.mkdir(parents=True)
    (review_dir / "semantic_candidates.jsonl").write_text("")
    build_learning_dataset(project_root)

    with pytest.raises(ValueError, match="zero examples"):
        run_learning_eval(project_root, base_only=True, dry_run=True)
    assert not (project_root / ".morpheus/training/evals").exists()


def test_dataset_build_rejects_duplicate_review_candidate_ids(tmp_path):
    project_root = copy_learning_project(tmp_path)
    store = ReviewStore(project_root)
    candidates = store.load_candidates()
    store.save_candidates([*candidates, candidates[0]])

    with pytest.raises(ValueError, match="duplicate candidate ids"):
        build_learning_dataset(project_root)


def test_review_change_during_build_publishes_no_dataset_or_external_output(
    tmp_path,
    monkeypatch,
):
    project_root = copy_learning_project(tmp_path)
    output_path = project_root / "exported.jsonl"
    original_check = dataset_module._source_hashes_are_current

    def change_review_after_artifacts(*args, **kwargs):
        store = ReviewStore(project_root)
        candidates = store.load_candidates()
        candidates[0] = candidates[0].model_copy(update={"status": "pending"})
        store.save_candidates(candidates)
        return original_check(*args, **kwargs)

    monkeypatch.setattr(
        dataset_module,
        "_source_hashes_are_current",
        change_review_after_artifacts,
    )

    with pytest.raises(ValueError, match="Review state changed"):
        build_learning_dataset(project_root, output=output_path)

    datasets_dir = project_root / ".morpheus/training/datasets"
    assert not output_path.exists()
    assert not [
        entry
        for entry in datasets_dir.iterdir()
        if not entry.name.startswith(".")
    ]


def test_dataset_output_rejects_final_symlink_without_touching_target(tmp_path):
    project_root = copy_learning_project(tmp_path)
    target = tmp_path / "outside-dataset.jsonl"
    target.write_text("outside sentinel\n")
    output_path = project_root / "exported.jsonl"
    output_path.symlink_to(target)

    with pytest.raises(ValueError, match="Dataset output must not be a symlink"):
        build_learning_dataset(project_root, output=output_path)

    assert output_path.is_symlink()
    assert target.read_text() == "outside sentinel\n"


def test_dataset_skips_noncanonical_posix_source_path_and_remains_valid(tmp_path):
    project_root = copy_learning_project(tmp_path)
    store = ReviewStore(project_root)
    candidates = store.load_candidates()
    store.save_candidates([
        candidate.model_copy(update={"source_path": "./README.md"})
        if candidate.id == "c_current"
        else candidate
        for candidate in candidates
    ])

    result = build_learning_dataset(project_root)

    dataset_dir = Path(result["dataset_dir"])
    skipped = _read_jsonl(dataset_dir / "skipped.jsonl")
    skipped_by_id = {item["candidate_id"]: item["reason"] for item in skipped}
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    validation = validate_dataset(project_root, dataset_dir)
    assert skipped_by_id["c_current"] == "invalid_source_path"
    assert "./README.md" not in manifest["source_paths"]
    assert validation["valid"] is True


def test_activation_revalidates_dataset_after_review_revocation(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    training = plan_training_run(project_root, dry_run=True)
    base_eval = run_learning_eval(project_root, base_only=True, dry_run=True)
    adapter_eval = run_learning_eval(
        project_root,
        adapter_id=training["adapter_id"],
        dry_run=True,
    )
    mark_eval_activation_eligible(Path(base_eval["eval_dir"]))
    mark_eval_activation_eligible(Path(adapter_eval["eval_dir"]))
    ReviewStore(project_root).reject(
        "c_current",
        reason="revoked after evaluation",
    )

    gate = check_activation_gate(project_root, training["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "dataset_not_current"
    assert "review_snapshot_changed" in gate["dataset_blockers"]


def test_adapter_cannot_be_evaluated_on_a_different_dataset_binding(tmp_path):
    project_root = copy_learning_project(tmp_path)
    first_dataset = build_learning_dataset(project_root)
    training = plan_training_run(project_root, dry_run=True)
    second_dataset = build_learning_dataset(project_root)
    assert second_dataset["dataset_id"] != first_dataset["dataset_id"]

    with pytest.raises(ValueError, match="Adapter dataset binding mismatch"):
        run_learning_eval(
            project_root,
            adapter_id=training["adapter_id"],
            dataset_id=second_dataset["dataset_id"],
            dry_run=True,
        )
    assert not (project_root / ".morpheus/training/evals").exists()


def test_activation_rechecks_adapter_manifest_dataset_binding(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    training = plan_training_run(project_root, dry_run=True)
    base_eval = run_learning_eval(project_root, base_only=True, dry_run=True)
    adapter_eval = run_learning_eval(
        project_root,
        adapter_id=training["adapter_id"],
        dry_run=True,
    )
    mark_eval_activation_eligible(Path(base_eval["eval_dir"]))
    mark_eval_activation_eligible(Path(adapter_eval["eval_dir"]))
    manifest_path = (
        project_root
        / ".morpheus/training/adapters"
        / training["adapter_id"]
        / "adapter_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_binding_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    gate = check_activation_gate(project_root, training["adapter_id"])

    assert gate["allowed"] is False
    assert gate["reason"] == "adapter_dataset_binding_mismatch"
    assert "adapter_dataset_binding_mismatch" in gate["adapter_blockers"]


def test_active_state_dataset_binds_state_and_evidence_artifacts(tmp_path):
    project_root = copy_learning_project(tmp_path)
    MorpheusConfig(project_root=project_root).init_default()
    apply_accepted_candidates(project_root)
    result = build_learning_dataset(project_root, source="active-state")
    dataset_dir = Path(result["dataset_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text())

    assert manifest["provenance"]["source_scope"] == "active_state_receipt"
    assert manifest["provenance"]["context_paths"] == [
        ".morpheus/WAKE.md",
        ".morpheus/evidence.jsonl",
        ".morpheus/state.json",
    ]
    assert validate_dataset(project_root, dataset_dir)["valid"] is True

    state_path = project_root / ".morpheus/state.json"
    state = json.loads(state_path.read_text())
    state["receipt_id"] = "rcpt_changed_after_dataset"
    state_path.write_text(json.dumps(state))
    validation = validate_dataset(project_root, dataset_dir)
    assert validation["valid"] is False
    assert "dataset_sources_changed" in validation["blockers"]
    assert "active_state_receipt_invalid" in validation["blockers"]


def test_active_state_change_during_build_publishes_no_dataset(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    MorpheusConfig(project_root=project_root).init_default()
    apply_accepted_candidates(project_root)
    original_capture = dataset_module.capture_active_state_authority
    captures = 0

    def change_before_final_capture(root):
        nonlocal captures
        captures += 1
        if captures == 2:
            state_path = project_root / ".morpheus/state.json"
            state = json.loads(state_path.read_text())
            state["receipt_id"] = "rcpt_concurrent_change"
            state_path.write_text(json.dumps(state))
        return original_capture(root)

    monkeypatch.setattr(
        dataset_module,
        "capture_active_state_authority",
        change_before_final_capture,
    )

    with pytest.raises(ValueError, match="receipt chain invalid"):
        build_learning_dataset(project_root, source="active-state")

    datasets_dir = project_root / ".morpheus/training/datasets"
    assert not [
        entry
        for entry in datasets_dir.iterdir()
        if not entry.name.startswith(".")
    ]


def test_accepted_review_dataset_ignores_unrelated_active_state_receipts(tmp_path):
    project_root = copy_learning_project(tmp_path)
    result = build_learning_dataset(project_root, source="accepted")
    dataset_dir = Path(result["dataset_dir"])
    MorpheusConfig(project_root=project_root).init_default()

    apply_accepted_candidates(project_root)

    validation = validate_dataset(project_root, dataset_dir)
    assert validation["valid"] is True
    assert validation["source_scope"] == "accepted_review_live"


def test_lab_dataset_uses_preserved_workspace_review_snapshot(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)
    lab = run_autonomous_lab(
        project_root,
        backend="fake",
        no_train=True,
        fixture_only=True,
    )
    lab_dir = Path(lab["lab_dir"])
    dataset_dir = lab_dir / "dataset"
    validation = validate_dataset(project_root, dataset_dir)
    assert validation["valid"] is True
    assert validation["source_scope"] == "lab_review_snapshot"

    workspace_store = ReviewStore(lab_dir / "workspace")
    candidates = workspace_store.load_candidates()
    candidates[0] = candidates[0].model_copy(update={"status": "pending"})
    workspace_store.save_candidates(candidates)

    validation = validate_dataset(project_root, dataset_dir)
    assert validation["valid"] is False
    assert "review_snapshot_changed" in validation["blockers"]
    with pytest.raises(ValueError, match="review_snapshot_changed"):
        plan_training_run(project_root, dry_run=True)


def test_newer_invalid_standalone_dataset_does_not_fall_back_to_valid_lab(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)
    lab = run_autonomous_lab(
        project_root,
        backend="fake",
        no_train=True,
        fixture_only=True,
    )
    lab_dir = Path(lab["lab_dir"])
    root_store = ReviewStore(project_root)
    root_store.save_candidates(
        ReviewStore(lab_dir / "workspace").load_candidates()
    )
    build_learning_dataset(project_root)
    root_store.reject(
        root_store.load_candidates()[0].id,
        reason="invalidate newest standalone dataset",
    )

    effective = latest_effective_dataset(project_root)

    assert effective["source"] == "standalone"
    assert effective["trainable"] is False
    assert "review_snapshot_changed" in effective["validation"]["blockers"]
    assert validate_dataset(project_root, lab_dir / "dataset")["valid"] is True
    with pytest.raises(ValueError, match="review_snapshot_changed"):
        plan_training_run(project_root, dry_run=True)
