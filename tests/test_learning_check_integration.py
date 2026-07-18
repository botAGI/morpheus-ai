import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.check import create_training_corrections
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.semantic.review import ReviewStore
from tests.test_check import STALE_INPUT, prepare_private_state, write_launch_repo
from tests.test_learning_dataset import read_jsonl


def test_check_stale_claim_creates_pending_correction_candidate(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        result = runner.invoke(
            app,
            [
                "check",
                "--input",
                str(STALE_INPUT),
                "--local",
                "--create-training-corrections",
                "--json",
            ],
        )

        assert result.exit_code == 1, result.output
        payload = json.loads(result.output)
        assert payload["training_corrections_created"] == 1
        candidates = ReviewStore(Path.cwd()).load_candidates()
        assert len(candidates) == 1
        assert candidates[0].status == "pending"
        assert candidates[0].kind == "outdated_claim"
        assert candidates[0].provider["name"] == "morpheus-check"
        assert candidates[0].provider["source_label"]
        assert candidates[0].memory_route == "stale_archive"
        team_report = json.loads(
            (Path.cwd() / ".morpheus" / "learning" / "team_loop_report.json").read_text()
        )
        assert team_report["input_source_counts"] == {"check_result": 1}
        assert team_report["created_input_receipt_count"] == 1


def test_check_without_correction_flag_writes_no_team_input_receipt(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        result = runner.invoke(
            app,
            ["check", "--input", str(STALE_INPUT), "--local", "--json"],
        )

        assert result.exit_code == 1, result.output
        assert "training_corrections_created" not in json.loads(result.output)
        assert not (Path.cwd() / ".morpheus" / "review" / "team_inputs").exists()
        assert not (Path.cwd() / ".morpheus" / "learning" / "team_loop_report.json").exists()


def test_repeated_check_correction_creation_is_idempotent(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        first = runner.invoke(
            app,
            ["check", "--input", str(STALE_INPUT), "--create-training-corrections", "--json"],
        )
        second = runner.invoke(
            app,
            ["check", "--input", str(STALE_INPUT), "--create-training-corrections", "--json"],
        )

        assert first.exit_code == 1, first.output
        assert second.exit_code == 1, second.output
        assert json.loads(first.output)["training_corrections_created"] == 1
        assert json.loads(second.output)["training_corrections_created"] == 0
        assert len(ReviewStore(Path.cwd()).load_candidates()) == 1
        team_report = json.loads(
            (Path.cwd() / ".morpheus" / "learning" / "team_loop_report.json").read_text()
        )
        assert team_report["created_input_receipt_count"] == 0
        assert team_report["existing_input_receipt_count"] == 1
        assert len(list(
            (Path.cwd() / ".morpheus" / "review" / "team_inputs").glob("*.jsonl")
        )) == 1


def test_check_corrections_for_same_claim_at_different_sources_have_distinct_ids(tmp_path):
    def check_result(path: str) -> dict:
        return {
            "active_state_receipt": "rcpt_test",
            "results": [{
                "claim": "Morpheus is mainly a personal AI agent.",
                "status": "stale",
                "reason": "claim matches outdated project state",
                "evidence": {"path": path, "line_start": 1},
            }],
        }

    first = create_training_corrections(tmp_path, check_result("README.md"))[0]
    first_artifact = tmp_path / first.source_path
    first_content = first_artifact.read_text()
    second = create_training_corrections(tmp_path, check_result("SPEC.md"))[0]

    assert first.id == "corr_177d5760fe7c6f1db886b609"
    assert first.id != second.id
    assert first.source_path != second.source_path
    assert first_artifact.read_text() == first_content
    assert len(ReviewStore(tmp_path).load_candidates()) == 2


def test_check_corrections_delegate_all_results_to_unified_input_audit(tmp_path):
    check_result = {
        "active_state_receipt": "rcpt_unified_check",
        "input_hash": "sha256:" + "2" * 64,
        "results": [
            {
                "claim": "Morpheus is mainly a personal AI agent.",
                "status": "stale",
                "reason": "claim matches outdated project state",
                "evidence": {"path": "README.md", "line_start": 1},
            },
            {
                "claim": "Morpheus keeps reviewed project state.",
                "status": "verified",
                "reason": "claim is supported by active Morpheus evidence",
                "evidence": {"path": "SPEC.md", "line_start": 2},
            },
        ],
    }

    first = create_training_corrections(tmp_path, check_result)
    first_report = json.loads(
        (tmp_path / ".morpheus" / "learning" / "team_loop_report.json").read_text()
    )
    second = create_training_corrections(tmp_path, check_result)
    second_report = json.loads(
        (tmp_path / ".morpheus" / "learning" / "team_loop_report.json").read_text()
    )

    assert len(first) == 1
    assert first[0].provider["name"] == "morpheus-check"
    assert second == []
    assert first_report["input_source_counts"] == {"check_result": 2}
    assert first_report["created_input_receipt_count"] == 2
    assert first_report["created_count"] == 1
    assert first_report["no_candidate_input_count"] == 1
    assert second_report["existing_input_receipt_count"] == 2
    assert second_report["existing_count"] == 1
    assert len(list(
        (tmp_path / ".morpheus" / "review" / "team_inputs").glob("*.jsonl")
    )) == 2
    assert len(ReviewStore(tmp_path).load_candidates()) == 1


def test_legacy_check_missing_path_and_reason_keep_golden_candidate_contract(tmp_path):
    check_result = {
        "active_state_receipt": "rcpt_legacy_shape",
        "results": [{
            "claim": "Morpheus is mainly a personal AI agent.",
            "status": "stale",
            "reason": None,
            "evidence": {"line_start": 3},
        }],
    }

    candidate = create_training_corrections(tmp_path, check_result)[0]
    artifact = tmp_path / candidate.source_path
    expected_line = (
        'Correction candidate: stale claim "Morpheus is mainly a personal AI agent." '
        "was flagged by morpheus check because None. Source: unknown:3."
    )

    assert candidate.id == "corr_2aff5e3eeec75b1df7a68c4e"
    assert candidate.source_path == (
        ".morpheus/review/check_corrections/corr_2aff5e3eeec75b1df7a68c4e.md"
    )
    assert candidate.source_revision == "check:rcpt_legacy_shape"
    assert candidate.provider == {
        "name": "morpheus-check",
        "model": "local",
        "source_label": "unknown:3",
    }
    assert candidate.status == "pending"
    assert candidate.evidence_excerpt == expected_line
    assert artifact.read_text() == expected_line + "\n"


@pytest.mark.parametrize("line_start", [True, False, 0, "", "3", 3.0])
def test_legacy_check_adapter_rejects_noninteger_line_without_writes(
    tmp_path,
    line_start,
):
    check_result = {
        "results": [{
            "claim": "Morpheus is mainly a personal AI agent.",
            "status": "stale",
            "reason": "claim matches outdated project state",
            "evidence": {"path": "README.md", "line_start": line_start},
        }],
    }

    with pytest.raises(ValueError, match="positive integer"):
        create_training_corrections(tmp_path, check_result)

    assert not (tmp_path / ".morpheus").exists()


def test_rejected_correction_is_not_trained(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)
        runner.invoke(
            app,
            ["check", "--input", str(STALE_INPUT), "--create-training-corrections"],
        )
        candidate = ReviewStore(Path.cwd()).load_candidates()[0]
        ReviewStore(Path.cwd()).reject(candidate.id, reason="not useful")

        result = build_learning_dataset(Path.cwd(), source="accepted", include_corrections=True)

        dataset_text = (Path(result["dataset_dir"]) / "dataset.instruction.jsonl").read_text()
        assert candidate.claim not in dataset_text
        skipped = read_jsonl(Path(result["dataset_dir"]) / "skipped.jsonl")
        assert skipped[0]["reason"] == "status_rejected"


def test_accepted_correction_becomes_negative_training_example(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)
        runner.invoke(
            app,
            ["check", "--input", str(STALE_INPUT), "--create-training-corrections"],
        )
        store = ReviewStore(Path.cwd())
        candidate = store.load_candidates()[0]
        store.accept(candidate.id, reviewed_by="tester")

        result = build_learning_dataset(Path.cwd(), source="accepted", include_corrections=True)

        examples = read_jsonl(Path(result["dataset_dir"]) / "dataset.instruction.jsonl")
        correction_examples = [
            item
            for item in examples
            if item["metadata"]["source_candidate_id"] == candidate.id
        ]
        assert correction_examples
        assert correction_examples[0]["metadata"]["example_type"] == "outdated_correction"
        assert correction_examples[0]["output"].startswith("No.")


def test_tampered_check_projection_never_enters_training_dataset(tmp_path):
    check_result = {
        "active_state_receipt": "rcpt_tamper_check",
        "results": [{
            "claim": "Morpheus is mainly a personal AI agent.",
            "status": "stale",
            "reason": "claim matches outdated project state",
            "evidence": {"path": "README.md", "line_start": 1},
        }],
    }
    candidate = create_training_corrections(tmp_path, check_result)[0]
    store = ReviewStore(tmp_path)
    candidate = store.accept(candidate.id, reviewed_by="tester")
    artifact = tmp_path / candidate.source_path
    tampered_line = "A replacement that is not a canonical check correction."
    artifact.write_text(tampered_line + "\n")
    store.save_candidates([candidate.model_copy(update={
        "source_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "evidence_excerpt": tampered_line,
        "evidence_sha256": hashlib.sha256(tampered_line.encode()).hexdigest(),
    })])

    result = build_learning_dataset(
        tmp_path,
        source="accepted",
        include_corrections=True,
    )

    skipped = read_jsonl(Path(result["dataset_dir"]) / "skipped.jsonl")
    assert len(skipped) == 1
    assert skipped[0]["candidate_id"] == candidate.id
    assert skipped[0]["reason"] == "check_correction_projection_mismatch"
    assert skipped[0]["memory_route"] == "negative_example"


def test_include_refusals_adds_unsupported_claim_eval_item(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)
        runner.invoke(
            app,
            ["check", "--input", str(STALE_INPUT), "--create-training-corrections"],
        )
        store = ReviewStore(Path.cwd())
        candidate = store.load_candidates()[0]
        store.accept(candidate.id, reviewed_by="tester")

        result = build_learning_dataset(Path.cwd(), include_refusals=True)

        eval_items = read_jsonl(Path(result["dataset_dir"]) / "eval.seed.jsonl")
        assert any(item["category"] == "unsupported_claim_refusal" for item in eval_items)
