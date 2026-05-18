import json
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app
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
