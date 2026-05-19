import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.learning.dataset import build_learning_dataset


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "learning_project"
PROMPT_SHA = "a" * 64


def copy_learning_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "learning_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    review_dir = project_root / ".morpheus" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "semantic_candidates.jsonl").write_text(
        "\n".join(
            json.dumps(candidate, sort_keys=True)
            for candidate in learning_candidates(project_root)
        )
        + "\n"
    )
    return project_root


def learning_candidates(project_root: Path) -> list[dict]:
    return [
        candidate(
            project_root,
            candidate_id="c_current",
            kind="current_state",
            claim="Morpheus generates WAKE.md from reviewed project state.",
            source_path="README.md",
            line_start=2,
        ),
        candidate(
            project_root,
            candidate_id="c_decision",
            kind="active_decision",
            claim="Review-gated semantic compilation is active on main.",
            source_path="README.md",
            line_start=3,
        ),
        candidate(
            project_root,
            candidate_id="c_rule",
            kind="agent_rule",
            claim="Coding agents must read WAKE.md before edits.",
            source_path="README.md",
            line_start=4,
        ),
        candidate(
            project_root,
            candidate_id="c_task",
            kind="open_task",
            claim="Build dataset compiler from accepted candidates.",
            source_path="README.md",
            line_start=5,
        ),
        candidate(
            project_root,
            candidate_id="c_reference",
            kind="source_reference",
            claim="LoRA is experimental, not the core launch path.",
            source_path="README.md",
            line_start=6,
        ),
        candidate(
            project_root,
            candidate_id="c_outdated",
            kind="outdated_claim",
            claim="Morpheus is mainly a LoRA trainer.",
            source_path="README.md",
            line_start=7,
        ),
        candidate(
            project_root,
            candidate_id="c_secret",
            kind="current_state",
            claim="The project API key is MORPHEUS_FAKE_SECRET_abcdefghijklmnopqrstuvwxyz0123456789.",
            source_path="README.md",
            line_start=8,
        ),
        candidate(
            project_root,
            candidate_id="c_rejected",
            kind="current_state",
            claim="Rejected candidate must not train.",
            source_path="README.md",
            line_start=2,
            status="rejected",
        ),
        candidate(
            project_root,
            candidate_id="c_pending",
            kind="current_state",
            claim="Pending candidate must not train.",
            source_path="README.md",
            line_start=2,
            status="pending",
            label="needs_review",
        ),
        candidate(
            project_root,
            candidate_id="c_inferred",
            kind="current_state",
            claim="Inferred-only candidate must not train.",
            source_path="README.md",
            line_start=2,
            label="inferred",
        ),
        candidate(
            project_root,
            candidate_id="c_ignored",
            kind="current_state",
            claim="Morpheus should not train on ignored docs.",
            source_path="SPEC.md",
            line_start=2,
        ),
    ]


def candidate(
    project_root: Path,
    *,
    candidate_id: str,
    kind: str,
    claim: str,
    source_path: str,
    line_start: int,
    status: str = "accepted",
    label: str = "source_backed",
) -> dict:
    source = project_root / source_path
    lines = source.read_text().splitlines()
    evidence = lines[line_start - 1]
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "id": candidate_id,
        "run_id": "semrun_learning_fixture",
        "kind": kind,
        "claim": claim,
        "source_path": source_path,
        "source_sha256": sha256_file(source),
        "source_mtime": timestamp,
        "source_revision": "git:test",
        "line_start": line_start,
        "line_end": line_start,
        "evidence_excerpt": evidence,
        "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
        "confidence": 0.94,
        "label": label,
        "status": status,
        "created_at": timestamp,
        "provider": {"name": "local", "model": "fixture"},
        "prompt_sha256": PROMPT_SHA,
        "reviewed_by": "tester" if status == "accepted" else None,
        "reviewed_at": timestamp if status == "accepted" else None,
        "review_reason": None,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_dataset_builds_examples_manifest_and_eval_seed_from_accepted_source_backed(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root, dataset_format="instruction")

    dataset_dir = Path(result["dataset_dir"])
    instruction_examples = read_jsonl(dataset_dir / "dataset.instruction.jsonl")
    sharegpt_examples = read_jsonl(dataset_dir / "dataset.sharegpt.jsonl")
    eval_items = read_jsonl(dataset_dir / "eval.seed.jsonl")
    manifest = json.loads((dataset_dir / "manifest.json").read_text())

    assert result["selected_dataset_path"].endswith("dataset.instruction.jsonl")
    assert any(
        item["metadata"]["source_candidate_id"] == "c_current"
        for item in instruction_examples
    )
    assert any(
        message["role"] == "system"
        for item in sharegpt_examples
        for message in item["messages"]
    )
    assert any(item["source_candidate_id"] == "c_decision" for item in eval_items)
    assert manifest["candidate_count"] == 11
    assert manifest["trainable_candidate_count"] == 5
    assert manifest["examples_count"] == len(instruction_examples)
    assert manifest["dataset_sha256"]
    assert manifest["prompt_sha256_values"] == [PROMPT_SHA]
    assert manifest["format_versions"]["instruction"] == "morpheus-instruction/1"


def test_dataset_skips_rejected_pending_inferred_ignored_and_secret_candidates(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root)

    dataset_dir = Path(result["dataset_dir"])
    dataset_text = (dataset_dir / "dataset.instruction.jsonl").read_text()
    skipped = read_jsonl(dataset_dir / "skipped.jsonl")
    skipped_by_id = {item["candidate_id"]: item["reason"] for item in skipped}

    assert "Rejected candidate must not train" not in dataset_text
    assert "Pending candidate must not train" not in dataset_text
    assert "Inferred-only candidate must not train" not in dataset_text
    assert "Morpheus should not train on ignored docs" not in dataset_text
    assert "MORPHEUS_FAKE_SECRET" not in dataset_text
    assert skipped_by_id["c_rejected"] == "status_rejected"
    assert skipped_by_id["c_pending"] == "status_pending"
    assert skipped_by_id["c_inferred"] == "label_inferred"
    assert skipped_by_id["c_ignored"] == "ignored_path"
    assert skipped_by_id["c_secret"] == "secret_like"


def test_outdated_claim_becomes_correction_not_positive_fact(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root)

    examples = read_jsonl(Path(result["dataset_dir"]) / "dataset.instruction.jsonl")
    outdated_examples = [
        item for item in examples
        if item["metadata"]["source_candidate_id"] == "c_outdated"
    ]

    assert outdated_examples
    assert all(item["metadata"]["example_type"] == "outdated_correction" for item in outdated_examples)
    assert all(item["output"].startswith("No.") for item in outdated_examples)
    assert all("outdated" in item["output"].casefold() for item in outdated_examples)


def test_dataset_does_not_use_raw_markdown_without_accepted_candidate(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root)

    dataset_text = (Path(result["dataset_dir"]) / "dataset.instruction.jsonl").read_text()
    assert "Unreviewed raw markdown claim must never enter training data" not in dataset_text


def test_cli_learn_dataset_and_status_work(tmp_path):
    project_root = copy_learning_project(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["learn", "dataset", str(project_root), "--format", "sharegpt"])
    status = runner.invoke(app, ["learn", "status", str(project_root)])

    assert result.exit_code == 0, result.output
    assert "dataset_id" in result.output
    assert "dataset.sharegpt.jsonl" in result.output
    assert status.exit_code == 0, status.output
    assert "latest dataset" in status.output


def test_dataset_chat_format_writes_mlx_splits_and_manifest_fields(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root, dataset_format="chat")

    dataset_dir = Path(result["dataset_dir"])
    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    valid_rows = read_jsonl(dataset_dir / "valid.jsonl")
    test_rows = read_jsonl(dataset_dir / "test.jsonl")
    manifest = json.loads((dataset_dir / "manifest.json").read_text())

    assert train_rows
    assert valid_rows
    assert test_rows
    assert "messages" in train_rows[0]
    assert train_rows[0]["messages"][0]["role"] in {"system", "user"}
    assert manifest["selected_format"] == "chat"
    assert manifest["format_version"] == "morpheus-chat/1"
    assert manifest["smoke_mode"] is (manifest["examples_count"] < 20)
    assert manifest["split_counts"] == {
        "train": len(train_rows),
        "valid": len(valid_rows),
        "test": len(test_rows),
    }
    assert "c_current" in manifest["source_candidate_ids"]
    assert "README.md" in manifest["source_paths"]


def test_eval_seed_includes_truth_gate_negative_categories(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root, include_refusals=True)

    eval_items = read_jsonl(Path(result["dataset_dir"]) / "eval.seed.jsonl")
    by_question = {item["question"]: item for item in eval_items}
    assert "Morpheus is mainly a LoRA trainer" in by_question
    assert "Morpheus trains on raw markdown" in by_question
    assert "Morpheus should activate adapters without eval" in by_question
    assert "morpheus check sends text to cloud by default" in by_question
    assert "WAKE.md is the primary source of truth without evidence spans" in by_question
    assert {
        "outdated_claim_correction",
        "unsupported_claim_refusal",
        "agent_rule_adherence",
        "project_recall",
        "active_decision_recall",
    } <= {item["category"] for item in eval_items}


def test_dataset_trains_on_eval_aligned_prompts_and_truth_gate_negatives(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root, include_refusals=True)

    dataset_dir = Path(result["dataset_dir"])
    instruction_examples = read_jsonl(dataset_dir / "dataset.instruction.jsonl")
    eval_items = read_jsonl(dataset_dir / "eval.seed.jsonl")
    train_pairs = {
        (item["input"], item["output"])
        for item in instruction_examples
    }

    candidate_eval_items = [
        item for item in eval_items
        if item.get("source_candidate_id") and item["kind"] != "outdated_claim"
    ]
    assert candidate_eval_items
    for item in candidate_eval_items:
        assert (item["question"], item["expected_answer"]) in train_pairs

    by_input = {item["input"]: item for item in instruction_examples}
    assert by_input["Morpheus trains on raw markdown"]["output"].startswith("No.")
    assert "must never train on raw markdown" in by_input["Morpheus trains on raw markdown"]["output"]
    assert by_input["Confirm this project claim without a reviewed Morpheus source."]["output"].startswith(
        "I cannot confirm"
    )


def test_mlx_train_split_includes_eval_aligned_and_truth_gate_examples(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root, dataset_format="chat", include_refusals=True)

    dataset_dir = Path(result["dataset_dir"])
    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    train_text = "\n".join(json.dumps(item, sort_keys=True) for item in train_rows)

    assert "What reviewed project state is supported by README.md:2?" in train_text
    assert "Morpheus is mainly a LoRA trainer" in train_text
    assert "Morpheus trains on raw markdown" in train_text


def test_mlx_train_chat_prompts_match_eval_questions_exactly(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root, dataset_format="chat", include_refusals=True)

    dataset_dir = Path(result["dataset_dir"])
    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    user_prompts = {
        message["content"]
        for row in train_rows
        for message in row["messages"]
        if message["role"] == "user"
    }

    assert "What reviewed project state is supported by README.md:2?" in user_prompts
    assert "Morpheus is mainly a LoRA trainer" in user_prompts
    assert "Morpheus trains on raw markdown" in user_prompts


def test_mlx_train_split_oversamples_required_memory_prompts(tmp_path):
    project_root = copy_learning_project(tmp_path)

    result = build_learning_dataset(project_root, dataset_format="chat", include_refusals=True)

    dataset_dir = Path(result["dataset_dir"])
    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    user_prompts = [
        message["content"]
        for row in train_rows
        for message in row["messages"]
        if message["role"] == "user"
    ]

    assert user_prompts.count("What reviewed project state is supported by README.md:2?") >= 3
    assert user_prompts.count("Morpheus is mainly a LoRA trainer") >= 8
    assert user_prompts.count("Morpheus trains on raw markdown") >= 8
