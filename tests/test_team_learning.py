import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest

import morpheus.core.learning.team as team_module
from morpheus.core.config import MorpheusConfig
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.quality import build_quality_report
from morpheus.core.learning.team import (
    TEAM_LOOP_POLICY_VERSION,
    run_team_learning_loop,
)
from morpheus.core.semantic.review import (
    ReviewStore,
    apply_accepted_candidates,
    run_semantic_review,
)
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.routing import route_candidate
from morpheus.core.semantic.verifier import verify_candidate_span
from tests.test_learning_dataset import candidate as learning_candidate
from tests.test_learning_dataset import read_jsonl


def feedback_item(**updates) -> dict:
    item = {
        "source_type": "human_correction",
        "external_id": "feedback-1",
        "claim": "Morpheus trains raw project Markdown directly.",
        "correction": "Morpheus trains only from accepted source-backed candidates.",
        "author": "reviewer@example.com",
        "url": "https://example.test/reviews/1",
    }
    item.update(updates)
    return item


def accepted_review_candidate(tmp_path) -> SemanticCandidate:
    source = tmp_path / "README.md"
    source.write_text("Morpheus keeps reviewed, source-backed project state.\n")
    return route_candidate(SemanticCandidate(**learning_candidate(
        tmp_path,
        candidate_id="accepted-review-1",
        kind="current_state",
        claim="Morpheus keeps reviewed, source-backed project state.",
        source_path="README.md",
        line_start=1,
    )))


def six_source_items(accepted_candidate_id: str) -> list[dict]:
    return [
        feedback_item(source_type="pr_comment", external_id="pr-comment-1"),
        feedback_item(
            source_type="rejected_agent_claim",
            external_id="rejected-agent-1",
        ),
        feedback_item(
            source_type="human_correction",
            external_id="human-correction-1",
        ),
        feedback_item(
            source_type="stale_claim_correction",
            external_id="stale-correction-1",
        ),
        {
            "source_type": "accepted_review_candidate",
            "candidate_id": accepted_candidate_id,
        },
        {
            "source_type": "check_result",
            "claim": "Morpheus trains raw project Markdown directly.",
            "status": "stale",
            "reason": "claim matches outdated project state",
            "evidence": {"path": "README.md", "line_start": 1},
            "active_state_receipt": "rcpt_six_sources",
            "input_hash": "sha256:" + "1" * 64,
        },
    ]


def test_team_loop_accepts_all_six_sources_in_one_idempotent_batch(tmp_path):
    accepted = accepted_review_candidate(tmp_path)
    store = ReviewStore(tmp_path)
    store.save_candidates([accepted])
    items = six_source_items(accepted.id)

    first = run_team_learning_loop(tmp_path, items)
    first_candidates = store.load_candidates()
    second = run_team_learning_loop(tmp_path, items)
    second_candidates = store.load_candidates()

    expected_source_counts = {
        "accepted_review_candidate": 1,
        "check_result": 1,
        "human_correction": 1,
        "pr_comment": 1,
        "rejected_agent_claim": 1,
        "stale_claim_correction": 1,
    }
    assert first["report"]["input_source_counts"] == expected_source_counts
    assert first["report"]["created_input_receipt_count"] == 6
    assert first["report"]["existing_input_receipt_count"] == 0
    assert first["report"]["created_count"] == 5
    assert first["report"]["existing_count"] == 0
    assert first["report"]["reconciled_candidate_ids"] == [accepted.id]
    assert second["report"]["input_source_counts"] == expected_source_counts
    assert second["report"]["created_input_receipt_count"] == 0
    assert second["report"]["existing_input_receipt_count"] == 6
    assert second["report"]["created_count"] == 0
    assert second["report"]["existing_count"] == 5
    assert second["report"]["reconciled_candidate_ids"] == [accepted.id]
    assert first_candidates == second_candidates
    assert len(second_candidates) == 6
    assert next(item for item in second_candidates if item.id == accepted.id) == accepted
    assert all(
        item.status == "pending"
        for item in second_candidates
        if item.id != accepted.id
    )
    assert len(list(
        (tmp_path / ".morpheus" / "review" / "team_inputs").glob("*.jsonl")
    )) == 6
    assert len(list(
        (tmp_path / ".morpheus" / "review" / "team_feedback").glob("*.jsonl")
    )) == 4
    assert len(list(
        (tmp_path / ".morpheus" / "review" / "check_corrections").glob("*.md")
    )) == 1


@pytest.mark.parametrize("status", ["pending", "rejected"])
def test_mixed_batch_rejects_unaccepted_reference_before_any_writes(tmp_path, status):
    candidate = accepted_review_candidate(tmp_path).model_copy(update={
        "status": status,
        "reviewed_by": None,
        "reviewed_at": None,
    })
    store = ReviewStore(tmp_path)
    store.save_candidates([candidate])
    baseline = store.load_candidates()
    items = [feedback_item(external_id="must-not-write"), {
        "source_type": "accepted_review_candidate",
        "candidate_id": candidate.id,
    }]

    with pytest.raises(ValueError, match="accepted"):
        run_team_learning_loop(tmp_path, items)

    assert store.load_candidates() == baseline
    assert not (tmp_path / ".morpheus" / "review" / "team_inputs").exists()
    assert not (tmp_path / ".morpheus" / "review" / "team_feedback").exists()


@pytest.mark.parametrize(
    "source_type",
    ["pr_comment", "rejected_agent_claim", "human_correction"],
)
def test_team_feedback_becomes_pending_source_backed_candidate(tmp_path, source_type):
    result = run_team_learning_loop(
        tmp_path,
        [feedback_item(source_type=source_type)],
    )

    candidate = ReviewStore(tmp_path).load_candidates()[0]
    artifact = tmp_path / candidate.source_path
    artifact_lines = artifact.read_text().splitlines()
    assert result["report"]["policy_version"] == TEAM_LOOP_POLICY_VERSION
    assert result["report"]["created_count"] == 1
    assert candidate.status == "pending"
    assert candidate.kind == "outdated_claim"
    assert candidate.label == "source_backed"
    assert candidate.semantic_class == "stale"
    assert candidate.memory_route == "stale_archive"
    assert candidate.correction_text == feedback_item()["correction"]
    assert candidate.provider["feedback_source"] == source_type
    assert len(artifact_lines) == 1
    assert candidate.evidence_excerpt == artifact_lines[0]
    assert verify_candidate_span(tmp_path, candidate).label == "source_backed"
    assert Path(result["json_path"]).is_file()
    assert Path(result["markdown_path"]).is_file()
    assert result["report"]["actions"] == {
        "dataset_generation_attempted": False,
        "training_attempted": False,
        "evaluation_attempted": False,
        "adapter_activation_attempted": False,
    }


def test_team_feedback_replay_is_idempotent(tmp_path):
    first = run_team_learning_loop(tmp_path, [feedback_item()])
    second = run_team_learning_loop(tmp_path, [feedback_item()])

    assert first["report"]["created_count"] == 1
    assert first["report"]["existing_count"] == 0
    assert second["report"]["created_count"] == 0
    assert second["report"]["existing_count"] == 1
    assert len(ReviewStore(tmp_path).load_candidates()) == 1


def test_legacy_team_feedback_canonical_id_is_unchanged(tmp_path):
    result = run_team_learning_loop(tmp_path, [feedback_item()])

    assert result["report"]["created_candidate_ids"] == [
        "teamfb_9a141c61c82e5172c4701964"
    ]


def test_team_input_receipt_tamper_blocks_replay_without_store_mutation(tmp_path):
    run_team_learning_loop(tmp_path, [feedback_item()])
    store = ReviewStore(tmp_path)
    baseline = store.load_candidates()
    receipt = next((store.review_dir / "team_inputs").glob("*.jsonl"))
    receipt.write_text("tampered input receipt\n")

    with pytest.raises(ValueError, match="content changed"):
        run_team_learning_loop(tmp_path, [feedback_item()])

    assert store.load_candidates() == baseline


def test_accepted_candidate_receipt_is_authority_bound_and_replayable(tmp_path):
    candidate = accepted_review_candidate(tmp_path)
    store = ReviewStore(tmp_path)
    store.save_candidates([candidate])
    event = {
        "source_type": "accepted_review_candidate",
        "candidate_id": candidate.id,
    }
    first = run_team_learning_loop(tmp_path, [event])
    receipt_path = next((store.review_dir / "team_inputs").glob("*.jsonl"))
    receipt_event = json.loads(receipt_path.read_text())

    second = run_team_learning_loop(tmp_path, [receipt_event])

    assert len(receipt_event["candidate_sha256"]) == 64
    assert first["report"]["created_input_receipt_count"] == 1
    assert second["report"]["existing_input_receipt_count"] == 1
    assert second["report"]["reconciled_candidate_ids"] == [candidate.id]
    assert store.load_candidates() == [candidate]


def test_team_feedback_replay_rejects_tampered_candidate_projection(tmp_path):
    item = feedback_item()
    run_team_learning_loop(tmp_path, [item])
    store = ReviewStore(tmp_path)
    candidate = store.load_candidates()[0]
    store.save_candidates([
        candidate.model_copy(update={"correction_text": "Tampered replacement."})
    ])

    with pytest.raises(ValueError, match="projection mismatch"):
        run_team_learning_loop(tmp_path, [item])


def test_team_feedback_replay_rejects_wrong_provider_id_squatting(tmp_path):
    item = feedback_item()
    run_team_learning_loop(tmp_path, [item])
    store = ReviewStore(tmp_path)
    candidate = store.load_candidates()[0]
    store.save_candidates([candidate.model_copy(update={
        "provider": {"name": "unrelated", "model": "local"},
    })])

    with pytest.raises(ValueError, match="projection mismatch"):
        run_team_learning_loop(tmp_path, [item])


def test_concurrent_team_feedback_runs_preserve_every_candidate(tmp_path):
    worker_count = 12
    start = Barrier(worker_count)

    def ingest(index: int) -> None:
        start.wait()
        run_team_learning_loop(
            tmp_path,
            [feedback_item(external_id=f"concurrent-{index}")],
        )

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        list(pool.map(ingest, range(worker_count)))

    candidates = ReviewStore(tmp_path).load_candidates()
    artifacts = list((tmp_path / ".morpheus" / "review" / "team_feedback").glob("*.jsonl"))
    receipts = list((tmp_path / ".morpheus" / "review" / "team_inputs").glob("*.jsonl"))
    assert len(candidates) == worker_count
    assert len({candidate.id for candidate in candidates}) == worker_count
    assert len(artifacts) == worker_count
    assert len(receipts) == worker_count


def test_semantic_scan_started_before_team_feedback_preserves_feedback(tmp_path):
    (tmp_path / "README.md").write_text("Morpheus keeps reviewed project state.\n")
    scan_started = Event()
    allow_scan_to_finish = Event()

    class BlockingProvider:
        name = "blocking-test"
        model = "local"

        def extract_candidates(self, source, **kwargs):
            scan_started.set()
            assert allow_scan_to_finish.wait(timeout=5)
            return []

    with ThreadPoolExecutor(max_workers=1) as pool:
        scan = pool.submit(run_semantic_review, tmp_path, provider=BlockingProvider())
        assert scan_started.wait(timeout=5)
        team_result = run_team_learning_loop(tmp_path, [feedback_item()])
        allow_scan_to_finish.set()
        scan.result(timeout=5)

    candidates = ReviewStore(tmp_path).load_candidates()
    assert team_result["report"]["created_count"] == 1
    assert [candidate.id for candidate in candidates] == (
        team_result["report"]["created_candidate_ids"]
    )


def test_changed_team_feedback_creates_immutable_new_version(tmp_path):
    first = run_team_learning_loop(tmp_path, [feedback_item()])
    second = run_team_learning_loop(
        tmp_path,
        [feedback_item(correction="Use reviewed candidates only; raw files remain evidence.")],
    )

    assert first["report"]["created_candidate_ids"] != second["report"]["created_candidate_ids"]
    assert len(ReviewStore(tmp_path).load_candidates()) == 2
    assert len(list((tmp_path / ".morpheus" / "review" / "team_feedback").glob("*.jsonl"))) == 2


def test_team_feedback_batch_conflict_writes_no_partial_artifacts(tmp_path):
    conflicting_item = feedback_item(external_id="feedback-conflict")
    run_team_learning_loop(tmp_path, [conflicting_item])
    store = ReviewStore(tmp_path)
    conflicting_candidate = store.load_candidates()[0]
    conflicting_artifact = tmp_path / conflicting_candidate.source_path
    store.save_candidates([])
    conflicting_artifact.write_text("conflicting immutable content\n")
    new_item = feedback_item(external_id="feedback-new")

    with pytest.raises(ValueError, match="different content"):
        run_team_learning_loop(tmp_path, [new_item, conflicting_item])

    assert ReviewStore(tmp_path).load_candidates() == []
    artifacts = list((tmp_path / ".morpheus" / "review" / "team_feedback").glob("*.jsonl"))
    assert artifacts == [conflicting_artifact]


def test_team_feedback_validation_is_atomic_before_writes(tmp_path):
    secret = feedback_item(
        external_id="feedback-secret",
        claim="The api key is sk-abcdefghijklmnopqrstuvwxyz123456.",
    )

    with pytest.raises(ValueError, match="secret-like"):
        run_team_learning_loop(tmp_path, [feedback_item(), secret])

    assert ReviewStore(tmp_path).load_candidates() == []
    assert not (tmp_path / ".morpheus" / "review" / "team_feedback").exists()


def test_check_result_secret_reason_is_atomic_before_writes(tmp_path):
    secret_check = {
        "source_type": "check_result",
        "claim": "Morpheus keeps reviewed state.",
        "status": "verified",
        "reason": "The token is ghp_abcdefghijklmnopqrstuvwxyz123456.",
    }

    with pytest.raises(ValueError, match="secret-like"):
        run_team_learning_loop(tmp_path, [feedback_item(), secret_check])

    assert ReviewStore(tmp_path).load_candidates() == []
    assert not (tmp_path / ".morpheus" / "review" / "team_inputs").exists()
    assert not (tmp_path / ".morpheus" / "review" / "team_feedback").exists()


def test_check_result_secret_in_nested_evidence_is_atomic_before_writes(tmp_path):
    secret_check = {
        "source_type": "check_result",
        "claim": "Morpheus keeps reviewed state.",
        "status": "verified",
        "reason": "claim is supported by active evidence",
        "evidence": {
            "path": "token=ghp_abcdefghijklmnopqrstuvwxyz123456/README.md",
            "line_start": 1,
        },
    }

    with pytest.raises(ValueError, match="secret-like"):
        run_team_learning_loop(tmp_path, [feedback_item(), secret_check])

    assert ReviewStore(tmp_path).load_candidates() == []
    assert not (tmp_path / ".morpheus" / "review" / "team_inputs").exists()
    assert not (tmp_path / ".morpheus" / "review" / "team_feedback").exists()


def test_stale_claim_correction_requires_explicit_nonblank_correction(tmp_path):
    item = feedback_item(
        source_type="stale_claim_correction",
        external_id="stale-missing-correction",
        correction=None,
    )

    with pytest.raises(ValueError):
        run_team_learning_loop(tmp_path, [item])

    assert ReviewStore(tmp_path).load_candidates() == []
    assert not (tmp_path / ".morpheus" / "review" / "team_inputs").exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"source_type": "email"},
        {"external_id": "   "},
        {"claim": "   "},
        {"unexpected_field": "must not be ignored"},
    ],
)
def test_team_feedback_rejects_invalid_contract_without_writes(tmp_path, updates):
    with pytest.raises(ValueError):
        run_team_learning_loop(tmp_path, [feedback_item(**updates)])

    assert ReviewStore(tmp_path).load_candidates() == []


def test_team_feedback_rejects_symlinked_feedback_directory(tmp_path):
    outside = tmp_path / "outside-feedback"
    outside.mkdir()
    review_dir = tmp_path / ".morpheus" / "review"
    review_dir.mkdir(parents=True)
    try:
        (review_dir / "team_feedback").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Team feedback path must not be a symlink"):
        run_team_learning_loop(tmp_path, [feedback_item()])

    assert list(outside.iterdir()) == []


def test_team_loop_rejects_symlinked_input_receipt_directory(tmp_path):
    outside = tmp_path / "outside-input-receipts"
    outside.mkdir()
    review_dir = tmp_path / ".morpheus" / "review"
    review_dir.mkdir(parents=True)
    try:
        (review_dir / "team_inputs").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Team input receipt path must not be a symlink"):
        run_team_learning_loop(tmp_path, [feedback_item()])

    assert list(outside.iterdir()) == []
    assert ReviewStore(tmp_path).load_candidates() == []


def test_team_loop_rejects_symlinked_check_correction_directory(tmp_path):
    outside = tmp_path / "outside-check-corrections"
    outside.mkdir()
    review_dir = tmp_path / ".morpheus" / "review"
    review_dir.mkdir(parents=True)
    try:
        (review_dir / "check_corrections").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    item = {
        "source_type": "check_result",
        "claim": "Morpheus trains raw project Markdown directly.",
        "status": "stale",
        "reason": "claim matches outdated project state",
    }

    with pytest.raises(ValueError, match="Check corrections path must not be a symlink"):
        run_team_learning_loop(tmp_path, [item])

    assert list(outside.iterdir()) == []
    assert not (review_dir / "team_inputs").exists()
    assert ReviewStore(tmp_path).load_candidates() == []


def test_check_result_audits_non_corrections_without_creating_candidates(tmp_path):
    items = [
        {
            "source_type": "check_result",
            "claim": "Morpheus keeps reviewed project state.",
            "status": "verified",
            "reason": "claim is supported by active Morpheus evidence",
        },
        {
            "source_type": "check_result",
            "claim": "Morpheus has an undocumented deployment target.",
            "status": "unknown",
            "reason": "no matching active evidence found",
        },
    ]

    result = run_team_learning_loop(tmp_path, items)

    assert result["report"]["created_input_receipt_count"] == 2
    assert result["report"]["created_count"] == 0
    assert result["report"]["no_candidate_input_count"] == 2
    assert ReviewStore(tmp_path).load_candidates() == []
    assert not (tmp_path / ".morpheus" / "review" / "check_corrections").exists()


def test_changed_check_reason_gets_new_receipt_without_duplicate_candidate(tmp_path):
    item = {
        "source_type": "check_result",
        "claim": "Morpheus trains raw project Markdown directly.",
        "status": "stale",
        "reason": "claim matches outdated project state",
        "evidence": {"path": "README.md", "line_start": 1},
        "active_state_receipt": "rcpt_check_reason",
    }

    first = run_team_learning_loop(tmp_path, [item])
    candidate = ReviewStore(tmp_path).load_candidates()[0]
    changed = {**item, "reason": "a newer check reached the same correction key"}
    second = run_team_learning_loop(tmp_path, [changed])

    assert first["report"]["created_count"] == 1
    assert second["report"]["created_input_receipt_count"] == 1
    assert second["report"]["created_count"] == 0
    assert second["report"]["existing_candidate_ids"] == [candidate.id]
    assert ReviewStore(tmp_path).load_candidates() == [candidate]
    assert len(list(
        (tmp_path / ".morpheus" / "review" / "team_inputs").glob("*.jsonl")
    )) == 2


def test_check_replay_rejects_wrong_provider_id_squatting(tmp_path):
    item = {
        "source_type": "check_result",
        "claim": "Morpheus trains raw project Markdown directly.",
        "status": "stale",
        "reason": "claim matches outdated project state",
    }
    run_team_learning_loop(tmp_path, [item])
    store = ReviewStore(tmp_path)
    candidate = store.load_candidates()[0]
    store.save_candidates([candidate.model_copy(update={
        "provider": {"name": "unrelated", "model": "local"},
    })])

    with pytest.raises(ValueError, match="projection mismatch"):
        run_team_learning_loop(tmp_path, [item])


@pytest.mark.parametrize("line_start", [True, "7", 7.0])
def test_check_result_line_start_is_strict_without_writes(tmp_path, line_start):
    item = {
        "source_type": "check_result",
        "claim": "Morpheus keeps reviewed state.",
        "status": "verified",
        "reason": "claim is supported by active evidence",
        "evidence": {"path": "README.md", "line_start": line_start},
    }

    with pytest.raises(ValueError):
        run_team_learning_loop(tmp_path, [item])

    assert not (tmp_path / ".morpheus").exists()


def test_mixed_batch_artifact_conflict_writes_no_new_receipt_or_candidate(tmp_path):
    check_item = {
        "source_type": "check_result",
        "claim": "Morpheus trains raw project Markdown directly.",
        "status": "incorrect",
        "reason": "claim contradicts reviewed project state",
        "evidence": {"path": "README.md", "line_start": 1},
    }
    run_team_learning_loop(tmp_path, [check_item])
    store = ReviewStore(tmp_path)
    check_candidate = store.load_candidates()[0]
    check_artifact = tmp_path / check_candidate.source_path
    store.save_candidates([])
    for receipt in (store.review_dir / "team_inputs").glob("*.jsonl"):
        receipt.unlink()
    check_artifact.write_text("conflicting immutable check content\n")

    with pytest.raises(ValueError, match="different content"):
        run_team_learning_loop(
            tmp_path,
            [feedback_item(external_id="must-remain-atomic"), check_item],
        )

    assert store.load_candidates() == []
    assert list((store.review_dir / "team_inputs").glob("*.jsonl")) == []
    assert not (store.review_dir / "team_feedback").exists()


@pytest.mark.parametrize("failure_call", range(1, 8))
def test_team_transaction_rolls_back_every_write_boundary(
    tmp_path,
    monkeypatch,
    failure_call,
):
    check_item = {
        "source_type": "check_result",
        "claim": "Morpheus trains raw project Markdown directly.",
        "status": "stale",
        "reason": "claim matches outdated project state",
    }
    original_apply = team_module._apply_team_transaction_entry
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError(f"injected write failure {failure_call}")
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(team_module, "_apply_team_transaction_entry", fail_once)

    with pytest.raises(OSError, match="injected write failure"):
        run_team_learning_loop(
            tmp_path,
            [feedback_item(external_id="transaction-feedback"), check_item],
        )

    store = ReviewStore(tmp_path)
    assert store.load_candidates() == []
    assert list((store.review_dir / "team_inputs").glob("*.jsonl")) == []
    assert list((store.review_dir / "team_feedback").glob("*.jsonl")) == []
    assert list((store.review_dir / "check_corrections").glob("*.md")) == []
    assert not (tmp_path / ".morpheus" / "learning" / "team_loop_report.json").exists()
    assert not (tmp_path / ".morpheus" / "learning" / "team_loop_report.md").exists()
    assert not (store.review_dir / ".team-learning-transaction.json").exists()
    assert list((tmp_path / ".morpheus").rglob("*.tmp")) == []


def test_team_transaction_rolls_back_when_commit_marker_write_fails(tmp_path, monkeypatch):
    original_write = team_module._atomic_project_write

    def fail_commit_marker(project_root, path, payload, **kwargs):
        if (
            path.name == ".team-learning-transaction.json"
            and kwargs.get("expected") is not None
        ):
            raise OSError("injected commit marker failure")
        return original_write(project_root, path, payload, **kwargs)

    monkeypatch.setattr(team_module, "_atomic_project_write", fail_commit_marker)

    with pytest.raises(OSError, match="injected commit marker failure"):
        run_team_learning_loop(tmp_path, [feedback_item()])

    store = ReviewStore(tmp_path)
    assert store.load_candidates() == []
    assert list((store.review_dir / "team_inputs").glob("*.jsonl")) == []
    assert list((store.review_dir / "team_feedback").glob("*.jsonl")) == []
    assert not (store.review_dir / ".team-learning-transaction.json").exists()


def test_team_transaction_cleans_published_prepare_marker_failure(tmp_path, monkeypatch):
    original_write = team_module._atomic_project_write

    def fail_after_prepare_marker(project_root, path, payload, **kwargs):
        result = original_write(project_root, path, payload, **kwargs)
        if (
            path.name == ".team-learning-transaction.json"
            and kwargs.get("expected") is None
        ):
            raise OSError("injected post-prepare failure")
        return result

    monkeypatch.setattr(team_module, "_atomic_project_write", fail_after_prepare_marker)

    with pytest.raises(OSError, match="injected post-prepare failure"):
        run_team_learning_loop(tmp_path, [feedback_item()])

    store = ReviewStore(tmp_path)
    assert store.load_candidates() == []
    assert list((store.review_dir / "team_inputs").glob("*.jsonl")) == []
    assert list((store.review_dir / "team_feedback").glob("*.jsonl")) == []
    assert not (store.review_dir / ".team-learning-transaction.json").exists()


def test_team_transaction_recovers_success_after_committed_marker_publish_error(
    tmp_path,
    monkeypatch,
):
    original_write = team_module._atomic_project_write

    def fail_after_committed_marker(project_root, path, payload, **kwargs):
        result = original_write(project_root, path, payload, **kwargs)
        if (
            path.name == ".team-learning-transaction.json"
            and kwargs.get("expected") is not None
        ):
            raise OSError("injected post-commit failure")
        return result

    monkeypatch.setattr(team_module, "_atomic_project_write", fail_after_committed_marker)

    result = run_team_learning_loop(tmp_path, [feedback_item()])

    store = ReviewStore(tmp_path)
    assert result["report"]["created_count"] == 1
    assert len(store.load_candidates()) == 1
    assert len(list((store.review_dir / "team_inputs").glob("*.jsonl"))) == 1
    assert len(list((store.review_dir / "team_feedback").glob("*.jsonl"))) == 1
    assert not (store.review_dir / ".team-learning-transaction.json").exists()


def test_prepared_team_transaction_recovers_backward_after_interruption(
    tmp_path,
    monkeypatch,
):
    original_apply = team_module._apply_team_transaction_entry
    calls = 0

    def interrupt_after_first_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(
        team_module,
        "_apply_team_transaction_entry",
        interrupt_after_first_write,
    )
    with pytest.raises(KeyboardInterrupt):
        run_team_learning_loop(tmp_path, [feedback_item()])
    monkeypatch.setattr(team_module, "_apply_team_transaction_entry", original_apply)

    result = run_team_learning_loop(tmp_path, [])

    store = ReviewStore(tmp_path)
    assert result["report"]["input_count"] == 0
    assert store.load_candidates() == []
    assert list((store.review_dir / "team_inputs").glob("*.jsonl")) == []
    assert list((store.review_dir / "team_feedback").glob("*.jsonl")) == []
    assert not (store.review_dir / ".team-learning-transaction.json").exists()


def test_review_accept_recovers_prepared_team_transaction_before_mutation(
    tmp_path,
    monkeypatch,
):
    original_apply = team_module._apply_team_transaction_entry
    calls = 0

    def interrupt_after_candidate_store_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(
        team_module,
        "_apply_team_transaction_entry",
        interrupt_after_candidate_store_write,
    )
    with pytest.raises(KeyboardInterrupt):
        run_team_learning_loop(tmp_path, [feedback_item()])
    monkeypatch.setattr(team_module, "_apply_team_transaction_entry", original_apply)

    store = ReviewStore(tmp_path)
    interrupted_candidates = store.load_candidates()
    assert len(interrupted_candidates) == 1
    assert interrupted_candidates[0].status == "pending"

    with pytest.raises(KeyError, match="candidate not found"):
        store.accept(interrupted_candidates[0].id, reviewed_by="tester")

    assert store.load_candidates() == []
    assert list((store.review_dir / "team_inputs").glob("*.jsonl")) == []
    assert list((store.review_dir / "team_feedback").glob("*.jsonl")) == []
    assert not (store.review_dir / ".team-learning-transaction.json").exists()


def test_committed_team_transaction_recovers_forward_after_interruption(
    tmp_path,
    monkeypatch,
):
    original_unlink = team_module._secure_project_unlink

    def interrupt_journal_removal(project_root, path, **kwargs):
        if path.name == ".team-learning-transaction.json":
            raise KeyboardInterrupt
        return original_unlink(project_root, path, **kwargs)

    monkeypatch.setattr(team_module, "_secure_project_unlink", interrupt_journal_removal)
    with pytest.raises(KeyboardInterrupt):
        run_team_learning_loop(tmp_path, [feedback_item()])
    monkeypatch.setattr(team_module, "_secure_project_unlink", original_unlink)

    result = run_team_learning_loop(tmp_path, [feedback_item()])

    store = ReviewStore(tmp_path)
    assert result["report"]["created_count"] == 0
    assert result["report"]["existing_count"] == 1
    assert len(store.load_candidates()) == 1
    assert len(list((store.review_dir / "team_inputs").glob("*.jsonl"))) == 1
    assert len(list((store.review_dir / "team_feedback").glob("*.jsonl"))) == 1
    assert not (store.review_dir / ".team-learning-transaction.json").exists()


@pytest.mark.parametrize("target_kind", ["candidate", "report"])
def test_team_transaction_rejects_post_preflight_symlink_target(
    tmp_path,
    monkeypatch,
    target_kind,
):
    outside = tmp_path / f"outside-{target_kind}.txt"
    outside.write_text("must remain unchanged\n")
    original_commit = team_module._commit_team_learning_transaction

    def inject_symlink(project_root, writes, **kwargs):
        if target_kind == "candidate":
            target = next(
                path for path, _ in writes if path.parent.name == "team_feedback"
            )
        else:
            target = next(
                path for path, _ in writes if path.name == "team_loop_report.json"
            )
        target.symlink_to(outside)
        return original_commit(project_root, writes, **kwargs)

    monkeypatch.setattr(
        team_module,
        "_commit_team_learning_transaction",
        inject_symlink,
    )

    with pytest.raises(ValueError, match="cannot be opened safely"):
        run_team_learning_loop(tmp_path, [feedback_item()])

    store = ReviewStore(tmp_path)
    assert outside.read_text() == "must remain unchanged\n"
    assert store.load_candidates() == []
    assert list((store.review_dir / "team_inputs").glob("*.jsonl")) == []
    assert not (store.review_dir / ".team-learning-transaction.json").exists()


def test_team_transaction_rejects_post_preflight_symlinked_parent(
    tmp_path,
    monkeypatch,
):
    outside = tmp_path / "outside-parent"
    outside.mkdir()
    original_commit = team_module._commit_team_learning_transaction

    def inject_parent_symlink(project_root, writes, **kwargs):
        target = next(
            path for path, _ in writes if path.parent.name == "team_feedback"
        )
        parent = target.parent
        parent.rename(parent.with_name("team_feedback.original"))
        parent.symlink_to(outside, target_is_directory=True)
        return original_commit(project_root, writes, **kwargs)

    monkeypatch.setattr(
        team_module,
        "_commit_team_learning_transaction",
        inject_parent_symlink,
    )

    with pytest.raises((OSError, ValueError)):
        run_team_learning_loop(tmp_path, [feedback_item()])

    store = ReviewStore(tmp_path)
    assert list(outside.iterdir()) == []
    assert store.load_candidates() == []
    assert list((store.review_dir / "team_inputs").glob("*.jsonl")) == []
    assert not (store.review_dir / ".team-learning-transaction.json").exists()


def test_team_loop_fails_closed_without_secure_descriptor_operations(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(os, "supports_dir_fd", set())

    with pytest.raises(RuntimeError, match="descriptor-relative filesystem"):
        run_team_learning_loop(tmp_path, [feedback_item()])

    assert not (
        tmp_path / ".morpheus" / "review" / "semantic_candidates.jsonl"
    ).exists()


def test_accepted_reference_requires_current_source_span(tmp_path):
    candidate = accepted_review_candidate(tmp_path)
    store = ReviewStore(tmp_path)
    store.save_candidates([candidate])
    (tmp_path / "README.md").write_text("The reviewed source changed.\n")

    with pytest.raises(ValueError, match="live source span"):
        run_team_learning_loop(tmp_path, [{
            "source_type": "accepted_review_candidate",
            "candidate_id": candidate.id,
        }])

    assert not (store.review_dir / "team_inputs").exists()


def test_accepted_reference_rejects_symlinked_live_source(tmp_path):
    candidate = accepted_review_candidate(tmp_path)
    store = ReviewStore(tmp_path)
    store.save_candidates([candidate])
    source = tmp_path / "README.md"
    target = tmp_path / "README.target.md"
    source.replace(target)
    try:
        source.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="must not be a symlink"):
        run_team_learning_loop(tmp_path, [{
            "source_type": "accepted_review_candidate",
            "candidate_id": candidate.id,
        }])

    assert not (store.review_dir / "team_inputs").exists()


def test_accepted_reference_digest_mismatch_is_atomic(tmp_path):
    candidate = accepted_review_candidate(tmp_path)
    store = ReviewStore(tmp_path)
    store.save_candidates([candidate])

    with pytest.raises(ValueError, match="digest mismatch"):
        run_team_learning_loop(tmp_path, [feedback_item(), {
            "source_type": "accepted_review_candidate",
            "candidate_id": candidate.id,
            "candidate_sha256": "0" * 64,
        }])

    assert store.load_candidates() == [candidate]
    assert not (store.review_dir / "team_inputs").exists()
    assert not (store.review_dir / "team_feedback").exists()


def test_team_loop_rejects_duplicate_review_candidate_ids_before_writes(tmp_path):
    candidate = accepted_review_candidate(tmp_path)
    store = ReviewStore(tmp_path)
    store.save_candidates([candidate, candidate])

    with pytest.raises(ValueError, match="duplicate candidate ids"):
        run_team_learning_loop(tmp_path, [feedback_item()])

    assert not (store.review_dir / "team_inputs").exists()
    assert not (store.review_dir / "team_feedback").exists()


def test_team_loop_rejects_symlinked_project_root(tmp_path):
    target = tmp_path / "target-project"
    target.mkdir()
    link = tmp_path / "linked-project"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Project root must not be a symlink"):
        run_team_learning_loop(link)

    assert not (target / ".morpheus").exists()


def test_team_loop_report_counts_review_and_check_inputs(tmp_path):
    run_team_learning_loop(tmp_path, [feedback_item()])
    store = ReviewStore(tmp_path)
    team_candidate = store.load_candidates()[0]
    store.accept(team_candidate.id, reviewed_by="tester")
    check_candidate = team_candidate.model_copy(update={
        "id": "corr_check_existing",
        "status": "pending",
        "reviewed_by": None,
        "reviewed_at": None,
        "provider": {
            "name": "morpheus-check",
            "model": "local",
            "source_label": "README.md:1",
        },
    })
    store.save_candidates([store.load_candidates()[0], check_candidate])

    result = run_team_learning_loop(tmp_path)

    report = result["report"]
    assert report["input_count"] == 0
    assert report["review_counts"] == {"accepted": 1, "pending": 1, "rejected": 0}
    assert report["accepted_review_candidate_count"] == 1
    assert report["check_correction_count"] == 1
    assert report["stale_correction_count"] == 2
    assert report["feedback_source_counts"] == {"human_correction": 1}
    assert json.loads(Path(result["json_path"]).read_text())["policy_version"] == TEAM_LOOP_POLICY_VERSION


@pytest.mark.parametrize("review_action", ["pending", "rejected"])
def test_unresolved_team_feedback_never_enters_dataset(tmp_path, review_action):
    run_team_learning_loop(tmp_path, [feedback_item()])
    store = ReviewStore(tmp_path)
    candidate = store.load_candidates()[0]
    if review_action == "rejected":
        store.reject(candidate.id, reason="not project truth", reviewed_by="tester")

    result = build_learning_dataset(
        tmp_path,
        include_corrections=True,
        include_refusals=False,
    )

    rows = read_jsonl(Path(result["dataset_dir"]) / "dataset.instruction.jsonl")
    assert all(
        row["metadata"].get("source_candidate_id") != candidate.id
        for row in rows
    )


def test_accepted_team_feedback_becomes_explicit_correction_examples(tmp_path):
    run_team_learning_loop(tmp_path, [feedback_item()])
    store = ReviewStore(tmp_path)
    candidate = store.load_candidates()[0]
    store.accept(candidate.id, reviewed_by="tester")

    result = build_learning_dataset(
        tmp_path,
        include_corrections=True,
        include_refusals=False,
    )

    dataset_dir = Path(result["dataset_dir"])
    examples = read_jsonl(dataset_dir / "dataset.instruction.jsonl")
    correction_example = next(
        item
        for item in examples
        if item["metadata"].get("source_candidate_id") == candidate.id
    )
    eval_item = next(
        item
        for item in read_jsonl(dataset_dir / "eval.seed.jsonl")
        if item.get("source_candidate_id") == candidate.id
    )
    heldout_item = next(
        item
        for item in read_jsonl(dataset_dir / "eval.heldout.jsonl")
        if item.get("source_candidate_id") == candidate.id
    )
    expected = "No. Morpheus trains only from accepted source-backed candidates."
    assert correction_example["output"] == expected
    assert eval_item["expected_answer"] == expected
    assert heldout_item["expected_answer"] == expected


def test_tampered_team_correction_is_not_source_bound_or_trainable(tmp_path):
    run_team_learning_loop(tmp_path, [feedback_item()])
    store = ReviewStore(tmp_path)
    candidate = store.accept(store.load_candidates()[0].id, reviewed_by="tester")
    store.save_candidates([
        candidate.model_copy(update={"correction_text": "A replacement not present in evidence."})
    ])

    result = build_learning_dataset(
        tmp_path,
        include_corrections=True,
        include_refusals=False,
    )

    skipped = read_jsonl(Path(result["dataset_dir"]) / "skipped.jsonl")
    assert skipped[0]["candidate_id"] == candidate.id
    assert skipped[0]["reason"] == "team_feedback_projection_mismatch"
    quality = build_quality_report(tmp_path)
    decision = quality["routing"]["decisions"][0]
    assert decision["memory_route"] == "human_review"
    assert decision["trainability_reason"] == "team_feedback_projection_mismatch"


def test_secret_in_tampered_correction_text_never_enters_dataset(tmp_path):
    run_team_learning_loop(tmp_path, [feedback_item()])
    store = ReviewStore(tmp_path)
    candidate = store.accept(store.load_candidates()[0].id, reviewed_by="tester")
    secret = "The api key is sk-abcdefghijklmnopqrstuvwxyz123456."
    store.save_candidates([candidate.model_copy(update={"correction_text": secret})])

    result = build_learning_dataset(
        tmp_path,
        include_corrections=True,
        include_refusals=False,
    )

    dataset_dir = Path(result["dataset_dir"])
    assert secret not in (dataset_dir / "dataset.instruction.jsonl").read_text()
    skipped = read_jsonl(dataset_dir / "skipped.jsonl")
    assert skipped[0]["candidate_id"] == candidate.id
    assert skipped[0]["reason"] == "secret_like"


def test_review_apply_skips_accepted_correction_from_active_state(tmp_path):
    (tmp_path / "README.md").write_text("Morpheus keeps reviewed project state.\n")
    MorpheusConfig(project_root=tmp_path).init_default()
    run_team_learning_loop(tmp_path, [feedback_item()])
    store = ReviewStore(tmp_path)
    candidate = store.load_candidates()[0]
    store.accept(candidate.id, reviewed_by="tester")

    result = apply_accepted_candidates(tmp_path)

    assert result["accepted_applied"] == 0
    assert result["accepted_corrections_skipped"] == 1
    state = json.loads((tmp_path / ".morpheus" / "state.json").read_text())
    assert all(claim["excerpt"] != candidate.claim for claim in state["claims"])
