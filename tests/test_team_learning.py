import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest

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
from morpheus.core.semantic.verifier import verify_candidate_span
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
    assert len(candidates) == worker_count
    assert len({candidate.id for candidate in candidates}) == worker_count
    assert len(artifacts) == worker_count


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
