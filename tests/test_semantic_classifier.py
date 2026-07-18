from datetime import datetime, timezone
import hashlib

import pytest

from morpheus.core.providers.fake import FakeProvider
from morpheus.core.semantic.classifier import classify_candidate
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.review import ReviewStore, run_semantic_review, semantic_report
from morpheus.core.semantic.routing import route_candidate


PROMPT_SHA = "1" * 64


def candidate(
    *,
    kind: str = "current_state",
    claim: str = "Morpheus builds source-backed project state.",
    source_path: str = "README.md",
    label: str = "source_backed",
    status: str = "pending",
    semantic_class: str = "unknown",
) -> SemanticCandidate:
    return SemanticCandidate(
        id="cand_test",
        run_id="semrun_test",
        kind=kind,
        claim=claim,
        source_path=source_path,
        source_sha256="a" * 64,
        source_mtime=datetime.now(timezone.utc),
        source_revision="git:test",
        line_start=1,
        line_end=1,
        evidence_excerpt=claim,
        evidence_sha256=hashlib.sha256(claim.encode()).hexdigest(),
        confidence=0.91,
        label=label,
        semantic_class=semantic_class,
        status=status,
        provider={"name": "test", "model": "fixture"},
        prompt_sha256=PROMPT_SHA,
    )


def test_classify_candidate_uses_kind_source_and_claim_signals():
    assert classify_candidate(candidate(kind="outdated_claim")) == "stale"
    assert classify_candidate(candidate(kind="open_task")) == "open_task"
    assert classify_candidate(candidate(kind="agent_rule")) == "convention"
    assert classify_candidate(candidate(
        kind="agent_rule",
        claim="Never train on raw markdown or secrets.",
    )) == "security"
    assert classify_candidate(candidate(source_path="pyproject.toml")) == "command"
    assert classify_candidate(candidate(claim="Never train on raw markdown or secrets.")) == "security"
    assert classify_candidate(candidate(claim="Morpheus serve exposes MCP truth tools.")) == "integration"
    assert classify_candidate(candidate(claim="WAKE.md is compiled project state.")) == "architecture"
    assert classify_candidate(candidate(claim="Morpheus is a verified learning layer.")) == "product"


def test_classify_candidate_routes_every_real_morpheus_cli_family_as_command():
    claims = [
        "morpheus init initializes .morpheus",
        "morpheus learn dataset . builds reviewed artifacts",
        "morpheus model-smoke --base-model qwen2.5:0.5b checks the model",
        "morpheus bootstrap-agent writes the agent bootstrap",
        "morpheus eval evaluates a legacy adapter",
    ]

    assert {
        classify_candidate(candidate(claim=claim))
        for claim in claims
    } == {"command"}


@pytest.mark.parametrize(
    "claim",
    [
        "Morpheus learns from reviewed project evidence.",
        "Morpheus trains adapter weights from reviewed datasets.",
        "Morpheus initialization produces compiled project state.",
        "Morpheus learning uses the truth layer.",
    ],
)
def test_classify_candidate_does_not_treat_command_prefix_prose_as_cli(claim):
    assert classify_candidate(candidate(claim=claim)) == "architecture"


@pytest.mark.parametrize(
    "claim",
    [
        "Morpheus provides a client library.",
        "Morpheus clicks through the workflow.",
        "The truffle formatter is optional.",
    ],
)
def test_classify_candidate_requires_boundaries_for_command_tool_terms(claim):
    assert classify_candidate(candidate(claim=claim)) != "command"


@pytest.mark.parametrize(
    "claim",
    [
        "The CLI prints reviewed state.",
        "Run ruff before committing.",
        "Use pytest for the suite.",
        "Pass --json to print machine output.",
    ],
)
def test_classify_candidate_preserves_exact_command_tool_terms(claim):
    assert classify_candidate(candidate(claim=claim)) == "command"


def test_semantic_report_counts_candidate_classes():
    candidates = [
        candidate(kind="outdated_claim"),
        candidate(kind="agent_rule"),
        candidate(source_path="pyproject.toml"),
    ]

    report = semantic_report(
        run_id="semrun_test",
        provider=FakeProvider(),
        sources_count=1,
        candidates=candidates,
    )

    assert report["by_class"] == {
        "command": 1,
        "convention": 1,
        "stale": 1,
    }


def test_run_semantic_review_persists_candidate_class(tmp_path):
    (tmp_path / "README.md").write_text(
        "Morpheus is a verified learning layer.\n"
        "DECISION: Never train on raw markdown.\n"
    )

    report = run_semantic_review(tmp_path, provider=FakeProvider())
    candidates = ReviewStore(tmp_path).load_candidates()

    assert report["by_class"]["product"] == 1
    assert report["by_class"]["security"] == 1
    assert {candidate.semantic_class for candidate in candidates} == {"product", "security"}


def test_route_candidate_blocks_pending_and_inferred_candidates_from_training():
    pending = route_candidate(candidate(status="pending"))
    inferred = route_candidate(candidate(status="accepted", label="inferred"))

    assert pending.trainability_status == "needs_review"
    assert pending.memory_route == "human_review"
    assert pending.trainability_reason == "status_pending"
    assert inferred.trainability_status == "excluded"
    assert inferred.memory_route == "excluded"
    assert inferred.trainability_reason == "label_inferred"


def test_route_candidate_keeps_stale_and_open_tasks_out_of_positive_training():
    stale = route_candidate(candidate(kind="outdated_claim", status="accepted"))
    task = route_candidate(candidate(kind="open_task", status="accepted"))

    assert stale.trainability_status == "negative_example"
    assert stale.memory_route == "negative_example"
    assert task.trainability_status == "retrievable"
    assert task.memory_route == "prompt_context"


def test_route_candidate_archives_stale_class_with_non_outdated_kind():
    stale = route_candidate(candidate(
        kind="current_state",
        status="accepted",
        semantic_class="stale",
    ))

    assert stale.trainability_status == "needs_review"
    assert stale.memory_route == "stale_archive"
    assert stale.trainability_reason == "stale_class_requires_outdated_claim_kind"


def test_route_candidate_assigns_every_public_memory_channel():
    decisions = {
        "active_decision": route_candidate(candidate(kind="active_decision", status="accepted")),
        "open_task": route_candidate(candidate(kind="open_task", status="accepted")),
        "source_reference": route_candidate(candidate(kind="source_reference", status="accepted")),
        "temporary": route_candidate(candidate(status="accepted", semantic_class="temporary")),
        "negative_example": route_candidate(candidate(kind="outdated_claim", status="accepted")),
        "stale_archive": route_candidate(candidate(kind="outdated_claim", status="pending")),
        "human_review": route_candidate(candidate(status="pending")),
        "excluded_rejected": route_candidate(candidate(status="rejected", label="needs_review")),
        "excluded_inferred": route_candidate(candidate(status="accepted", label="inferred")),
        "excluded_unsafe": route_candidate(candidate(
            status="accepted",
            claim="The API token is secret-value-1234567890.",
        )),
        "adapter_training": route_candidate(candidate(status="accepted")),
    }

    assert decisions["active_decision"].memory_route == "adapter_training"
    assert decisions["open_task"].memory_route == "prompt_context"
    assert decisions["source_reference"].memory_route == "retrieval"
    assert decisions["temporary"].memory_route == "eval_only"
    assert decisions["negative_example"].memory_route == "negative_example"
    assert decisions["stale_archive"].memory_route == "stale_archive"
    assert decisions["human_review"].memory_route == "human_review"
    assert decisions["excluded_rejected"].memory_route == "excluded"
    assert decisions["excluded_inferred"].memory_route == "excluded"
    assert decisions["excluded_unsafe"].memory_route == "excluded"
    assert decisions["adapter_training"].memory_route == "adapter_training"


def test_route_candidate_uses_freshly_classified_temporary_class():
    routed = route_candidate(candidate(
        status="accepted",
        claim="This temporary migration fact expires after rollout.",
    ))

    assert routed.semantic_class == "temporary"
    assert routed.trainability_status == "eval_only"
    assert routed.memory_route == "eval_only"


def test_semantic_report_counts_trainability_and_memory_routes():
    candidates = [
        route_candidate(candidate(status="accepted")),
        route_candidate(candidate(kind="outdated_claim", status="accepted")),
        route_candidate(candidate(status="pending")),
    ]

    report = semantic_report(
        run_id="semrun_test",
        provider=FakeProvider(),
        sources_count=1,
        candidates=candidates,
    )

    assert report["by_trainability"] == {
        "negative_example": 1,
        "needs_review": 1,
        "trainable": 1,
    }
    assert report["by_route"] == {
        "adapter_training": 1,
        "human_review": 1,
        "negative_example": 1,
    }


def test_review_accept_recomputes_trainability_route(tmp_path):
    (tmp_path / "README.md").write_text("Morpheus generates WAKE.md for AI agents.\n")
    run_semantic_review(tmp_path, provider=FakeProvider())
    store = ReviewStore(tmp_path)
    candidate_id = store.load_candidates()[0].id

    store.accept(candidate_id, reviewed_by="tester")
    accepted = store.load_candidates()[0]

    assert accepted.status == "accepted"
    assert accepted.trainability_status == "trainable"
    assert accepted.memory_route == "adapter_training"
    assert accepted.trainability_reason == "accepted_source_backed_stable_claim"
