import json
from pathlib import Path

import httpx

from morpheus.core.providers.fake import FakeProvider
from morpheus.core.providers.local import LocalProvider
from morpheus.core.providers.null import NullProvider
from morpheus.core.providers.ollama import OllamaProvider
from morpheus.core.semantic.review import ReviewStore, run_semantic_review
from morpheus.core.semantic.scanner import scan_semantic_sources


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_null_provider_extracts_no_candidates(tmp_path):
    write(tmp_path / "README.md", "Morpheus generates WAKE.md.\n")
    source = scan_semantic_sources(tmp_path)[0]

    candidates = NullProvider().extract_candidates(
        source,
        run_id="semrun_test",
        prompt_sha256="1" * 64,
        source_revision="git:unknown",
    )

    assert candidates == []


def test_ollama_provider_extracts_candidates_from_local_generate_response(tmp_path):
    write(tmp_path / "README.md", "Morpheus generates WAKE.md for AI agents.\n")
    source = scan_semantic_sources(tmp_path)[0]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        payload = json.loads(request.content)
        assert payload["stream"] is False
        return httpx.Response(
            200,
            json={
                "response": json.dumps({
                    "candidates": [
                        {
                            "kind": "current_state",
                            "claim": "Morpheus generates WAKE.md for AI agents.",
                            "line_start": 1,
                            "line_end": 1,
                            "evidence_excerpt": "Morpheus generates WAKE.md for AI agents.",
                            "confidence": 0.91,
                        }
                    ]
                })
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(client=client, base_url="http://127.0.0.1:11434", model="qwen-test")

    candidates = provider.extract_candidates(
        source,
        run_id="semrun_test",
        prompt_sha256="1" * 64,
        source_revision="git:unknown",
    )

    assert len(candidates) == 1
    assert candidates[0].provider == {"name": "ollama", "model": "qwen-test"}
    assert candidates[0].source_path == "README.md"


def test_fake_provider_extracts_fixture_candidates_with_source_spans(tmp_path):
    write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# Demo",
                "Morpheus generates WAKE.md for AI agents.",
                "DECISION: Keep semantic mode review-gated.",
                "TODO: Add richer stale-claim detection.",
                "",
            ]
        ),
    )
    source = scan_semantic_sources(tmp_path)[0]

    candidates = FakeProvider().extract_candidates(
        source,
        run_id="semrun_test",
        prompt_sha256="1" * 64,
        source_revision="git:unknown",
    )

    assert [candidate.kind for candidate in candidates] == [
        "current_state",
        "active_decision",
        "open_task",
    ]
    assert {candidate.line_start for candidate in candidates} == {2, 3, 4}
    assert all(candidate.source_path == "README.md" for candidate in candidates)
    assert all(candidate.prompt_sha256 == "1" * 64 for candidate in candidates)


def test_local_provider_caps_candidates_per_source_for_reviewability(tmp_path):
    write(
        tmp_path / "README.md",
        "\n".join(f"Morpheus generates WAKE.md candidate {idx}." for idx in range(20)),
    )
    source = scan_semantic_sources(tmp_path)[0]

    candidates = LocalProvider().extract_candidates(
        source,
        run_id="semrun_test",
        prompt_sha256="1" * 64,
        source_revision="git:unknown",
    )

    assert len(candidates) == 8


def test_local_provider_treats_wake_md_as_context_not_candidate_source(tmp_path):
    write(tmp_path / "WAKE.md", "Morpheus generates WAKE.md candidate.\n")
    source = scan_semantic_sources(tmp_path)[0]

    candidates = LocalProvider().extract_candidates(
        source,
        run_id="semrun_test",
        prompt_sha256="1" * 64,
        source_revision="git:unknown",
    )

    assert candidates == []


def test_run_semantic_review_writes_jsonl_report_and_draft(tmp_path):
    write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# Demo",
                "Morpheus generates WAKE.md for AI agents.",
                "DECISION: Keep semantic mode review-gated.",
                "TODO: Add richer stale-claim detection.",
                "",
            ]
        ),
    )
    store = ReviewStore(tmp_path)

    report = run_semantic_review(tmp_path, provider=FakeProvider())

    assert report["provider"]["name"] == "fake"
    assert report["candidates_total"] == 3
    assert report["source_backed_total"] == 3
    assert store.candidates_path.is_file()
    assert store.report_path.is_file()
    assert store.draft_wake_path.is_file()
    assert "Morpheus generates WAKE.md" in store.draft_wake_path.read_text()

    lines = store.candidates_path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["label"] == "source_backed"


def test_review_store_accepts_and_rejects_candidates(tmp_path):
    write(
        tmp_path / "README.md",
        "Morpheus generates WAKE.md for AI agents.\nTODO: Add semantic review.\n",
    )
    run_semantic_review(tmp_path, provider=FakeProvider())
    store = ReviewStore(tmp_path)
    candidates = store.load_candidates()

    store.accept(candidates[0].id, reviewed_by="tester")
    store.reject(candidates[1].id, reason="too broad", reviewed_by="tester")
    updated = {candidate.id: candidate for candidate in store.load_candidates()}

    assert updated[candidates[0].id].status == "accepted"
    assert updated[candidates[0].id].reviewed_by == "tester"
    assert updated[candidates[1].id].status == "rejected"
    assert updated[candidates[1].id].review_reason == "too broad"


def test_review_store_diff_reports_pending_accepted_rejected(tmp_path):
    write(
        tmp_path / "README.md",
        "Morpheus generates WAKE.md for AI agents.\nTODO: Add semantic review.\n",
    )
    run_semantic_review(tmp_path, provider=FakeProvider())
    store = ReviewStore(tmp_path)
    candidates = store.load_candidates()
    store.accept(candidates[0].id, reviewed_by="tester")

    diff = store.diff()

    assert diff["accepted"] == 1
    assert diff["pending"] == 1
    assert diff["rejected"] == 0


def test_review_apply_rechecks_source_spans_before_promoting(tmp_path):
    write(tmp_path / "README.md", "Morpheus generates WAKE.md for AI agents.\n")
    run_semantic_review(tmp_path, provider=FakeProvider())
    store = ReviewStore(tmp_path)
    candidate = store.load_candidates()[0]
    store.accept(candidate.id, reviewed_by="tester")
    write(tmp_path / "README.md", "The source changed after review.\n")

    from morpheus.core.config import MorpheusConfig
    from morpheus.core.semantic.review import apply_accepted_candidates

    MorpheusConfig(project_root=tmp_path).init_default()
    result = apply_accepted_candidates(tmp_path)
    updated = store.load_candidates()[0]

    assert result["accepted_applied"] == 0
    assert updated.status == "pending"
    assert updated.label == "needs_review"
    assert updated.review_reason == "source span changed before apply"
