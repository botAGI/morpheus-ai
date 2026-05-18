import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.scanner import scan_semantic_sources
from morpheus.core.semantic.verifier import verify_candidate_span


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_semantic_scanner_groups_high_signal_sources(tmp_path):
    write(tmp_path / "README.md", "# Demo\n")
    write(tmp_path / "WAKE.md", "# WAKE\n")
    write(tmp_path / "docs" / "WHY.md", "Docs\n")
    write(tmp_path / "pyproject.toml", "[project]\nname='demo'\n")
    write(tmp_path / ".github" / "workflows" / "ci.yml", "name: CI\n")
    write(tmp_path / "morpheus" / "cli.py", "print('cli')\n")
    write(tmp_path / "tests" / "fixtures" / "README.md", "fixture\n")
    write(tmp_path / "notes.txt", "low signal\n")

    sources = scan_semantic_sources(tmp_path)

    by_path = {source.path: source for source in sources}
    assert by_path["README.md"].category == "docs_state_sources"
    assert by_path["WAKE.md"].category == "docs_state_sources"
    assert by_path["docs/WHY.md"].category == "docs_state_sources"
    assert by_path["pyproject.toml"].category == "build_manifest_sources"
    assert by_path[".github/workflows/ci.yml"].category == "workflow_sources"
    assert by_path["morpheus/cli.py"].category == "cli_api_sources"
    assert "tests/fixtures/README.md" not in by_path
    assert "notes.txt" not in by_path


def test_semantic_scanner_respects_morpheusignore_and_skips_secrets(tmp_path):
    write(tmp_path / ".morpheusignore", "docs/private.md\n")
    write(tmp_path / "README.md", "# Public\n")
    write(tmp_path / "docs" / "private.md", "DECISION: do not scan\n")
    write(tmp_path / ".env", "OPENAI_API_KEY=secret\n")
    write(tmp_path / "deploy.key", "-----BEGIN PRIVATE KEY-----\nsecret\n")
    write(tmp_path / "docs" / "token.md", "token = 'sk-" + "a" * 80 + "'\n")

    sources = scan_semantic_sources(tmp_path)

    paths = {source.path for source in sources}
    assert paths == {"README.md"}


def test_semantic_candidate_requires_source_span_and_prompt_hash(tmp_path):
    write(tmp_path / "README.md", "Morpheus writes WAKE.md.\n")
    source = scan_semantic_sources(tmp_path)[0]

    candidate = SemanticCandidate(
        id="cand_20260518T120000Z_abcd1234",
        run_id="semrun_20260518T120000Z",
        kind="current_state",
        claim="Morpheus writes WAKE.md.",
        source_path=source.path,
        source_sha256=source.sha256,
        source_mtime=source.modified_at,
        source_revision="git:unknown",
        line_start=1,
        line_end=1,
        evidence_excerpt="Morpheus writes WAKE.md.",
        evidence_sha256="0" * 64,
        confidence=0.9,
        label="needs_review",
        status="pending",
        created_at=datetime.now(timezone.utc),
        provider={"name": "fake", "model": "fixture"},
        prompt_sha256="1" * 64,
    )

    payload = candidate.model_dump(mode="json")
    assert payload["source_path"] == "README.md"
    assert payload["line_start"] == 1
    assert payload["prompt_sha256"] == "1" * 64

    del payload["prompt_sha256"]
    with pytest.raises(ValidationError):
        SemanticCandidate.model_validate(payload)


def test_verify_candidate_span_marks_exact_match_source_backed(tmp_path):
    write(tmp_path / "README.md", "Intro\nMorpheus writes WAKE.md.\n")
    source = scan_semantic_sources(tmp_path)[0]
    candidate = SemanticCandidate(
        id="cand_20260518T120000Z_abcd1234",
        run_id="semrun_20260518T120000Z",
        kind="current_state",
        claim="Morpheus writes WAKE.md.",
        source_path="README.md",
        source_sha256=source.sha256,
        source_mtime=source.modified_at,
        source_revision="git:unknown",
        line_start=2,
        line_end=2,
        evidence_excerpt="Morpheus writes WAKE.md.",
        evidence_sha256="0" * 64,
        confidence=0.9,
        label="needs_review",
        status="pending",
        created_at=datetime.now(timezone.utc),
        provider={"name": "fake", "model": "fixture"},
        prompt_sha256="1" * 64,
    )

    verified = verify_candidate_span(tmp_path, candidate)

    assert verified.label == "source_backed"
    assert verified.status == "pending"
    assert verified.evidence_sha256 != "0" * 64


def test_verify_candidate_span_does_not_self_confirm_wake_md(tmp_path):
    write(tmp_path / "WAKE.md", "Morpheus generates WAKE.md.\n")
    source = scan_semantic_sources(tmp_path)[0]
    candidate = SemanticCandidate(
        id="cand_20260518T120000Z_abcd1234",
        run_id="semrun_20260518T120000Z",
        kind="current_state",
        claim="Morpheus generates WAKE.md.",
        source_path="WAKE.md",
        source_sha256=source.sha256,
        source_mtime=source.modified_at,
        source_revision="git:unknown",
        line_start=1,
        line_end=1,
        evidence_excerpt="Morpheus generates WAKE.md.",
        evidence_sha256="0" * 64,
        confidence=0.9,
        label="needs_review",
        status="pending",
        created_at=datetime.now(timezone.utc),
        provider={"name": "fake", "model": "fixture"},
        prompt_sha256="1" * 64,
    )

    verified = verify_candidate_span(tmp_path, candidate)

    assert verified.label == "needs_review"


def test_verify_candidate_span_uses_fuzzy_match_and_rejects_bad_spans(tmp_path):
    write(tmp_path / "README.md", "Morpheus generates WAKE.md for AI agents.\n")
    source = scan_semantic_sources(tmp_path)[0]
    payload = {
        "id": "cand_20260518T120000Z_abcd1234",
        "run_id": "semrun_20260518T120000Z",
        "kind": "current_state",
        "claim": "Morpheus generates WAKE.md.",
        "source_path": "README.md",
        "source_sha256": source.sha256,
        "source_mtime": source.modified_at,
        "source_revision": "git:unknown",
        "line_start": 1,
        "line_end": 1,
        "evidence_excerpt": "Morpheus generates WAKE.md for agents.",
        "evidence_sha256": "0" * 64,
        "confidence": 0.9,
        "label": "needs_review",
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "provider": {"name": "fake", "model": "fixture"},
        "prompt_sha256": "1" * 64,
    }

    fuzzy = verify_candidate_span(tmp_path, SemanticCandidate(**payload))
    assert fuzzy.label == "source_backed"

    payload["line_start"] = 20
    payload["line_end"] = 20
    bad = verify_candidate_span(tmp_path, SemanticCandidate(**payload))
    assert bad.label == "needs_review"


def test_semantic_candidates_are_jsonl_serializable(tmp_path):
    write(tmp_path / "README.md", "Morpheus writes WAKE.md.\n")
    source = scan_semantic_sources(tmp_path)[0]
    candidate = SemanticCandidate(
        id="cand_20260518T120000Z_abcd1234",
        run_id="semrun_20260518T120000Z",
        kind="current_state",
        claim="Morpheus writes WAKE.md.",
        source_path=source.path,
        source_sha256=source.sha256,
        source_mtime=source.modified_at,
        source_revision="git:unknown",
        line_start=1,
        line_end=1,
        evidence_excerpt="Morpheus writes WAKE.md.",
        evidence_sha256="0" * 64,
        confidence=0.9,
        label="needs_review",
        status="pending",
        created_at=datetime.now(timezone.utc),
        provider={"name": "fake", "model": "fixture"},
        prompt_sha256="1" * 64,
    )

    line = candidate.model_dump_json()

    assert json.loads(line)["id"] == "cand_20260518T120000Z_abcd1234"
