"""
Tests for morpheus.core.models
"""
from datetime import datetime, timezone
from morpheus.core.models import Source, Claim, Evidence, ProjectState, Receipt


def test_source_model():
    s = Source(
        id="src_001",
        path="README.md",
        kind="markdown",
        sha256="abc123",
        size_bytes=1024,
        line_count=50,
        modified_at=datetime.now(timezone.utc)
    )
    assert s.id == "src_001"
    assert s.path == "README.md"
    assert s.kind == "markdown"
    assert s.sha256 == "abc123"


def test_claim_defaults():
    c = Claim(
        id="clm_001",
        source_id="src_001",
        line_start=10,
        line_end=10,
        excerpt="TODO: implement this"
    )
    assert c.status == "active"
    assert c.category == "fact"
    assert c.inference is False


def test_evidence_model():
    e = Evidence(
        id="ev_001",
        claim_id="clm_001",
        source_id="src_001",
        path="README.md",
        line_start=10,
        line_end=10,
        excerpt="TODO: implement this",
        source_sha256="abc123",
        excerpt_sha256="def456",
        timestamp=datetime.now(timezone.utc)
    )
    assert e.id == "ev_001"
    assert e.claim_id == "clm_001"


def test_project_state():
    s = Source(
        id="src_001",
        path="README.md",
        kind="markdown",
        sha256="abc123",
        size_bytes=1024,
        line_count=50,
        modified_at=datetime.now(timezone.utc)
    )
    c = Claim(
        id="clm_001",
        source_id="src_001",
        line_start=10,
        line_end=10,
        excerpt="TODO: test",
        category="task"
    )
    
    state = ProjectState(
        sources=[s],
        claims=[c],
        evidence=[],
        compiled_at=datetime.now(timezone.utc),
        receipt_id="rcpt_test_001"
    )
    
    assert len(state.sources) == 1
    assert len(state.claims) == 1
    assert state.receipt_id == "rcpt_test_001"


def test_receipt_model():
    r = Receipt(
        schema_version="morpheus-receipt/1",
        receipt_id="rcpt_test_001",
        project={"name": "test"},
        wake_md_sha256="wake123",
        state_json_sha256="state123",
        evidence_jsonl_sha256="ev123",
        sources=[],
        claim_count={"active": 5, "superseded": 1, "unverified": 0},
        tool={"name": "morpheus", "version": "0.1.0"},
        issued_at="2026-05-13T12:00:00Z",
        previous_receipt_sha256=None,
        signature={"algo": "ed25519", "key_id": "test", "signature_b64": "sig123"}
    )
    
    assert r.schema_version == "morpheus-receipt/1"
    assert r.receipt_id == "rcpt_test_001"
    assert r.claim_count["active"] == 5
    assert r.issued_at == "2026-05-13T12:00:00Z"
