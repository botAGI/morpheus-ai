"""
Tests for morpheus.core.provenance
"""
import pytest
import tempfile
from pathlib import Path
from morpheus.core.provenance import (
    compute_sha256_file,
    compute_sha256_bytes,
    build_receipt
)


def test_compute_sha256_bytes():
    result = compute_sha256_bytes(b"hello world")
    assert len(result) == 64
    assert result == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_compute_sha256_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("hello world")
        f.flush()
        path = Path(f.name)
    
    try:
        result = compute_sha256_file(path)
        assert len(result) == 64
        assert result == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    finally:
        path.unlink()


def test_build_receipt_basic():
    state = {
        "project": {"name": "test-project"},
        "claims": [
            {"id": "clm_001", "status": "active"},
            {"id": "clm_002", "status": "active"},
            {"id": "clm_003", "status": "superseded"}
        ],
        "evidence": []
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        priv_path = Path(tmpdir) / "private.key"
        
        # Generate a test key - RAW bytes format (32 bytes)
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
        private_key = ed25519.Ed25519PrivateKey.generate()
        priv_bytes = private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption()
        )
        priv_path.write_bytes(priv_bytes)
        
        receipt = build_receipt(
            state_dict=state,
            wake_md_sha="wake_sha_12345678",
            sources_data=[{"id": "src_001", "path": "README.md"}],
            private_key_path=priv_path,
            prev_hash=None
        )
        
        assert receipt["schema_version"] == "morpheus-receipt/1"
        assert "rcpt_" in receipt["receipt_id"]
        assert receipt["claim_count"]["active"] == 2
        assert receipt["claim_count"]["superseded"] == 1
        assert receipt["previous_receipt_sha256"] is None
        assert receipt["signature"]["algo"] == "ed25519"
        assert len(receipt["signature"]["signature_b64"]) > 0


def test_build_receipt_with_previous():
    state = {"project": {"name": "test"}, "claims": [], "evidence": []}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        priv_path = Path(tmpdir) / "private.key"
        
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
        private_key = ed25519.Ed25519PrivateKey.generate()
        priv_bytes = private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption()
        )
        priv_path.write_bytes(priv_bytes)
        
        receipt = build_receipt(
            state_dict=state,
            wake_md_sha="wake_sha",
            sources_data=[],
            private_key_path=priv_path,
            prev_hash="rcpt_previous_123"
        )
        
        assert receipt["previous_receipt_sha256"] == "rcpt_previous_123"


def test_build_receipt_no_key():
    state = {"project": {"name": "test"}, "claims": [], "evidence": []}
    
    receipt = build_receipt(
        state_dict=state,
        wake_md_sha="wake_sha",
        sources_data=[],
        private_key_path=Path("/nonexistent/key"),
        prev_hash=None
    )
    
    # Should still build receipt, just with empty signature
    assert receipt["schema_version"] == "morpheus-receipt/1"
    assert receipt["signature"]["signature_b64"] == ""
