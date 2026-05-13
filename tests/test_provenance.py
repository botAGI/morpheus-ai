"""
Tests for morpheus.core.provenance
"""
import hashlib
import tempfile
from pathlib import Path

import pytest

from morpheus.core.provenance import (
    compute_sha256_file,
    compute_sha256_bytes,
    build_receipt,
    receipt_file_name,
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


def test_compute_sha256_file_streams_large_files(monkeypatch):
    content = b"a" * (1024 * 1024 + 17)
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    def fail_read_bytes(self):
        raise AssertionError("compute_sha256_file should stream file content")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    try:
        result = compute_sha256_file(path)
        assert len(result) == 64
        assert result == hashlib.sha256(content).hexdigest()
    finally:
        path.unlink()


def test_receipt_file_name_rejects_path_separators():
    for receipt_id in ["../evil", "nested/evil", "nested\\evil", "", ".", ".."]:
        with pytest.raises(ValueError, match="invalid receipt id"):
            receipt_file_name(receipt_id)


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
        
        previous_sha = "f" * 64
        receipt = build_receipt(
            state_dict=state,
            wake_md_sha="wake_sha",
            sources_data=[],
            private_key_path=priv_path,
            prev_hash=previous_sha,
        )
        
        assert receipt["previous_receipt_sha256"] == previous_sha


def test_build_receipt_requires_private_key():
    state = {"project": {"name": "test"}, "claims": [], "evidence": []}

    with pytest.raises(FileNotFoundError, match="private signing key"):
        build_receipt(
            state_dict=state,
            wake_md_sha="wake_sha",
            sources_data=[],
            private_key_path=Path("/nonexistent/key"),
            prev_hash=None,
        )


def test_build_receipt_rejects_invalid_receipt_id_before_signing(tmp_path):
    private_key_path = tmp_path / "private.key"

    with pytest.raises(ValueError, match="invalid receipt id"):
        build_receipt(
            state_dict={"claims": [], "evidence": []},
            wake_md_sha="wake_sha",
            sources_data=[],
            private_key_path=private_key_path,
            receipt_id="../evil",
        )
