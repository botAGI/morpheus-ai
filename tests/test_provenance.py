"""
Tests for morpheus.core.provenance
"""
import hashlib
import tempfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from morpheus.core.provenance import (
    compute_sha256_file,
    compute_sha256_bytes,
    build_receipt,
    latest_receipt_file,
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


def test_compute_sha256_file_rejects_symlinked_files(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="must not be a symlink"):
        compute_sha256_file(link)


def test_compute_sha256_file_rejects_symlinked_parent_directory(tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "source.txt").write_text("secret")
    linked_dir = tmp_path / "linked"
    try:
        linked_dir.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Hash input path must not contain a symlink"):
        compute_sha256_file(linked_dir / "source.txt")


def test_receipt_file_name_rejects_path_separators():
    for receipt_id in ["../evil", "nested/evil", "nested\\evil", "", ".", ".."]:
        with pytest.raises(ValueError, match="invalid receipt id"):
            receipt_file_name(receipt_id)


def test_latest_receipt_file_rejects_non_object_receipt_json(tmp_path):
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    (receipts_dir / "receipt_bad.json").write_text("[]")

    with pytest.raises(ValueError, match="receipt_bad.json: expected JSON object"):
        latest_receipt_file(receipts_dir)


def test_latest_receipt_file_reports_unreadable_receipt_files(tmp_path):
    receipts_dir = tmp_path / "receipts"
    (receipts_dir / "receipt_bad.json").mkdir(parents=True)

    with pytest.raises(ValueError, match="receipt_bad.json: unreadable receipt"):
        latest_receipt_file(receipts_dir)


def test_latest_receipt_file_rejects_symlinked_receipt_file(tmp_path):
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    outside_receipt = tmp_path / "outside-receipt.json"
    outside_receipt.write_text('{"receipt_id": "outside", "previous_receipt_sha256": null}')
    (receipts_dir / "receipt_outside.json").symlink_to(outside_receipt)

    with pytest.raises(ValueError, match="receipt_outside.json: must not be a symlink"):
        latest_receipt_file(receipts_dir)


def test_latest_receipt_file_rejects_receipts_path_file(tmp_path):
    receipts_dir = tmp_path / "receipts"
    receipts_dir.write_text("not a directory")

    with pytest.raises(ValueError, match="receipts path is not a directory"):
        latest_receipt_file(receipts_dir)


def test_latest_receipt_file_rejects_receipts_path_symlink(tmp_path):
    outside_receipts = tmp_path / "outside-receipts"
    outside_receipts.mkdir()
    receipts_dir = tmp_path / "receipts"
    receipts_dir.symlink_to(outside_receipts, target_is_directory=True)

    with pytest.raises(ValueError, match="receipts path must not be a symlink"):
        latest_receipt_file(receipts_dir)


def test_latest_receipt_file_rejects_symlinked_receipts_ancestor(tmp_path):
    outside_morpheus = tmp_path / "outside-morpheus"
    outside_receipts = outside_morpheus / "receipts"
    outside_receipts.mkdir(parents=True)
    (outside_receipts / "receipt_outside.json").write_text(
        '{"receipt_id": "outside", "previous_receipt_sha256": null}'
    )
    morpheus_dir = tmp_path / ".morpheus"
    try:
        morpheus_dir.symlink_to(outside_morpheus, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="receipts path must not contain a symlink"):
        latest_receipt_file(morpheus_dir / "receipts")


def test_latest_receipt_file_rejects_non_string_previous_hash(tmp_path):
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    (receipts_dir / "receipt_bad.json").write_text(
        '{"receipt_id": "bad", "previous_receipt_sha256": {"not": "a hash"}}'
    )

    with pytest.raises(
        ValueError,
        match="receipt_bad.json: previous_receipt_sha256 must be string or null",
    ):
        latest_receipt_file(receipts_dir)


def test_latest_receipt_file_rejects_chain_without_root(tmp_path):
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    (receipts_dir / "receipt_orphan.json").write_text(
        '{"receipt_id": "orphan", "previous_receipt_sha256": "' + ("a" * 64) + '"}'
    )

    with pytest.raises(
        ValueError,
        match="expected exactly one receipt chain root, found 0",
    ):
        latest_receipt_file(receipts_dir)


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


def test_build_receipt_ignores_malformed_claims_and_evidence(tmp_path):
    private_key = ed25519.Ed25519PrivateKey.generate()
    private_key_path = tmp_path / "private.key"
    private_key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )

    receipt = build_receipt(
        state_dict={
            "claims": [
                {"status": "active"},
                "not a claim object",
                {"status": "superseded"},
            ],
            "evidence": None,
        },
        wake_md_sha="wake_sha",
        sources_data=[],
        private_key_path=private_key_path,
    )

    assert receipt["claim_count"] == {"active": 1, "superseded": 1, "unverified": 0}


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


def test_build_receipt_rejects_private_key_symlink(tmp_path):
    private_key = ed25519.Ed25519PrivateKey.generate()
    outside_key = tmp_path / "outside.key"
    outside_key.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    private_key_path = tmp_path / "private.key"
    private_key_path.symlink_to(outside_key)

    with pytest.raises(ValueError, match="private signing key must not be a symlink"):
        build_receipt(
            state_dict={"claims": [], "evidence": []},
            wake_md_sha="wake_sha",
            sources_data=[],
            private_key_path=private_key_path,
        )


def test_build_receipt_rejects_symlinked_private_key_ancestor(tmp_path):
    private_key = ed25519.Ed25519PrivateKey.generate()
    outside_keys = tmp_path / "outside-keys"
    outside_keys.mkdir()
    (outside_keys / "local.key").write_bytes(
        private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    keys_dir = tmp_path / "keys"
    try:
        keys_dir.symlink_to(outside_keys, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="private signing key path must not contain a symlink"):
        build_receipt(
            state_dict={"claims": [], "evidence": []},
            wake_md_sha="wake_sha",
            sources_data=[],
            private_key_path=keys_dir / "local.key",
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


def test_build_receipt_rejects_non_string_previous_hash_before_signing(tmp_path):
    private_key_path = tmp_path / "private.key"

    with pytest.raises(ValueError, match="previous_receipt_sha256 must be string or null"):
        build_receipt(
            state_dict={"claims": [], "evidence": []},
            wake_md_sha="wake_sha",
            sources_data=[],
            private_key_path=private_key_path,
            prev_hash={"not": "a hash"},
        )
