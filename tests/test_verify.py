"""
Tests for morpheus.core.verify.
"""
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from morpheus.core.provenance import build_receipt, compute_sha256_file
from morpheus.core.verify import verify_receipt_chain


def _write_keypair(keys_dir: Path) -> Path:
    keys_dir.mkdir(parents=True)
    private_key = ed25519.Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )

    private_key_path = keys_dir / "local.key"
    private_key_path.write_bytes(private_bytes)
    (keys_dir / "local.pub").write_bytes(public_bytes)
    return private_key_path


def _write_receipt(receipts_dir: Path, name: str, receipt: dict) -> Path:
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / name
    receipt_path.write_text(json.dumps(receipt))
    return receipt_path


def test_verify_receipt_chain_accepts_valid_signed_receipt(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    receipt = build_receipt(
        state_dict={"claims": [{"id": "clm_0001", "status": "active"}], "evidence": []},
        wake_md_sha="a" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert valid
    assert errors == []


def test_verify_receipt_chain_rejects_tampered_signed_receipt(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="b" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    receipt["wake_md_sha256"] = "c" * 64
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "invalid ed25519 signature" in errors[0]


def test_verify_receipt_chain_rejects_tampered_signed_metadata(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="3" * 64,
        sources_data=[{"id": "src_001", "path": "README.md"}],
        private_key_path=private_key_path,
    )
    receipt["sources"] = [{"id": "src_001", "path": "tampered.md"}]
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "invalid ed25519 signature" in errors[0]


def test_verify_receipt_chain_validates_previous_receipt_link(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    first = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="d" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    first_path = _write_receipt(morpheus_dir / "receipts", "receipt_001.json", first)
    second = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="e" * 64,
        sources_data=[],
        private_key_path=private_key_path,
        prev_hash=compute_sha256_file(first_path),
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_002.json", second)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert valid
    assert errors == []


def test_verify_receipt_chain_detects_previous_receipt_file_tampering(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    first = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="1" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    first_path = _write_receipt(morpheus_dir / "receipts", "receipt_001.json", first)
    second = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="2" * 64,
        sources_data=[],
        private_key_path=private_key_path,
        prev_hash=compute_sha256_file(first_path),
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_002.json", second)

    valid_before_tamper, errors_before_tamper = verify_receipt_chain(morpheus_dir)
    assert valid_before_tamper
    assert errors_before_tamper == []

    tampered_first = dict(first)
    tampered_first["sources"] = [{"id": "src_tampered", "path": "tampered.md"}]
    first_path.write_text(json.dumps(tampered_first))

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert any("previous_receipt_sha256 mismatch" in error for error in errors)


def test_verify_receipt_chain_requires_key_for_signed_receipts(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="f" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    (morpheus_dir / "keys" / "local.key").unlink()
    (morpheus_dir / "keys" / "local.pub").unlink()
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "missing local public key" in errors[0]


def test_verify_receipt_chain_rejects_unsigned_receipt(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="0" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    receipt["signature"]["signature_b64"] = ""
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "missing ed25519 signature" in errors[0]
