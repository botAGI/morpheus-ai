"""
Tests for morpheus.core.verify.
"""
import json
from pathlib import Path

import pytest
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


def test_verify_receipt_chain_rejects_symlinked_morpheus_dir(tmp_path):
    outside_morpheus = tmp_path / "outside-morpheus"
    private_key_path = _write_keypair(outside_morpheus / "keys")
    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="a" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    _write_receipt(outside_morpheus / "receipts", "receipt_001.json", receipt)
    morpheus_dir = tmp_path / ".morpheus"
    try:
        morpheus_dir.symlink_to(outside_morpheus, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert errors == ["morpheus path must not be a symlink"]


def test_verify_receipt_chain_rejects_symlinked_morpheus_parent(tmp_path):
    outside_project = tmp_path / "outside-project"
    outside_morpheus = outside_project / ".morpheus"
    private_key_path = _write_keypair(outside_morpheus / "keys")
    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="a" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    _write_receipt(outside_morpheus / "receipts", "receipt_001.json", receipt)
    linked_project = tmp_path / "linked-project"
    try:
        linked_project.symlink_to(outside_project, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    valid, errors = verify_receipt_chain(linked_project / ".morpheus")

    assert not valid
    assert errors == [f"morpheus path must not contain a symlink: {linked_project}"]


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


def test_verify_receipt_chain_rejects_latest_wake_artifact_mismatch(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")
    wake_path = morpheus_dir / "WAKE.md"
    wake_path.write_text("# WAKE\n\nvalid\n")

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha=compute_sha256_file(wake_path),
        sources_data=[],
        private_key_path=private_key_path,
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)
    wake_path.write_text("# WAKE\n\ntampered\n")

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "latest WAKE.md sha256 mismatch" in errors[0]


def test_verify_receipt_chain_reports_unreadable_latest_wake_artifact(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="0" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)
    (morpheus_dir / "WAKE.md").mkdir()

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert any("latest WAKE.md unreadable" in error for error in errors)


def test_verify_receipt_chain_rejects_symlinked_latest_wake_artifact(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")
    outside_wake = tmp_path / "outside-WAKE.md"
    outside_wake.write_text("# WAKE\n\noutside\n")
    (morpheus_dir / "WAKE.md").symlink_to(outside_wake)

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha=compute_sha256_file(outside_wake),
        sources_data=[],
        private_key_path=private_key_path,
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert any("latest WAKE.md must not be a symlink" in error for error in errors)


def test_verify_receipt_chain_rejects_latest_state_artifact_mismatch(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")
    state_path = morpheus_dir / "state.json"
    state_path.write_text('{"claims": [], "evidence": []}')

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="0" * 64,
        sources_data=[],
        private_key_path=private_key_path,
        state_json_sha=compute_sha256_file(state_path),
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)
    state_path.write_text('{"claims": [{"id": "tampered"}], "evidence": []}')

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "latest state.json sha256 mismatch" in errors[0]


def test_verify_receipt_chain_rejects_latest_evidence_artifact_mismatch(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")
    evidence_path = morpheus_dir / "evidence.jsonl"
    evidence_path.write_text('{"id":"ev_0001"}\n')

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": [{"id": "ev_0001"}]},
        wake_md_sha="0" * 64,
        sources_data=[],
        private_key_path=private_key_path,
        evidence_jsonl_sha=compute_sha256_file(evidence_path),
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)
    evidence_path.write_text('{"id":"ev_tampered"}\n')

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "latest evidence.jsonl sha256 mismatch" in errors[0]


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


def test_verify_receipt_chain_orders_receipts_by_previous_hash_not_filename(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    first = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="4" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    first_path = _write_receipt(morpheus_dir / "receipts", "receipt_z_first.json", first)
    second = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="5" * 64,
        sources_data=[],
        private_key_path=private_key_path,
        prev_hash=compute_sha256_file(first_path),
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_a_second.json", second)

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


def test_verify_receipt_chain_rejects_non_string_previous_hash(tmp_path):
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
    second["previous_receipt_sha256"] = {"not": "a hash"}
    _write_receipt(morpheus_dir / "receipts", "receipt_002.json", second)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "receipt_002.json: previous_receipt_sha256 must be string or null" in errors


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


def test_verify_receipt_chain_reports_unreadable_public_key(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="f" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    (morpheus_dir / "keys" / "local.pub").unlink()
    (morpheus_dir / "keys" / "local.pub").mkdir()
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert any("failed to load ed25519 key" in error for error in errors)


def test_verify_receipt_chain_rejects_keys_path_symlink(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    outside_keys = tmp_path / "outside-keys"
    private_key_path = _write_keypair(outside_keys)
    (morpheus_dir / "keys").symlink_to(outside_keys, target_is_directory=True)

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="f" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert any("keys path must not be a symlink" in error for error in errors)


def test_verify_receipt_chain_rejects_public_key_symlink(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")
    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="f" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    external_public_key = tmp_path / "external-local.pub"
    external_public_key.write_bytes((morpheus_dir / "keys" / "local.pub").read_bytes())
    (morpheus_dir / "keys" / "local.pub").unlink()
    try:
        (morpheus_dir / "keys" / "local.pub").symlink_to(external_public_key)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert any("public key must not be a symlink" in error for error in errors)


def test_verify_receipt_chain_rejects_private_key_symlink_fallback(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")
    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="f" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    external_private_key = tmp_path / "external-local.key"
    external_private_key.write_bytes(private_key_path.read_bytes())
    (morpheus_dir / "keys" / "local.pub").unlink()
    private_key_path.unlink()
    try:
        private_key_path.symlink_to(external_private_key)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert any("private key must not be a symlink" in error for error in errors)


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


def test_verify_receipt_chain_rejects_non_object_signature(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")

    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="0" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    receipt["signature"] = "not a signature object"
    _write_receipt(morpheus_dir / "receipts", "receipt_001.json", receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "receipt_001.json: signature must be JSON object" in errors


def test_verify_receipt_chain_rejects_non_object_receipt_json(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    _write_keypair(morpheus_dir / "keys")
    receipts_dir = morpheus_dir / "receipts"
    receipts_dir.mkdir(parents=True)
    (receipts_dir / "receipt_bad.json").write_text("[]")

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "receipt_bad.json: expected JSON object" in errors


def test_verify_receipt_chain_reports_unreadable_receipt_files(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    _write_keypair(morpheus_dir / "keys")
    receipts_dir = morpheus_dir / "receipts"
    (receipts_dir / "receipt_bad.json").mkdir(parents=True)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert any("receipt_bad.json: unreadable receipt" in error for error in errors)


def test_verify_receipt_chain_rejects_symlinked_receipt_files(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = _write_keypair(morpheus_dir / "keys")
    receipt = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="f" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    outside_receipt = tmp_path / "outside-receipt.json"
    outside_receipt.write_text(json.dumps(receipt))
    receipts_dir = morpheus_dir / "receipts"
    receipts_dir.mkdir(parents=True)
    (receipts_dir / "receipt_outside.json").symlink_to(outside_receipt)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert "receipt_outside.json: must not be a symlink" in errors


def test_verify_receipt_chain_rejects_receipts_path_file(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    _write_keypair(morpheus_dir / "keys")
    receipts_dir = morpheus_dir / "receipts"
    receipts_dir.write_text("not a directory")

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert errors == ["receipts path is not a directory"]


def test_verify_receipt_chain_rejects_receipts_path_symlink(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    _write_keypair(morpheus_dir / "keys")
    outside_receipts = tmp_path / "outside-receipts"
    outside_receipts.mkdir()
    receipts_dir = morpheus_dir / "receipts"
    receipts_dir.symlink_to(outside_receipts, target_is_directory=True)

    valid, errors = verify_receipt_chain(morpheus_dir)

    assert not valid
    assert errors == ["receipts path must not be a symlink"]
