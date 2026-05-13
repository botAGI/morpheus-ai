"""
Provenance: SHA256 utilities and receipt signing with ed25519.
"""
import hashlib
import base64
from pathlib import Path
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


def compute_sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_receipt(state_dict: dict, wake_md_sha: str, sources_data: list, private_key_path: Path, prev_hash: str = None) -> dict:
    """Build and sign a receipt."""
    import json
    from morpheus.core.models import Receipt

    state_json_bytes = json.dumps(state_dict, default=str).encode()
    state_sha = compute_sha256_bytes(state_json_bytes)

    evidence_items = [e for e in state_dict.get("evidence", [])]
    evidence_bytes = json.dumps(evidence_items, default=str).encode()
    evidence_sha = compute_sha256_bytes(evidence_bytes)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_hash = wake_md_sha[:8]
    receipt_id = f"rcpt_{ts}_{short_hash}"

    if not private_key_path.exists():
        raise FileNotFoundError(f"private signing key not found: {private_key_path}")

    payload = f"{wake_md_sha}{state_sha}{evidence_sha}{prev_hash or ''}".encode()
    private_bytes = private_key_path.read_bytes()
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
    sig = private_key.sign(payload)
    signature_b64 = base64.b64encode(sig).decode()

    tool_info = {"name": "morpheus", "version": "0.1.0"}

    claim_counts = {"active": 0, "superseded": 0, "unverified": 0}
    for c in state_dict.get("claims", []):
        cat = c.get("status", "active")
        if cat in claim_counts:
            claim_counts[cat] += 1

    return {
        "schema_version": "morpheus-receipt/1",
        "receipt_id": receipt_id,
        "project": {"name": ".", "root_sha": wake_md_sha[:16]},
        "wake_md_sha256": wake_md_sha,
        "state_json_sha256": state_sha,
        "evidence_jsonl_sha256": evidence_sha,
        "sources": sources_data,
        "claim_count": claim_counts,
        "tool": tool_info,
        "issued_at": ts,
        "previous_receipt_sha256": prev_hash,
        "signature": {
            "algo": "ed25519",
            "key_id": "local",
            "signature_b64": signature_b64,
        },
    }
