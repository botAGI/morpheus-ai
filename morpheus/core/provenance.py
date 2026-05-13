"""
Provenance: SHA256 utilities and receipt signing with ed25519.
"""
import hashlib
import base64
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric import ed25519


def compute_sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evidence_jsonl_bytes(evidence_items: list) -> bytes:
    """Serialize evidence records as deterministic JSONL bytes."""
    lines = [
        json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
        for item in evidence_items
    ]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode()


def _list_or_empty(value) -> list:
    return value if isinstance(value, list) else []


def receipt_signature_payload(receipt: dict) -> bytes:
    """Return the canonical receipt payload protected by the ed25519 signature."""
    signed_receipt = {key: value for key, value in receipt.items() if key != "signature"}
    return json.dumps(
        signed_receipt,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


def receipt_file_name(receipt_id: str) -> str:
    """Return a collision-resistant receipt artifact filename."""
    if not receipt_id or receipt_id in {".", ".."} or "/" in receipt_id or "\\" in receipt_id:
        raise ValueError(f"invalid receipt id: {receipt_id!r}")
    return f"receipt_{receipt_id}.json"


def new_receipt_id() -> str:
    """Return a unique receipt id independent of artifact hashes."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"rcpt_{ts}_{uuid.uuid4().hex[:8]}"


def latest_receipt_file(receipts_dir: Path) -> Path | None:
    """Return the receipt chain tail by previous hash links."""
    if receipts_dir.is_symlink():
        raise ValueError("receipts path must not be a symlink")
    if receipts_dir.exists() and not receipts_dir.is_dir():
        raise ValueError("receipts path is not a directory")

    receipt_files = sorted(receipts_dir.glob("receipt_*.json"))
    if not receipt_files:
        return None

    records = []
    for receipt_file in receipt_files:
        if receipt_file.is_symlink():
            raise ValueError(f"{receipt_file.name}: must not be a symlink")

        try:
            receipt_text = receipt_file.read_text()
        except OSError as exc:
            raise ValueError(f"{receipt_file.name}: unreadable receipt ({exc})") from exc

        receipt = json.loads(receipt_text)
        if not isinstance(receipt, dict):
            raise ValueError(f"{receipt_file.name}: expected JSON object")
        previous = receipt.get("previous_receipt_sha256")
        if previous not in (None, "") and not isinstance(previous, str):
            raise ValueError(
                f"{receipt_file.name}: previous_receipt_sha256 must be string or null"
            )
        try:
            receipt_sha = compute_sha256_file(receipt_file)
        except OSError as exc:
            raise ValueError(f"{receipt_file.name}: unreadable receipt ({exc})") from exc
        records.append({
            "path": receipt_file,
            "sha256": receipt_sha,
            "previous": previous,
        })

    roots = [record for record in records if record["previous"] in (None, "")]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one receipt chain root, found {len(roots)}")

    referenced_hashes = {
        record["previous"]
        for record in records
        if record["previous"] not in (None, "")
    }
    tails = [record for record in records if record["sha256"] not in referenced_hashes]
    if len(tails) != 1:
        raise ValueError(f"expected exactly one receipt chain tail, found {len(tails)}")

    return tails[0]["path"]


def build_receipt(
    state_dict: dict,
    wake_md_sha: str,
    sources_data: list,
    private_key_path: Path,
    prev_hash: str = None,
    receipt_id: str | None = None,
    state_json_sha: str | None = None,
    evidence_jsonl_sha: str | None = None,
) -> dict:
    """Build and sign a receipt."""
    state_json_bytes = json.dumps(state_dict, default=str).encode()
    state_sha = state_json_sha or compute_sha256_bytes(state_json_bytes)

    evidence_items = _list_or_empty(state_dict.get("evidence"))
    evidence_sha = evidence_jsonl_sha or compute_sha256_bytes(evidence_jsonl_bytes(evidence_items))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt_id = receipt_id or new_receipt_id()
    receipt_file_name(receipt_id)

    if prev_hash is not None and not isinstance(prev_hash, str):
        raise ValueError("previous_receipt_sha256 must be string or null")

    if private_key_path.is_symlink():
        raise ValueError("private signing key must not be a symlink")
    if not private_key_path.exists():
        raise FileNotFoundError(f"private signing key not found: {private_key_path}")

    tool_info = {"name": "morpheus", "version": "0.1.0"}

    claim_counts = {"active": 0, "superseded": 0, "unverified": 0}
    for c in _list_or_empty(state_dict.get("claims")):
        if not isinstance(c, dict):
            continue
        cat = c.get("status", "active")
        if cat in claim_counts:
            claim_counts[cat] += 1

    receipt = {
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
    }

    private_bytes = private_key_path.read_bytes()
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
    sig = private_key.sign(receipt_signature_payload(receipt))
    receipt["signature"] = {
        "algo": "ed25519",
        "key_id": "local",
        "signature_b64": base64.b64encode(sig).decode(),
    }

    return receipt
