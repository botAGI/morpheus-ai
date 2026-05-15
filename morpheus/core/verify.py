"""
Receipt chain verification.
"""
import base64
import binascii
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from morpheus.core.provenance import compute_sha256_file, receipt_signature_payload
from morpheus.core.safe_io import reject_symlink_components


def verify_receipt_chain(morpheus_dir: Path) -> tuple[bool, list[str]]:
    """Verify the full receipt chain in .morpheus/receipts/."""
    if morpheus_dir.is_symlink():
        return False, ["morpheus path must not be a symlink"]
    try:
        reject_symlink_components(morpheus_dir, "morpheus path")
    except ValueError as exc:
        return False, [str(exc)]

    receipts_dir = morpheus_dir / "receipts"
    errors = []

    if not receipts_dir.exists():
        return False, ["receipts dir missing"]
    if receipts_dir.is_symlink():
        return False, ["receipts path must not be a symlink"]
    if not receipts_dir.is_dir():
        return False, ["receipts path is not a directory"]

    receipt_files = sorted(receipts_dir.glob("receipt_*.json"))
    if not receipt_files:
        return False, ["no receipts found"]

    public_key, key_error = _load_public_key(morpheus_dir / "keys")
    receipt_records = []

    for receipt_file in receipt_files:
        if receipt_file.is_symlink():
            errors.append(f"{receipt_file.name}: must not be a symlink")
            continue

        try:
            receipt_text = receipt_file.read_text()
        except OSError as exc:
            errors.append(f"{receipt_file.name}: unreadable receipt ({exc})")
            continue

        try:
            receipt = json.loads(receipt_text)
        except json.JSONDecodeError as exc:
            errors.append(f"{receipt_file.name}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(receipt, dict):
            errors.append(f"{receipt_file.name}: expected JSON object")
            continue

        try:
            receipt_sha = compute_sha256_file(receipt_file)
        except OSError as exc:
            errors.append(f"{receipt_file.name}: unreadable receipt ({exc})")
            continue

        receipt_records.append({
            "path": receipt_file,
            "receipt": receipt,
            "sha256": receipt_sha,
        })

    ordered_records, ordering_errors = _order_receipt_records(receipt_records)
    errors.extend(ordering_errors)

    for i, record in enumerate(ordered_records):
        receipt_file = record["path"]
        receipt = record["receipt"]

        # Check required fields
        required = [
            "receipt_id",
            "wake_md_sha256",
            "state_json_sha256",
            "evidence_jsonl_sha256",
            "signature",
        ]
        for field in required:
            if field not in receipt:
                errors.append(f"{receipt_file.name}: missing field '{field}'")

        # Check previous link against the actual previous receipt artifact hash.
        if i == 0:
            if receipt.get("previous_receipt_sha256") is not None:
                errors.append(f"{receipt_file.name}: first receipt has previous_receipt_sha256")
        else:
            prev_sha = ordered_records[i - 1]["sha256"]
            if receipt.get("previous_receipt_sha256") != prev_sha:
                errors.append(f"{receipt_file.name}: previous_receipt_sha256 mismatch")

        # Verify signature if present
        sig = receipt.get("signature", {})
        if not isinstance(sig, dict):
            errors.append(f"{receipt_file.name}: signature must be JSON object")
        elif not sig.get("signature_b64"):
            errors.append(f"{receipt_file.name}: missing ed25519 signature")
        elif key_error:
            errors.append(f"{receipt_file.name}: {key_error}")
        else:
            signature_error = _verify_receipt_signature(receipt, public_key)
            if signature_error:
                errors.append(f"{receipt_file.name}: {signature_error}")

    _verify_latest_artifact_hash(
        morpheus_dir=morpheus_dir,
        ordered_records=ordered_records,
        relative_path="WAKE.md",
        receipt_hash_field="wake_md_sha256",
        error_label="latest WAKE.md sha256 mismatch",
        errors=errors,
    )
    _verify_latest_artifact_hash(
        morpheus_dir=morpheus_dir,
        ordered_records=ordered_records,
        relative_path="state.json",
        receipt_hash_field="state_json_sha256",
        error_label="latest state.json sha256 mismatch",
        errors=errors,
    )
    _verify_latest_artifact_hash(
        morpheus_dir=morpheus_dir,
        ordered_records=ordered_records,
        relative_path="evidence.jsonl",
        receipt_hash_field="evidence_jsonl_sha256",
        error_label="latest evidence.jsonl sha256 mismatch",
        errors=errors,
    )

    return (len(errors) == 0, errors)


def _verify_latest_artifact_hash(
    morpheus_dir: Path,
    ordered_records: list[dict],
    relative_path: str,
    receipt_hash_field: str,
    error_label: str,
    errors: list[str],
) -> None:
    artifact_path = morpheus_dir / relative_path
    if not ordered_records:
        return
    if artifact_path.is_symlink():
        errors.append(f"latest {relative_path} must not be a symlink")
        return
    if not artifact_path.exists():
        errors.append(f"latest {relative_path} missing")
        return

    latest_receipt = ordered_records[-1]["receipt"]
    try:
        actual_sha = compute_sha256_file(artifact_path)
    except OSError as exc:
        errors.append(f"latest {relative_path} unreadable ({exc})")
        return
    if latest_receipt.get(receipt_hash_field) != actual_sha:
        errors.append(error_label)


def _order_receipt_records(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Order receipt records by previous_receipt_sha256 links instead of filename."""
    errors = []
    for record in records:
        prev_sha = record["receipt"].get("previous_receipt_sha256")
        if prev_sha not in (None, "") and not isinstance(prev_sha, str):
            errors.append(
                f"{record['path'].name}: previous_receipt_sha256 must be string or null"
            )
    if errors:
        return sorted(records, key=lambda record: record["path"].name), errors

    if len(records) <= 1:
        return records, []

    roots = [
        record for record in records
        if record["receipt"].get("previous_receipt_sha256") in (None, "")
    ]
    if len(roots) != 1:
        errors.append(f"expected exactly one root receipt, found {len(roots)}")
        return sorted(records, key=lambda record: record["path"].name), errors

    children_by_prev_sha: dict[str, list[dict]] = {}
    for record in records:
        prev_sha = record["receipt"].get("previous_receipt_sha256")
        if prev_sha not in (None, ""):
            children_by_prev_sha.setdefault(prev_sha, []).append(record)

    ordered = []
    seen = set()
    current = roots[0]
    while current is not None:
        current_sha = current["sha256"]
        if current_sha in seen:
            errors.append("receipt chain contains a cycle")
            break

        seen.add(current_sha)
        ordered.append(current)

        children = children_by_prev_sha.get(current_sha, [])
        if len(children) > 1:
            errors.append(f"{current['path'].name}: multiple receipts reference this receipt")
            break

        current = children[0] if children else None

    leftovers = [record for record in records if record["sha256"] not in seen]
    if leftovers:
        errors.append("receipt chain is disconnected")
        ordered.extend(sorted(leftovers, key=lambda record: record["path"].name))

    return ordered, errors


def _load_public_key(keys_dir: Path) -> tuple[ed25519.Ed25519PublicKey | None, str | None]:
    """Load the local public key, deriving it from the private key for older projects."""
    if keys_dir.is_symlink():
        return None, "keys path must not be a symlink"

    public_key_path = keys_dir / "local.pub"
    private_key_path = keys_dir / "local.key"

    try:
        if public_key_path.exists():
            if public_key_path.is_symlink():
                return None, "public key must not be a symlink"
            return ed25519.Ed25519PublicKey.from_public_bytes(public_key_path.read_bytes()), None

        if private_key_path.exists():
            if private_key_path.is_symlink():
                return None, "private key must not be a symlink"
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_path.read_bytes())
            return private_key.public_key(), None
    except (OSError, ValueError) as exc:
        return None, f"failed to load ed25519 key ({exc})"

    return None, "missing local public key for ed25519 signature verification"


def _verify_receipt_signature(
    receipt: dict,
    public_key: ed25519.Ed25519PublicKey | None,
) -> str | None:
    if public_key is None:
        return "missing public key"

    signature = receipt.get("signature", {})
    if signature.get("algo") != "ed25519":
        return f"unsupported signature algorithm '{signature.get('algo', 'unknown')}'"

    try:
        signature_bytes = base64.b64decode(signature["signature_b64"], validate=True)
    except (KeyError, ValueError, binascii.Error) as exc:
        return f"invalid base64 signature ({exc})"

    try:
        public_key.verify(signature_bytes, receipt_signature_payload(receipt))
    except InvalidSignature:
        return "invalid ed25519 signature"

    return None
