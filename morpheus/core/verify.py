"""
Receipt chain verification.
"""
import base64
import binascii
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


def verify_receipt_chain(morpheus_dir: Path) -> tuple[bool, list[str]]:
    """Verify the full receipt chain in .morpheus/receipts/."""
    receipts_dir = morpheus_dir / "receipts"
    errors = []

    if not receipts_dir.exists():
        return False, ["receipts dir missing"]

    receipt_files = sorted(receipts_dir.glob("receipt_*.json"))
    if not receipt_files:
        return False, ["no receipts found"]

    public_key, key_error = _load_public_key(morpheus_dir / "keys")

    for i, receipt_file in enumerate(receipt_files):
        try:
            receipt = json.loads(receipt_file.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{receipt_file.name}: invalid JSON ({exc.msg})")
            continue

        # Check required fields
        required = ["receipt_id", "wake_md_sha256", "state_json_sha256", "signature"]
        for field in required:
            if field not in receipt:
                errors.append(f"{receipt_file.name}: missing field '{field}'")

        # Check previous link if not first
        if i > 0:
            prev_file = receipt_files[i - 1]
            prev_receipt = json.loads(prev_file.read_text())
            prev_id = prev_receipt.get("receipt_id")
            if receipt.get("previous_receipt_sha256") != prev_id:
                errors.append(f"{receipt_file.name}: previous_receipt_sha256 mismatch")

        # Verify signature if present
        sig = receipt.get("signature", {})
        if sig.get("signature_b64"):
            if key_error:
                errors.append(f"{receipt_file.name}: {key_error}")
            else:
                signature_error = _verify_receipt_signature(receipt, public_key)
                if signature_error:
                    errors.append(f"{receipt_file.name}: {signature_error}")

    return (len(errors) == 0, errors)


def _load_public_key(keys_dir: Path) -> tuple[ed25519.Ed25519PublicKey | None, str | None]:
    """Load the local public key, deriving it from the private key for older projects."""
    public_key_path = keys_dir / "local.pub"
    private_key_path = keys_dir / "local.key"

    try:
        if public_key_path.exists():
            return ed25519.Ed25519PublicKey.from_public_bytes(public_key_path.read_bytes()), None

        if private_key_path.exists():
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_path.read_bytes())
            return private_key.public_key(), None
    except ValueError as exc:
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

    payload = (
        f"{receipt.get('wake_md_sha256', '')}"
        f"{receipt.get('state_json_sha256', '')}"
        f"{receipt.get('evidence_jsonl_sha256', '')}"
        f"{receipt.get('previous_receipt_sha256') or ''}"
    ).encode()

    try:
        public_key.verify(signature_bytes, payload)
    except InvalidSignature:
        return "invalid ed25519 signature"

    return None
