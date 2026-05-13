"""
Receipt chain verification.
"""
import json
from pathlib import Path
from morpheus.core.provenance import compute_sha256_bytes


def verify_receipt_chain(morpheus_dir: Path) -> tuple[bool, list[str]]:
    """Verify the full receipt chain in .morpheus/receipts/."""
    receipts_dir = morpheus_dir / "receipts"
    errors = []

    if not receipts_dir.exists():
        return False, ["receipts dir missing"]

    receipt_files = sorted(receipts_dir.glob("receipt_*.json"))
    if not receipt_files:
        return False, ["no receipts found"]

    for i, receipt_file in enumerate(receipt_files):
        receipt = json.loads(receipt_file.read_text())

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
            errors.append(f"{receipt_file.name}: signature verification not implemented (ed25519)")

    return (len(errors) == 0, errors)
