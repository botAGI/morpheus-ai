"""Fail-closed identity checks for registered adapter weight artifacts."""
import json
import os
import stat
from hashlib import sha256
from pathlib import Path, PurePosixPath

from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths


ADAPTER_ARTIFACT_SCHEMA = "morpheus-adapter-artifact/1"
_WEIGHT_ARTIFACT_FIELDS = frozenset({"path", "sha256", "size"})


def validate_registered_adapter_artifact(
    adapter_dir: Path,
    *,
    expected_adapter_id: str,
) -> dict:
    """Validate the registry manifest and its one declared weight artifact."""
    if (
        not _safe_adapter_id(expected_adapter_id)
        or adapter_dir.name != expected_adapter_id
        or ".." in adapter_dir.parts
    ):
        return {
            "valid": False,
            "blockers": ["adapter_id_invalid"],
            "artifact": None,
        }
    manifest_path = adapter_dir / "adapter_manifest.json"
    try:
        reject_symlink_paths([adapter_dir, manifest_path], "Adapter artifact")
        reject_symlink_components(adapter_dir, "Adapter artifact")
        first = manifest_path.read_bytes()
        second = manifest_path.read_bytes()
        if first != second:
            raise ValueError("adapter manifest changed while reading")
        manifest = json.loads(first)
        if not isinstance(manifest, dict):
            raise ValueError("adapter manifest must be a JSON object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "valid": False,
            "blockers": [f"adapter_manifest_invalid:{exc}"],
            "artifact": None,
        }
    result = validate_adapter_artifact_manifest(
        adapter_dir,
        manifest,
        expected_adapter_id=expected_adapter_id,
    )
    return {
        **result,
        "manifest_sha256": sha256(first).hexdigest(),
    }


def validate_adapter_artifact_manifest(
    adapter_dir: Path,
    manifest: dict,
    *,
    expected_adapter_id: str,
) -> dict:
    """Validate one already-read adapter manifest against stable weight bytes."""
    blockers: list[str] = []
    if manifest.get("adapter_id") != expected_adapter_id:
        blockers.append("adapter_id_mismatch")
    if manifest.get("artifact_schema") != ADAPTER_ARTIFACT_SCHEMA:
        blockers.append("artifact_schema_invalid")
    if manifest.get("training_status") != "trained":
        blockers.append("adapter_not_trained")

    declared = manifest.get("weight_artifact")
    if not isinstance(declared, dict):
        blockers.append("weight_artifact_missing")
        return {"valid": False, "blockers": blockers, "artifact": None}
    if set(declared) != _WEIGHT_ARTIFACT_FIELDS:
        blockers.append("weight_artifact_schema_invalid")

    relative_path = declared.get("path")
    declared_sha = declared.get("sha256")
    declared_size = declared.get("size")
    if not _safe_weight_basename(relative_path):
        blockers.append("weight_artifact_path_invalid")
    if not _valid_sha256(declared_sha):
        blockers.append("weight_artifact_sha256_invalid")
    if type(declared_size) is not int or declared_size <= 0:
        blockers.append("weight_artifact_size_invalid")
    if blockers:
        return {"valid": False, "blockers": blockers, "artifact": None}

    weight_path = adapter_dir / relative_path
    try:
        actual = _read_regular_weight_identity(weight_path)
    except (OSError, ValueError) as exc:
        return {
            "valid": False,
            "blockers": [f"weight_artifact_invalid:{exc}"],
            "artifact": None,
        }
    if actual["size"] != declared_size:
        blockers.append("weight_artifact_size_mismatch")
    if actual["sha256"] != declared_sha:
        blockers.append("weight_artifact_sha256_mismatch")
    artifact = {
        "path": relative_path,
        "sha256": actual["sha256"],
        "size": actual["size"],
    }
    return {
        "valid": not blockers,
        "blockers": blockers,
        "artifact": artifact if not blockers else None,
    }


def _read_regular_weight_identity(path: Path) -> dict:
    reject_symlink_paths([path], "Adapter weight")
    reject_symlink_components(path, "Adapter weight")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"adapter weight must be a regular file: {path}")
    if before.st_size <= 0:
        raise ValueError(f"adapter weight must not be empty: {path}")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"adapter weight must be a regular file: {path}")
        if _stat_identity(opened) != _stat_identity(before):
            raise ValueError("adapter weight changed before it was opened")
        digest = sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        closed_read = os.fstat(descriptor)
        if _stat_identity(closed_read) != _stat_identity(opened):
            raise ValueError("adapter weight changed while it was read")
    finally:
        os.close(descriptor)

    after = path.lstat()
    if _stat_identity(after) != _stat_identity(opened):
        raise ValueError("adapter weight changed after it was read")
    if total != opened.st_size or total <= 0:
        raise ValueError("adapter weight size changed while it was read")
    return {"sha256": digest.hexdigest(), "size": total}


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _safe_weight_basename(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    parsed = PurePosixPath(value)
    return bool(
        not parsed.is_absolute()
        and len(parsed.parts) == 1
        and parsed.name == value
        and value not in {".", ".."}
        and parsed.suffix == ".safetensors"
    )


def _safe_adapter_id(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and value
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
        and Path(value).name == value
    )


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
