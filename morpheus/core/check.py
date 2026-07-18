"""Local truth checks for agent claims against Morpheus state."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re

import toml

from morpheus.core.learning.team import (
    CHECK_CORRECTION_PROMPT_SHA256 as _CHECK_CORRECTION_PROMPT_SHA256,
    run_team_learning_loop,
)
from morpheus.core.provenance import compute_sha256_file, latest_receipt_file
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.classifier import classify_claim
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.semantic.routing import (
    ROUTING_POLICY_VERSION,
    route_check_result,
)


CLAIM_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
MARKER_RE = re.compile(r"^(TODO|DECISION|FIXME|NOTE|HACK|XXX):\s*", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")
CI_ENV_VARS = ("MORPHEUS_CI", "CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "CIRCLECI")
CHECK_CORRECTION_PROMPT_SHA256 = _CHECK_CORRECTION_PROMPT_SHA256
def discover_project_root(start: Path) -> Path:
    """Walk upward until a Morpheus, package, or git root is found."""
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in [".morpheus", "pyproject.toml", ".git"]):
            return candidate
    return current


def ci_mode_from_env(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    return any(env.get(name, "").lower() in {"1", "true", "yes"} for name in CI_ENV_VARS)


def check_exit_code(
    result: dict,
    *,
    ci_mode: bool,
    allow_stale_state: bool,
    strict_freshness: bool,
    fail_on_unknown: bool,
) -> int:
    if result["state_freshness"] == "stale" and ci_mode and not allow_stale_state:
        return 2
    if result["state_freshness"] == "unknown" and ci_mode and strict_freshness:
        return 2
    if result["claims_stale"] or result["claims_contradicted"]:
        return 1
    if fail_on_unknown and result["claims_not_found"]:
        return 1
    return 0


def check_text(
    text: str,
    *,
    project_root: Path,
    fail_on_unknown: bool = False,
) -> dict:
    """Check input text against local Morpheus state without provider calls."""
    project_root = discover_project_root(project_root)
    context = _load_check_context(project_root)
    claims = _extract_claims(text)
    results = [_route_check_claim(_classify_claim(claim, context)) for claim in claims]
    freshness, warning = _state_freshness(project_root)
    receipt_id = context["state"].get("receipt_id") or context["latest_receipt_id"]
    payload = {
        "input_hash": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "active_state_receipt": receipt_id,
        "state_freshness": freshness,
        "state_warning": warning,
        "modes_used": ["local"],
        "claims_extracted": len(claims),
        "claims_supported": sum(1 for item in results if item["status"] == "verified"),
        "claims_contradicted": sum(1 for item in results if item["status"] == "incorrect"),
        "claims_stale": sum(1 for item in results if item["status"] == "stale"),
        "claims_not_found": sum(1 for item in results if item["status"] == "unknown"),
        "by_class": dict(sorted(Counter(item["semantic_class"] for item in results).items())),
        "by_route": dict(sorted(Counter(item["memory_route"] for item in results).items())),
        "routing_policy_version": ROUTING_POLICY_VERSION,
        "fail_on_unknown": fail_on_unknown,
        "results": results,
    }
    return payload


def _route_check_claim(result: dict) -> dict:
    route, reason = route_check_result(
        str(result.get("status") or "unknown"),
        str(result.get("semantic_class") or "unknown"),
    )
    return {
        **result,
        "memory_route": route,
        "routing_reason": reason,
    }


def render_check_summary(result: dict) -> str:
    lines = [
        "Morpheus check",
        f"Receipt: {result.get('active_state_receipt') or 'unknown'}",
        f"State: {result['state_freshness']}",
    ]
    if result.get("state_warning"):
        lines.append(f"Warning: {result['state_warning']}")
    lines.append(
        "Claims: "
        f"{result['claims_supported']} verified, "
        f"{result['claims_stale']} stale, "
        f"{result['claims_contradicted']} incorrect, "
        f"{result['claims_not_found']} unknown"
    )
    for item in result["results"]:
        evidence = item.get("evidence")
        if evidence:
            span = f"{evidence['path']}:{evidence['line_start']}"
        else:
            span = "no source span"
        lines.append(f"- {item['status']}: {item['claim']} ({span})")
    return "\n".join(lines)


def render_check_annotated(result: dict) -> str:
    lines = [
        "<!-- morpheus-check -->",
        f"State freshness: `{result['state_freshness']}`",
        "",
    ]
    if result.get("state_warning"):
        lines.extend([f"> {result['state_warning']}", ""])
    for item in result["results"]:
        evidence = item.get("evidence")
        if evidence:
            span = f"{evidence['path']}:{evidence['line_start']}"
        else:
            span = "no source span"
        lines.extend([
            f"> {item['claim']}",
            f"`{item['status']}` - {item['reason']} ({span})",
            "",
        ])
    return "\n".join(lines).rstrip()


def create_training_corrections(project_root: Path, check_result: dict) -> list[SemanticCandidate]:
    """Route local check results through the unified reviewed-input orchestrator."""
    project_root = discover_project_root(project_root)
    items = []
    for item in check_result.get("results", []):
        if not isinstance(item, dict):
            raise ValueError("Check results must contain JSON objects")
        claim = str(item.get("claim") or "").strip()
        event = {
            "source_type": "check_result",
            "claim": claim,
            "status": item.get("status"),
            "reason": _legacy_check_reason(item),
            "active_state_receipt": check_result.get("active_state_receipt"),
            "input_hash": check_result.get("input_hash"),
        }
        evidence = _legacy_check_evidence(item)
        if evidence is not None:
            event["evidence"] = evidence
        items.append(event)

    result = run_team_learning_loop(project_root, items)
    created_ids = result["report"]["created_candidate_ids"]
    if not created_ids:
        return []
    store = ReviewStore(project_root)
    with store.transaction():
        candidates = store.load_candidates()
    by_id = {candidate.id: candidate for candidate in candidates}
    missing_ids = [candidate_id for candidate_id in created_ids if candidate_id not in by_id]
    if missing_ids:
        raise ValueError(
            "Unified check candidates disappeared after commit: " + ", ".join(missing_ids)
        )
    return [
        by_id[candidate_id]
        for candidate_id in created_ids
        if by_id[candidate_id].provider.get("name") == "morpheus-check"
    ]


def _legacy_check_reason(item: dict) -> str:
    """Preserve legacy reason rendering while rejecting noncanonical artifact text."""
    reason = str(item.get("reason"))
    if not reason.strip() or reason != reason.strip() or "\n" in reason or "\r" in reason:
        raise ValueError("Check correction reason must be a nonblank single line")
    return reason


def _legacy_check_evidence(item: dict) -> dict | None:
    """Preserve the old source-label fallback for canonical legacy payloads."""
    raw_evidence = item.get("evidence")
    if not isinstance(raw_evidence, dict) or not raw_evidence:
        return None
    path = str(raw_evidence.get("path") or "unknown")
    if not path.strip() or path != path.strip() or "\n" in path or "\r" in path:
        raise ValueError("Check correction evidence path must be a nonblank single line")
    raw_line = raw_evidence.get("line_start")
    if raw_line is None:
        raw_line = 1
    if isinstance(raw_line, bool) or not isinstance(raw_line, int) or raw_line < 1:
        raise ValueError("Check correction evidence line_start must be a positive integer")
    return {"path": path, "line_start": raw_line}


def _load_check_context(project_root: Path) -> dict:
    morpheus_dir = project_root / ".morpheus"
    if morpheus_dir.is_symlink():
        raise ValueError(".morpheus path must not be a symlink")
    reject_symlink_components(morpheus_dir, ".morpheus path")
    if not morpheus_dir.is_dir():
        raise ValueError("Morpheus state not found. Run: morpheus wake .")

    state = _read_json_object(morpheus_dir / "state.json", "state.json")
    evidence_rows = _read_jsonl(morpheus_dir / "evidence.jsonl", "evidence.jsonl")
    claim_by_id = {
        str(claim.get("id")): claim
        for claim in state.get("claims", [])
        if isinstance(claim, dict)
    }
    evidence_by_claim = {
        str(row.get("claim_id")): row
        for row in evidence_rows
        if isinstance(row, dict)
    }
    active_claims = []
    stale_claims = []
    for claim in state.get("claims", []):
        if not isinstance(claim, dict):
            continue
        item = _claim_item(claim, evidence_by_claim.get(str(claim.get("id"))))
        if claim.get("category") == "outdated" or claim.get("status") in {"superseded", "outdated"}:
            stale_claims.append(item)
        elif claim.get("status", "active") == "active":
            active_claims.append(item)

    wake_stale = []
    for wake_path in [project_root / "WAKE.md", morpheus_dir / "WAKE.md"]:
        wake_stale.extend(_wake_outdated_claims(wake_path, project_root))

    package_metadata = _package_metadata(project_root / "pyproject.toml", project_root)
    latest_receipt_id = _latest_receipt_id(morpheus_dir)
    return {
        "state": state,
        "claim_by_id": claim_by_id,
        "active_claims": active_claims,
        "stale_claims": stale_claims + wake_stale,
        "package_metadata": package_metadata,
        "latest_receipt_id": latest_receipt_id,
    }


def _classify_claim(claim: str, context: dict) -> dict:
    package_result = _classify_package_claim(claim, context["package_metadata"])
    if package_result is not None:
        return package_result

    stale = _best_match(claim, context["stale_claims"], threshold=0.72)
    if stale:
        return {
            "claim": claim,
            "status": "stale",
            "semantic_class": stale["semantic_class"],
            "reason": "claim matches outdated project state",
            "evidence": stale["evidence"],
        }

    active = _best_match(claim, context["active_claims"], threshold=0.64)
    if active:
        return {
            "claim": claim,
            "status": "verified",
            "semantic_class": active["semantic_class"],
            "reason": "claim is supported by active Morpheus evidence",
            "evidence": active["evidence"],
        }

    contradiction = _active_subject_contradiction(claim, context["active_claims"])
    if contradiction:
        return {
            "claim": claim,
            "status": "incorrect",
            "semantic_class": contradiction["semantic_class"],
            "reason": "claim contradicts an active source-backed project claim",
            "evidence": contradiction["evidence"],
        }

    return {
        "claim": claim,
        "status": "unknown",
        "semantic_class": "unknown",
        "reason": "no matching active evidence found",
        "evidence": None,
    }


def _classify_package_claim(claim: str, package_metadata: dict | None) -> dict | None:
    if not package_metadata:
        return None
    folded = claim.casefold()
    name = package_metadata.get("name")
    version = package_metadata.get("version")
    if name and name.casefold() in folded:
        return {
            "claim": claim,
            "status": "verified",
            "semantic_class": "command",
            "reason": "claim matches pyproject package metadata",
            "evidence": package_metadata["evidence"],
        }
    if "package" in folded or "distribution" in folded:
        match = re.search(r"\b(?:package|distribution)(?:\s+name)?\s+(?:is|=|called)\s+([a-z0-9_.-]+)", folded)
        if match and name and match.group(1) != name.casefold():
            return {
                "claim": claim,
                "status": "incorrect",
                "semantic_class": "command",
                "reason": f"package metadata says distribution is {name}",
                "evidence": package_metadata["evidence"],
            }
    if version and version in folded and "version" in folded:
        return {
            "claim": claim,
            "status": "verified",
            "semantic_class": "command",
            "reason": "claim matches pyproject version metadata",
            "evidence": package_metadata["evidence"],
        }
    return None


def _state_freshness(project_root: Path) -> tuple[str, str | None]:
    morpheus_dir = project_root / ".morpheus"
    receipts_dir = morpheus_dir / "receipts"
    try:
        latest = latest_receipt_file(receipts_dir)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        return "unknown", f"Latest receipt cannot be inspected: {exc}"
    if latest is None:
        return "unknown", "Latest receipt does not exist. Run: morpheus wake ."
    receipt = _read_json_object(latest, latest.name)
    sources = receipt.get("sources")
    if not isinstance(sources, list) or not sources:
        return "unknown", "Latest receipt does not contain v0.2 source hash metadata. Run: morpheus wake ."
    for source in sources:
        if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
            return "unknown", "Latest receipt contains incomplete source hash metadata. Run: morpheus wake ."
        source_path = project_root / str(source["path"])
        if source_path.is_symlink():
            return "stale", f"State is stale: source path is a symlink: {source['path']}"
        if not source_path.exists():
            return "stale", f"State is stale: source missing: {source['path']}"
        try:
            current_sha = compute_sha256_file(source_path)
        except (OSError, ValueError) as exc:
            return "stale", f"State is stale: source unreadable: {source['path']} ({exc})"
        if current_sha != source["sha256"]:
            return "stale", f"State is stale: source changed: {source['path']}"
    return "fresh", None


def _read_json_object(path: Path, label: str) -> dict:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} invalid: expected JSON object")
    return data


def _read_jsonl(path: Path, label: str) -> list[dict]:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number} invalid JSON: {exc.msg}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _latest_receipt_id(morpheus_dir: Path) -> str | None:
    try:
        latest = latest_receipt_file(morpheus_dir / "receipts")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if latest is None:
        return None
    try:
        return _read_json_object(latest, latest.name).get("receipt_id")
    except ValueError:
        return None


def _claim_item(claim: dict, evidence: dict | None) -> dict:
    text = _strip_marker(str(claim.get("excerpt", "")))
    if evidence:
        source_span = _evidence_span(evidence)
    else:
        source_span = {
            "path": str(claim.get("path") or "state.json"),
            "line_start": int(claim.get("line_start") or 1),
            "line_end": int(claim.get("line_end") or claim.get("line_start") or 1),
            "excerpt": str(claim.get("excerpt", "")),
        }
    return {
        "text": text,
        "semantic_class": classify_claim(
            kind=_candidate_kind_from_claim(claim),
            claim=text,
            source_path=source_span["path"],
        ),
        "evidence": source_span,
    }


def _candidate_kind_from_claim(claim: dict) -> str:
    category = str(claim.get("category") or "").casefold()
    if category in {"task", "todo"}:
        return "open_task"
    if category in {"outdated", "stale"}:
        return "outdated_claim"
    if category == "decision":
        return "active_decision"
    return "current_state"


def _evidence_span(evidence: dict) -> dict:
    line_start = int(evidence.get("line_start") or 1)
    return {
        "path": str(evidence.get("path") or "evidence.jsonl"),
        "line_start": line_start,
        "line_end": int(evidence.get("line_end") or line_start),
        "excerpt": str(evidence.get("excerpt", "")),
        "claim_id": str(evidence.get("claim_id") or ""),
    }


def _wake_outdated_claims(path: Path, project_root: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        return []
    reject_symlink_components(path, "WAKE.md")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    in_outdated = False
    claims = []
    rel_path = path.relative_to(project_root).as_posix()
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("## "):
            in_outdated = stripped.lower() == "## outdated claims"
            continue
        if not in_outdated or not stripped.startswith("-"):
            continue
        text = stripped.lstrip("- ").strip()
        text = re.sub(r"\s+Outdated\.?$", "", text, flags=re.IGNORECASE).strip()
        text = text.strip('"')
        if not text:
            continue
        claims.append({
            "text": text,
            "semantic_class": "stale",
            "evidence": {
                "path": rel_path,
                "line_start": line_number,
                "line_end": line_number,
                "excerpt": stripped,
            },
        })
    return claims


def _package_metadata(path: Path, project_root: Path) -> dict | None:
    if path.is_symlink() or not path.is_file():
        return None
    reject_symlink_components(path, "pyproject.toml")
    try:
        data = toml.loads(path.read_text())
    except (OSError, toml.TomlDecodeError):
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    name = project.get("name")
    version = project.get("version")
    if not name and not version:
        return None
    line_start = _find_first_line(path, [f'name = "{name}"', f"version = \"{version}\""])
    return {
        "name": str(name) if name is not None else None,
        "version": str(version) if version is not None else None,
        "evidence": {
            "path": path.relative_to(project_root).as_posix(),
            "line_start": line_start,
            "line_end": line_start,
            "excerpt": f"name={name!r} version={version!r}",
        },
    }


def _find_first_line(path: Path, needles: list[str]) -> int:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return 1
    clean_needles = [needle for needle in needles if "None" not in needle]
    for line_number, line in enumerate(lines, 1):
        if any(needle in line for needle in clean_needles):
            return line_number
    return 1


def _extract_claims(text: str) -> list[str]:
    claims = []
    for chunk in CLAIM_SPLIT_RE.split(text.strip()):
        claim = chunk.strip()
        if not claim:
            continue
        words = WORD_RE.findall(claim.casefold())
        if len(words) < 3:
            continue
        claims.append(claim.rstrip())
    return claims


def _best_match(claim: str, items: list[dict], *, threshold: float) -> dict | None:
    claim_norm = _normalize(claim)
    best = None
    best_score = 0.0
    for item in items:
        item_norm = _normalize(item["text"])
        if not item_norm:
            continue
        if item_norm in claim_norm or claim_norm in item_norm:
            score = 1.0
        else:
            score = max(
                SequenceMatcher(None, claim_norm, item_norm).ratio(),
                _token_overlap(claim_norm, item_norm),
            )
        if score > best_score:
            best = item
            best_score = score
    return best if best is not None and best_score >= threshold else None


def _active_subject_contradiction(claim: str, active_claims: list[dict]) -> dict | None:
    subject, complement = _simple_is_statement(claim)
    if subject is None or complement is None:
        return None
    for item in active_claims:
        active_subject, active_complement = _simple_is_statement(item["text"])
        if active_subject != subject or active_complement is None:
            continue
        if _token_overlap(complement, active_complement) <= 0.2:
            return item
    return None


def _simple_is_statement(text: str) -> tuple[str | None, str | None]:
    normalized = _normalize(text)
    match = re.match(r"^(.+?)\s+(?:is|are)\s+(.+)$", normalized)
    if not match:
        return None, None
    subject = _strip_articles(match.group(1))
    complement = _strip_articles(match.group(2))
    if not subject or not complement:
        return None, None
    return subject, complement


def _strip_articles(text: str) -> str:
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(WORD_RE.findall(left))
    right_tokens = set(WORD_RE.findall(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _strip_marker(text: str) -> str:
    return MARKER_RE.sub("", text).strip()


def _normalize(text: str) -> str:
    text = _strip_marker(text).casefold()
    return " ".join(WORD_RE.findall(text))
