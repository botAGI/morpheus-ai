"""Deterministic semantic provider for tests and local alpha dogfood."""
from datetime import datetime, timezone
import hashlib
import re

from morpheus.core.semantic.models import SemanticCandidate, SemanticSource


class FakeProvider:
    name = "fake"
    model = "fixture"

    def extract_candidates(
        self,
        source: SemanticSource,
        *,
        run_id: str,
        prompt_sha256: str,
        source_revision: str,
    ) -> list[SemanticCandidate]:
        candidates = []
        for line_number, line in enumerate(source.content.splitlines(), 1):
            claim = _claim_from_line(line, source.path)
            if claim is None:
                continue
            kind, text = claim
            evidence = line.strip()
            digest = hashlib.sha256(
                f"{source.path}:{line_number}:{text}".encode()
            ).hexdigest()
            candidates.append(
                SemanticCandidate(
                    id=f"cand_{run_id.removeprefix('semrun_')}_{digest[:8]}",
                    run_id=run_id,
                    kind=kind,
                    claim=text,
                    source_path=source.path,
                    source_sha256=source.sha256,
                    source_mtime=source.modified_at,
                    source_revision=source_revision,
                    line_start=line_number,
                    line_end=line_number,
                    evidence_excerpt=evidence,
                    evidence_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
                    confidence=0.82,
                    label="needs_review",
                    status="pending",
                    created_at=datetime.now(timezone.utc),
                    provider={"name": self.name, "model": self.model},
                    prompt_sha256=prompt_sha256,
                )
            )
        return candidates


def _claim_from_line(line: str, source_path: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("```"):
        return None
    if stripped.startswith("<!--") or stripped.startswith("!["):
        return None
    lowered = stripped.casefold()
    if "ignore previous instructions" in lowered or "disregard previous instructions" in lowered:
        return None
    if "uvx --from morpheus-wake" in lowered or "morpheus wake" in lowered:
        return "source_reference", stripped.rstrip(".")
    if source_path == "AGENTS.md" and stripped.startswith(("-", "1.", "2.", "3.", "4.", "5.")):
        if "morpheus" in lowered or "wake.md" in lowered:
            return "agent_rule", stripped.lstrip("- ").strip().rstrip(".")
    marker = re.match(r"^(DECISION|TODO|NOTE|OUTDATED|AGENT_RULE):\s*(.+)$", stripped, re.I)
    if marker:
        marker_name = marker.group(1).upper()
        kind = {
            "DECISION": "active_decision",
            "TODO": "open_task",
            "NOTE": "current_state",
            "OUTDATED": "outdated_claim",
            "AGENT_RULE": "agent_rule",
        }[marker_name]
        return kind, marker.group(2).strip()
    if "morpheus" in lowered or "wake.md" in lowered:
        return "current_state", stripped.rstrip(".")
    return None
