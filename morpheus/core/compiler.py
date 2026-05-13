"""
Source compiler: extracts sources, claims, and evidence from project files.
"""
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from morpheus.core.models import Source, Claim, Evidence, ProjectState


EVIDENCE_MARKERS = ["TODO:", "DECISION:", "FIXME:", "NOTE:", "HACK:"]
MARKER_CATEGORIES = {
    "TODO:": "task",
    "DECISION:": "decision",
    "FIXME:": "fixme",
    "NOTE:": "note",
    "HACK:": "hack",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def compile_project(project_root: Path) -> ProjectState:
    """Scan project sources and extract claims."""
    sources = []
    claims = []
    evidence = []

    claim_counter = 0
    evidence_counter = 0

    for path in sorted(project_root.rglob("*")):
        if path.is_file() and not _is_excluded(path):
            sha = compute_sha256(path)
            content = path.read_text(errors="ignore")
            lines = content.splitlines()
            src = Source(
                id=f"src_{len(sources)+1:03d}",
                path=str(path.relative_to(project_root)),
                kind=path.suffix.lstrip(".") or "text",
                sha256=sha,
                size_bytes=len(content.encode()),
                line_count=len(lines),
                modified_at=datetime.fromtimestamp(path.stat().st_mtime),
            )
            sources.append(src)

            file_claims, file_evidence = _extract_claims(
                src,
                lines,
                claim_start=claim_counter,
                evidence_start=evidence_counter,
            )
            claims.extend(file_claims)
            evidence.extend(file_evidence)
            claim_counter += len(file_claims)
            evidence_counter += len(file_evidence)

    return ProjectState(
        sources=sources,
        claims=claims,
        evidence=evidence,
        compiled_at=datetime.now(timezone.utc),
    )


def _is_excluded(path: Path) -> bool:
    exclusions = {".git", "node_modules", "__pycache__", ".morpheus", ".venv", "venv", ".tox", ".eggs", "*.pyc"}
    return any(part in exclusions or path.match(pat) for part in path.parts for pat in exclusions)


def _extract_claims(
    source: Source,
    lines: list[str],
    claim_start: int = 0,
    evidence_start: int = 0,
):
    claims = []
    evidence = []
    claim_id_counter = claim_start
    evidence_id_counter = evidence_start

    for i, line in enumerate(lines, 1):
        for marker in EVIDENCE_MARKERS:
            if marker in line:
                claim_id_counter += 1
                evidence_id_counter += 1
                cid = f"clm_{claim_id_counter:04d}"
                claim = Claim(
                    id=cid,
                    source_id=source.id,
                    line_start=i,
                    line_end=i,
                    excerpt=line.strip(),
                    category=MARKER_CATEGORIES[marker],
                    status="active",
                    inference=False,
                    created_at=datetime.now(timezone.utc),
                )
                claims.append(claim)

                import hashlib as hl
                exc = line.strip().encode()
                ev = Evidence(
                    id=f"ev_{evidence_id_counter:04d}",
                    claim_id=cid,
                    source_id=source.id,
                    path=source.path,
                    line_start=i,
                    line_end=i,
                    excerpt=line.strip(),
                    source_sha256=source.sha256,
                    excerpt_sha256=hl.sha256(exc).hexdigest(),
                    timestamp=datetime.now(timezone.utc),
                )
                evidence.append(ev)
    return claims, evidence
