"""
Linear integration - reads cached issues and extracts evidence.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.integrations.cache import cache_rows, load_cache_payload
from morpheus.integrations.dates import parse_cache_datetime
from morpheus.integrations.evidence import INTEGRATION_EVIDENCE_KEYWORDS, matched_keyword_excerpts


LINEAR_EVIDENCE_KEYWORDS = INTEGRATION_EVIDENCE_KEYWORDS


class LinearIntegration:
    def __init__(self, token_path: Path | None = None):
        self.token_path = token_path or Path.home() / ".morpheus" / "linear_token.txt"

    def authenticate(self) -> bool:
        if not self.token_path.is_file() or self.token_path.is_symlink():
            return False
        try:
            reject_symlink_components(self.token_path, "Linear token path")
        except ValueError:
            return False
        return True

    def get_issues(self, days: int = 30, max_results: int = 100) -> list[dict]:
        """Fetch recent issues from a local cache."""
        if days < 0 or max_results <= 0:
            return []

        cache_path = self.token_path.parent / "linear_cache.json"
        if cache_path.exists():
            return self._load_from_cache(cache_path, days)[:max_results]
        return []

    def _load_from_cache(self, cache_path: Path, days: int) -> list[dict]:
        try:
            reject_symlink_paths([cache_path], "Linear cache path")
            reject_symlink_components(cache_path, "Linear cache path")
            data = load_cache_payload(cache_path)
        except ValueError:
            return []
        if data is None:
            return []
        rows = cache_rows(data, "issues", "items", "data")
        if not rows:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        dated_issues = []
        for issue in rows:
            if not isinstance(issue, dict):
                continue
            updated_at = _parse_cache_datetime(
                issue.get("updated_at")
                or issue.get("updatedAt")
                or issue.get("created_at")
                or issue.get("createdAt")
                or issue.get("date")
            )
            if updated_at and updated_at > cutoff:
                dated_issues.append((updated_at, issue))
        dated_issues.sort(key=lambda item: item[0], reverse=True)
        return [issue for _, issue in dated_issues]

    def extract_evidence(self, issue: dict) -> list[dict]:
        """Extract claim-like statements from a Linear issue."""
        title = issue.get("title") or ""
        description = issue.get("description") or ""
        text = f"{description}\n{title}".strip()
        if not isinstance(text, str):
            text = str(text)
        return [
            {
                "type": "linear_claim",
                "source": "linear",
                "issue_id": issue.get("id"),
                "identifier": issue.get("identifier"),
                "keyword": keyword,
                "excerpt": excerpt,
                "url": issue.get("url"),
            }
            for keyword, excerpt in matched_keyword_excerpts(text, keywords=LINEAR_EVIDENCE_KEYWORDS)
        ]

    def _get_token(self) -> Optional[str]:
        if self.authenticate():
            try:
                return self.token_path.read_text().strip()
            except OSError:
                return None
        return None


def _parse_cache_datetime(value: str | int | float | None) -> datetime | None:
    return parse_cache_datetime(value)
