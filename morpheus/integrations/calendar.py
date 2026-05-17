"""
Google Calendar integration - reads events as evidence.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone

from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.integrations.cache import cache_rows, load_cache_payload
from morpheus.integrations.dates import parse_cache_datetime
from morpheus.integrations.evidence import matched_keyword_excerpts

EVENT_EVIDENCE_TEXT_FIELDS = ("description", "summary")


class CalendarIntegration:
    SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
    
    def __init__(self, token_path: Path | None = None):
        self.token_path = token_path or Path.home() / ".morpheus" / "calendar_token.json"
        self.credentials_path = Path.home() / ".morpheus" / "calendar_credentials.json"
    
    def authenticate(self):
        if self.token_path.is_symlink() or not self.token_path.is_file():
            raise RuntimeError(
                "Calendar not authenticated. Run: morpheus integrate calendar"
            )
        try:
            reject_symlink_components(self.token_path, "Calendar token path")
        except ValueError as exc:
            raise RuntimeError(
                "Calendar not authenticated. Run: morpheus integrate calendar"
            ) from exc
        return True
    
    def get_events(self, days: int = 30, max_results: int = 100) -> list[dict]:
        """Fetch upcoming/recent events"""
        if days < 0 or max_results <= 0:
            return []

        cache_path = self.token_path.parent / "calendar_cache.json"
        if cache_path.exists():
            return self._load_from_cache(cache_path, days)[:max_results]

        if not self.authenticate():
            return []

        return []
    
    def _load_from_cache(self, cache_path: Path, days: int) -> list[dict]:
        try:
            reject_symlink_paths([cache_path], "Calendar cache path")
            reject_symlink_components(cache_path, "Calendar cache path")
            data = load_cache_payload(cache_path)
        except ValueError:
            return []
        if data is None:
            return []
        rows = cache_rows(data, "events", "items", "data")
        if not rows:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        dated_events = []
        for event in rows:
            if not isinstance(event, dict):
                continue
            start = _parse_cache_datetime(event.get("start"))
            if start and start > cutoff:
                dated_events.append((start, event))
        dated_events.sort(key=lambda item: item[0], reverse=True)
        return [event for _, event in dated_events]
    
    def extract_evidence(self, event: dict) -> list[dict]:
        """Extract claim-like statements from event"""
        evidence = []
        text = event_evidence_text(event)
        for keyword, excerpt in matched_keyword_excerpts(text):
            evidence.append({
                "type": "event_claim",
                "source": "calendar",
                "event_id": event.get("id"),
                "keyword": keyword,
                "excerpt": excerpt,
                "url": event.get("htmlLink") or event.get("url"),
            })
        return evidence


def event_evidence_text(event: dict) -> str:
    """Return the human-visible calendar text fields used for claim extraction."""
    parts = []
    for field in EVENT_EVIDENCE_TEXT_FIELDS:
        value = event.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            parts.append(value)
    return "\n".join(parts)


def _parse_cache_datetime(value: object) -> datetime | None:
    return parse_cache_datetime(value, datetime_type=datetime)
