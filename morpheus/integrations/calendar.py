"""
Google Calendar integration - reads events as evidence.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone

from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths

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
        if not self.authenticate():
            return []
        
        cache_path = self.token_path.parent / "calendar_cache.json"
        if cache_path.exists():
            return self._load_from_cache(cache_path, days)[:max_results]
        
        return []
    
    def _load_from_cache(self, cache_path: Path, days: int) -> list[dict]:
        import json
        try:
            reject_symlink_paths([cache_path], "Calendar cache path")
            reject_symlink_components(cache_path, "Calendar cache path")
            data = json.loads(cache_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        events = []
        for event in data:
            if not isinstance(event, dict):
                continue
            start = _parse_cache_datetime(event.get("start"))
            if start and start > cutoff:
                events.append(event)
        return events
    
    def extract_evidence(self, event: dict) -> list[dict]:
        """Extract claim-like statements from event"""
        evidence = []
        description = event.get("description") or ""
        summary = event.get("summary") or ""
        text = f"{description} {summary}"
        for keyword in ["DECISION:", "AGREED:", "TODO:", "ACTION:", "WILL:"]:
            if keyword in text.upper():
                evidence.append({
                    "type": "event_claim",
                    "source": "calendar",
                    "event_id": event.get("id"),
                    "keyword": keyword,
                    "excerpt": text[:500]
                })
        return evidence


def _parse_cache_datetime(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
