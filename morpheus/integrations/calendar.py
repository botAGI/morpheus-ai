"""
Google Calendar integration - reads events as evidence.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone

class CalendarIntegration:
    SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
    
    def __init__(self, token_path: Path | None = None):
        self.token_path = token_path or Path.home() / ".morpheus" / "calendar_token.json"
        self.credentials_path = Path.home() / ".morpheus" / "calendar_credentials.json"
    
    def authenticate(self):
        if not self.token_path.exists():
            raise RuntimeError(
                "Calendar not authenticated. Run: morpheus integrate calendar"
            )
        return True
    
    def get_events(self, days: int = 30, max_results: int = 100) -> list[dict]:
        """Fetch upcoming/recent events"""
        if not self.authenticate():
            return []
        
        cache_path = self.token_path.parent / "calendar_cache.json"
        if cache_path.exists():
            return self._load_from_cache(cache_path, days)
        
        return []
    
    def _load_from_cache(self, cache_path: Path, days: int) -> list[dict]:
        import json
        data = json.loads(cache_path.read_text())
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [e for e in data if datetime.fromisoformat(e.get("start", "2000")) > cutoff]
    
    def extract_evidence(self, event: dict) -> list[dict]:
        """Extract claim-like statements from event"""
        evidence = []
        text = event.get("description", "") + " " + event.get("summary", "")
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
