"""
Gmail integration - reads emails and extracts evidence.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone

from morpheus.core.safe_io import reject_symlink_paths

class GmailIntegration:
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
    
    def __init__(self, token_path: Path | None = None):
        self.token_path = token_path or Path.home() / ".morpheus" / "gmail_token.json"
        self.credentials_path = Path.home() / ".morpheus" / "gmail_credentials.json"
    
    def authenticate(self):
        """OAuth2 flow - for now just check if token exists"""
        if self.token_path.is_symlink() or not self.token_path.is_file():
            raise RuntimeError(
                "Gmail not authenticated. Run: morpheus integrate gmail\n"
                "You need credentials.json from Google Cloud Console"
            )
        return True
    
    def get_emails(self, days: int = 30, max_results: int = 50) -> list[dict]:
        """Fetch recent emails"""
        if max_results <= 0:
            return []
        if not self.authenticate():
            return []
        
        # For MVP: just read from local cache if available
        cache_path = self.token_path.parent / "gmail_cache.json"
        if cache_path.exists():
            return self._load_from_cache(cache_path, days)[:max_results]
        
        # Placeholder - real implementation uses Google API
        return []
    
    def _load_from_cache(self, cache_path: Path, days: int) -> list[dict]:
        import json
        try:
            reject_symlink_paths([cache_path], "Gmail cache path")
            data = json.loads(cache_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        emails = []
        for email in data:
            if not isinstance(email, dict):
                continue
            email_date = _parse_cache_datetime(email.get("date"))
            if email_date and email_date > cutoff:
                emails.append(email)
        return emails
    
    def extract_evidence(self, email: dict) -> list[dict]:
        """Extract claim-like statements from email"""
        evidence = []
        text = email.get("snippet") or ""
        if not isinstance(text, str):
            text = str(text)
        # Look for decisions, tasks, commitments
        for keyword in ["DECISION:", "DECIDED:", "TODO:", "WILL:", "COMMIT:", "AGREED:"]:
            if keyword in text.upper():
                evidence.append({
                    "type": "email_claim",
                    "source": "gmail",
                    "email_id": email.get("id"),
                    "keyword": keyword,
                    "excerpt": text
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
