"""
Gmail integration - reads emails and extracts evidence.
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import httpx

class GmailIntegration:
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
    
    def __init__(self, token_path: Path | None = None):
        self.token_path = token_path or Path.home() / ".morpheus" / "gmail_token.json"
        self.credentials_path = Path.home() / ".morpheus" / "gmail_credentials.json"
    
    def authenticate(self):
        """OAuth2 flow - for now just check if token exists"""
        if not self.token_path.exists():
            raise RuntimeError(
                "Gmail not authenticated. Run: morpheus integrate gmail\n"
                "You need credentials.json from Google Cloud Console"
            )
        return True
    
    def get_emails(self, days: int = 30, max_results: int = 50) -> list[dict]:
        """Fetch recent emails"""
        if not self.authenticate():
            return []
        
        # For MVP: just read from local cache if available
        cache_path = self.token_path.parent / "gmail_cache.json"
        if cache_path.exists():
            return self._load_from_cache(cache_path, days)
        
        # Placeholder - real implementation uses Google API
        return []
    
    def _load_from_cache(self, cache_path: Path, days: int) -> list[dict]:
        import json
        data = json.loads(cache_path.read_text())
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [e for e in data if datetime.fromisoformat(e.get("date", "2000")) > cutoff]
    
    def extract_evidence(self, email: dict) -> list[dict]:
        """Extract claim-like statements from email"""
        evidence = []
        text = email.get("snippet", "")
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
