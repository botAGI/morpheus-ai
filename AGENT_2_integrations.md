# Agent 2 — Integrations Module

## Your Task

Implement integrations in `/Users/testbot/.openclaw/workspace/morpheus-ai/morpheus/integrations/`.

### Files to create:

### 1. `__init__.py`
```python
from .gmail import GmailIntegration
from .calendar import CalendarIntegration
from .github import GitHubIntegration
from .filesystem import FileSystemWatcher

__all__ = ["GmailIntegration", "CalendarIntegration", "GitHubIntegration", "FileSystemWatcher"]
```

### 2. `gmail.py`
Gmail integration with OAuth2:

```python
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
```

### 3. `calendar.py`
Google Calendar integration:

```python
"""
Google Calendar integration - reads events as evidence.
"""
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

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
        cutoff = datetime.utcnow() - timedelta(days=days)
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
```

### 4. `github.py`
GitHub integration via Personal Access Token:

```python
"""
GitHub integration - reads issues, PRs, commits.
"""
from pathlib import Path
from datetime import datetime
from typing import Optional

class GitHubIntegration:
    def __init__(self, token_path: Path | None = None):
        self.token_path = token_path or Path.home() / ".morpheus" / "github_token.txt"
        self.api_url = "https://api.github.com"
    
    def authenticate(self) -> bool:
        return self.token_path.exists()
    
    def get_repo(self, owner: str, repo: str) -> dict:
        """Get repo info"""
        token = self._get_token()
        import httpx
        headers = {"Authorization": f"token {token}"} if token else {}
        resp = httpx.get(f"{self.api_url}/repos/{owner}/{repo}", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    
    def get_issues(self, owner: str, repo: str, state: str = "all", days: int = 30) -> list[dict]:
        """Get issues"""
        token = self._get_token()
        import httpx
        headers = {"Authorization": f"token {token}"} if token else {}
        resp = httpx.get(
            f"{self.api_url}/repos/{owner}/{repo}/issues",
            headers=headers,
            params={"state": state, "per_page": 100},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    
    def get_pulls(self, owner: str, repo: str, state: str = "all") -> list[dict]:
        """Get pull requests"""
        token = self._get_token()
        import httpx
        headers = {"Authorization": f"token {token}"} if token else {}
        resp = httpx.get(
            f"{self.api_url}/repos/{owner}/{repo}/pulls",
            headers=headers,
            params={"state": state, "per_page": 100},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    
    def _get_token(self) -> Optional[str]:
        if self.token_path.exists():
            return self.token_path.read_text().strip()
        return None
```

### 5. `filesystem.py`
Filesystem watcher for local files:

```python
"""
Filesystem integration - watches local files for changes.
"""
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

class FileSystemWatcher:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.cache_file = self.root / ".morpheus" / "fs_cache.json"
        self.file_hashes: dict[str, str] = {}
    
    def scan(self) -> list[dict]:
        """Scan all files, return changed ones"""
        import json
        changed = []
        
        # Load previous hashes
        if self.cache_file.exists():
            self.file_hashes = json.loads(self.cache_file.read_text())
        
        current_hashes = {}
        
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if ".morpheus" in path.parts:
                continue
            
            rel_path = str(path.relative_to(self.root))
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            current_hashes[rel_path] = file_hash
            
            is_new = rel_path not in self.file_hashes
            is_changed = rel_path in self.file_hashes and self.file_hashes[rel_path] != file_hash
            
            if is_new or is_changed:
                changed.append({
                    "path": rel_path,
                    "status": "new" if is_new else "modified",
                    "hash": file_hash,
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                })
        
        # Save new hashes
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(current_hashes, indent=2))
        
        return changed
    
    def extract_claims(self, path: str) -> list[dict]:
        """Extract claims from a file"""
        full_path = self.root / path
        if not full_path.exists():
            return []
        
        content = full_path.read_text("utf-8", errors="replace")
        lines = content.splitlines()
        claims = []
        
        for i, line in enumerate(lines, 1):
            for marker in ["TODO:", "FIXME:", "DECISION:", "NOTE:", "XXX:"]:
                if marker in line:
                    claims.append({
                        "path": path,
                        "line": i,
                        "marker": marker,
                        "excerpt": line.strip()
                    })
        
        return claims
```

## Instructions

1. Create all files in `/Users/testbot/.openclaw/workspace/morpheus-ai/morpheus/integrations/`
2. Use exact imports as shown
3. Add module-level docstrings
4. Implement real OAuth/API logic where marked as placeholder
5. Keep placeholders clearly marked
