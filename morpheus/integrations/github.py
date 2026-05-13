"""
GitHub integration - reads issues, PRs, commits.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

class GitHubIntegration:
    def __init__(self, token_path: Path | None = None):
        self.token_path = token_path or Path.home() / ".morpheus" / "github_token.txt"
        self.api_url = "https://api.github.com"
    
    def authenticate(self) -> bool:
        return self.token_path.is_file()
    
    def get_repo(self, owner: str, repo: str) -> dict:
        """Get repo info"""
        token = self._get_token()
        import httpx
        headers = {"Authorization": f"token {token}"} if token else {}
        resp = httpx.get(f"{self.api_url}/repos/{owner}/{repo}", headers=headers, timeout=10)
        resp.raise_for_status()
        data = _safe_response_json(resp)
        return data if isinstance(data, dict) else {}
    
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
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_issues = []
        data = _safe_response_json(resp)
        if not isinstance(data, list):
            return []
        for issue in data:
            if not isinstance(issue, dict):
                continue
            if "pull_request" in issue:
                continue
            updated_at = _parse_github_datetime(issue.get("updated_at"))
            if updated_at and updated_at > cutoff:
                recent_issues.append(issue)
        return recent_issues
    
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
        data = _safe_response_json(resp)
        if not isinstance(data, list):
            return []
        return [pull for pull in data if isinstance(pull, dict)]
    
    def _get_token(self) -> Optional[str]:
        if self.token_path.is_file():
            try:
                return self.token_path.read_text().strip()
            except OSError:
                return None
        return None


def _safe_response_json(response):
    try:
        return response.json()
    except ValueError:
        return None


def _parse_github_datetime(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
