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
