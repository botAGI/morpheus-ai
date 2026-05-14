"""
GitHub integration - reads issues, PRs, commits.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

MAX_GITHUB_PAGES = 10
GITHUB_EVIDENCE_KEYWORDS = (
    "DECISION:",
    "DECIDED:",
    "TODO:",
    "FIXME:",
    "NOTE:",
    "ACTION:",
    "WILL:",
    "COMMIT:",
    "AGREED:",
)


class GitHubIntegration:
    def __init__(self, token_path: Path | None = None):
        self.token_path = token_path or Path.home() / ".morpheus" / "github_token.txt"
        self.api_url = "https://api.github.com"
    
    def authenticate(self) -> bool:
        return self.token_path.is_file() and not self.token_path.is_symlink()
    
    def get_repo(self, owner: str, repo: str) -> dict:
        """Get repo info"""
        token = self._get_token()
        headers = {"Authorization": f"token {token}"} if token else {}
        data = _github_get_json(f"{self.api_url}/repos/{owner}/{repo}", headers=headers)
        return data if isinstance(data, dict) else {}
    
    def get_issues(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        days: int = 30,
        max_results: int = 100,
    ) -> list[dict]:
        """Get issues"""
        if max_results <= 0:
            return []
        token = self._get_token()
        headers = {"Authorization": f"token {token}"} if token else {}
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        data = _github_get_json_list(
            f"{self.api_url}/repos/{owner}/{repo}/issues",
            headers=headers,
            params={"state": state, "per_page": _github_per_page(max_results)},
            stop_when=lambda items: _github_recent_issue_count(items, cutoff) >= max_results,
        )
        recent_issues = []
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
                if len(recent_issues) >= max_results:
                    break
        return recent_issues
    
    def get_pulls(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        max_results: int = 100,
    ) -> list[dict]:
        """Get pull requests"""
        if max_results <= 0:
            return []
        token = self._get_token()
        headers = {"Authorization": f"token {token}"} if token else {}
        data = _github_get_json_list(
            f"{self.api_url}/repos/{owner}/{repo}/pulls",
            headers=headers,
            params={"state": state, "per_page": _github_per_page(max_results)},
            stop_when=lambda items: _github_pull_count(items) >= max_results,
        )
        if not isinstance(data, list):
            return []
        pulls = []
        for pull in data:
            if not isinstance(pull, dict):
                continue
            pulls.append(pull)
            if len(pulls) >= max_results:
                break
        return pulls

    def get_commits(
        self,
        owner: str,
        repo: str,
        days: int = 30,
        max_results: int = 100,
    ) -> list[dict]:
        """Get recent commits"""
        if max_results <= 0:
            return []
        token = self._get_token()
        headers = {"Authorization": f"token {token}"} if token else {}
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        data = _github_get_json_list(
            f"{self.api_url}/repos/{owner}/{repo}/commits",
            headers=headers,
            params={"since": cutoff.isoformat(), "per_page": _github_per_page(max_results)},
            stop_when=lambda items: _github_recent_commit_count(items, cutoff) >= max_results,
        )
        if not isinstance(data, list):
            return []

        commits = []
        for commit in data:
            if not isinstance(commit, dict):
                continue
            committed_at = _github_commit_datetime(commit)
            if committed_at and committed_at > cutoff:
                commits.append(commit)
                if len(commits) >= max_results:
                    break
        return commits

    def extract_evidence(self, item: dict, item_type: str | None = None) -> list[dict]:
        """Extract claim-like statements from a GitHub issue, pull request, or commit."""
        item_type = item_type or _github_item_type(item)
        text = _github_item_text(item, item_type)
        if not text:
            return []

        upper_text = text.upper()
        item_id = item.get("sha") if item_type == "commit" else item.get("number", item.get("id"))
        return [
            {
                "type": "github_claim",
                "source": "github",
                "item_type": item_type,
                "item_id": item_id,
                "keyword": keyword,
                "excerpt": text[:500],
                "url": item.get("html_url"),
            }
            for keyword in GITHUB_EVIDENCE_KEYWORDS
            if keyword in upper_text
        ]
    
    def _get_token(self) -> Optional[str]:
        if self.authenticate():
            try:
                return self.token_path.read_text().strip()
            except OSError:
                return None
        return None


def _github_get_json(url: str, *, headers: dict, params: dict | None = None):
    response = _github_get_response(url, headers=headers, params=params)
    if response is None:
        return None
    return _safe_response_json(response)


def _github_get_json_list(
    url: str,
    *,
    headers: dict,
    params: dict | None = None,
    max_pages: int = MAX_GITHUB_PAGES,
    stop_when=None,
):
    items = []
    next_url = url
    next_params = params

    for _ in range(max_pages):
        response = _github_get_response(next_url, headers=headers, params=next_params)
        if response is None:
            return None

        data = _safe_response_json(response)
        if not isinstance(data, list):
            return None

        items.extend(data)
        if stop_when is not None and stop_when(items):
            break
        next_url = _github_next_url(response)
        if not next_url:
            break
        next_params = None

    return items


def _github_get_response(url: str, *, headers: dict, params: dict | None = None):
    import httpx

    try:
        kwargs = {"headers": headers, "timeout": 10}
        if params is not None:
            kwargs["params"] = params
        response = httpx.get(url, **kwargs)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return response


def _github_next_url(response) -> str | None:
    links = getattr(response, "links", None)
    if isinstance(links, dict):
        next_link = links.get("next")
        if isinstance(next_link, dict):
            next_url = next_link.get("url")
            if isinstance(next_url, str) and next_url:
                return next_url

    headers = getattr(response, "headers", {})
    link_header = headers.get("link") if hasattr(headers, "get") else None
    if not isinstance(link_header, str):
        return None

    for part in link_header.split(","):
        link_part, *params = part.split(";")
        if not any(param.strip().lower() in {'rel="next"', "rel=next"} for param in params):
            continue
        link_part = link_part.strip()
        if link_part.startswith("<") and ">" in link_part:
            return link_part[1:link_part.index(">")]
    return None


def _safe_response_json(response):
    try:
        return response.json()
    except ValueError:
        return None


def _github_per_page(max_results: int) -> int:
    return min(max_results, 100)


def _github_item_text(item: dict, item_type: str) -> str:
    fields = []
    if item_type == "commit":
        payload = item.get("commit")
        if isinstance(payload, dict):
            fields.append(payload.get("message"))
    else:
        fields.extend([item.get("title"), item.get("body")])

    parts = [_github_text_part(field) for field in fields]
    return "\n".join(part for part in parts if part)


def _github_item_type(item: dict) -> str:
    if isinstance(item.get("commit"), dict) and item.get("sha"):
        return "commit"
    if "pull_request" in item or ("head" in item and "base" in item):
        return "pull_request"
    if "number" in item:
        return "issue"
    return "github"


def _github_text_part(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _github_recent_issue_count(items: list, cutoff: datetime) -> int:
    count = 0
    for issue in items:
        if not isinstance(issue, dict) or "pull_request" in issue:
            continue
        updated_at = _parse_github_datetime(issue.get("updated_at"))
        if updated_at and updated_at > cutoff:
            count += 1
    return count


def _github_pull_count(items: list) -> int:
    return sum(1 for pull in items if isinstance(pull, dict))


def _github_recent_commit_count(items: list, cutoff: datetime) -> int:
    count = 0
    for commit in items:
        if not isinstance(commit, dict):
            continue
        committed_at = _github_commit_datetime(commit)
        if committed_at and committed_at > cutoff:
            count += 1
    return count


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


def _github_commit_datetime(commit: dict) -> datetime | None:
    payload = commit.get("commit")
    if not isinstance(payload, dict):
        return None

    for actor_key in ("committer", "author"):
        actor = payload.get(actor_key)
        if not isinstance(actor, dict):
            continue
        committed_at = _parse_github_datetime(actor.get("date"))
        if committed_at:
            return committed_at
    return None
