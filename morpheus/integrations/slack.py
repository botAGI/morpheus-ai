"""
Slack integration - reads cached messages and extracts evidence.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.integrations.dates import parse_cache_datetime


SLACK_EVIDENCE_KEYWORDS = (
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


class SlackIntegration:
    def __init__(self, token_path: Path | None = None):
        self.token_path = token_path or Path.home() / ".morpheus" / "slack_token.txt"

    def authenticate(self) -> bool:
        if not self.token_path.is_file() or self.token_path.is_symlink():
            return False
        try:
            reject_symlink_components(self.token_path, "Slack token path")
        except ValueError:
            return False
        return True

    def get_messages(self, days: int = 30, max_results: int = 100) -> list[dict]:
        """Fetch recent messages from a local cache."""
        if days < 0 or max_results <= 0:
            return []

        cache_path = self.token_path.parent / "slack_cache.json"
        if cache_path.exists():
            return self._load_from_cache(cache_path, days)[:max_results]
        return []

    def _load_from_cache(self, cache_path: Path, days: int) -> list[dict]:
        import json

        try:
            reject_symlink_paths([cache_path], "Slack cache path")
            reject_symlink_components(cache_path, "Slack cache path")
            data = json.loads(cache_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        dated_messages = []
        for message in data:
            if not isinstance(message, dict):
                continue
            timestamp = _parse_cache_datetime(message.get("ts") or message.get("date"))
            if timestamp and timestamp > cutoff:
                dated_messages.append((timestamp, message))
        dated_messages.sort(key=lambda item: item[0], reverse=True)
        return [message for _, message in dated_messages]

    def extract_evidence(self, message: dict) -> list[dict]:
        """Extract claim-like statements from a Slack message."""
        text = message.get("text") or ""
        if not isinstance(text, str):
            text = str(text)
        upper_text = text.upper()
        return [
            {
                "type": "slack_claim",
                "source": "slack",
                "message_id": message.get("id") or message.get("ts"),
                "channel": message.get("channel"),
                "user": message.get("user"),
                "keyword": keyword,
                "excerpt": text[:500],
                "url": message.get("permalink") or message.get("url"),
            }
            for keyword in SLACK_EVIDENCE_KEYWORDS
            if keyword in upper_text
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
