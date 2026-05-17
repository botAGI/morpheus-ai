"""
Slack integration - reads cached messages and extracts evidence.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.integrations.cache import cache_rows, load_cache_payload
from morpheus.integrations.dates import parse_cache_datetime
from morpheus.integrations.evidence import INTEGRATION_EVIDENCE_KEYWORDS, matched_keyword_excerpts


SLACK_EVIDENCE_KEYWORDS = INTEGRATION_EVIDENCE_KEYWORDS
SLACK_NESTED_TEXT_KEYS = {"fallback", "pretext", "text", "title", "value"}


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
        try:
            reject_symlink_paths([cache_path], "Slack cache path")
            reject_symlink_components(cache_path, "Slack cache path")
            data = load_cache_payload(cache_path)
        except ValueError:
            return []
        if data is None:
            return []
        rows = cache_rows(data, "messages", "items", "data")
        if not rows:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        dated_messages = []
        for message in rows:
            if not isinstance(message, dict):
                continue
            timestamp = _parse_cache_datetime(message.get("ts") or message.get("date"))
            if timestamp and timestamp > cutoff:
                dated_messages.append((timestamp, message))
        dated_messages.sort(key=lambda item: item[0], reverse=True)
        return [message for _, message in dated_messages]

    def extract_evidence(self, message: dict) -> list[dict]:
        """Extract claim-like statements from a Slack message."""
        evidence = []
        for text in slack_message_text_parts(message):
            evidence.extend({
                "type": "slack_claim",
                "source": "slack",
                "message_id": message.get("id") or message.get("ts"),
                "channel": message.get("channel"),
                "user": message.get("user"),
                "keyword": keyword,
                "excerpt": excerpt,
                "url": message.get("permalink") or message.get("url"),
            }
            for keyword, excerpt in matched_keyword_excerpts(text, keywords=SLACK_EVIDENCE_KEYWORDS))
        return evidence

    def _get_token(self) -> Optional[str]:
        if self.authenticate():
            try:
                return self.token_path.read_text().strip()
            except OSError:
                return None
        return None


def _parse_cache_datetime(value: str | int | float | None) -> datetime | None:
    return parse_cache_datetime(value)


def slack_message_text(message: dict) -> str:
    """Return human-visible Slack message text from top-level text, blocks, and attachments."""
    return "\n".join(slack_message_text_parts(message))


def slack_message_text_parts(message: dict) -> list[str]:
    """Return ordered human-visible Slack message text segments."""
    parts: list[str] = []
    _append_slack_text(parts, message.get("text"))
    _append_slack_nested_text(parts, message.get("blocks"), seen=set())
    _append_slack_nested_text(parts, message.get("attachments"), seen=set())
    return parts


def _append_slack_nested_text(parts: list[str], payload: object, *, seen: set[int]) -> None:
    if isinstance(payload, (dict, list)):
        payload_id = id(payload)
        if payload_id in seen:
            return
        seen.add(payload_id)

    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in SLACK_NESTED_TEXT_KEYS:
                if isinstance(value, (dict, list)):
                    _append_slack_nested_text(parts, value, seen=seen)
                else:
                    _append_slack_text(parts, value)
                continue
            if isinstance(value, (dict, list)):
                _append_slack_nested_text(parts, value, seen=seen)
        return

    if isinstance(payload, list):
        for item in payload:
            _append_slack_nested_text(parts, item, seen=seen)


def _append_slack_text(parts: list[str], value: object) -> None:
    if value is None:
        return
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if text and text not in parts:
        parts.append(text)
