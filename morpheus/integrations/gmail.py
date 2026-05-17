"""
Gmail integration - reads emails and extracts evidence.
"""
import base64
import binascii
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime, timedelta, timezone

from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.integrations.cache import cache_rows, load_cache_payload
from morpheus.integrations.dates import parse_cache_datetime
from morpheus.integrations.evidence import matched_keyword_excerpts

EMAIL_EVIDENCE_TEXT_FIELDS = ("subject", "snippet", "body", "text")
HTML_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
}


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
        try:
            reject_symlink_components(self.token_path, "Gmail token path")
        except ValueError as exc:
            raise RuntimeError(
                "Gmail not authenticated. Run: morpheus integrate gmail\n"
                "You need credentials.json from Google Cloud Console"
            ) from exc
        return True
    
    def get_emails(self, days: int = 30, max_results: int = 50) -> list[dict]:
        """Fetch recent emails"""
        if days < 0 or max_results <= 0:
            return []

        cache_path = self.token_path.parent / "gmail_cache.json"
        if cache_path.exists():
            return self._load_from_cache(cache_path, days)[:max_results]

        if not self.authenticate():
            return []

        # Placeholder - real implementation uses Google API
        return []
    
    def _load_from_cache(self, cache_path: Path, days: int) -> list[dict]:
        try:
            reject_symlink_paths([cache_path], "Gmail cache path")
            reject_symlink_components(cache_path, "Gmail cache path")
            data = load_cache_payload(cache_path)
        except ValueError:
            return []
        if data is None:
            return []
        rows = cache_rows(data, "emails", "messages", "items", "data")
        if not rows:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        dated_emails = []
        for email in rows:
            if not isinstance(email, dict):
                continue
            date_value = email.get("date")
            if date_value is None:
                date_value = email.get("internalDate")
            if date_value is None:
                date_value = gmail_payload_header(email, "Date")
            email_date = _parse_cache_datetime(date_value)
            if email_date and email_date > cutoff:
                dated_emails.append((email_date, email))
        dated_emails.sort(key=lambda item: item[0], reverse=True)
        return [email for _, email in dated_emails]
    
    def extract_evidence(self, email: dict) -> list[dict]:
        """Extract claim-like statements from email"""
        evidence = []
        text = email_evidence_text(email)
        for keyword, excerpt in matched_keyword_excerpts(text):
            evidence.append({
                "type": "email_claim",
                "source": "gmail",
                "email_id": email.get("id"),
                "keyword": keyword,
                "excerpt": excerpt
            })
        return evidence


def email_evidence_text(email: dict) -> str:
    """Return the human-visible email text fields used for claim extraction."""
    parts = []
    for field in EMAIL_EVIDENCE_TEXT_FIELDS:
        value = email.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            parts.append(value)
    payload_subject = gmail_payload_header(email, "Subject")
    if payload_subject and payload_subject not in parts:
        parts.insert(0, payload_subject)
    payload_text = gmail_payload_text(email)
    if payload_text and payload_text not in parts:
        parts.append(payload_text)
    return "\n".join(parts)


def gmail_payload_text(email: dict) -> str:
    """Return decoded native Gmail API text/plain payload body content."""
    payload = email.get("payload")
    if not isinstance(payload, dict):
        return ""
    text_parts, html_parts = _gmail_payload_text_parts(payload)
    return "\n".join(text_parts or html_parts)


def _gmail_payload_text_parts(payload: dict) -> tuple[list[str], list[str]]:
    text_parts = []
    html_parts = []
    mime_type = str(payload.get("mimeType") or "").split(";", 1)[0].strip().casefold()
    body = payload.get("body")
    if isinstance(body, dict):
        decoded = _decode_gmail_body_data(body.get("data"))
        if decoded and (not mime_type or mime_type == "text/plain"):
            text_parts.append(decoded)
        elif decoded and mime_type == "text/html":
            html_text = _gmail_html_to_text(decoded)
            if html_text:
                html_parts.append(html_text)

    parts = payload.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            child_text_parts, child_html_parts = _gmail_payload_text_parts(part)
            text_parts.extend(child_text_parts)
            html_parts.extend(child_html_parts)
    return text_parts, html_parts


def _decode_gmail_body_data(data: object) -> str:
    if not isinstance(data, str):
        return ""
    compact_data = "".join(data.split())
    if not compact_data:
        return ""
    padding = "=" * (-len(compact_data) % 4)
    try:
        decoded = base64.urlsafe_b64decode(compact_data + padding)
    except (binascii.Error, ValueError):
        return ""
    return decoded.decode("utf-8", errors="replace").strip()


def _gmail_html_to_text(value: str) -> str:
    parser = _GmailHTMLTextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text()


class _GmailHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        normalized = tag.casefold()
        if normalized in {"script", "style"}:
            self._skip_depth += 1
            return
        if normalized in HTML_BLOCK_TAGS:
            self._append_break()

    def handle_endtag(self, tag):
        normalized = tag.casefold()
        if normalized in {"script", "style"}:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if normalized in HTML_BLOCK_TAGS:
            self._append_break()

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._chunks and self._chunks[-1] not in {"\n", " "}:
            self._chunks.append(" ")
        self._chunks.append(text)

    def text(self) -> str:
        rendered = "".join(self._chunks)
        return "\n".join(line.strip() for line in rendered.splitlines() if line.strip())

    def _append_break(self):
        if self._chunks and self._chunks[-1] != "\n":
            self._chunks.append("\n")


def gmail_payload_header(email: dict, name: str) -> str | None:
    """Return a native Gmail API payload header value by case-insensitive name."""
    payload = email.get("payload")
    if not isinstance(payload, dict):
        return None
    headers = payload.get("headers")
    if not isinstance(headers, list):
        return None

    target_name = name.strip().casefold()
    if not target_name:
        return None
    for header in headers:
        if not isinstance(header, dict):
            continue
        header_name = header.get("name")
        if not isinstance(header_name, str):
            continue
        if header_name.strip().casefold() != target_name:
            continue
        value = header.get("value")
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            return value
    return None


def _parse_cache_datetime(value: object) -> datetime | None:
    return parse_cache_datetime(value, datetime_type=datetime)
