"""Shared evidence marker helpers for external integrations."""
import re

INTEGRATION_EVIDENCE_KEYWORDS = (
    "DECISION:",
    "DECIDED:",
    "TODO:",
    "FIXME:",
    "NOTE:",
    "ACTION:",
    "WILL:",
    "COMMIT:",
    "AGREED:",
    "HACK:",
    "XXX:",
)

MAX_EVIDENCE_EXCERPT_LENGTH = 500


def matched_keyword_excerpts(
    text: str,
    *,
    keywords: tuple[str, ...] = INTEGRATION_EVIDENCE_KEYWORDS,
    max_length: int = MAX_EVIDENCE_EXCERPT_LENGTH,
) -> list[tuple[str, str]]:
    """Return detected evidence keywords with excerpts that include the matched marker."""
    matches = []
    seen_keywords = set()
    normalized_keywords = []
    for keyword in keywords:
        normalized_keyword = keyword.strip()
        if not normalized_keyword:
            continue
        keyword_key = normalized_keyword.upper()
        if keyword_key in seen_keywords:
            continue
        seen_keywords.add(keyword_key)
        normalized_keywords.append(normalized_keyword)

    for keyword_order, keyword in enumerate(normalized_keywords):
        first_occurrence = True
        pattern = re.compile(rf"(?<!\w){re.escape(keyword)}", re.IGNORECASE)
        for match in pattern.finditer(text):
            index = match.start()
            excerpt = evidence_excerpt(
                text if first_occurrence else text[index:],
                keyword,
                index=index if first_occurrence else 0,
                max_length=max_length,
            )
            matches.append((index, keyword_order, keyword, excerpt))
            first_occurrence = False
    return [(keyword, excerpt) for _, _, keyword, excerpt in sorted(matches)]


def evidence_excerpt(text: str, keyword: str, *, index: int | None = None, max_length: int = 500) -> str:
    """Return a bounded excerpt, shifting forward when needed to keep keyword visible."""
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text

    keyword_index = text.upper().find(keyword.upper()) if index is None else index
    if keyword_index < 0 or keyword_index + len(keyword) <= max_length:
        return text[:max_length]
    return text[keyword_index:keyword_index + max_length]
