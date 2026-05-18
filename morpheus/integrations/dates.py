"""
Shared date parsing helpers for local integration cache files.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

MILLISECONDS_EPOCH_THRESHOLD = 10_000_000_000


def parse_cache_datetime(value: object, *, datetime_type: Any = datetime) -> datetime | None:
    """Parse ISO/RFC3339 or Unix timestamp cache values as timezone-aware datetimes."""
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, dict):
        return _parse_mapping_datetime(value, datetime_type=datetime_type)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _parse_epoch_seconds(value)
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lstrip("+-").replace(".", "", 1).isdigit():
        return _parse_epoch_seconds(normalized)

    try:
        parsed = datetime_type.fromisoformat(_normalize_utc_suffix(normalized))
    except (TypeError, ValueError):
        parsed = _parse_rfc2822_datetime(normalized)
        return parsed or _parse_epoch_seconds(normalized)

    return _normalize_datetime(parsed)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_utc_suffix(value: str) -> str:
    if value.endswith(("Z", "z")):
        return value[:-1] + "+00:00"
    return value


def _parse_mapping_datetime(value: dict, *, datetime_type: Any) -> datetime | None:
    for key in ("dateTime", "date"):
        if key not in value:
            continue
        parsed = parse_cache_datetime(value[key], datetime_type=datetime_type)
        if parsed is not None:
            return parsed
    return None


def _parse_epoch_seconds(value: object) -> datetime | None:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None

    candidates = [timestamp]
    if abs(timestamp) >= MILLISECONDS_EPOCH_THRESHOLD:
        candidates.insert(0, timestamp / 1000)

    for candidate in candidates:
        try:
            return datetime.fromtimestamp(candidate, timezone.utc)
        except (OSError, OverflowError, ValueError):
            continue
    return None


def _parse_rfc2822_datetime(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
