"""
Shared date parsing helpers for local integration cache files.
"""
from datetime import datetime, timezone
from typing import Any


def parse_cache_datetime(value: object, *, datetime_type: Any = datetime) -> datetime | None:
    """Parse ISO/RFC3339 or Unix timestamp cache values as timezone-aware datetimes."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _parse_epoch_seconds(value)
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    if not normalized:
        return None

    try:
        parsed = datetime_type.fromisoformat(normalized.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return _parse_epoch_seconds(normalized)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_epoch_seconds(value: object) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), timezone.utc)
    except (OSError, OverflowError, TypeError, ValueError):
        return None
