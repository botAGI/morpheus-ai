"""
Tests for external integrations.
"""
import json
from datetime import datetime, timedelta, timezone

from morpheus.integrations.calendar import CalendarIntegration
from morpheus.integrations.gmail import GmailIntegration


def test_gmail_cache_loads_timezone_dates_and_skips_invalid_rows(tmp_path):
    now = datetime.now(timezone.utc)
    cache_path = tmp_path / "gmail_cache.json"
    cache_path.write_text(
        json.dumps(
            [
                {"id": "new", "date": now.isoformat(), "snippet": "TODO: keep this"},
                {
                    "id": "old",
                    "date": (now - timedelta(days=45)).isoformat(),
                    "snippet": "TODO: too old",
                },
                {"id": "bad-date", "date": "not-a-date", "snippet": "TODO: bad"},
                ["not", "a", "message"],
            ]
        )
    )

    emails = GmailIntegration(token_path=tmp_path / "token.json")._load_from_cache(
        cache_path,
        days=30,
    )

    assert [email["id"] for email in emails] == ["new"]


def test_calendar_cache_loads_timezone_dates_and_skips_invalid_rows(tmp_path):
    now = datetime.now(timezone.utc)
    cache_path = tmp_path / "calendar_cache.json"
    cache_path.write_text(
        json.dumps(
            [
                {"id": "new", "start": now.isoformat(), "summary": "DECISION: keep this"},
                {
                    "id": "old",
                    "start": (now - timedelta(days=45)).isoformat(),
                    "summary": "DECISION: too old",
                },
                {"id": "bad-date", "start": "not-a-date", "summary": "DECISION: bad"},
                "not an event",
            ]
        )
    )

    events = CalendarIntegration(token_path=tmp_path / "token.json")._load_from_cache(
        cache_path,
        days=30,
    )

    assert [event["id"] for event in events] == ["new"]
