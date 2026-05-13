"""
Tests for external integrations.
"""
import json
from datetime import datetime, timedelta, timezone

from morpheus.integrations.calendar import CalendarIntegration
from morpheus.integrations.github import GitHubIntegration
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


def test_gmail_get_emails_respects_max_results_for_cache(tmp_path):
    now = datetime.now(timezone.utc)
    token_path = tmp_path / "gmail_token.json"
    token_path.write_text("{}")
    (tmp_path / "gmail_cache.json").write_text(
        json.dumps(
            [
                {"id": "first", "date": now.isoformat(), "snippet": "TODO: first"},
                {"id": "second", "date": now.isoformat(), "snippet": "TODO: second"},
            ]
        )
    )

    emails = GmailIntegration(token_path=token_path).get_emails(days=30, max_results=1)

    assert [email["id"] for email in emails] == ["first"]


def test_gmail_extract_evidence_handles_null_snippet(tmp_path):
    evidence = GmailIntegration(token_path=tmp_path / "token.json").extract_evidence(
        {"id": "email-1", "snippet": None}
    )

    assert evidence == []


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


def test_calendar_get_events_respects_max_results_for_cache(tmp_path):
    now = datetime.now(timezone.utc)
    token_path = tmp_path / "calendar_token.json"
    token_path.write_text("{}")
    (tmp_path / "calendar_cache.json").write_text(
        json.dumps(
            [
                {"id": "first", "start": now.isoformat(), "summary": "TODO: first"},
                {"id": "second", "start": now.isoformat(), "summary": "TODO: second"},
            ]
        )
    )

    events = CalendarIntegration(token_path=token_path).get_events(days=30, max_results=1)

    assert [event["id"] for event in events] == ["first"]


def test_calendar_extract_evidence_handles_null_text_fields(tmp_path):
    evidence = CalendarIntegration(token_path=tmp_path / "token.json").extract_evidence(
        {"id": "event-1", "description": None, "summary": None}
    )

    assert evidence == []


def test_github_get_issues_filters_by_recent_update(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"number": 1, "updated_at": now.isoformat()},
                {"number": 2, "updated_at": (now - timedelta(days=45)).isoformat()},
                {"number": 3, "updated_at": "not-a-date"},
                "not an issue",
            ]

    def fake_get(url, *, headers, params, timeout):
        assert url == "https://api.github.com/repos/owner/repo/issues"
        assert params == {"state": "all", "per_page": 100}
        assert timeout == 10
        return Response()

    monkeypatch.setattr("httpx.get", fake_get)

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=30,
    )

    assert [issue["number"] for issue in issues] == [1]
