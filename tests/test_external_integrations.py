"""
Tests for external integrations.
"""
import json
from datetime import datetime, timedelta, timezone

from morpheus.integrations.calendar import CalendarIntegration
from morpheus.integrations.github import GitHubIntegration
from morpheus.integrations.gmail import GmailIntegration


def test_github_get_repo_returns_empty_dict_for_non_object_response(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return ["not", "a", "repo"]

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())

    repo = GitHubIntegration(token_path=tmp_path / "missing-token").get_repo(
        "owner",
        "repo",
    )

    assert repo == {}


def test_github_get_repo_returns_empty_dict_for_malformed_json_response(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("invalid json")

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())

    repo = GitHubIntegration(token_path=tmp_path / "missing-token").get_repo(
        "owner",
        "repo",
    )

    assert repo == {}


def test_github_authenticate_requires_token_file(tmp_path):
    token_path = tmp_path / "github_token.txt"
    token_path.mkdir()

    assert not GitHubIntegration(token_path=token_path).authenticate()


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
                {"id": "bad-type", "date": 1234567890, "snippet": "TODO: bad type"},
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


def test_gmail_authenticate_rejects_token_directory(tmp_path):
    token_path = tmp_path / "gmail_token.json"
    token_path.mkdir()

    try:
        GmailIntegration(token_path=token_path).authenticate()
    except RuntimeError as exc:
        assert "Gmail not authenticated" in str(exc)
    else:
        raise AssertionError("authenticate should reject token directories")


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
                {"id": "bad-type", "start": 1234567890, "summary": "DECISION: bad type"},
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


def test_calendar_authenticate_rejects_token_directory(tmp_path):
    token_path = tmp_path / "calendar_token.json"
    token_path.mkdir()

    try:
        CalendarIntegration(token_path=token_path).authenticate()
    except RuntimeError as exc:
        assert "Calendar not authenticated" in str(exc)
    else:
        raise AssertionError("authenticate should reject token directories")


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
                {"number": 4, "updated_at": 1234567890},
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


def test_github_get_issues_returns_empty_list_for_non_list_response(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return 42

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=30,
    )

    assert issues == []


def test_github_get_issues_returns_empty_list_for_malformed_json_response(
    monkeypatch,
    tmp_path,
):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("invalid json")

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=30,
    )

    assert issues == []


def test_github_get_issues_skips_pull_request_issue_rows(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"number": 1, "updated_at": now.isoformat()},
                {
                    "number": 2,
                    "updated_at": now.isoformat(),
                    "pull_request": {"url": "https://api.github.com/pulls/2"},
                },
            ]

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=30,
    )

    assert [issue["number"] for issue in issues] == [1]


def test_github_get_pulls_returns_only_object_rows(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"number": 1}, "not a pull", {"number": 2}]

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())

    pulls = GitHubIntegration(token_path=tmp_path / "missing-token").get_pulls(
        "owner",
        "repo",
    )

    assert [pull["number"] for pull in pulls] == [1, 2]


def test_github_get_pulls_returns_empty_list_for_malformed_json_response(
    monkeypatch,
    tmp_path,
):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("invalid json")

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())

    pulls = GitHubIntegration(token_path=tmp_path / "missing-token").get_pulls(
        "owner",
        "repo",
    )

    assert pulls == []
