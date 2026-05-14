"""
Tests for external integrations.
"""
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

import morpheus.integrations.calendar as calendar_module
import morpheus.integrations.gmail as gmail_module
from morpheus.integrations.calendar import CalendarIntegration
from morpheus.integrations.github import GitHubIntegration
from morpheus.integrations.gmail import GmailIntegration


def github_status_error_response(status_code: int = 403):
    request = httpx.Request("GET", "https://api.github.com/test")
    response = httpx.Response(status_code, request=request)

    class Response:
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                f"GitHub returned {status_code}",
                request=request,
                response=response,
            )

        def json(self):
            raise AssertionError("GitHub response body should not be parsed after HTTP errors")

    return Response()


def github_json_response(payload, *, link: str | None = None):
    request = httpx.Request("GET", "https://api.github.com/test")
    headers = {"Link": link} if link else {}
    return httpx.Response(200, request=request, json=payload, headers=headers)


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


def test_github_get_repo_returns_empty_dict_for_network_errors(monkeypatch, tmp_path):
    def raise_request_error(*args, **kwargs):
        raise httpx.RequestError("network down")

    monkeypatch.setattr("httpx.get", raise_request_error)

    repo = GitHubIntegration(token_path=tmp_path / "missing-token").get_repo(
        "owner",
        "repo",
    )

    assert repo == {}


def test_github_get_repo_returns_empty_dict_for_http_status_errors(monkeypatch, tmp_path):
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: github_status_error_response())

    repo = GitHubIntegration(token_path=tmp_path / "missing-token").get_repo(
        "owner",
        "repo",
    )

    assert repo == {}


def test_github_authenticate_requires_token_file(tmp_path):
    token_path = tmp_path / "github_token.txt"
    token_path.mkdir()

    assert not GitHubIntegration(token_path=token_path).authenticate()


def test_github_authenticate_rejects_token_symlink(tmp_path):
    external_token = tmp_path / "external-token.txt"
    external_token.write_text("secret")
    token_path = tmp_path / "github_token.txt"
    token_path.symlink_to(external_token)
    integration = GitHubIntegration(token_path=token_path)

    assert not integration.authenticate()
    assert integration._get_token() is None


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


def test_gmail_cache_loads_rfc3339_z_dates(tmp_path):
    cache_path = tmp_path / "gmail_cache.json"
    cache_path.write_text(
        json.dumps(
            [
                {
                    "id": "gmail-z",
                    "date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "snippet": "TODO: keep z timestamp",
                }
            ]
        )
    )

    emails = GmailIntegration(token_path=tmp_path / "token.json")._load_from_cache(
        cache_path,
        days=30,
    )

    assert [email["id"] for email in emails] == ["gmail-z"]


def test_gmail_parse_cache_datetime_normalizes_z_for_python310(monkeypatch):
    class LegacyDateTime:
        @staticmethod
        def fromisoformat(value):
            if value.endswith("Z"):
                raise ValueError("Invalid isoformat string")
            return datetime.fromisoformat(value)

    monkeypatch.setattr(gmail_module, "datetime", LegacyDateTime)

    parsed = gmail_module._parse_cache_datetime("2026-05-14T10:30:00Z")

    assert parsed == datetime(2026, 5, 14, 10, 30, tzinfo=timezone.utc)


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


def test_gmail_get_emails_returns_empty_list_for_non_positive_max_results(tmp_path):
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

    emails = GmailIntegration(token_path=token_path).get_emails(days=30, max_results=-1)

    assert emails == []


def test_gmail_get_emails_returns_empty_list_for_negative_days_without_auth(tmp_path):
    emails = GmailIntegration(token_path=tmp_path / "missing-token.json").get_emails(days=-1)

    assert emails == []


def test_gmail_authenticate_rejects_token_directory(tmp_path):
    token_path = tmp_path / "gmail_token.json"
    token_path.mkdir()

    try:
        GmailIntegration(token_path=token_path).authenticate()
    except RuntimeError as exc:
        assert "Gmail not authenticated" in str(exc)
    else:
        raise AssertionError("authenticate should reject token directories")


def test_gmail_authenticate_rejects_token_symlink(tmp_path):
    external_token = tmp_path / "external-token.json"
    external_token.write_text("{}")
    token_path = tmp_path / "gmail_token.json"
    token_path.symlink_to(external_token)

    try:
        GmailIntegration(token_path=token_path).authenticate()
    except RuntimeError as exc:
        assert "Gmail not authenticated" in str(exc)
    else:
        raise AssertionError("authenticate should reject token symlinks")


def test_gmail_cache_rejects_symlinked_cache(tmp_path):
    now = datetime.now(timezone.utc)
    external_cache = tmp_path / "external-cache.json"
    external_cache.write_text(
        json.dumps([{"id": "external", "date": now.isoformat(), "snippet": "TODO: secret"}])
    )
    cache_path = tmp_path / "gmail_cache.json"
    cache_path.symlink_to(external_cache)

    emails = GmailIntegration(token_path=tmp_path / "token.json")._load_from_cache(
        cache_path,
        days=30,
    )

    assert emails == []


def test_gmail_cache_rejects_symlinked_cache_parent(tmp_path):
    now = datetime.now(timezone.utc)
    external_cache_dir = tmp_path / "external-cache-dir"
    external_cache_dir.mkdir()
    (external_cache_dir / "gmail_cache.json").write_text(
        json.dumps([{"id": "external", "date": now.isoformat(), "snippet": "TODO: secret"}])
    )
    linked_cache_dir = tmp_path / "linked-cache-dir"
    try:
        linked_cache_dir.symlink_to(external_cache_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    emails = GmailIntegration(token_path=tmp_path / "token.json")._load_from_cache(
        linked_cache_dir / "gmail_cache.json",
        days=30,
    )

    assert emails == []


def test_gmail_extract_evidence_handles_null_snippet(tmp_path):
    evidence = GmailIntegration(token_path=tmp_path / "token.json").extract_evidence(
        {"id": "email-1", "snippet": None}
    )

    assert evidence == []


def test_gmail_extract_evidence_truncates_long_snippets(tmp_path):
    snippet = "TODO: " + ("x" * 600)

    evidence = GmailIntegration(token_path=tmp_path / "token.json").extract_evidence(
        {"id": "email-1", "snippet": snippet}
    )

    assert evidence == [
        {
            "type": "email_claim",
            "source": "gmail",
            "email_id": "email-1",
            "keyword": "TODO:",
            "excerpt": snippet[:500],
        }
    ]


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


def test_calendar_cache_loads_rfc3339_z_dates(tmp_path):
    cache_path = tmp_path / "calendar_cache.json"
    cache_path.write_text(
        json.dumps(
            [
                {
                    "id": "calendar-z",
                    "start": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "summary": "TODO: keep z timestamp",
                }
            ]
        )
    )

    events = CalendarIntegration(token_path=tmp_path / "token.json")._load_from_cache(
        cache_path,
        days=30,
    )

    assert [event["id"] for event in events] == ["calendar-z"]


def test_calendar_parse_cache_datetime_normalizes_z_for_python310(monkeypatch):
    class LegacyDateTime:
        @staticmethod
        def fromisoformat(value):
            if value.endswith("Z"):
                raise ValueError("Invalid isoformat string")
            return datetime.fromisoformat(value)

    monkeypatch.setattr(calendar_module, "datetime", LegacyDateTime)

    parsed = calendar_module._parse_cache_datetime("2026-05-14T10:30:00Z")

    assert parsed == datetime(2026, 5, 14, 10, 30, tzinfo=timezone.utc)


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


def test_calendar_get_events_returns_empty_list_for_non_positive_max_results(tmp_path):
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

    events = CalendarIntegration(token_path=token_path).get_events(days=30, max_results=-1)

    assert events == []


def test_calendar_get_events_returns_empty_list_for_negative_days_without_auth(tmp_path):
    events = CalendarIntegration(token_path=tmp_path / "missing-token.json").get_events(days=-1)

    assert events == []


def test_calendar_authenticate_rejects_token_directory(tmp_path):
    token_path = tmp_path / "calendar_token.json"
    token_path.mkdir()

    try:
        CalendarIntegration(token_path=token_path).authenticate()
    except RuntimeError as exc:
        assert "Calendar not authenticated" in str(exc)
    else:
        raise AssertionError("authenticate should reject token directories")


def test_calendar_authenticate_rejects_token_symlink(tmp_path):
    external_token = tmp_path / "external-token.json"
    external_token.write_text("{}")
    token_path = tmp_path / "calendar_token.json"
    token_path.symlink_to(external_token)

    try:
        CalendarIntegration(token_path=token_path).authenticate()
    except RuntimeError as exc:
        assert "Calendar not authenticated" in str(exc)
    else:
        raise AssertionError("authenticate should reject token symlinks")


def test_calendar_cache_rejects_symlinked_cache(tmp_path):
    now = datetime.now(timezone.utc)
    external_cache = tmp_path / "external-cache.json"
    external_cache.write_text(
        json.dumps([{"id": "external", "start": now.isoformat(), "summary": "TODO: secret"}])
    )
    cache_path = tmp_path / "calendar_cache.json"
    cache_path.symlink_to(external_cache)

    events = CalendarIntegration(token_path=tmp_path / "token.json")._load_from_cache(
        cache_path,
        days=30,
    )

    assert events == []


def test_calendar_cache_rejects_symlinked_cache_parent(tmp_path):
    now = datetime.now(timezone.utc)
    external_cache_dir = tmp_path / "external-cache-dir"
    external_cache_dir.mkdir()
    (external_cache_dir / "calendar_cache.json").write_text(
        json.dumps([{"id": "external", "start": now.isoformat(), "summary": "TODO: secret"}])
    )
    linked_cache_dir = tmp_path / "linked-cache-dir"
    try:
        linked_cache_dir.symlink_to(external_cache_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    events = CalendarIntegration(token_path=tmp_path / "token.json")._load_from_cache(
        linked_cache_dir / "calendar_cache.json",
        days=30,
    )

    assert events == []


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


def test_github_get_issues_follows_next_page_links(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    second_page = "https://api.github.com/repos/owner/repo/issues?page=2"
    calls = []

    def fake_get(url, *, headers, params=None, timeout):
        calls.append((url, params))
        assert headers == {}
        assert timeout == 10
        if len(calls) == 1:
            return github_json_response(
                [{"number": 1, "updated_at": now.isoformat()}],
                link=f'<{second_page}>; rel="next"',
            )
        return github_json_response([{"number": 2, "updated_at": now.isoformat()}])

    monkeypatch.setattr("httpx.get", fake_get)

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        state="open",
        days=30,
    )

    assert [issue["number"] for issue in issues] == [1, 2]
    assert calls == [
        (
            "https://api.github.com/repos/owner/repo/issues",
            {"state": "open", "per_page": 100},
        ),
        (second_page, None),
    ]


def test_github_get_issues_ignores_cross_origin_next_page_links(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    calls = []

    def fake_get(url, *, headers, params=None, timeout):
        calls.append((url, params))
        if url != "https://api.github.com/repos/owner/repo/issues":
            raise AssertionError("GitHub pagination should not follow cross-origin links")
        return github_json_response(
            [{"number": 1, "updated_at": now.isoformat()}],
            link='<https://evil.example/repos/owner/repo/issues?page=2>; rel="next"',
        )

    monkeypatch.setattr("httpx.get", fake_get)

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=30,
    )

    assert [issue["number"] for issue in issues] == [1]
    assert calls == [
        (
            "https://api.github.com/repos/owner/repo/issues",
            {"state": "all", "per_page": 100},
        )
    ]


def test_github_get_issues_respects_max_results(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)

    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: github_json_response(
            [
                {"number": 1, "updated_at": now.isoformat()},
                {"number": 2, "updated_at": now.isoformat()},
            ]
        ),
    )

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=30,
        max_results=1,
    )

    assert [issue["number"] for issue in issues] == [1]


def test_github_get_issues_stops_fetching_pages_at_max_results(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    second_page = "https://api.github.com/repos/owner/repo/issues?page=2"

    def fake_get(url, *, headers, params=None, timeout):
        if url == second_page:
            raise AssertionError("GitHub issues should stop after max_results")
        return github_json_response(
            [{"number": 1, "updated_at": now.isoformat()}],
            link=f'<{second_page}>; rel="next"',
        )

    monkeypatch.setattr("httpx.get", fake_get)

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=30,
        max_results=1,
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


def test_github_get_issues_returns_empty_list_for_network_errors(monkeypatch, tmp_path):
    def raise_request_error(*args, **kwargs):
        raise httpx.RequestError("network down")

    monkeypatch.setattr("httpx.get", raise_request_error)

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=30,
    )

    assert issues == []


def test_github_get_issues_returns_empty_list_for_http_status_errors(monkeypatch, tmp_path):
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: github_status_error_response())

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=30,
    )

    assert issues == []


def test_github_get_issues_returns_empty_list_for_non_positive_max_results(
    monkeypatch,
    tmp_path,
):
    def fail_request(*args, **kwargs):
        raise AssertionError("GitHub issues should not be fetched")

    monkeypatch.setattr("httpx.get", fail_request)

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        max_results=0,
    )

    assert issues == []


def test_github_get_issues_returns_empty_list_for_negative_days_without_request(
    monkeypatch,
    tmp_path,
):
    def fail_request(*args, **kwargs):
        raise AssertionError("GitHub issues should not be fetched")

    monkeypatch.setattr("httpx.get", fail_request)

    issues = GitHubIntegration(token_path=tmp_path / "missing-token").get_issues(
        "owner",
        "repo",
        days=-1,
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


def test_github_get_pulls_follows_next_page_links(monkeypatch, tmp_path):
    second_page = "https://api.github.com/repos/owner/repo/pulls?page=2"
    calls = []

    def fake_get(url, *, headers, params=None, timeout):
        calls.append((url, params))
        assert headers == {}
        assert timeout == 10
        if len(calls) == 1:
            return github_json_response(
                [{"number": 1}],
                link=f'<{second_page}>; rel="next"',
            )
        return github_json_response([{"number": 2}])

    monkeypatch.setattr("httpx.get", fake_get)

    pulls = GitHubIntegration(token_path=tmp_path / "missing-token").get_pulls(
        "owner",
        "repo",
        state="closed",
    )

    assert [pull["number"] for pull in pulls] == [1, 2]
    assert calls == [
        (
            "https://api.github.com/repos/owner/repo/pulls",
            {"state": "closed", "per_page": 100},
        ),
        (second_page, None),
    ]


def test_github_get_pulls_respects_max_results(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: github_json_response([{"number": 1}, {"number": 2}]),
    )

    pulls = GitHubIntegration(token_path=tmp_path / "missing-token").get_pulls(
        "owner",
        "repo",
        max_results=1,
    )

    assert [pull["number"] for pull in pulls] == [1]


def test_github_get_pulls_stops_fetching_pages_at_max_results(monkeypatch, tmp_path):
    second_page = "https://api.github.com/repos/owner/repo/pulls?page=2"

    def fake_get(url, *, headers, params=None, timeout):
        if url == second_page:
            raise AssertionError("GitHub pulls should stop after max_results")
        return github_json_response([{"number": 1}], link=f'<{second_page}>; rel="next"')

    monkeypatch.setattr("httpx.get", fake_get)

    pulls = GitHubIntegration(token_path=tmp_path / "missing-token").get_pulls(
        "owner",
        "repo",
        max_results=1,
    )

    assert [pull["number"] for pull in pulls] == [1]


def test_github_get_pulls_returns_empty_list_for_non_positive_max_results(
    monkeypatch,
    tmp_path,
):
    def fail_request(*args, **kwargs):
        raise AssertionError("GitHub pulls should not be fetched")

    monkeypatch.setattr("httpx.get", fail_request)

    pulls = GitHubIntegration(token_path=tmp_path / "missing-token").get_pulls(
        "owner",
        "repo",
        max_results=0,
    )

    assert pulls == []


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


def test_github_get_pulls_returns_empty_list_for_http_status_errors(monkeypatch, tmp_path):
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: github_status_error_response())

    pulls = GitHubIntegration(token_path=tmp_path / "missing-token").get_pulls(
        "owner",
        "repo",
    )

    assert pulls == []


def test_github_get_pulls_returns_empty_list_for_network_errors(monkeypatch, tmp_path):
    def raise_request_error(*args, **kwargs):
        raise httpx.RequestError("network down")

    monkeypatch.setattr("httpx.get", raise_request_error)

    pulls = GitHubIntegration(token_path=tmp_path / "missing-token").get_pulls(
        "owner",
        "repo",
    )

    assert pulls == []


def test_github_get_commits_returns_recent_paginated_object_rows(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    second_page = "https://api.github.com/repos/owner/repo/commits?page=2"
    calls = []

    def fake_get(url, *, headers, params=None, timeout):
        calls.append((url, params))
        assert headers == {}
        assert timeout == 10
        if len(calls) == 1:
            return github_json_response(
                [
                    {"sha": "new-a", "commit": {"committer": {"date": now.isoformat()}}},
                    {
                        "sha": "old",
                        "commit": {
                            "committer": {
                                "date": (now - timedelta(days=45)).isoformat(),
                            }
                        },
                    },
                    "not a commit",
                ],
                link=f'<{second_page}>; rel="next"',
            )
        return github_json_response(
            [
                {"sha": "new-b", "commit": {"author": {"date": now.isoformat()}}},
                {"sha": "bad-date", "commit": {"committer": {"date": "not-a-date"}}},
                {"sha": "missing-date", "commit": {}},
            ]
        )

    monkeypatch.setattr("httpx.get", fake_get)

    commits = GitHubIntegration(token_path=tmp_path / "missing-token").get_commits(
        "owner",
        "repo",
        days=30,
    )

    assert [commit["sha"] for commit in commits] == ["new-a", "new-b"]
    assert calls[0][0] == "https://api.github.com/repos/owner/repo/commits"
    assert calls[0][1]["per_page"] == 100
    since = datetime.fromisoformat(calls[0][1]["since"])
    assert now - timedelta(days=31) < since < now
    assert calls[1] == (second_page, None)


def test_github_get_commits_respects_max_results(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)

    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: github_json_response(
            [
                {"sha": "new-a", "commit": {"committer": {"date": now.isoformat()}}},
                {"sha": "new-b", "commit": {"committer": {"date": now.isoformat()}}},
            ]
        ),
    )

    commits = GitHubIntegration(token_path=tmp_path / "missing-token").get_commits(
        "owner",
        "repo",
        days=30,
        max_results=1,
    )

    assert [commit["sha"] for commit in commits] == ["new-a"]


def test_github_get_commits_stops_fetching_pages_at_max_results(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    second_page = "https://api.github.com/repos/owner/repo/commits?page=2"

    def fake_get(url, *, headers, params=None, timeout):
        if url == second_page:
            raise AssertionError("GitHub commits should stop after max_results")
        return github_json_response(
            [{"sha": "new-a", "commit": {"committer": {"date": now.isoformat()}}}],
            link=f'<{second_page}>; rel="next"',
        )

    monkeypatch.setattr("httpx.get", fake_get)

    commits = GitHubIntegration(token_path=tmp_path / "missing-token").get_commits(
        "owner",
        "repo",
        days=30,
        max_results=1,
    )

    assert [commit["sha"] for commit in commits] == ["new-a"]


def test_github_get_commits_returns_empty_list_for_non_positive_max_results(
    monkeypatch,
    tmp_path,
):
    def fail_request(*args, **kwargs):
        raise AssertionError("GitHub commits should not be fetched")

    monkeypatch.setattr("httpx.get", fail_request)

    commits = GitHubIntegration(token_path=tmp_path / "missing-token").get_commits(
        "owner",
        "repo",
        max_results=0,
    )

    assert commits == []


def test_github_get_commits_returns_empty_list_for_negative_days_without_request(
    monkeypatch,
    tmp_path,
):
    def fail_request(*args, **kwargs):
        raise AssertionError("GitHub commits should not be fetched")

    monkeypatch.setattr("httpx.get", fail_request)

    commits = GitHubIntegration(token_path=tmp_path / "missing-token").get_commits(
        "owner",
        "repo",
        days=-1,
    )

    assert commits == []


def test_github_extract_evidence_reads_issue_and_pull_request_text(tmp_path):
    item = {
        "number": 42,
        "title": "DECISION: keep signed receipts",
        "body": "TODO: document the verification flow",
        "html_url": "https://github.com/owner/repo/issues/42",
    }

    evidence = GitHubIntegration(token_path=tmp_path / "missing-token").extract_evidence(
        item,
        item_type="issue",
    )

    assert evidence == [
        {
            "type": "github_claim",
            "source": "github",
            "item_type": "issue",
            "item_id": 42,
            "keyword": "DECISION:",
            "excerpt": "DECISION: keep signed receipts\nTODO: document the verification flow",
            "url": "https://github.com/owner/repo/issues/42",
        },
        {
            "type": "github_claim",
            "source": "github",
            "item_type": "issue",
            "item_id": 42,
            "keyword": "TODO:",
            "excerpt": "DECISION: keep signed receipts\nTODO: document the verification flow",
            "url": "https://github.com/owner/repo/issues/42",
        },
    ]


def test_github_extract_evidence_reads_commit_messages_and_handles_null_text(tmp_path):
    commit = {
        "sha": "abc123",
        "html_url": "https://github.com/owner/repo/commit/abc123",
        "title": None,
        "commit": {
            "message": "Fix parser\n\nAGREED: preserve malformed rows for review",
        },
    }

    evidence = GitHubIntegration(token_path=tmp_path / "missing-token").extract_evidence(
        commit,
        item_type="commit",
    )

    assert evidence == [
        {
            "type": "github_claim",
            "source": "github",
            "item_type": "commit",
            "item_id": "abc123",
            "keyword": "AGREED:",
            "excerpt": "Fix parser\n\nAGREED: preserve malformed rows for review",
            "url": "https://github.com/owner/repo/commit/abc123",
        }
    ]


def test_github_extract_evidence_infers_commit_item_type(tmp_path):
    commit = {
        "sha": "abc123",
        "commit": {"message": "TODO: wire commit evidence into compilation"},
    }

    evidence = GitHubIntegration(token_path=tmp_path / "missing-token").extract_evidence(commit)

    assert evidence == [
        {
            "type": "github_claim",
            "source": "github",
            "item_type": "commit",
            "item_id": "abc123",
            "keyword": "TODO:",
            "excerpt": "TODO: wire commit evidence into compilation",
            "url": None,
        }
    ]
