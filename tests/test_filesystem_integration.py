"""
Tests for morpheus.integrations.filesystem.
"""
import json

import pytest

from morpheus.integrations.filesystem import FileSystemWatcher


def test_scan_reports_new_modified_and_deleted_files(tmp_path):
    watched = tmp_path / "watched"
    watched.mkdir()
    note = watched / "note.txt"
    note.write_text("TODO: write docs\n")

    watcher = FileSystemWatcher(watched)

    first_scan = watcher.scan()

    assert watcher.file_hashes == {"note.txt": first_scan[0]["hash"]}
    assert first_scan == [
        {
            "path": "note.txt",
            "status": "new",
            "hash": watcher.file_hashes["note.txt"],
            "size": len("TODO: write docs\n"),
            "modified": first_scan[0]["modified"],
        }
    ]

    note.write_text("TODO: write docs\nFIXME: repair sync\n")
    modified_scan = watcher.scan()

    assert [change["status"] for change in modified_scan] == ["modified"]
    assert modified_scan[0]["path"] == "note.txt"

    note.unlink()
    deleted_scan = watcher.scan()

    assert deleted_scan == [
        {
            "path": "note.txt",
            "status": "deleted",
            "hash": modified_scan[0]["hash"],
            "size": 0,
            "modified": None,
        }
    ]


def test_scan_ignores_morpheus_cache_and_recovers_from_bad_cache(tmp_path):
    (tmp_path / ".morpheus").mkdir()
    (tmp_path / ".morpheus" / "fs_cache.json").write_text("{not json")
    (tmp_path / ".morpheus" / "internal.txt").write_text("TODO: ignore me")
    (tmp_path / "readme.md").write_text("hello")

    changes = FileSystemWatcher(tmp_path).scan()

    assert [change["path"] for change in changes] == ["readme.md"]
    assert changes[0]["status"] == "new"

    cache = json.loads((tmp_path / ".morpheus" / "fs_cache.json").read_text())
    assert list(cache) == ["readme.md"]


def test_scan_ignores_symlinked_files_outside_root(tmp_path):
    watched = tmp_path / "watched"
    outside = tmp_path / "outside"
    watched.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("TODO: do not index through symlink\n")
    link = watched / "linked-secret.txt"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    changes = FileSystemWatcher(watched).scan()

    assert changes == []


def test_scan_skips_files_that_cannot_be_hashed(tmp_path, monkeypatch):
    bad = tmp_path / "bad.md"
    good = tmp_path / "good.md"
    bad.write_text("TODO: transient unreadable file\n")
    good.write_text("TODO: keep scanning readable files\n")
    watcher = FileSystemWatcher(tmp_path)
    original_sha256 = FileSystemWatcher._sha256

    def raise_for_bad_file(self, path):
        if path == bad:
            raise OSError("permission denied")
        return original_sha256(self, path)

    monkeypatch.setattr(FileSystemWatcher, "_sha256", raise_for_bad_file)

    changes = watcher.scan()

    assert [change["path"] for change in changes] == ["good.md"]
    assert json.loads((tmp_path / ".morpheus" / "fs_cache.json").read_text()) == {
        "good.md": changes[0]["hash"]
    }


def test_extract_claims_rejects_paths_outside_root(tmp_path):
    watched = tmp_path / "watched"
    outside = tmp_path / "outside"
    watched.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("TODO: do not read outside root\n")

    claims = FileSystemWatcher(watched).extract_claims("../outside/secret.txt")

    assert claims == []


def test_extract_claims_ignores_excluded_internal_paths(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    (morpheus_dir / "internal.txt").write_text("TODO: do not read internal state\n")

    claims = FileSystemWatcher(tmp_path).extract_claims(".morpheus/internal.txt")

    assert claims == []


def test_extract_claims_returns_empty_for_directories(tmp_path):
    source_dir = tmp_path / "notes"
    source_dir.mkdir()

    claims = FileSystemWatcher(tmp_path).extract_claims("notes")

    assert claims == []


def test_extract_claims_returns_empty_for_unreadable_files(tmp_path, monkeypatch):
    source = tmp_path / "notes.md"
    source.write_text("TODO: unreadable\n")

    original_read_text = type(source).read_text

    def raise_for_source(path, *args, **kwargs):
        if path == source:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(source), "read_text", raise_for_source)

    claims = FileSystemWatcher(tmp_path).extract_claims("notes.md")

    assert claims == []


def test_extract_claims_returns_marker_locations(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("intro\nDECISION: use receipts\nplain\nXXX: investigate edge case\n")

    claims = FileSystemWatcher(tmp_path).extract_claims("notes.md")

    assert claims == [
        {
            "path": "notes.md",
            "line": 2,
            "marker": "DECISION:",
            "excerpt": "DECISION: use receipts",
        },
        {
            "path": "notes.md",
            "line": 4,
            "marker": "XXX:",
            "excerpt": "XXX: investigate edge case",
        },
    ]
