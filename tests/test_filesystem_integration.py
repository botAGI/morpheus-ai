"""
Tests for morpheus.integrations.filesystem.
"""
import json

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
