"""
Tests for morpheus.integrations.filesystem.
"""
import hashlib
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


def test_scan_excludes_common_secret_and_generated_files(tmp_path):
    (tmp_path / ".env").write_text("TODO: do not index env secrets\n")
    (tmp_path / "local.key").write_text("TODO: do not index signing keys\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.txt").write_text("TODO: do not index dependencies\n")
    (tmp_path / "valid.txt").write_text("TODO: keep source\n")

    changes = FileSystemWatcher(tmp_path).scan()

    assert [change["path"] for change in changes] == ["valid.txt"]
    assert FileSystemWatcher(tmp_path).extract_claims(".env") == []


def test_scan_excludes_repo_hygiene_artifacts_by_default(tmp_path):
    ignored_dirs = [
        ".idea",
        ".vscode",
        "target",
        "ui/target",
        "reports",
        "morpheus_adapters",
        "htmlcov",
        "env",
    ]
    for ignored_dir in ignored_dirs:
        path = tmp_path / ignored_dir
        path.mkdir(parents=True)
        (path / "generated.txt").write_text("TODO: ignore generated artifact\n")

    for ignored_file in [
        ".DS_Store",
        "debug.log",
        "server.pid",
        "dataset.jsonl",
        "WAKE.md",
    ]:
        (tmp_path / ignored_file).write_text("TODO: ignore generated file\n")

    (tmp_path / "valid.txt").write_text("TODO: keep source\n")

    changes = FileSystemWatcher(tmp_path).scan()

    assert [change["path"] for change in changes] == ["valid.txt"]


def test_scan_does_not_report_deleted_entries_for_excluded_cached_paths(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    (morpheus_dir / "fs_cache.json").write_text(
        json.dumps({".morpheus/internal.txt": "old-hash"})
    )

    changes = FileSystemWatcher(tmp_path).scan()

    assert changes == []
    assert json.loads((morpheus_dir / "fs_cache.json").read_text()) == {}


def test_scan_ignores_malformed_cache_entries(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    (morpheus_dir / "fs_cache.json").write_text(
        json.dumps({"missing.txt": {"not": "a hash"}, 123: "bad path"})
    )

    changes = FileSystemWatcher(tmp_path).scan()

    assert changes == []
    assert json.loads((morpheus_dir / "fs_cache.json").read_text()) == {}


def test_scan_ignores_cache_entries_outside_root(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    cache_path = morpheus_dir / "fs_cache.json"
    cache_path.write_text(
        json.dumps({
            "../outside.md": "a" * 64,
            str(tmp_path.parent / "outside.md"): "b" * 64,
        })
    )

    changes = FileSystemWatcher(tmp_path).scan()

    assert changes == []
    assert json.loads(cache_path.read_text()) == {}


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


def test_scan_rejects_symlinked_root_without_writing_target(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TODO: do not scan symlinked roots\n")
    watched = tmp_path / "watched"
    try:
        watched.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    changes = FileSystemWatcher(watched).scan()

    assert changes == []
    assert not (outside / ".morpheus" / "fs_cache.json").exists()


def test_scan_rejects_symlinked_root_parent_without_writing_target(tmp_path):
    outside_parent = tmp_path / "outside-parent"
    watched_target = outside_parent / "watched"
    watched_target.mkdir(parents=True)
    (watched_target / "secret.txt").write_text("TODO: do not scan symlinked parents\n")
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(outside_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    changes = FileSystemWatcher(linked_parent / "watched").scan()

    assert changes == []
    assert not (watched_target / ".morpheus" / "fs_cache.json").exists()


def test_extract_claims_rejects_symlinked_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TODO: do not read through symlinked root\n")
    watched = tmp_path / "watched"
    try:
        watched.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    claims = FileSystemWatcher(watched).extract_claims("secret.txt")

    assert claims == []


def test_extract_claims_rejects_symlinked_root_parent(tmp_path):
    outside_parent = tmp_path / "outside-parent"
    watched_target = outside_parent / "watched"
    watched_target.mkdir(parents=True)
    (watched_target / "secret.txt").write_text("TODO: do not read through parent\n")
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(outside_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    claims = FileSystemWatcher(linked_parent / "watched").extract_claims("secret.txt")

    assert claims == []


def test_sha256_rejects_symlinked_files(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="must not be a symlink"):
        FileSystemWatcher(tmp_path)._sha256(link)


def test_sha256_rejects_symlinked_parent_directories(tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "source.txt").write_text("secret")
    linked_dir = tmp_path / "linked"
    try:
        linked_dir.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(
        ValueError, match="Filesystem source path must not contain a symlink"
    ):
        FileSystemWatcher(tmp_path)._sha256(linked_dir / "source.txt")


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


def test_scan_preserves_cache_when_directory_traversal_fails(tmp_path, monkeypatch):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    cached = {"old.md": "a" * 64}
    cache_path = morpheus_dir / "fs_cache.json"
    cache_path.write_text(json.dumps(cached))
    watcher = FileSystemWatcher(tmp_path)

    def raise_during_traversal(path, *args, **kwargs):
        if path == tmp_path:
            raise OSError("transient traversal failure")
        return original_rglob(path, *args, **kwargs)

    original_rglob = type(tmp_path).rglob
    monkeypatch.setattr(type(tmp_path), "rglob", raise_during_traversal)

    changes = watcher.scan()

    assert changes == []
    assert watcher.file_hashes == cached
    assert json.loads(cache_path.read_text()) == cached


def test_scan_reports_changes_when_cache_file_cannot_be_written(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    (morpheus_dir / "fs_cache.json").mkdir()
    (tmp_path / "readme.md").write_text("TODO: keep scanning despite cache write failure\n")

    changes = FileSystemWatcher(tmp_path).scan()

    assert [change["path"] for change in changes] == ["readme.md"]
    assert changes[0]["status"] == "new"


def test_scan_rejects_symlinked_cache_without_writing_target(tmp_path):
    watched = tmp_path / "watched"
    outside = tmp_path / "outside-cache.json"
    watched.mkdir()
    morpheus_dir = watched / ".morpheus"
    morpheus_dir.mkdir()
    outside.write_text("preserve me")
    try:
        (morpheus_dir / "fs_cache.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    (watched / "readme.md").write_text("TODO: keep scanning despite blocked cache\n")

    changes = FileSystemWatcher(watched).scan()

    assert [change["path"] for change in changes] == ["readme.md"]
    assert changes[0]["status"] == "new"
    assert outside.read_text() == "preserve me"


def test_scan_rejects_symlinked_cache_parent_without_writing_target(tmp_path):
    watched = tmp_path / "watched"
    outside_cache_dir = tmp_path / "outside-cache-dir"
    watched.mkdir()
    outside_cache_dir.mkdir()
    try:
        (watched / ".morpheus").symlink_to(outside_cache_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    (watched / "readme.md").write_text("TODO: keep scanning despite blocked cache\n")

    changes = FileSystemWatcher(watched).scan()

    assert [change["path"] for change in changes] == ["readme.md"]
    assert changes[0]["status"] == "new"
    assert not (outside_cache_dir / "fs_cache.json").exists()


def test_scan_ignores_cache_from_symlinked_parent_directory(tmp_path):
    watched = tmp_path / "watched"
    outside_cache_dir = tmp_path / "outside-cache-dir"
    watched.mkdir()
    outside_cache_dir.mkdir()
    contents = "TODO: report as new despite external cache\n"
    outside_cache = outside_cache_dir / "fs_cache.json"
    outside_cache.write_text(
        json.dumps({"readme.md": hashlib.sha256(contents.encode()).hexdigest()})
    )
    try:
        (watched / ".morpheus").symlink_to(outside_cache_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    (watched / "readme.md").write_text(contents)

    changes = FileSystemWatcher(watched).scan()

    assert [change["path"] for change in changes] == ["readme.md"]
    assert changes[0]["status"] == "new"
    assert json.loads(outside_cache.read_text()) == {
        "readme.md": hashlib.sha256(contents.encode()).hexdigest()
    }


def test_scan_ignores_symlinked_cache_contents(tmp_path):
    watched = tmp_path / "watched"
    outside = tmp_path / "outside-cache.json"
    watched.mkdir()
    morpheus_dir = watched / ".morpheus"
    morpheus_dir.mkdir()
    contents = "TODO: report this as new despite symlinked cache\n"
    outside.write_text(
        json.dumps({"readme.md": hashlib.sha256(contents.encode()).hexdigest()})
    )
    try:
        (morpheus_dir / "fs_cache.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    (watched / "readme.md").write_text(contents)

    changes = FileSystemWatcher(watched).scan()

    assert [change["path"] for change in changes] == ["readme.md"]
    assert changes[0]["status"] == "new"


def test_extract_claims_rejects_paths_outside_root(tmp_path):
    watched = tmp_path / "watched"
    outside = tmp_path / "outside"
    watched.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("TODO: do not read outside root\n")

    claims = FileSystemWatcher(watched).extract_claims("../outside/secret.txt")

    assert claims == []


def test_extract_claims_rejects_absolute_paths_inside_root(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("TODO: do not expose absolute source paths\n")

    claims = FileSystemWatcher(tmp_path).extract_claims(str(source))

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
    source.write_text(
        "intro\nDECISION: use receipts\nplain\nHACK: preserve odd API shape\n"
        "XXX: investigate edge case\n"
    )

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
            "marker": "HACK:",
            "excerpt": "HACK: preserve odd API shape",
        },
        {
            "path": "notes.md",
            "line": 5,
            "marker": "XXX:",
            "excerpt": "XXX: investigate edge case",
        },
    ]
