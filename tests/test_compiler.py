"""
Tests for morpheus.core.compiler
"""
import hashlib
import tempfile
from pathlib import Path

import pytest

from morpheus.core.compiler import (
    compute_sha256,
    compile_project,
    EVIDENCE_MARKERS
)


def test_compute_sha256():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("hello world")
        f.flush()
        path = Path(f.name)
    
    try:
        hash1 = compute_sha256(path)
        assert len(hash1) == 64
        assert hash1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        
        # Same content = same hash
        hash2 = compute_sha256(path)
        assert hash1 == hash2
        
        # Different file = different hash
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f2:
            f2.write("different content")
            f2.flush()
            path2 = Path(f2.name)
        
        try:
            hash3 = compute_sha256(path2)
            assert hash1 != hash3
        finally:
            path2.unlink()
    finally:
        path.unlink()


def test_compute_sha256_streams_large_files(monkeypatch):
    content = b"a" * (1024 * 1024 + 17)
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    def fail_read_bytes(self):
        raise AssertionError("compute_sha256 should stream file content")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    try:
        hash1 = compute_sha256(path)
        assert len(hash1) == 64
        assert hash1 == hashlib.sha256(content).hexdigest()
    finally:
        try:
            path.unlink()
        except AssertionError:
            pytest.fail("temporary file cleanup should not use patched read_bytes")


def test_compute_sha256_rejects_symlinked_files(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="must not be a symlink"):
        compute_sha256(link)


def test_compute_sha256_rejects_symlinked_parent_directory(tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "source.txt").write_text("secret")
    linked_dir = tmp_path / "linked"
    try:
        linked_dir.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Source file must not contain a symlink"):
        compute_sha256(linked_dir / "source.txt")


def test_compile_project_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test files
        (tmppath / "README.md").write_text("# Project\n\nTest project.")
        (tmppath / "main.py").write_text("print('hello')")
        
        state = compile_project(tmppath)
        
        assert len(state.sources) == 2
        paths = {s.path for s in state.sources}
        assert "README.md" in paths
        assert "main.py" in paths


def test_compile_project_rejects_symlinked_project_root_without_scanning_target(tmp_path):
    outside = tmp_path / "outside-project"
    outside.mkdir()
    (outside / "README.md").write_text("TODO: do not compile through symlinked root\n")
    project_root = tmp_path / "linked-project"
    try:
        project_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Project root must not be a symlink"):
        compile_project(project_root)


def test_compile_project_rejects_symlinked_project_root_parent(tmp_path):
    outside_parent = tmp_path / "outside-parent"
    outside_project = outside_parent / "project"
    outside_project.mkdir(parents=True)
    (outside_project / "README.md").write_text(
        "TODO: do not compile through symlinked project parent\n"
    )
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(outside_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Project root must not contain a symlink"):
        compile_project(linked_parent / "project")


def test_compile_project_records_actual_file_size_for_non_utf8_bytes():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        data = b"\xff\xfeTODO: binary marker\n"
        binary_path = tmppath / "data.bin"
        binary_path.write_bytes(data)

        state = compile_project(tmppath)

        source = next(source for source in state.sources if source.path == "data.bin")
        assert source.size_bytes == len(data)


def test_compile_project_skips_unreadable_files(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        unreadable = tmppath / "unreadable.md"
        unreadable.write_text("TODO: skip unreadable\n")
        (tmppath / "readable.md").write_text("TODO: keep readable\n")

        import morpheus.core.compiler as compiler_module

        original_compute_sha256 = compiler_module.compute_sha256

        def raise_for_unreadable(path):
            if path.resolve() == unreadable.resolve():
                raise OSError("permission denied")
            return original_compute_sha256(path)

        monkeypatch.setattr(compiler_module, "compute_sha256", raise_for_unreadable)

        state = compile_project(tmppath)

        assert [source.path for source in state.sources] == ["readable.md"]
        assert [claim.excerpt for claim in state.claims] == ["TODO: keep readable"]


def test_compile_project_ignores_symlinked_files_outside_project():
    with tempfile.TemporaryDirectory() as project_dir, tempfile.TemporaryDirectory() as outside_dir:
        project_path = Path(project_dir)
        secret_path = Path(outside_dir) / "secret.txt"
        secret_path.write_text("TODO: do not compile external symlink\n")
        link_path = project_path / "linked-secret.txt"
        try:
            link_path.symlink_to(secret_path)
        except OSError as exc:
            pytest.skip(f"symlink creation unsupported: {exc}")

        state = compile_project(project_path)

        assert "linked-secret.txt" not in {source.path for source in state.sources}


def test_compile_project_excludes():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create excluded dirs
        (tmppath / ".morpheus").mkdir()
        (tmppath / ".morpheus" / "config.toml").write_text("[project]")
        
        # Create .git dir properly
        git_dir = tmppath / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]")
        
        (tmppath / "node_modules").mkdir()
        (tmppath / "node_modules" / "pkg").write_text("package")
        
        (tmppath / "__pycache__").mkdir()
        (tmppath / "__pycache__" / "mod.pyc").write_text("\x00\x01")
        
        # Create valid file
        (tmppath / "valid.txt").write_text("content")
        
        state = compile_project(tmppath)
        
        valid_paths = {s.path for s in state.sources}
        assert "valid.txt" in valid_paths
        assert ".morpheus/config.toml" not in valid_paths
        assert ".git/config" not in valid_paths
        assert "node_modules/pkg" not in valid_paths
        assert "__pycache__/mod.pyc" not in valid_paths


def test_compile_project_excludes_common_local_tool_outputs_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        ignored_dirs = [
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            "test-results",
            "build",
            "dist",
        ]
        for ignored_dir in ignored_dirs:
            path = tmppath / ignored_dir
            path.mkdir()
            (path / "generated.txt").write_text("TODO: ignore generated local output\n")

        (tmppath / "valid.txt").write_text("TODO: keep source\n")

        state = compile_project(tmppath)

        assert {source.path for source in state.sources} == {"valid.txt"}
        assert [claim.excerpt for claim in state.claims] == ["TODO: keep source"]


def test_compile_project_excludes_repo_hygiene_artifacts_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        ignored_dirs = [
            ".idea",
            ".vscode",
            "target",
            "ui/target",
            "ui/dist",
            "reports",
            "morpheus_adapters",
        ]
        for ignored_dir in ignored_dirs:
            path = tmppath / ignored_dir
            path.mkdir(parents=True)
            (path / "generated.txt").write_text("TODO: ignore generated artifact\n")

        for ignored_file in [
            ".DS_Store",
            "debug.log",
            "server.pid",
            "dataset.jsonl",
            "WAKE.md",
        ]:
            (tmppath / ignored_file).write_text("TODO: ignore generated file\n")

        (tmppath / "valid.txt").write_text("TODO: keep source\n")

        state = compile_project(tmppath)

        assert {source.path for source in state.sources} == {"valid.txt"}
        assert [claim.excerpt for claim in state.claims] == ["TODO: keep source"]


def test_compile_project_excludes_common_secret_files_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        for file_name in [
            ".env",
            ".env.local",
            "local.key",
            "identity.pem",
            "id_rsa",
            "id_ed25519",
            "certificate.p12",
            "keystore.pfx",
        ]:
            (tmppath / file_name).write_text("TODO: do not compile secrets\n")
        (tmppath / "valid.txt").write_text("TODO: keep source\n")

        state = compile_project(tmppath)

        assert {source.path for source in state.sources} == {"valid.txt"}
        assert [claim.excerpt for claim in state.claims] == ["TODO: keep source"]


def test_compile_project_extracts_claims():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        (tmppath / "tasks.md").write_text("""# Tasks

TODO: implement feature X
DECISION: using Python for backend

## Notes

FIXME: this is broken
NOTE: remember to document
""")
        
        state = compile_project(tmppath)
        
        assert len(state.claims) >= 3
        
        categories = {c.category for c in state.claims}
        assert "task" in categories
        assert "decision" in categories
        assert "fixme" in categories
        assert "note" in categories
        
        # Check evidence exists
        assert len(state.evidence) >= 3
        
        # Evidence references valid claims
        claim_ids = {c.id for c in state.claims}
        for ev in state.evidence:
            assert ev.claim_id in claim_ids


def test_compile_project_extracts_xxx_markers_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "notes.md").write_text("intro\nXXX: investigate edge case\n")

        state = compile_project(tmppath)

        assert [claim.excerpt for claim in state.claims] == [
            "XXX: investigate edge case"
        ]
        assert state.claims[0].category == "xxx"
        assert [evidence.excerpt for evidence in state.evidence] == [
            "XXX: investigate edge case"
        ]


def test_compile_project_matches_markers_case_insensitively():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "notes.md").write_text(
            "todo: handle lowercase markers\nDecision: keep mixed-case markers\n"
        )

        state = compile_project(tmppath)

        assert [claim.excerpt for claim in state.claims] == [
            "todo: handle lowercase markers",
            "Decision: keep mixed-case markers",
        ]
        assert [claim.category for claim in state.claims] == ["task", "decision"]
        assert [evidence.excerpt for evidence in state.evidence] == [
            "todo: handle lowercase markers",
            "Decision: keep mixed-case markers",
        ]


def test_compile_project_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        state = compile_project(tmppath)
        
        assert len(state.sources) == 0
        assert len(state.claims) == 0
        assert len(state.evidence) == 0


def test_evidence_markers():
    assert "TODO:" in EVIDENCE_MARKERS
    assert "DECISION:" in EVIDENCE_MARKERS
    assert "FIXME:" in EVIDENCE_MARKERS
    assert "XXX:" in EVIDENCE_MARKERS


def test_compile_project_generates_stable_unique_claim_and_evidence_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        (tmppath / "b.py").write_text("TODO: second file\nFIXME: second fix\n")
        (tmppath / "a.py").write_text("TODO: first file\n")

        state = compile_project(tmppath)

        assert [s.path for s in state.sources] == ["a.py", "b.py"]
        assert [c.id for c in state.claims] == ["clm_0001", "clm_0002", "clm_0003"]
        assert [e.id for e in state.evidence] == ["ev_0001", "ev_0002", "ev_0003"]
        assert len({c.id for c in state.claims}) == len(state.claims)
        assert len({e.id for e in state.evidence}) == len(state.evidence)


def test_compile_project_respects_configured_exclude_patterns():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = ["."]
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus", "generated", "*.log"]
evidence_markers = ["TODO:", "DECISION:", "FIXME:", "NOTE:", "HACK:"]
integrations = {}
"""
        )
        (tmppath / "generated").mkdir()
        (tmppath / "generated" / "ignored.py").write_text("TODO: ignore generated output\n")
        (tmppath / "debug.log").write_text("TODO: ignore logs\n")
        (tmppath / "src").mkdir()
        (tmppath / "src" / "keep.py").write_text("TODO: keep source\n")

        state = compile_project(tmppath)

        assert [source.path for source in state.sources] == ["src/keep.py"]
        assert [claim.excerpt for claim in state.claims] == ["TODO: keep source"]


def test_compile_project_respects_configured_evidence_markers():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = ["."]
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus"]
evidence_markers = ["ACTION:"]
integrations = {}
"""
        )
        (tmppath / "notes.md").write_text("TODO: ignore default marker\nACTION: follow up\n")

        state = compile_project(tmppath)

        assert [claim.excerpt for claim in state.claims] == ["ACTION: follow up"]
        assert state.claims[0].category == "action"


def test_compile_project_respects_empty_configured_evidence_markers():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = ["."]
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus"]
evidence_markers = []
integrations = {}
"""
        )
        (tmppath / "notes.md").write_text("TODO: explicitly ignored marker\n")

        state = compile_project(tmppath)

        assert [source.path for source in state.sources] == ["notes.md"]
        assert state.claims == []
        assert state.evidence == []


def test_compile_project_ignores_blank_configured_evidence_markers():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = ["."]
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus"]
evidence_markers = ["", "   ", "TODO:"]
integrations = {}
"""
        )
        (tmppath / "notes.md").write_text("plain line\nTODO: keep marker\n")

        state = compile_project(tmppath)

        assert [claim.excerpt for claim in state.claims] == ["TODO: keep marker"]
        assert [evidence.excerpt for evidence in state.evidence] == ["TODO: keep marker"]


def test_compile_project_deduplicates_configured_evidence_markers():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = ["."]
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus"]
evidence_markers = ["TODO:", "TODO:", "FIXME:", "TODO:"]
integrations = {}
"""
        )
        (tmppath / "notes.md").write_text("TODO: keep one claim per marker\n")

        state = compile_project(tmppath)

        assert [claim.excerpt for claim in state.claims] == [
            "TODO: keep one claim per marker"
        ]
        assert [evidence.excerpt for evidence in state.evidence] == [
            "TODO: keep one claim per marker"
        ]


def test_compile_project_deduplicates_case_variant_evidence_markers():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = ["."]
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus"]
evidence_markers = ["TODO:", "todo:"]
integrations = {}
"""
        )
        (tmppath / "notes.md").write_text("ToDo: keep one claim for case variants\n")

        state = compile_project(tmppath)

        assert [claim.excerpt for claim in state.claims] == [
            "ToDo: keep one claim for case variants"
        ]
        assert [claim.category for claim in state.claims] == ["task"]


def test_compile_project_strips_configured_evidence_markers_before_matching():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = ["."]
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus"]
evidence_markers = [" TODO: ", " TODO: "]
integrations = {}
"""
        )
        (tmppath / "notes.md").write_text("TODO: marker from padded config\n")

        state = compile_project(tmppath)

        assert [claim.excerpt for claim in state.claims] == [
            "TODO: marker from padded config"
        ]
        assert [claim.category for claim in state.claims] == ["task"]
        assert [evidence.excerpt for evidence in state.evidence] == [
            "TODO: marker from padded config"
        ]


def test_compile_project_respects_configured_watch_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = ["src", "docs/decision.md"]
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus"]
evidence_markers = ["TODO:", "DECISION:"]
integrations = {}
"""
        )
        (tmppath / "src").mkdir()
        (tmppath / "src" / "app.py").write_text("TODO: watched source\n")
        (tmppath / "docs").mkdir()
        (tmppath / "docs" / "decision.md").write_text("DECISION: watched file\n")
        (tmppath / "docs" / "ignored.md").write_text("TODO: unwatched sibling\n")
        (tmppath / "README.md").write_text("TODO: unwatched root\n")

        state = compile_project(tmppath)

        assert [source.path for source in state.sources] == [
            "docs/decision.md",
            "src/app.py",
        ]
        assert [claim.excerpt for claim in state.claims] == [
            "DECISION: watched file",
            "TODO: watched source",
        ]


def test_compile_project_skips_watch_dirs_that_fail_during_traversal(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = ["bad", "good"]
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus"]
evidence_markers = ["TODO:"]
integrations = {}
"""
        )
        bad_dir = tmppath / "bad"
        good_dir = tmppath / "good"
        bad_dir.mkdir()
        good_dir.mkdir()
        (bad_dir / "ignored.md").write_text("TODO: skipped traversal\n")
        (good_dir / "kept.md").write_text("TODO: keep scanning\n")

        path_type = type(tmppath)
        original_rglob = path_type.rglob

        def raise_for_bad_dir(path, pattern):
            if path.resolve() == bad_dir.resolve():
                raise OSError("permission denied")
            return original_rglob(path, pattern)

        monkeypatch.setattr(path_type, "rglob", raise_for_bad_dir)

        state = compile_project(tmppath)

        assert [source.path for source in state.sources] == ["good/kept.md"]
        assert [claim.excerpt for claim in state.claims] == ["TODO: keep scanning"]


def test_compile_project_respects_empty_configured_watch_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        morpheus_dir = tmppath / ".morpheus"
        morpheus_dir.mkdir()
        (morpheus_dir / "morpheus.toml").write_text(
            """
watch_dirs = []
exclude_patterns = [".git", "node_modules", "__pycache__", ".morpheus"]
evidence_markers = ["TODO:"]
integrations = {}
"""
        )
        (tmppath / "README.md").write_text("TODO: not watched\n")

        state = compile_project(tmppath)

        assert state.sources == []
        assert state.claims == []
        assert state.evidence == []
