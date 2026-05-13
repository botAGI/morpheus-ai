"""
Tests for morpheus.core.compiler
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
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
