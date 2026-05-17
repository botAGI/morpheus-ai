from pathlib import Path
import shutil
import subprocess

import pytest


def tracked_files() -> set[str]:
    if not shutil.which("git"):
        pytest.skip("git is required for repository hygiene checks")
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def test_public_docs_include_bilingual_readme_and_testing_checklist():
    assert Path("README.md").is_file()
    assert Path("README.ru.md").is_file()
    assert Path("WAKE.md").is_file()
    assert Path("docs/WHY_WAKE.md").is_file()
    assert Path("docs/TESTING.md").is_file()


def test_public_wake_is_tracked_as_showcase_artifact():
    assert "WAKE.md" in tracked_files()


def test_public_git_index_excludes_local_agent_artifacts():
    tracked = tracked_files()

    forbidden_exact = {
        "SOUL.md",
        "USER.md",
        "IDENTITY.md",
        "MEMORY.md",
    }
    assert not (tracked & forbidden_exact)
    assert not any(path.startswith("memory/") for path in tracked)
    assert not any(path.startswith("AGENT_") for path in tracked)
    assert not any(path.startswith("docs/superpowers/") for path in tracked)
    assert not any("__pycache__/" in path for path in tracked)
    assert not any(path.endswith((".pyc", ".pyo")) for path in tracked)


def test_gitignore_covers_local_memory_and_development_artifacts():
    gitignore = Path(".gitignore").read_text()

    for pattern in [
        ".morpheus/",
        "SOUL.md",
        "USER.md",
        "IDENTITY.md",
        "MEMORY.md",
        "memory/",
        "AGENT_*.md",
        "docs/superpowers/",
        "*.zip",
    ]:
        assert pattern in gitignore
