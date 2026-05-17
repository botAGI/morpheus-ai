from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]


def read_project_file(relative_path: str) -> str:
    path = ROOT / relative_path
    assert path.is_file(), f"Missing release scaffold file: {relative_path}"
    return path.read_text(encoding="utf-8")


def assert_contains_all(content: str, snippets: list[str]) -> None:
    missing = [snippet for snippet in snippets if snippet not in content]
    assert not missing, f"Missing expected snippets: {missing}"


def test_ci_workflow_runs_lint_tests_and_package_build():
    workflow = read_project_file(".github/workflows/ci.yml")

    assert_contains_all(
        workflow,
        [
            "pull_request:",
            "push:",
            "permissions:",
            "contents: read",
            "3.10",
            "3.11",
            "3.12",
            'python -m pip install -e ".[dev]"',
            "ruff check .",
            "pytest tests/",
            "python -m build",
            "python -m twine check dist/*",
        ],
    )


def test_release_workflow_uses_trusted_publishing_without_pypi_token():
    workflow = read_project_file(".github/workflows/release.yml")

    assert_contains_all(
        workflow,
        [
            "workflow_dispatch:",
            "tags:",
            "v*.*.*",
            "contents: read",
            "id-token: write",
            "environment:",
            "name: pypi",
            "https://pypi.org/p/morpheus-wake",
            "pypa/gh-action-pypi-publish@release/v1",
        ],
    )
    assert "morpheus-ai" not in workflow
    assert "PYPI_TOKEN" not in workflow
    assert "password:" not in workflow


def test_distribution_name_avoids_existing_pypi_project():
    pyproject = read_project_file("pyproject.toml")

    assert 'name = "morpheus-wake"' in pyproject
    assert 'name = "morpheus-ai"' not in pyproject
    assert 'morpheus = "morpheus.cli:app"' in pyproject


def test_quickstart_uses_distribution_name_and_morpheus_command():
    readme = read_project_file("README.md")
    readme_ru = read_project_file("README.ru.md")

    for content in [readme, readme_ru]:
        assert "uvx --from morpheus-wake morpheus wake ." in content
        assert "pipx run --spec morpheus-wake morpheus wake ." in content
        assert "python -m pip install -e \".[dev]\"" in content


def test_dependabot_tracks_actions_and_python_dependencies():
    config = read_project_file(".github/dependabot.yml")

    assert_contains_all(
        config,
        [
            'package-ecosystem: "github-actions"',
            'package-ecosystem: "pip"',
            'directory: "/"',
            'interval: "weekly"',
        ],
    )


def test_github_actions_use_node_24_compatible_majors():
    ci = read_project_file(".github/workflows/ci.yml")
    release = read_project_file(".github/workflows/release.yml")

    for workflow in [ci, release]:
        assert "actions/checkout@v6" in workflow
        assert "actions/setup-python@v6" in workflow
        assert "actions/checkout@v4" not in workflow
        assert "actions/setup-python@v5" not in workflow
    assert "actions/upload-artifact@v7" in ci
    assert "actions/upload-artifact@v7" in release
    assert "actions/download-artifact@v8" in release
    assert "actions/upload-artifact@v4" not in ci + release
    assert "actions/download-artifact@v4" not in release


def test_container_scaffold_runs_morpheus_as_non_root_service():
    dockerfile = read_project_file("Dockerfile")
    dockerignore = read_project_file(".dockerignore")

    assert_contains_all(
        dockerfile,
        [
            "FROM python:3.12-slim",
            "useradd --create-home --shell /usr/sbin/nologin morpheus",
            "USER morpheus",
            "EXPOSE 8000 5173",
            '"morpheus",',
            '"serve",',
            '"--ui",',
            '"--host",',
            '"0.0.0.0",',
        ],
    )
    assert_contains_all(
        dockerignore,
        [".git", ".venv", "__pycache__", ".pytest_cache", "dist", "build", ".morpheus", "WAKE.md"],
    )


def test_project_has_release_security_and_contributor_docs():
    license_text = read_project_file("LICENSE")
    security = read_project_file("SECURITY.md")
    contributing = read_project_file("CONTRIBUTING.md")
    changelog = read_project_file("CHANGELOG.md")
    makefile = read_project_file("Makefile")

    assert_contains_all(license_text, ["MIT License", "Morpheus Team"])
    assert_contains_all(
        security,
        [
            "Supported Versions",
            "Reporting a Vulnerability",
            "Local-first Security Model",
            "MCP/A2A",
        ],
    )
    assert_contains_all(
        contributing,
        [
            "Development Setup",
            "Quality Gates",
            "Release Process",
            "Trusted Publishing",
        ],
    )
    assert_contains_all(changelog, ["## [Unreleased]", "A2A", "MCP", "Quickstart"])
    assert_contains_all(makefile, ["install-dev:", "lint:", "test:", "verify:", "build:", "serve:"])


def test_changelog_has_v010_release_section():
    changelog = read_project_file("CHANGELOG.md")

    assert "## [Unreleased]\n\n## [0.1.0] - 2026-05-17" in changelog
    assert "### Added" in changelog
    assert "### Fixed" in changelog
    assert "WAKE.md" in changelog
    assert "morpheus-wake" in changelog


def test_v010_release_notes_cover_launch_highlights():
    notes = read_project_file("docs/release-notes/v0.1.0.md")

    assert notes.startswith("# v0.1.0 — WAKE.md for AI agents\n")
    assert_contains_all(
        notes,
        [
            "Generate WAKE.md from project state",
            "One-command `morpheus wake .` flow",
            "Public/private WAKE modes",
            "Verifiable provenance receipts",
            "Agent handoff",
            "MCP/A2A-style local interop",
            "UI launchpad",
            "Cache-backed integrations",
            "Experimental training is explicitly not the core launch path",
        ],
    )


def test_demo_scaffold_is_safe_and_self_contained():
    demo_readme = read_project_file("demo/README.md")
    transcript = read_project_file("demo/transcript.md")
    script_path = ROOT / "demo/record_demo.sh"
    script = read_project_file("demo/record_demo.sh")

    assert os.access(script_path, os.X_OK), "demo/record_demo.sh should be executable"
    assert_contains_all(
        demo_readme,
        ["asciinema", "agg", "demo.cast", "demo.gif", "morpheus wake ."],
    )
    assert_contains_all(
        transcript,
        [
            "Without Morpheus",
            "With Morpheus",
            "Read WAKE.md and continue",
        ],
    )
    assert_contains_all(
        script,
        [
            "mktemp -d",
            "DECISION:",
            "TODO:",
            "NOTE:",
            "morpheus wake .",
            "morpheus verify --all",
            "morpheus stale .",
            "Paste this into an agent: Read WAKE.md and continue.",
        ],
    )
    forbidden = ["OPENAI_API_KEY", "OBSIDIAN", "OpenClaw", "Hermes", "curl http"]
    for snippet in forbidden:
        assert snippet not in script
