from pathlib import Path
import os


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "0.2.0b2"


def read_project_file(relative_path: str) -> str:
    path = ROOT / relative_path
    assert path.is_file(), f"Missing release scaffold file: {relative_path}"
    return path.read_text(encoding="utf-8")


def assert_contains_all(content: str, snippets: list[str]) -> None:
    missing = [snippet for snippet in snippets if snippet not in content]
    assert not missing, f"Missing expected snippets: {missing}"


def workflow_job_section(workflow: str, job_name: str) -> str:
    marker = f"  {job_name}:\n"
    lines = workflow.splitlines(keepends=True)
    start = lines.index(marker)
    end = len(lines)
    for index, line in enumerate(lines[start + 1:], start + 1):
        if line.startswith("  ") and not line.startswith("    "):
            end = index
            break
    return "".join(lines[start:end])


def test_package_and_runtime_versions_match_release_contract():
    from morpheus import __version__

    project = tomllib.loads(read_project_file("pyproject.toml"))
    assert project["project"]["version"] == RELEASE_VERSION
    assert __version__ == RELEASE_VERSION


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
    publish_job = workflow_job_section(workflow, "publish-pypi")

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
    assert "needs: build" in publish_job
    assert (
        "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
        in publish_job
    )
    assert "pypa/gh-action-pypi-publish@release/v1" in publish_job


def test_release_workflow_verifies_tag_and_quality_before_publishing():
    workflow = read_project_file(".github/workflows/release.yml")

    required_steps = [
        "name: Install package and release tooling",
        'python -m pip install -e ".[dev]" build twine',
        "name: Verify tag matches package version",
        "github.ref_type == 'tag'",
        "github.ref_name",
        "import tomllib",
        "ruff check .",
        "pytest tests/ -q",
        "python -m build",
        "python -m twine check dist/*",
    ]
    assert_contains_all(workflow, required_steps)

    publish_job = workflow.index("publish-pypi:")
    for snippet in required_steps:
        assert workflow.index(snippet) < publish_job


def test_release_docs_define_canonical_strict_pre_tag_and_publish_protocol():
    release = read_project_file("docs/RELEASE.md")
    testing = read_project_file("docs/TESTING.md")

    assert_contains_all(
        release,
        [
            ".venv/bin/ruff check .",
            ".venv/bin/pytest tests/ -q",
            ".venv/bin/morpheus stale .",
            ".venv/bin/morpheus compile",
            "sed -n '1,240p' WAKE.md",
            ".venv/bin/morpheus diagnostics --json",
            ".venv/bin/morpheus agent-connect --json",
            ".venv/bin/morpheus wake . --private",
            ".venv/bin/morpheus verify --all",
            "make verify",
            "make build",
            ".venv/bin/python -m twine check dist/*",
            "morpheus_wake-*.dist-info/METADATA",
            f"Morpheus AI v{RELEASE_VERSION}",
            "Push the verified release commit to `main`",
            "exact commit SHA",
            "Python 3.10",
            "Python 3.11",
            "Python 3.12",
            "Package build",
            "before creating or pushing an annotated tag",
            "tag -> tag workflow -> verify PyPI wheel and sdist -> "
            "GitHub Release -> pinned post-publish smoke",
            "Never reuse a public tag",
            "Never replace an artifact accepted by PyPI",
        ],
    )
    assert_contains_all(
        testing,
        [
            "[canonical release sequence](RELEASE.md)",
            ".venv/bin/ruff check .",
            ".venv/bin/pytest tests/ -q",
            ".venv/bin/morpheus stale .",
            ".venv/bin/morpheus compile",
            "sed -n '1,240p' WAKE.md",
            ".venv/bin/morpheus diagnostics --json",
            ".venv/bin/morpheus agent-connect --json",
            ".venv/bin/morpheus wake . --private",
            ".venv/bin/morpheus verify --all",
            "make verify",
            "make build",
            ".venv/bin/python -m twine check dist/*",
            "morpheus_wake-*.dist-info/METADATA",
            f"Morpheus AI v{RELEASE_VERSION}",
            "Push the verified release commit to `main`",
            "exact commit SHA",
            "Python 3.10",
            "Python 3.11",
            "Python 3.12",
            "Package build",
        ],
    )


def test_public_roadmap_surfaces_mark_v03_through_v07_complete():
    expected_markers = {
        "README.md": [
            f"**v0.{minor} (complete in the current code)**"
            for minor in range(3, 8)
        ],
        "README.ru.md": [
            f"**v0.{minor} (завершён в текущем коде)**"
            for minor in range(3, 8)
        ],
        "SPEC.md": [
            "**v0.3 Semantic classifier (complete in current code)**",
            "**v0.4 Dataset quality dashboard (complete in current code)**",
            "**v0.5 Adapter memory benchmark (complete in current code)**",
            "**v0.6 Agent memory routing (complete in current code)**",
            "**v0.7 Team learning loop (complete in current code)**",
        ],
        "docs/ROADMAP.md": [
            "## v0.3: Semantic Classifier As Product Core — Complete In Current Code",
            "## v0.4: Dataset Quality Dashboard — Complete In Current Code",
            "## v0.5: Adapter Memory Benchmark — Complete In Current Code",
            "## v0.6: Agent Memory Routing — Complete In Current Code",
            "## v0.7: Team Learning Loop — Complete In Current Code",
        ],
        "docs/WHY_WAKE.md": [
            f"**v0.{minor} — complete in the current code**"
            for minor in range(3, 8)
        ],
        "WAKE.md": [
            f"v0.{minor}" for minor in range(3, 8)
        ] + [
            "1. v0.3 semantic classifier as product core — complete in current code.",
            "2. v0.4 dataset quality dashboard — complete in current code.",
        ],
    }

    for relative_path, markers in expected_markers.items():
        content = read_project_file(relative_path)
        assert_contains_all(content, markers)

    roadmap = read_project_file("docs/ROADMAP.md")
    v03 = roadmap.split("## v0.3:", 1)[1].split("## v0.4:", 1)[0]
    v04 = roadmap.split("## v0.4:", 1)[1].split("## v0.5:", 1)[0]
    for section in [v03, v04]:
        assert "Verified acceptance criteria:" in section
        criteria = section.split("Verified acceptance criteria:", 1)[1]
        assert "- [x]" in criteria
        assert all(
            line.startswith("- [x]")
            for line in criteria.splitlines()
            if line.startswith("- ")
        )
    assert_contains_all(
        v04,
        [
            "dataset manifests record class, trainability, and route counts",
            "shared quality report computes per-candidate routing and trainability reasons",
            "aggregate `top_blockers`",
            "CLI and API expose the shared report",
            "browser dashboard renders aggregate quality counts, gates, and blockers",
            "not per-claim reasons",
        ],
    )
    assert "dataset manifests include category counts and top blockers" not in v04
    assert "dashboard can explain why a claim is not trainable" not in v04

    english_surfaces = [
        read_project_file(path)
        for path in [
            "README.md",
            "SPEC.md",
            "docs/ROADMAP.md",
            "docs/WHY_WAKE.md",
            "WAKE.md",
        ]
    ]
    for content in english_surfaces:
        assert "No milestone after v0.7 is currently defined." in content

    assert "После v0.7 следующий milestone пока не определён." in read_project_file(
        "README.ru.md"
    )


def test_public_roadmap_surfaces_do_not_call_implemented_pipeline_the_next_step():
    surfaces = [
        read_project_file(path).casefold()
        for path in [
            "README.md",
            "README.ru.md",
            "SPEC.md",
            "docs/ROADMAP.md",
            "docs/WHY_WAKE.md",
            "WAKE.md",
        ]
    ]

    for content in surfaces:
        assert "next product axis" not in content
        assert "the next step is" not in content
        assert (
            "moving toward a verified classification-to-training pipeline"
            not in content
        )
        assert "следующая продуктовая ось" not in content


def test_distribution_name_avoids_existing_pypi_project():
    pyproject = read_project_file("pyproject.toml")

    assert 'name = "morpheus-wake"' in pyproject
    assert f'version = "{RELEASE_VERSION}"' in pyproject
    assert 'name = "morpheus-ai"' not in pyproject
    assert 'morpheus = "morpheus.cli:app"' in pyproject
    assert '"typer>=0.12.0,<0.26"' in pyproject
    assert "typer[all]" not in pyproject
    assert_contains_all(
        pyproject,
        [
            "[project.urls]",
            'Homepage = "https://github.com/botAGI/morpheus-ai"',
            'Repository = "https://github.com/botAGI/morpheus-ai"',
            'Issues = "https://github.com/botAGI/morpheus-ai/issues"',
            'Changelog = "https://github.com/botAGI/morpheus-ai/blob/main/CHANGELOG.md"',
            'Release = "https://github.com/botAGI/morpheus-ai/releases"',
        ],
    )


def test_quickstart_uses_distribution_name_and_morpheus_command():
    readme = read_project_file("README.md")
    readme_ru = read_project_file("README.ru.md")

    assert "https://github.com/botAGI/morpheus-ai/blob/main/README.ru.md" in readme
    assert "https://github.com/botAGI/morpheus-ai/blob/main/WAKE.md" in readme
    assert "https://raw.githubusercontent.com/botAGI/morpheus-ai/main/demo/morpheus-demo.gif" in readme
    assert "](README.ru.md)" not in readme
    assert "](WAKE.md)" not in readme
    assert "](demo/morpheus-demo.gif)" not in readme
    for content in [readme, readme_ru]:
        assert f"uvx --from 'morpheus-wake=={RELEASE_VERSION}' morpheus wake ." in content
        assert f"pipx run --spec 'morpheus-wake=={RELEASE_VERSION}' morpheus wake ." in content
        assert "python -m pip install -e \".[dev]\"" in content


def test_current_docs_distinguish_package_version_from_release_tag():
    readme = read_project_file("README.md")
    readme_ru = read_project_file("README.ru.md")
    beta_exit = read_project_file("docs/BETA_EXIT_PLAN.md")
    readiness = read_project_file("docs/RELEASE_READINESS_2026-07-20.md")

    for content in [readme, readme_ru]:
        assert f"Current beta package: {RELEASE_VERSION}." in content
        assert f"Current beta package: v{RELEASE_VERSION}." not in content
    assert beta_exit.count(f"current beta package `{RELEASE_VERSION}`") == 2
    assert f"current beta package `v{RELEASE_VERSION}`" not in beta_exit
    assert f"package version is `{RELEASE_VERSION}` and the Git tag is `v{RELEASE_VERSION}`" in readiness


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
    assert_contains_all(
        makefile,
        [
            "install-dev:",
            "lint:",
            "test:",
            "verify:",
            "build:",
            "serve:",
            "UV ?=",
            "pip install --python",
        ],
    )


def test_make_verify_runs_lint_and_tests_through_project_python():
    makefile = read_project_file("Makefile")

    assert_contains_all(
        makefile,
        [
            "$(PYTHON) -m ruff check .",
            "$(PYTHON) -m pytest tests/ -q",
        ],
    )


def test_changelog_has_current_b2_and_historical_release_sections():
    changelog = read_project_file("CHANGELOG.md")

    assert "## [Unreleased]" in changelog
    assert "## [0.2.0b2] - 2026-07-20" in changelog
    assert changelog.index("## [Unreleased]") < changelog.index("## [0.2.0b2] - 2026-07-20")
    assert changelog.index("## [0.2.0b2] - 2026-07-20") < changelog.index("## [0.2.0b1] - 2026-05-20")
    assert "morpheus-active-state-review-authority/1" in changelog
    assert "morpheus-team-learning/2" in changelog
    assert "## [0.2.0b1] - 2026-05-20" in changelog
    assert "Review-gated semantic compile alpha" in changelog
    assert "Repeat-2 dogfood MLX stability report" in changelog
    assert "## [0.1.1] - 2026-05-18" in changelog
    assert "## [0.1.0] - 2026-05-17" in changelog
    assert "### Added" in changelog
    assert "### Fixed" in changelog
    assert "Removed local testbot path from public AGENTS.md." in changelog
    assert "WAKE.md" in changelog
    assert "morpheus-wake" in changelog


def test_v020b2_release_notes_state_beta_boundaries_and_local_gates():
    notes = read_project_file("docs/release-notes/v0.2.0b2.md")

    assert notes.startswith("# v0.2.0b2 — First verify. Then learn.\n")
    assert_contains_all(
        notes,
        [
            "`morpheus-active-state-review-authority/1`",
            "`morpheus-benchmark-categories/1`",
            "`morpheus-team-learning/2`",
            "No accepted source span means no training example.",
            "No eval pass means no adapter activation.",
            "No rollback means no production activation.",
            "Cloud integrations remain opt-in.",
            "uvx --from 'morpheus-wake==0.2.0b2' morpheus wake . --private",
            "ruff check .",
            "pytest tests/ -q",
            "morpheus wake . --private",
            "morpheus verify --all",
        ],
    )


def test_current_docs_use_exact_production_activation_invariant():
    for relative_path in [
        "README.md",
        "docs/LEARNING_CORE.md",
        "docs/ROADMAP.md",
        "docs/release-notes/v0.2.0b2.md",
        "morpheus/training/README.md",
    ]:
        content = read_project_file(relative_path)
        assert "No rollback means no production activation." in content

    readme_ru = read_project_file("README.ru.md")
    assert "Нет rollback - нет production activation." in readme_ru


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
            "Visual terminal demo in the README",
            "MCP/A2A-style local interop",
            "UI launchpad",
            "Cache-backed integrations",
            "Experimental training is explicitly not the core launch path",
        ],
    )


def test_demo_scaffold_is_safe_and_self_contained():
    demo_readme = read_project_file("demo/README.md")
    transcript = read_project_file("demo/transcript.md")
    cast = read_project_file("demo/morpheus-demo.cast")
    script_path = ROOT / "demo/record_demo.sh"
    script = read_project_file("demo/record_demo.sh")
    gif_path = ROOT / "demo/morpheus-demo.gif"

    assert os.access(script_path, os.X_OK), "demo/record_demo.sh should be executable"
    assert gif_path.is_file(), "demo/morpheus-demo.gif should be committed for README"
    assert gif_path.stat().st_size > 10_000, "demo GIF should not be empty"
    assert_contains_all(
        demo_readme,
        [
            "asciinema",
            "agg",
            "demo.cast",
            "demo.gif",
            "morpheus-demo.cast",
            "morpheus-demo.gif",
            "morpheus wake .",
        ],
    )
    assert_contains_all(
        cast,
        [
            "uvx --from morpheus-wake morpheus wake .",
            "Read WAKE.md and continue.",
            "Agent State Compiler",
        ],
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


def test_daily_training_script_uses_safe_learning_lab_gate():
    script_path = ROOT / "scripts" / "daily_training.sh"
    script = read_project_file("scripts/daily_training.sh")

    assert os.access(script_path, os.X_OK), "scripts/daily_training.sh should be executable"
    assert_contains_all(
        script,
        [
            "morpheus wake . --private",
            "morpheus verify --all",
            "morpheus learn lab . --dogfood --no-train",
            "morpheus learn status",
        ],
    )
    forbidden = [
        "morpheus consolidate",
        "morpheus train",
        "dataset.jsonl",
        "morpheus_adapters/daily",
    ]
    for snippet in forbidden:
        assert snippet not in script


def test_mcp_truth_tools_smoke_script_is_local_and_complete():
    script_path = ROOT / "scripts" / "mcp_truth_tools_smoke.py"
    script = read_project_file("scripts/mcp_truth_tools_smoke.py")

    assert os.access(script_path, os.X_OK), "scripts/mcp_truth_tools_smoke.py should be executable"
    assert_contains_all(
        script,
        [
            "127.0.0.1",
            "morpheus_check_text",
            "morpheus_get_active_state",
            "morpheus_get_evidence_for_claim",
            "morpheus_get_wake",
            "MCP_TRUTH_TOOLS_SMOKE_PASS",
        ],
    )
    forbidden = ["OPENAI_API_KEY", "anthropic", "api.openai.com", "0.0.0.0"]
    for snippet in forbidden:
        assert snippet not in script
