from pathlib import Path
import shutil
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

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
    assert Path("docs/RELEASE.md").is_file()
    assert Path("docs/release-notes/v0.1.0.md").is_file()
    assert Path("docs/TESTING.md").is_file()
    assert Path("docs/AUTONOMOUS_LAB.md").is_file()
    assert Path("docs/LEARNING_CORE.md").is_file()


def test_public_wake_is_tracked_as_showcase_artifact():
    assert "WAKE.md" in tracked_files()


def test_root_wake_includes_source_references():
    wake = Path("WAKE.md").read_text()

    assert "## Source References" in wake
    for source in [
        "README.md",
        "README.ru.md",
        "SPEC.md",
        "docs/WHY_WAKE.md",
        "docs/RELEASE.md",
        "CHANGELOG.md",
    ]:
        assert source in wake


def test_root_wake_next_work_is_launch_ordered():
    wake = Path("WAKE.md").read_text()

    lab = wake.index("1. Preserve the repeat-2 live dogfood MLX pass")
    mcp = wake.index("2. Keep the dedicated live MCP truth-tools smoke gate")
    stability = wake.index("3. Prepare beta release notes and CI confirmation")
    split = wake.index("4. Keep CLI/API split")
    assert lab < mcp < stability < split


def test_readme_first_screen_uses_wake_framing():
    readme = Path("README.md").read_text()
    first_screen = "\n".join(readme.splitlines()[:18])

    assert readme.startswith("# Morpheus\n")
    assert "Stop coding agents from hallucinating about your repo." in first_screen
    assert "First verify. Then learn." in first_screen
    assert "`WAKE.md` tells agents where we are." in first_screen


def test_readme_demo_points_to_current_truth_layer_loop():
    expected = "morpheus learn lab"

    for path in [Path("README.md"), Path("README.ru.md")]:
        content = path.read_text()
        assert expected in content
        assert "morpheus check" in content
        assert "First verify. Then learn." in content
        assert "publish v0.1.0, then start semantic compile mode" not in content
        assert "publish v0.1.0, add the visual demo" not in content
        assert "update README, SPEC, and public repo metadata" not in content


def test_readme_cli_reference_lists_v02a1_commands():
    required = [
        "morpheus check",
        "morpheus check --input FILE",
        "morpheus check --json",
        "morpheus review list",
        "morpheus review apply",
        "morpheus learn lab .",
        "morpheus learn dataset .",
        "morpheus learn status",
        "morpheus learn train . --dry-run",
        "morpheus learn eval .",
    ]
    for path in [Path("README.md"), Path("README.ru.md")]:
        content = path.read_text()
        assert all(command in content for command in required)


def test_package_description_matches_truth_layer_positioning():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["description"] == (
        "Source-grounded truth layer for coding agents: verify project claims, "
        "compile WAKE.md, and run local learning experiments"
    )


def test_spec_headings_are_v02a1_current():
    spec = Path("SPEC.md").read_text()

    assert "## 3. Non-Goals\n" in spec
    assert "### Semantic Compiler Alpha\n" in spec
    assert "Non-Goals For v0.1" not in spec
    assert "Planned Semantic Compiler" not in spec


def test_public_docs_name_mcp_truth_tools():
    required_tools = [
        "morpheus_check_text",
        "morpheus_get_active_state",
        "morpheus_get_evidence_for_claim",
        "morpheus_get_wake",
    ]
    for path in [Path("README.md"), Path("README.ru.md"), Path("SPEC.md")]:
        content = path.read_text()
        assert all(tool in content for tool in required_tools)


def test_public_docs_explain_semantic_provider_modes():
    required = [
        "MORPHEUS_SEMANTIC_PROVIDER",
        "local",
        "null",
        "ollama",
        "offline",
        "cloud",
    ]
    for path in [Path("README.md"), Path("README.ru.md"), Path("SPEC.md")]:
        content = path.read_text()
        assert all(term in content for term in required)


def test_agents_bootstrap_uses_localhost_by_default():
    agents = Path("AGENTS.md").read_text()

    assert "/Users/testbot/projects/morpheus-ai" not in agents
    assert "project_root=<PROJECT_ROOT>" in agents
    assert "--host 127.0.0.1" in agents
    assert "0.0.0.0" in agents
    assert "explicit user-approved trusted LAN" in agents
    default_line = next(line for line in agents.splitlines() if "If the API/UI are unavailable" in line)
    assert "--host 0.0.0.0" not in default_line


def test_public_docs_do_not_regress_to_memory_layer_pitch():
    stale_pitch = "Local-first memory compiler for AI agents"

    for path in [Path("README.md"), Path("README.ru.md"), Path("SPEC.md"), Path("WAKE.md")]:
        assert stale_pitch not in path.read_text()


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
