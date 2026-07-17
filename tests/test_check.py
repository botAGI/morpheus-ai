import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.cli import read_check_input


ROOT = Path(__file__).resolve().parents[1]
STALE_INPUT = ROOT / "tests" / "fixtures" / "check_stale_input.txt"
CORRECT_INPUT = ROOT / "tests" / "fixtures" / "check_correct_input.txt"


def write_launch_repo() -> None:
    Path("README.md").write_text(
        "\n".join(
            [
                "# Demo",
                "DECISION: Morpheus is the Agent State Compiler.",
                "DECISION: The distribution package is morpheus-wake.",
                "DECISION: Never train on raw markdown or secrets.",
                "",
            ]
        )
    )
    Path("pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "morpheus-wake"',
                'version = "0.1.1"',
                "",
            ]
        )
    )
    Path("WAKE.md").write_text(
        "\n".join(
            [
                "# WAKE.md - Project State",
                "",
                "## Current State",
                "",
                "Morpheus is the Agent State Compiler.",
                "",
                "## Outdated Claims",
                "",
                '- "Morpheus is mainly a personal AI agent." Outdated.',
                "",
            ]
        )
    )


def prepare_private_state(runner: CliRunner) -> None:
    result = runner.invoke(app, ["wake", ".", "--private"])
    assert result.exit_code == 0, result.output


def test_check_verified_file_input_exits_zero_and_reports_json(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        result = runner.invoke(
            app,
            [
                "check",
                "--input",
                str(CORRECT_INPUT),
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert set(payload) == {
            "input_hash",
            "checked_at",
            "active_state_receipt",
            "state_freshness",
            "state_warning",
            "modes_used",
            "claims_extracted",
            "claims_supported",
            "claims_contradicted",
            "claims_stale",
            "claims_not_found",
            "by_class",
            "by_route",
            "routing_policy_version",
            "fail_on_unknown",
            "results",
        }
        assert payload["modes_used"] == ["local"]
        assert payload["state_freshness"] == "fresh"
        assert payload["claims_supported"] == 1
        assert payload["claims_stale"] == 0
        assert payload["by_class"]["product"] == 1
        assert payload["results"][0]["status"] == "verified"
        assert payload["results"][0]["semantic_class"] == "product"
        assert payload["results"][0]["memory_route"] == "retrieval"
        assert payload["results"][0]["routing_reason"] == "verified_project_knowledge"
        assert payload["by_route"] == {"retrieval": 1}
        assert payload["routing_policy_version"] == "morpheus-memory-routing/1"
        assert payload["results"][0]["evidence"]["path"] == "README.md"


def test_check_stale_file_input_exits_one_with_source_span(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        result = runner.invoke(app, ["check", "--input", str(STALE_INPUT), "--json"])

        assert result.exit_code == 1, result.output
        payload = json.loads(result.output)
        assert payload["claims_stale"] == 1
        assert payload["by_class"]["stale"] == 1
        assert payload["results"][0]["status"] == "stale"
        assert payload["results"][0]["semantic_class"] == "stale"
        assert payload["results"][0]["memory_route"] == "stale_archive"
        assert payload["results"][0]["evidence"]["path"] == "WAKE.md"
        assert payload["results"][0]["evidence"]["line_start"] == 9


def test_check_incorrect_package_metadata_claim_exits_one(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        result = runner.invoke(
            app,
            ["check", "--json"],
            input="The distribution package is morpheus-ai.\n",
        )

        assert result.exit_code == 1, result.output
        payload = json.loads(result.output)
        assert payload["claims_contradicted"] == 1
        assert payload["results"][0]["status"] == "incorrect"
        assert payload["results"][0]["memory_route"] == "human_review"
        assert payload["results"][0]["evidence"]["path"] == "pyproject.toml"


def test_check_incorrect_active_claim_contradiction_exits_one(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        result = runner.invoke(
            app,
            ["check", "--json"],
            input="Morpheus is a runtime memory system.\n",
        )

        assert result.exit_code == 1, result.output
        payload = json.loads(result.output)
        assert payload["claims_contradicted"] == 1
        assert payload["results"][0]["status"] == "incorrect"
        assert payload["results"][0]["evidence"]["path"] == "README.md"


def test_check_reads_stdin_without_ci_mode_and_does_not_call_semantic_provider(
    tmp_path,
    monkeypatch,
):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)
        monkeypatch.setenv("MORPHEUS_SEMANTIC_PROVIDER", "fake")
        monkeypatch.setattr(
            "morpheus.cli.semantic_provider_from_env",
            lambda: (_ for _ in ()).throw(AssertionError("provider should not be called")),
        )

        result = runner.invoke(
            app,
            ["check", "--json"],
            input="This repo uses an unverified launch claim.\n",
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["modes_used"] == ["local"]
        assert payload["claims_not_found"] == 1
        assert payload["results"][0]["status"] == "unknown"
        assert payload["results"][0]["memory_route"] == "human_review"


def test_check_routes_verified_safety_rules_to_prompt_context(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        result = runner.invoke(
            app,
            ["check", "--json"],
            input="Never train on raw markdown or secrets.\n",
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["results"][0]["status"] == "verified"
        assert payload["results"][0]["semantic_class"] == "security"
        assert payload["results"][0]["memory_route"] == "prompt_context"
        assert payload["by_route"] == {"prompt_context": 1}


def test_check_fail_on_unknown_exits_one(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        result = runner.invoke(
            app,
            ["check", "--fail-on-unknown"],
            input="This repo uses an unverified launch claim.\n",
        )

        assert result.exit_code == 1
        assert "unknown" in result.output


def test_check_stale_state_in_ci_exits_two_unless_allowed(tmp_path, monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)
        Path("README.md").write_text(
            "DECISION: Morpheus changed after the latest receipt.\n"
        )
        monkeypatch.setenv("MORPHEUS_CI", "1")

        result = runner.invoke(app, ["check"], input="Morpheus is the Agent State Compiler.\n")

        assert result.exit_code == 2
        assert "State is stale" in result.output

        allowed = runner.invoke(
            app,
            ["check", "--allow-stale-state", "--json"],
            input="Morpheus is the Agent State Compiler.\n",
        )

        assert allowed.exit_code in {0, 1}, allowed.output
        assert json.loads(allowed.output)["state_freshness"] == "stale"


def test_check_without_input_prints_help_and_exits_two(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_launch_repo()
        prepare_private_state(runner)

        result = runner.invoke(app, ["check"])

        assert result.exit_code == 2
        assert "Usage:" in result.output
        assert "morpheus check --input" in result.output


def test_read_check_input_does_not_read_interactive_tty(monkeypatch):
    class TtyStdin:
        def isatty(self) -> bool:
            return True

        def read(self) -> str:
            raise AssertionError("interactive stdin should not be read")

    monkeypatch.setattr(sys, "stdin", TtyStdin())

    assert read_check_input(None) == ""
