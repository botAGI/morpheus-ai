import json
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app


def test_compile_semantic_review_writes_review_artifacts(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text(
            "Morpheus generates WAKE.md for AI agents.\n"
            "DECISION: Keep semantic compilation review-gated.\n"
        )

        init_result = runner.invoke(app, ["init"])
        result = runner.invoke(app, ["compile", "--semantic", "--review"])

        assert init_result.exit_code == 0, init_result.output
        assert result.exit_code == 0, result.output
        assert "Semantic review" in result.output
        assert Path(".morpheus/review/semantic_candidates.jsonl").is_file()
        assert Path(".morpheus/review/WAKE.draft.md").is_file()
        report = json.loads(Path(".morpheus/review/semantic_report.json").read_text())
        assert report["candidates_total"] == 2
        assert report["source_backed_total"] == 2


def test_wake_semantic_review_runs_one_command_flow(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("Morpheus generates WAKE.md for AI agents.\n")

        result = runner.invoke(app, ["wake", ".", "--semantic", "--review", "--private"])

        assert result.exit_code == 0, result.output
        assert Path(".morpheus/WAKE.md").is_file()
        assert Path(".morpheus/review/semantic_candidates.jsonl").is_file()
        assert "Semantic review" in result.output


def test_review_list_show_accept_reject_and_diff(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text(
            "Morpheus generates WAKE.md for AI agents.\n"
            "TODO: Add semantic review.\n"
        )
        assert runner.invoke(app, ["init"]).exit_code == 0
        assert runner.invoke(app, ["compile", "--semantic", "--review"]).exit_code == 0

        list_result = runner.invoke(app, ["review", "list", "--json"])
        assert list_result.exit_code == 0, list_result.output
        candidates = json.loads(list_result.output)
        assert len(candidates) == 2

        show_result = runner.invoke(app, ["review", "show", candidates[0]["id"]])
        assert show_result.exit_code == 0, show_result.output
        assert "Morpheus generates WAKE.md" in show_result.output

        accept_result = runner.invoke(app, ["review", "accept", candidates[0]["id"]])
        reject_result = runner.invoke(
            app,
            ["review", "reject", candidates[1]["id"], "--reason", "too broad"],
        )
        diff_result = runner.invoke(app, ["review", "diff", "--json"])

        assert accept_result.exit_code == 0, accept_result.output
        assert reject_result.exit_code == 0, reject_result.output
        assert diff_result.exit_code == 0, diff_result.output
        assert json.loads(diff_result.output) == {
            "pending": 0,
            "accepted": 1,
            "rejected": 1,
        }


def test_review_apply_promotes_accepted_candidates_to_active_state(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text(
            "Morpheus generates WAKE.md for AI agents.\n"
            "TODO: Add semantic review.\n"
        )
        assert runner.invoke(app, ["init"]).exit_code == 0
        assert runner.invoke(app, ["compile", "--semantic", "--review"]).exit_code == 0
        candidates = json.loads(runner.invoke(app, ["review", "list", "--json"]).output)
        assert runner.invoke(app, ["review", "accept", candidates[0]["id"]]).exit_code == 0

        apply_result = runner.invoke(app, ["review", "apply"])

        assert apply_result.exit_code == 0, apply_result.output
        state = json.loads(Path(".morpheus/state.json").read_text())
        assert any(
            claim["excerpt"] == "Morpheus generates WAKE.md for AI agents"
            for claim in state["claims"]
        )
        assert "Receipt chain valid" in runner.invoke(app, ["verify", "--all"]).output
