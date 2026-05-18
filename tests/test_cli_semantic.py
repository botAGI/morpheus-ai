import json
import hashlib
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app
from morpheus.core.semantic.review import ReviewStore


PROMPT_SHA = "b" * 64


def write_candidate_fixture(project_root: Path) -> list[dict]:
    readme = project_root / "README.md"
    agents = project_root / "AGENTS.md"
    readme.write_text(
        "Morpheus generates WAKE.md from reviewed project state.\n"
        "DECISION: Review-gated semantic compilation is active on main.\n"
        "TODO: Expand richer stale-claim detection.\n",
    )
    agents.write_text("- Read WAKE.md before edits.\n")
    review_dir = project_root / ".morpheus" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        candidate_row(
            project_root,
            candidate_id="cand_current",
            kind="current_state",
            claim="Morpheus generates WAKE.md from reviewed project state.",
            source_path="README.md",
            line_start=1,
            confidence=0.95,
        ),
        candidate_row(
            project_root,
            candidate_id="cand_decision",
            kind="active_decision",
            claim="Review-gated semantic compilation is active on main.",
            source_path="README.md",
            line_start=2,
            confidence=0.96,
        ),
        candidate_row(
            project_root,
            candidate_id="cand_task",
            kind="open_task",
            claim="Expand richer stale-claim detection.",
            source_path="README.md",
            line_start=3,
            confidence=0.80,
        ),
        candidate_row(
            project_root,
            candidate_id="cand_rule",
            kind="agent_rule",
            claim="Read WAKE.md before edits.",
            source_path="AGENTS.md",
            line_start=1,
            confidence=0.94,
        ),
    ]
    (review_dir / "semantic_candidates.jsonl").write_text(
        "\n".join(json.dumps(candidate, sort_keys=True) for candidate in candidates)
        + "\n"
    )
    return candidates


def candidate_row(
    project_root: Path,
    *,
    candidate_id: str,
    kind: str,
    claim: str,
    source_path: str,
    line_start: int,
    confidence: float,
    status: str = "pending",
    label: str = "source_backed",
) -> dict:
    source = project_root / source_path
    line = source.read_text().splitlines()[line_start - 1].strip()
    return {
        "id": candidate_id,
        "run_id": "semrun_cli_fixture",
        "kind": kind,
        "claim": claim,
        "source_path": source_path,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_mtime": "2026-05-18T00:00:00+00:00",
        "source_revision": "git:test",
        "line_start": line_start,
        "line_end": line_start,
        "evidence_excerpt": line,
        "evidence_sha256": hashlib.sha256(line.encode()).hexdigest(),
        "confidence": confidence,
        "label": label,
        "status": status,
        "created_at": "2026-05-18T00:00:00+00:00",
        "provider": {"name": "local", "model": "fixture"},
        "prompt_sha256": PROMPT_SHA,
        "reviewed_by": None,
        "reviewed_at": None,
        "review_reason": None,
    }


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


def test_review_list_source_backed_and_trainable_filters(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        candidates = write_candidate_fixture(Path.cwd())

        source_backed = runner.invoke(app, ["review", "list", "--source-backed", "--json"])
        trainable_before = runner.invoke(app, ["review", "list", "--trainable", "--json"])
        assert runner.invoke(app, ["review", "accept", candidates[0]["id"]]).exit_code == 0
        trainable_after = runner.invoke(app, ["review", "list", "--trainable", "--json"])

        assert source_backed.exit_code == 0, source_backed.output
        assert trainable_before.exit_code == 0, trainable_before.output
        assert trainable_after.exit_code == 0, trainable_after.output
        assert len(json.loads(source_backed.output)) == 4
        assert json.loads(trainable_before.output) == []
        assert [item["id"] for item in json.loads(trainable_after.output)] == [candidates[0]["id"]]


def test_review_batch_accept_reject_and_export_pack(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        candidates = write_candidate_fixture(Path.cwd())
        accept_file = Path(".morpheus/review/accept_ids.txt")
        reject_file = Path(".morpheus/review/reject_ids.txt")
        accept_file.write_text(f"{candidates[0]['id']}\n{candidates[1]['id']}\n")
        reject_file.write_text(f"{candidates[2]['id']}\n")

        accept_result = runner.invoke(app, ["review", "accept-batch", "--file", str(accept_file)])
        reject_result = runner.invoke(
            app,
            ["review", "reject-batch", "--file", str(reject_file), "--reason", "too broad"],
        )
        pack_result = runner.invoke(app, ["review", "export-pack"])
        diff_result = runner.invoke(app, ["review", "diff", "--json"])

        assert accept_result.exit_code == 0, accept_result.output
        assert reject_result.exit_code == 0, reject_result.output
        assert pack_result.exit_code == 0, pack_result.output
        assert Path(".morpheus/review/review_pack.md").is_file()
        pack = Path(".morpheus/review/review_pack.md").read_text()
        assert f"## {candidates[0]['id']}" in pack
        assert "Accept command:" in pack
        assert json.loads(diff_result.output) == {
            "pending": 1,
            "accepted": 2,
            "rejected": 1,
        }


def test_review_suggest_accept_strict_writes_only_verified_low_risk_ids(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        candidates = write_candidate_fixture(Path.cwd())

        result = runner.invoke(app, ["review", "suggest-accept", "--strict"])

        assert result.exit_code == 0, result.output
        suggested = Path(".morpheus/review/suggested_accept_ids.txt").read_text().splitlines()
        assert candidates[0]["id"] in suggested
        assert candidates[1]["id"] in suggested
        assert candidates[3]["id"] in suggested
        assert candidates[2]["id"] not in suggested
        statuses = [
            item["status"]
            for item in json.loads(runner.invoke(app, ["review", "list", "--json"]).output)
        ]
        assert statuses == ["pending", "pending", "pending", "pending"]


def test_review_doctor_explains_zero_strict_suggestions(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_candidate_fixture(Path.cwd())
        store = ReviewStore(Path.cwd())
        candidates = store.load_candidates()
        candidates = [
            candidate.model_copy(update={"confidence": 0.82})
            for candidate in candidates
        ]
        store.save_candidates(candidates)

        result = runner.invoke(app, ["review", "doctor"])

        assert result.exit_code == 0, result.output
        payload = json.loads(Path(".morpheus/review/review_doctor.json").read_text())
        assert payload["summary"]["total"] == 4
        assert payload["summary"]["strict_suggestions"] == 0
        assert payload["aggregate"]["top_strict_failure_reasons"]["confidence_below_threshold"] == 4
        assert "confidence_below_threshold" in Path(".morpheus/review/review_doctor.md").read_text()


def test_review_propose_writes_reports_without_changing_statuses(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        candidates = write_candidate_fixture(Path.cwd())

        result = runner.invoke(app, ["review", "propose", "--max", "30", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        proposed_ids = Path(".morpheus/review/proposed_accept_ids.txt").read_text().splitlines()
        assert payload["counts"]["ACCEPT_SAFE"] >= 3
        assert candidates[0]["id"] in proposed_ids
        assert Path(".morpheus/review/proposal_report.md").is_file()
        assert Path(".morpheus/review/proposal_report.json").is_file()
        statuses = [
            item["status"]
            for item in json.loads(runner.invoke(app, ["review", "list", "--json"]).output)
        ]
        assert statuses == ["pending", "pending", "pending", "pending"]


def test_review_propose_marks_long_multiclaim_as_needs_split(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        candidates = write_candidate_fixture(Path.cwd())
        store = ReviewStore(Path.cwd())
        long_candidate = candidate_row(
            Path.cwd(),
            candidate_id="cand_multiclaim",
            kind="current_state",
            claim=(
                "Morpheus generates WAKE.md from reviewed project state and "
                "review-gated semantic compilation is active on main and "
                "agents must refuse unsupported claims without evidence spans."
            ),
            source_path="README.md",
            line_start=1,
            confidence=0.95,
        )
        store.save_candidates([
            *store.load_candidates(),
            store.load_candidates()[0].model_validate(long_candidate),
        ])

        result = runner.invoke(app, ["review", "propose", "--max", "30"])

        assert result.exit_code == 0, result.output
        report = json.loads(Path(".morpheus/review/proposal_report.json").read_text())
        by_id = {item["id"]: item for item in report["proposals"]}
        assert by_id["cand_multiclaim"]["category"] == "NEEDS_SPLIT"
        split = json.loads(Path(".morpheus/review/split_suggestions.json").read_text())
        assert split["suggestions"][0]["original_candidate_id"] == "cand_multiclaim"
        assert len(split["suggestions"][0]["suggested_atomic_claims"]) >= 2
        assert "cand_multiclaim" not in Path(".morpheus/review/proposed_accept_ids.txt").read_text()
        assert candidates[0]["id"] in Path(".morpheus/review/proposed_accept_ids.txt").read_text()


def test_review_propose_does_not_positive_accept_outdated_claim(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_candidate_fixture(Path.cwd())
        store = ReviewStore(Path.cwd())
        outdated = candidate_row(
            Path.cwd(),
            candidate_id="cand_outdated",
            kind="outdated_claim",
            claim="Morpheus is mainly a LoRA trainer.",
            source_path="README.md",
            line_start=1,
            confidence=0.95,
        )
        store.save_candidates([
            *store.load_candidates(),
            store.load_candidates()[0].model_validate(outdated),
        ])

        result = runner.invoke(app, ["review", "propose", "--max", "30"])

        assert result.exit_code == 0, result.output
        report = json.loads(Path(".morpheus/review/proposal_report.json").read_text())
        by_id = {item["id"]: item for item in report["proposals"]}
        assert by_id["cand_outdated"]["category"] != "ACCEPT_SAFE"
        assert "outdated_claim_correction_only" in by_id["cand_outdated"]["reasons"]
        assert "cand_outdated" not in Path(".morpheus/review/proposed_accept_ids.txt").read_text()


def test_review_interactive_refuses_non_tty(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_candidate_fixture(Path.cwd())

        result = runner.invoke(app, ["review", "interactive"])

        assert result.exit_code == 2
        assert "Non-interactive terminal" in result.output
        assert "morpheus review propose" in result.output


def test_cli_learn_dataset_empty_prints_training_blocked_message(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_candidate_fixture(Path.cwd())

        result = runner.invoke(app, ["learn", "dataset", ".", "--from", "accepted", "--format", "instruction"])

        assert result.exit_code == 0, result.output
        assert "Training blocked: accepted candidates < 20 or examples < 100." in result.output
        assert "morpheus review interactive --proposed" in result.output


def test_proposed_candidates_require_explicit_accept_before_dataset_examples(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(app, ["init"]).exit_code == 0
        write_candidate_fixture(Path.cwd())
        propose = runner.invoke(app, ["review", "propose", "--max", "30"])
        assert propose.exit_code == 0, propose.output
        before = runner.invoke(app, ["learn", "dataset", ".", "--from", "accepted", "--format", "instruction"])
        assert '"examples_count": 0' in before.output

        accept = runner.invoke(
            app,
            ["review", "accept-batch", "--file", ".morpheus/review/proposed_accept_ids.txt"],
        )
        apply = runner.invoke(app, ["review", "apply"])
        after = runner.invoke(app, ["learn", "dataset", ".", "--from", "accepted", "--format", "instruction"])

        assert accept.exit_code == 0, accept.output
        assert apply.exit_code == 0, apply.output
        assert after.exit_code == 0, after.output
        assert '"examples_count": 0' not in after.output
