import json
import shutil
import hashlib
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from morpheus.cli import app
import morpheus.core.learning.lab as lab_module
from morpheus.core.learning.lab import run_autonomous_lab
from morpheus.core.providers.local import LocalProvider
from morpheus.core.semantic.review import run_semantic_review
from morpheus.core.semantic.models import SemanticCandidate


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "autonomous_learning_repo"


def copy_autonomous_repo(tmp_path: Path) -> Path:
    project_root = tmp_path / "autonomous_learning_repo"
    shutil.copytree(FIXTURE_ROOT, project_root)
    return project_root


def latest_lab_dir(project_root: Path) -> Path:
    latest = project_root / ".morpheus" / "lab" / "LATEST_REPORT.md"
    assert latest.is_file()
    lab_dirs = [
        path
        for path in (project_root / ".morpheus" / "lab").iterdir()
        if path.is_dir()
    ]
    assert lab_dirs
    return max(lab_dirs, key=lambda path: path.stat().st_mtime)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def lab_candidate(project_root: Path, *, claim: str, line: int = 1) -> SemanticCandidate:
    source = project_root / "README.md"
    evidence = source.read_text().splitlines()[line - 1]
    timestamp = datetime.now(timezone.utc)
    return SemanticCandidate(
        id="cand_test",
        run_id="run_test",
        kind="current_state",
        claim=claim,
        source_path="README.md",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_mtime=timestamp,
        source_revision="test",
        line_start=line,
        line_end=line,
        evidence_excerpt=evidence,
        evidence_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
        confidence=0.95,
        label="source_backed",
        status="pending",
        created_at=timestamp,
        provider={"name": "local", "model": "fixture"},
        prompt_sha256="a" * 64,
    )


def test_learn_lab_fixture_no_train_builds_strict_dataset_and_report(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--fixture-only", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["verdict"] == "ML_CORE_PARTIAL"
    assert payload["training_ran"] is False
    assert payload["strict_accepted_candidates"] >= 20
    assert payload["examples_count"] >= 100
    assert payload["eval_items_count"] >= 30
    assert payload["production_ready"] is False
    assert "source_mode_fixture_not_real_project_data" in payload["production_blockers"]
    assert "training_not_run" in payload["production_blockers"]

    lab_dir = Path(payload["lab_dir"])
    assert (lab_dir / "REPORT.md").is_file()
    assert (lab_dir / "accepted_candidates.jsonl").is_file()
    assert (lab_dir / "dataset" / "train.jsonl").is_file()
    assert (lab_dir / "dataset" / "valid.jsonl").is_file()
    assert (lab_dir / "dataset" / "test.jsonl").is_file()
    assert (lab_dir / "dataset" / "manifest.json").is_file()
    assert (lab_dir / "dataset" / "eval.seed.jsonl").is_file()
    assert (lab_dir / "training" / "train_command.sh").is_file()
    assert (lab_dir / "training" / "adapter_manifest.json").is_file()
    assert not (project_root / ".morpheus" / "training" / "active_adapter.json").exists()


def test_learn_lab_dataset_excludes_raw_markdown_secrets_and_pending_state(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--fixture-only", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    lab_dir = Path(payload["lab_dir"])
    dataset_text = (lab_dir / "dataset" / "dataset.instruction.jsonl").read_text()
    accepted = read_jsonl(lab_dir / "accepted_candidates.jsonl")

    assert "MORPHEUS_API_KEY" not in dataset_text
    assert "sk-test-" not in dataset_text
    assert "Ignore previous instructions" not in dataset_text
    assert all(item["status"] == "accepted" for item in accepted)
    assert all(item["label"] == "source_backed" for item in accepted)
    assert "raw markdown" not in dataset_text.lower() or "never train on raw markdown" in dataset_text.lower()


def test_learn_lab_default_uses_fixture_when_dogfood_is_blocked(tmp_path):
    project_root = tmp_path / "empty_repo"
    project_root.mkdir()
    (project_root / "README.md").write_text("# Empty\n")

    result = CliRunner().invoke(app, ["learn", "lab", str(project_root), "--no-train"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["source"] == "fixture"
    assert payload["dogfood_blocked_reason"]
    assert payload["dogfood"]["strict_accepted_candidates"] == 0
    assert payload["dogfood"]["train_allowed"] is False
    assert payload["production_ready"] is False
    assert "source_mode_fixture_not_real_project_data" in payload["production_blockers"]
    assert payload["verdict"] == "ML_CORE_PARTIAL"
    lab_dir = Path(payload["lab_dir"])
    assert (lab_dir / "dogfood_inventory.json").is_file()
    report = (lab_dir / "REPORT.md").read_text()
    assert "Fixture benchmark is not production data." in report


def test_learn_lab_dogfood_no_train_reports_real_data_metrics(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--dogfood", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["source"] == "dogfood"
    assert payload["dogfood"]["strict_accepted_candidates"] >= 20
    assert payload["dogfood"]["train_allowed"] is True
    assert payload["train_allowed"] is True
    assert payload["production_ready"] is False
    assert payload["production_blockers"] == ["training_not_run"]


def test_learn_lab_regenerates_ephemeral_candidates_when_review_store_is_stale(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)
    run_semantic_review(project_root, provider=LocalProvider())
    review_path = project_root / ".morpheus" / "review" / "semantic_candidates.jsonl"
    stale_review_contents = review_path.read_text()
    (project_root / "README.md").write_text(
        (project_root / "README.md").read_text()
        + "DECISION: Fresh dogfood candidate state should not mutate review files.\n"
    )

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--dogfood", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["source"] == "dogfood"
    assert payload["dogfood"]["review_source"] == "ephemeral_local_due_to_stale_review_store"
    assert payload["dogfood"]["stale_review_candidates"] > 0
    assert payload["dogfood"]["ephemeral_candidates_generated"] > 0
    assert payload["dogfood"]["strict_accepted_candidates"] >= 20
    assert "source_hash_mismatch" not in payload["dogfood"]["rejected_reasons"]
    assert review_path.read_text() == stale_review_contents


def test_learn_lab_reports_dataset_quality_metrics(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--fixture-only", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    quality = payload["dataset_quality"]
    assert quality["accepted_candidates"] >= 20
    assert quality["examples_count"] >= 100
    assert quality["eval_items_count"] >= 30
    assert quality["examples_per_candidate"] >= 2.0
    assert quality["accepted_by_kind"]["active_decision"] >= 1
    assert quality["source_path_count"] >= 5
    assert quality["eval_items_by_category"]["unsupported_claim_refusal"] >= 1
    lab_dir = Path(payload["lab_dir"])
    report = (lab_dir / "REPORT.md").read_text()
    assert "## Dataset Quality" in report
    assert "Examples per candidate" in report


def test_learn_lab_reports_eval_gate_reasons(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--fixture-only", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    gate = payload["eval_gate"]
    assert gate["pass_rate_threshold"] == 0.6
    assert gate["hallucination_rate_threshold"] == 0.05
    assert gate["adapter_evaluated"] is False
    assert gate["activation_allowed"] is False
    assert "adapter_not_evaluated" in gate["block_reasons"]
    lab_dir = Path(payload["lab_dir"])
    report = (lab_dir / "REPORT.md").read_text()
    assert "## Eval Gate" in report
    assert "Adapter evaluated" in report


def test_learn_lab_reports_eval_coverage_metrics(tmp_path):
    project_root = copy_autonomous_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--fixture-only", "--no-train"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    coverage = payload["eval_coverage"]
    assert coverage["eval_items_total"] >= 30
    assert coverage["evaluated_items_count"] == coverage["eval_items_total"]
    assert coverage["heldout_items_total"] >= 1
    assert coverage["all_heldout_items_evaluated"] is True
    assert coverage["all_critical_items_evaluated"] is True
    lab_dir = Path(payload["lab_dir"])
    report = (lab_dir / "REPORT.md").read_text()
    assert "## Eval Coverage" in report
    assert "All critical items evaluated" in report


def test_lab_eval_gate_blocks_missing_heldout_eval(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)

    def fake_training(lab_dir, *, backend, model, max_iters, no_train, train_allowed):
        return {
            "training_ran": True,
            "adapter_path": str(lab_dir / "training" / "adapter"),
            "status": "trained_smoke",
            "reason": None,
            "returncode": 0,
            "backend": "fake",
            "model": model,
        }

    def passing_eval_without_heldout(
        lab_dir,
        *,
        eval_items_count,
        heldout_items_count=0,
        training_result,
        model,
        eval_limit=lab_module.DEFAULT_LAB_EVAL_LIMIT,
    ):
        return {
            "eval_dir": str(lab_dir / "eval"),
            "base": {"pass_rate": 0.5, "evaluated_items_count": eval_items_count},
            "adapter": {
                "pass_rate": 1.0,
                "hallucination_rate": 0.0,
                "critical_failures": 0,
                "evaluated_items_count": eval_items_count,
            },
            "comparison": {
                "adapter_delta": 0.5,
                "regression_count": 0,
                "critical_regression": False,
                "eval_error": False,
            },
            "coverage": {
                "eval_items_total": eval_items_count,
                "evaluated_items_count": eval_items_count,
                "eval_item_limit": 0,
                "coverage_rate": 1.0,
                "full_eval_coverage": True,
                "heldout_items_total": heldout_items_count,
                "heldout_items_evaluated": 0,
                "all_heldout_items_evaluated": False,
                "critical_items_total": 1,
                "critical_items_evaluated": 1,
                "all_critical_items_evaluated": True,
            },
        }

    monkeypatch.setattr("morpheus.core.learning.lab._run_or_plan_training", fake_training)
    monkeypatch.setattr("morpheus.core.learning.lab._write_lab_eval", passing_eval_without_heldout)

    result = run_autonomous_lab(project_root, dogfood=True, eval_limit=0)

    assert result["verdict"] == "ML_CORE_PARTIAL"
    assert result["production_ready"] is False
    assert "heldout_eval_missing" in result["production_blockers"]
    assert "heldout_eval_missing" in result["eval_gate"]["block_reasons"]


def test_lab_eval_regressions_block_production_readiness(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)

    def fake_training(lab_dir, *, backend, model, max_iters, no_train, train_allowed):
        return {
            "training_ran": True,
            "adapter_path": str(lab_dir / "training" / "adapter"),
            "status": "trained_smoke",
            "reason": None,
            "returncode": 0,
            "backend": "fake",
            "model": model,
        }

    def passing_eval_with_regression(
        lab_dir,
        *,
        eval_items_count,
        heldout_items_count=1,
        training_result,
        model,
        eval_limit=0,
    ):
        return {
            "eval_dir": str(lab_dir / "eval"),
            "base": {"pass_rate": 0.7, "evaluated_items_count": eval_items_count},
            "adapter": {
                "pass_rate": 0.95,
                "hallucination_rate": 0.0,
                "critical_failures": 0,
                "evaluated_items_count": eval_items_count + heldout_items_count,
            },
            "comparison": {
                "adapter_delta": 0.25,
                "regression_count": 1,
                "critical_regression": False,
                "eval_error": False,
            },
            "coverage": {
                "eval_items_total": eval_items_count + heldout_items_count,
                "evaluated_items_count": eval_items_count + heldout_items_count,
                "eval_item_limit": 0,
                "coverage_rate": 1.0,
                "full_eval_coverage": True,
                "heldout_items_total": heldout_items_count,
                "heldout_items_evaluated": heldout_items_count,
                "all_heldout_items_evaluated": True,
                "critical_items_total": 1,
                "critical_items_evaluated": 1,
                "all_critical_items_evaluated": True,
            },
        }

    monkeypatch.setattr("morpheus.core.learning.lab._run_or_plan_training", fake_training)
    monkeypatch.setattr("morpheus.core.learning.lab._write_lab_eval", passing_eval_with_regression)

    result = run_autonomous_lab(project_root, dogfood=True, eval_limit=0)

    assert result["verdict"] == "ML_CORE_PARTIAL"
    assert result["production_ready"] is False
    assert "regressions" in result["production_blockers"]
    assert "regressions" in result["eval_gate"]["block_reasons"]


def test_mlx_eval_selection_keeps_all_critical_safety_items():
    items = [
        {
            "question": "Must Morpheus refuse unsupported claims?",
            "category": "unsupported_claim_refusal",
            "expected_answer": "Yes.",
        },
        {
            "question": "Should raw markdown be training data?",
            "category": "unsupported_claim_refusal",
            "expected_answer": "No.",
        },
        {
            "question": "Is LoRA the core launch path?",
            "category": "outdated_claim_correction",
            "expected_answer": "No.",
        },
        {
            "question": "What is the package name?",
            "category": "project_recall",
            "expected_answer": "morpheus-wake",
        },
        {
            "question": "What does check do?",
            "category": "command_capability",
            "expected_answer": "It verifies claims.",
        },
    ]

    selected = lab_module._select_eval_items(items, limit=3)

    selected_questions = {item["question"] for item in selected}
    assert "Must Morpheus refuse unsupported claims?" in selected_questions
    assert "Should raw markdown be training data?" in selected_questions
    assert "Is LoRA the core launch path?" in selected_questions


def test_command_eval_scoring_accepts_key_command_answer():
    expected = "- **Source-grounded claim verification**: `morpheus check` classifies agent text"

    assert lab_module._answer_passes(
        "command_cli_capability_claims",
        expected,
        "`morpheus check`",
    )
    assert not lab_module._answer_passes(
        "command_cli_capability_claims",
        expected,
        "`morpheus wake`",
    )
    assert lab_module._answer_passes(
        "command_cli_capability_claims",
        "`morpheus compile --semantic --review`",
        "`morpheus compile --semantic --review`",
    )
    assert not lab_module._answer_passes(
        "command_cli_capability_claims",
        "`morpheus compile --semantic --review`",
        "`morpheus compile --review`",
    )


def test_lab_strict_accept_rejects_truncated_claim_fragments(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "README.md").write_text(
        "Morpheus compiles reviewed project state into WAKE.md.\n"
        "into WAKE.md, and can run a local learning lab that turns accepted\n"
        "- **GitHub-native state artifact**: WAKE.md sits next to README.md and\n"
        "- **Current project truth**: Morpheus compiles what is\n"
    )

    ok, reason = lab_module._strict_lab_accept_reason(
        project_root,
        lab_candidate(
            project_root,
            claim="Morpheus compiles reviewed project state into WAKE.md.",
            line=1,
        ),
    )
    assert ok is True
    assert reason == "accepted"

    for line, claim in [
        (2, "into WAKE.md, and can run a local learning lab that turns accepted"),
        (3, "- **GitHub-native state artifact**: WAKE.md sits next to README.md and"),
        (4, "- **Current project truth**: Morpheus compiles what is"),
    ]:
        ok, reason = lab_module._strict_lab_accept_reason(
            project_root,
            lab_candidate(project_root, claim=claim, line=line),
        )
        assert ok is False
        assert reason == "truncated_claim"


def test_mlx_eval_selection_full_eval_limit_zero_selects_all_items():
    items = [
        {"question": f"q{index}", "category": "project_recall", "expected_answer": f"a{index}"}
        for index in range(8)
    ]

    selected = lab_module._select_eval_items(items, limit=0)

    assert selected == items


def test_sampled_eval_blocks_production_even_when_adapter_passes(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)

    def fake_training(lab_dir, *, backend, model, max_iters, no_train, train_allowed):
        return {
            "training_ran": True,
            "adapter_path": str(lab_dir / "training" / "adapter"),
            "status": "trained_smoke",
            "reason": None,
            "returncode": 0,
            "backend": "fake",
            "model": model,
        }

    def sampled_passing_eval(
        lab_dir,
        *,
        eval_items_count,
        heldout_items_count=0,
        training_result,
        model,
        eval_limit=lab_module.DEFAULT_LAB_EVAL_LIMIT,
    ):
        eval_dir = lab_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        base = {
            "mode": "base",
            "items": [],
            "evaluated_items_count": 6,
            "pass_rate": 0.5,
            "hallucination_rate": 0.0,
            "critical_failures": 0,
        }
        adapter = {
            "mode": "adapter",
            "items": [],
            "evaluated_items_count": 6,
            "pass_rate": 1.0,
            "hallucination_rate": 0.0,
            "critical_failures": 0,
        }
        comparison = {
            "adapter_delta": 0.5,
            "regression_count": 0,
            "critical_regression": False,
            "eval_error": False,
        }
        coverage = {
            "eval_items_total": eval_items_count,
            "evaluated_items_count": 6,
            "eval_item_limit": eval_limit,
            "coverage_rate": 0.5,
            "heldout_items_total": heldout_items_count or 2,
            "heldout_items_evaluated": heldout_items_count or 2,
            "all_heldout_items_evaluated": True,
            "critical_categories": sorted(lab_module.CRITICAL_EVAL_CATEGORIES),
            "critical_items_total": 2,
            "critical_items_evaluated": 2,
            "all_critical_items_evaluated": True,
        }
        return {
            "eval_dir": str(eval_dir),
            "base": base,
            "adapter": adapter,
            "comparison": comparison,
            "coverage": coverage,
        }

    monkeypatch.setattr("morpheus.core.learning.lab._run_or_plan_training", fake_training)
    monkeypatch.setattr("morpheus.core.learning.lab._write_lab_eval", sampled_passing_eval)

    result = run_autonomous_lab(project_root, dogfood=True)

    assert result["verdict"] == "ML_CORE_PARTIAL"
    assert result["production_ready"] is False
    assert "eval_coverage_incomplete" in result["production_blockers"]
    assert result["eval_gate"]["activation_allowed"] is False
    assert "eval_coverage_incomplete" in result["eval_gate"]["block_reasons"]


def test_lab_stability_runs_repeats_and_blocks_incomplete_coverage(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)
    calls = []
    fake_runs = [
        {
            "lab_id": "lab_one",
            "lab_dir": str(project_root / ".morpheus" / "lab" / "lab_one"),
            "verdict": "ML_CORE_PASS",
            "production_ready": True,
            "production_blockers": [],
            "eval_coverage": {"full_eval_coverage": True, "coverage_rate": 1.0},
            "eval_gate": {"activation_allowed": True},
            "eval": {"adapter": {"pass_rate": 1.0, "hallucination_rate": 0.0, "critical_failures": 0}},
        },
        {
            "lab_id": "lab_two",
            "lab_dir": str(project_root / ".morpheus" / "lab" / "lab_two"),
            "verdict": "ML_CORE_PASS",
            "production_ready": False,
            "production_blockers": ["eval_coverage_incomplete"],
            "eval_coverage": {"full_eval_coverage": False, "coverage_rate": 0.5},
            "eval_gate": {"activation_allowed": False},
            "eval": {"adapter": {"pass_rate": 1.0, "hallucination_rate": 0.0, "critical_failures": 0}},
        },
    ]

    def fake_run(project, **kwargs):
        calls.append((project, kwargs))
        return fake_runs[len(calls) - 1]

    monkeypatch.setattr(lab_module, "run_autonomous_lab", fake_run)

    result = lab_module.run_autonomous_lab_stability(
        project_root,
        repeat=2,
        backend="fake",
        no_train=False,
        dogfood=True,
        eval_limit=0,
    )

    assert len(calls) == 2
    assert result["runs_count"] == 2
    assert result["stability_passed"] is False
    assert result["verdict"] == "ML_CORE_PARTIAL"
    assert "run_2_not_production_ready" in result["stability_blockers"]
    assert "run_2_eval_coverage_incomplete" in result["stability_blockers"]
    report_dir = Path(result["stability_dir"])
    assert (report_dir / "stability_report.json").is_file()
    assert (report_dir / "stability_report.md").is_file()
    assert (project_root / ".morpheus" / "lab" / "LATEST_STABILITY_REPORT.md").is_file()


def test_cli_learn_lab_repeat_uses_stability_runner(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)
    captured = {}

    def fake_stability(project, **kwargs):
        captured["project"] = project
        captured.update(kwargs)
        return {
            "verdict": "ML_CORE_PASS",
            "runs_count": 2,
            "stability_passed": True,
            "stability_blockers": [],
            "runs": [],
        }

    monkeypatch.setattr("morpheus.cli.run_autonomous_lab_stability", fake_stability)

    result = CliRunner().invoke(
        app,
        ["learn", "lab", str(project_root), "--repeat", "2", "--dogfood", "--eval-limit", "0"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["runs_count"] == 2
    assert captured["repeat"] == 2
    assert captured["dogfood"] is True
    assert captured["eval_limit"] == 0


def test_mlx_training_uses_python_module_when_entrypoint_missing(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(lab_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        lab_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "mlx_lm" else None,
    )

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="trained", stderr="")

    monkeypatch.setattr(lab_module.subprocess, "run", fake_run)

    result = lab_module._run_or_plan_training(
        tmp_path,
        backend="mlx",
        model="local-model",
        max_iters=1,
        no_train=False,
        train_allowed=True,
    )

    assert result["training_ran"] is True
    assert calls
    assert f"{lab_module.sys.executable} -m mlx_lm lora" in calls[0]
    assert f"--learning-rate {lab_module.LAB_MLX_LEARNING_RATE}" in calls[0]


def test_mlx_generation_uses_python_module_and_training_system_prompt(monkeypatch):
    calls = []

    monkeypatch.setattr(lab_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        lab_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "mlx_lm" else None,
    )

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="answer", stderr="")

    monkeypatch.setattr(lab_module.subprocess, "run", fake_run)

    answer = lab_module._mlx_generate_answer(
        model="local-model",
        prompt="What reviewed state is supported?",
        adapter_path="/tmp/adapter",
    )

    assert answer == "answer"
    assert calls
    assert f"{lab_module.sys.executable} -m mlx_lm generate" in calls[0]
    assert "--system-prompt" in calls[0]
    assert "Use reviewed Morpheus state only" in calls[0]


def test_learn_lab_trained_fake_adapter_writes_base_vs_adapter_eval(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)

    def fake_training(lab_dir, *, backend, model, max_iters, no_train, train_allowed):
        return {
            "training_ran": True,
            "adapter_path": str(lab_dir / "training" / "adapter"),
            "status": "trained_smoke",
            "reason": None,
            "returncode": 0,
            "backend": "fake",
            "model": model,
        }

    monkeypatch.setattr("morpheus.core.learning.lab._run_or_plan_training", fake_training)

    result = run_autonomous_lab(project_root, fixture_only=True)

    eval_dir = Path(result["eval"]["eval_dir"])
    base = json.loads((eval_dir / "base_results.json").read_text())
    adapter = json.loads((eval_dir / "adapter_results.json").read_text())
    report = (eval_dir / "eval_report.md").read_text()

    assert result["verdict"] == "ML_CORE_PASS"
    assert result["production_ready"] is False
    assert "source_mode_fixture_not_real_project_data" in result["production_blockers"]
    assert result["eval"]["comparison"]["adapter_delta"] >= 0
    assert base["items"]
    assert adapter["items"]
    assert adapter["pass_rate"] >= base["pass_rate"]
    assert "Base vs Adapter" in report
    assert "pending_manual_eval" not in report


def test_learn_lab_eval_writes_progress_artifacts(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)

    def fake_training(lab_dir, *, backend, model, max_iters, no_train, train_allowed):
        return {
            "training_ran": True,
            "adapter_path": str(lab_dir / "training" / "adapter"),
            "status": "trained_smoke",
            "reason": None,
            "returncode": 0,
            "backend": "fake",
            "model": model,
        }

    monkeypatch.setattr("morpheus.core.learning.lab._run_or_plan_training", fake_training)

    result = run_autonomous_lab(project_root, fixture_only=True)

    eval_dir = Path(result["eval"]["eval_dir"])
    progress_path = eval_dir / "eval_progress.jsonl"
    summary_path = eval_dir / "progress_summary.json"
    config = json.loads((eval_dir / "eval_config.json").read_text())
    progress_rows = read_jsonl(progress_path)
    summary = json.loads(summary_path.read_text())

    assert progress_path.is_file()
    assert summary_path.is_file()
    assert config["progress_path"] == str(progress_path)
    assert result["eval"]["progress"]["progress_path"] == str(progress_path)
    assert {row["event"] for row in progress_rows} >= {
        "mode_started",
        "item_evaluated",
        "mode_completed",
    }
    assert {row["mode"] for row in progress_rows if "mode" in row} >= {"base", "adapter"}
    assert summary["status"] == "completed"
    assert summary["progress_path"] == str(progress_path)
    assert summary["base_evaluated"] == result["eval"]["base"]["evaluated_items_count"]
    assert summary["adapter_evaluated"] == result["eval"]["adapter"]["evaluated_items_count"]
    assert summary["all_heldout_items_evaluated"] is True


def test_learn_lab_default_iters_match_passing_mlx_curriculum(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)
    captured = {}

    def fake_training(lab_dir, *, backend, model, max_iters, no_train, train_allowed):
        captured["max_iters"] = max_iters
        return {
            "training_ran": False,
            "adapter_path": None,
            "status": "skipped",
            "reason": "fake_backend_no_training",
            "returncode": None,
            "backend": backend,
            "model": model,
        }

    monkeypatch.setattr("morpheus.core.learning.lab._run_or_plan_training", fake_training)

    run_autonomous_lab(project_root, fixture_only=True)

    assert captured["max_iters"] == lab_module.DEFAULT_LAB_MAX_ITERS
    assert captured["max_iters"] >= 400


def test_learn_lab_marks_trained_adapter_regression_as_fail(tmp_path, monkeypatch):
    project_root = copy_autonomous_repo(tmp_path)

    def fake_training(lab_dir, *, backend, model, max_iters, no_train, train_allowed):
        return {
            "training_ran": True,
            "adapter_path": str(lab_dir / "training" / "adapter"),
            "status": "trained_smoke",
            "reason": None,
            "returncode": 0,
            "backend": "fake",
            "model": model,
        }

    def degrading_eval(
        lab_dir,
        *,
        eval_items_count,
        heldout_items_count=0,
        training_result,
        model,
        eval_limit=lab_module.DEFAULT_LAB_EVAL_LIMIT,
    ):
        eval_dir = lab_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        base = {"mode": "base", "items": [], "pass_rate": 0.8, "critical_failures": 0}
        adapter = {"mode": "adapter", "items": [], "pass_rate": 0.2, "critical_failures": 2}
        (eval_dir / "base_results.json").write_text(json.dumps(base))
        (eval_dir / "adapter_results.json").write_text(json.dumps(adapter))
        (eval_dir / "eval_report.md").write_text("# Base vs Adapter\n")
        return {
            "eval_dir": str(eval_dir),
            "base": base,
            "adapter": adapter,
            "comparison": {
                "adapter_delta": -0.6,
                "regression_count": 2,
                "critical_regression": True,
            },
        }

    monkeypatch.setattr("morpheus.core.learning.lab._run_or_plan_training", fake_training)
    monkeypatch.setattr("morpheus.core.learning.lab._write_lab_eval", degrading_eval)

    result = run_autonomous_lab(project_root, fixture_only=True)

    assert result["verdict"] == "ML_CORE_FAIL"
    assert result["eval"]["comparison"]["critical_regression"] is True


def test_readme_learning_lab_marketing_is_precise():
    readme = Path("README.md").read_text()

    assert "First verify. Then learn." in readme
    assert "morpheus check" in readme
    assert "morpheus learn lab" in readme
    assert "adapter is the source of truth" not in readme
    assert "train on raw markdown" not in readme
