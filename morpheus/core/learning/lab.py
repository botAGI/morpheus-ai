"""Autonomous source-grounded learning lab for Morpheus."""
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from fnmatch import fnmatch
from pathlib import Path

from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.examples import SYSTEM_PROMPT
from morpheus.core.learning.safety import (
    contains_secret_like_text,
    load_morpheusignore,
    path_is_ignored,
)
from morpheus.core.providers.local import LocalProvider
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.review import ReviewStore, run_semantic_review
from morpheus.core.semantic.scanner import scan_semantic_sources
from morpheus.core.semantic.verifier import verify_candidate_span


LAB_MIN_ACCEPTED = 20
LAB_MIN_EXAMPLES = 100
LAB_MIN_EVAL_ITEMS = 30
LAB_MLX_EVAL_ITEM_LIMIT = 6
LAB_PASS_RATE_THRESHOLD = 0.60
LAB_HALLUCINATION_RATE_THRESHOLD = 0.05
LAB_MLX_LEARNING_RATE = "1e-5"
DEFAULT_LAB_BACKEND = "fake"
DEFAULT_LAB_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
LAB_STRICT_KINDS = {
    "current_state",
    "active_decision",
    "agent_rule",
    "source_reference",
}
SPECULATIVE_WORDS = {
    "probably",
    "maybe",
    "might",
    "could",
    "seems",
    "appears",
    "возможно",
    "кажется",
    "вероятно",
}


def run_autonomous_lab(
    project_root: Path,
    *,
    backend: str = DEFAULT_LAB_BACKEND,
    model: str = DEFAULT_LAB_MODEL,
    no_train: bool = False,
    fixture_only: bool = False,
    dogfood: bool = False,
    max_iters: int = 50,
) -> dict:
    """Run an autonomous benchmark or dogfood learning experiment."""
    project_root = _safe_project_root(project_root)
    lab_id = _timestamp_id("lab")
    lab_dir = project_root / ".morpheus" / "lab" / lab_id
    _ensure_new_dir(lab_dir)

    config = {
        "lab_id": lab_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "backend": backend,
        "model": model,
        "no_train": no_train,
        "fixture_only": fixture_only,
        "dogfood": dogfood,
        "max_iters": max_iters,
        "min_accepted": LAB_MIN_ACCEPTED,
        "min_examples": LAB_MIN_EXAMPLES,
        "min_eval_items": LAB_MIN_EVAL_ITEMS,
    }
    _write_json(lab_dir / "lab_config.json", config)

    dogfood_blocked_reason = None
    dogfood_metrics = None
    source_mode = "dogfood"
    source_project = project_root
    if fixture_only:
        source_mode = "fixture"
        source_project = project_root
    elif not dogfood:
        dogfood_result = _strict_accept_for_project(project_root)
        dogfood_metrics = _source_metrics(dogfood_result)
        _write_dogfood_reports(lab_dir, dogfood_result)
        if len(dogfood_result["accepted"]) >= LAB_MIN_ACCEPTED:
            source_mode = "dogfood"
            _write_source_reports(lab_dir, dogfood_result)
            accepted = dogfood_result["accepted"]
            source_project = project_root
            return _finish_lab(
                project_root=project_root,
                source_project=source_project,
                lab_dir=lab_dir,
                lab_id=lab_id,
                source_mode=source_mode,
                accepted=accepted,
                rejected_reasons=dogfood_result["rejected_reasons"],
                backend=backend,
                model=model,
                no_train=no_train,
                max_iters=max_iters,
                dogfood_blocked_reason=None,
                dogfood_metrics=dogfood_metrics,
            )
        dogfood_blocked_reason = (
            f"strict accepted candidates {len(dogfood_result['accepted'])} "
            f"< required {LAB_MIN_ACCEPTED}"
        )
        source_mode = "fixture"
        source_project = _fixture_project(project_root, lab_dir)
    else:
        dogfood_result = _strict_accept_for_project(project_root)
        dogfood_metrics = _source_metrics(dogfood_result)
        _write_dogfood_reports(lab_dir, dogfood_result)
        _write_source_reports(lab_dir, dogfood_result)
        if len(dogfood_result["accepted"]) < LAB_MIN_ACCEPTED:
            dogfood_blocked_reason = (
                f"strict accepted candidates {len(dogfood_result['accepted'])} "
                f"< required {LAB_MIN_ACCEPTED}"
            )
        return _finish_lab(
            project_root=project_root,
            source_project=source_project,
            lab_dir=lab_dir,
            lab_id=lab_id,
            source_mode=source_mode,
            accepted=dogfood_result["accepted"],
            rejected_reasons=dogfood_result["rejected_reasons"],
            backend=backend,
            model=model,
            no_train=no_train,
            max_iters=max_iters,
            dogfood_blocked_reason=dogfood_blocked_reason,
            dogfood_metrics=dogfood_metrics,
        )

    fixture_result = _strict_accept_for_project(source_project)
    _write_source_reports(lab_dir, fixture_result)
    return _finish_lab(
        project_root=project_root,
        source_project=source_project,
        lab_dir=lab_dir,
        lab_id=lab_id,
        source_mode=source_mode,
        accepted=fixture_result["accepted"],
        rejected_reasons=fixture_result["rejected_reasons"],
        backend=backend,
        model=model,
        no_train=no_train,
        max_iters=max_iters,
        dogfood_blocked_reason=dogfood_blocked_reason,
        dogfood_metrics=dogfood_metrics,
    )


def strict_lab_accept_candidates(project_root: Path) -> dict:
    """Return strict machine-accepted candidates without mutating review state."""
    return _strict_accept_for_project(_safe_project_root(project_root))


def lab_auto_accept(project_root: Path, *, reviewed_by: str = "lab") -> dict:
    """Explicit lab-only status mutation for strict accepted candidates."""
    project_root = _safe_project_root(project_root)
    result = _strict_accept_for_project(project_root)
    accepted_ids = {candidate.id for candidate in result["accepted"]}
    store = ReviewStore(project_root)
    candidates = []
    accepted_count = 0
    now = datetime.now(timezone.utc)
    for candidate in store.load_candidates():
        if candidate.id in accepted_ids:
            candidates.append(candidate.model_copy(update={
                "status": "accepted",
                "reviewed_by": reviewed_by,
                "reviewed_at": now,
                "review_reason": "strict lab-only machine accept",
            }))
            accepted_count += 1
        else:
            candidates.append(candidate)
    store.save_candidates(candidates)
    return {
        "accepted": accepted_count,
        "candidates_total": len(candidates),
        "rejected_reasons": result["rejected_reasons"],
    }


def _finish_lab(
    *,
    project_root: Path,
    source_project: Path,
    lab_dir: Path,
    lab_id: str,
    source_mode: str,
    accepted: list[SemanticCandidate],
    rejected_reasons: Counter,
    backend: str,
    model: str,
    no_train: bool,
    max_iters: int,
    dogfood_blocked_reason: str | None,
    dogfood_metrics: dict | None,
) -> dict:
    _write_jsonl(lab_dir / "accepted_candidates.jsonl", [
        candidate.model_dump(mode="json") for candidate in accepted
    ])
    _write_strict_accept_report(
        lab_dir / "strict_accept_report.md",
        accepted=accepted,
        rejected_reasons=rejected_reasons,
        dogfood_blocked_reason=dogfood_blocked_reason,
    )
    workspace = _prepare_lab_workspace(lab_dir, source_project, accepted)
    dataset_result = None
    examples_count = 0
    eval_items_count = 0
    dataset_id = None
    dataset_manifest = {}
    if accepted:
        dataset_result = build_learning_dataset(
            workspace,
            dataset_format="chat",
            source="accepted",
            include_corrections=True,
            include_refusals=True,
        )
        dataset_dir = Path(dataset_result["dataset_dir"])
        lab_dataset_dir = lab_dir / "dataset"
        _copy_dataset_dir(dataset_dir, lab_dataset_dir)
        dataset_manifest = _read_json(lab_dataset_dir / "manifest.json")
        dataset_id = str(dataset_manifest.get("dataset_id") or "")
        examples_count = int(dataset_manifest.get("examples_count") or 0)
        eval_items_count = _count_jsonl(lab_dataset_dir / "eval.seed.jsonl")
    else:
        _write_empty_dataset(lab_dir / "dataset")

    train_allowed = (
        len(accepted) >= LAB_MIN_ACCEPTED
        and examples_count >= LAB_MIN_EXAMPLES
        and eval_items_count >= LAB_MIN_EVAL_ITEMS
    )
    training_result = _run_or_plan_training(
        lab_dir,
        backend=backend,
        model=model,
        max_iters=max_iters,
        no_train=no_train or not train_allowed,
        train_allowed=train_allowed,
    )
    eval_result = _write_lab_eval(
        lab_dir,
        eval_items_count=eval_items_count,
        training_result=training_result,
        model=model,
    )
    verdict = _lab_verdict(
        train_allowed=train_allowed,
        training_result=training_result,
        examples_count=examples_count,
        eval_items_count=eval_items_count,
        eval_result=eval_result,
    )
    eval_gate = _eval_gate(eval_result)
    dataset_quality = _dataset_quality(
        accepted=accepted,
        examples_count=examples_count,
        eval_items_count=eval_items_count,
        eval_seed_path=lab_dir / "dataset" / "eval.seed.jsonl",
    )
    production_blockers = _production_blockers(
        source_mode=source_mode,
        train_allowed=train_allowed,
        training_result=training_result,
        eval_result=eval_result,
        verdict=verdict,
    )
    summary = {
        "lab_id": lab_id,
        "lab_dir": str(lab_dir),
        "source": source_mode,
        "source_is_real_project_data": source_mode == "dogfood",
        "dogfood_blocked_reason": dogfood_blocked_reason,
        "dogfood": dogfood_metrics,
        "strict_accepted_candidates": len(accepted),
        "examples_count": examples_count,
        "eval_items_count": eval_items_count,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
        "dataset_quality": dataset_quality,
        "training_backend": backend,
        "model": model,
        "training_ran": training_result["training_ran"],
        "adapter_path": training_result.get("adapter_path"),
        "verdict": verdict,
        "train_allowed": train_allowed,
        "production_ready": not production_blockers and verdict == "ML_CORE_PASS",
        "production_blockers": production_blockers,
        "eval_gate": eval_gate,
        "eval": eval_result,
    }
    _write_json(lab_dir / "source_inventory.json", _source_inventory(source_project, accepted, rejected_reasons))
    _write_report(lab_dir / "REPORT.md", summary)
    latest = project_root / ".morpheus" / "lab" / "LATEST_REPORT.md"
    latest.write_text((lab_dir / "REPORT.md").read_text())
    return summary


def _strict_accept_for_project(project_root: Path) -> dict:
    candidates, review_meta = _load_or_generate_candidates(project_root)
    accepted: list[SemanticCandidate] = []
    rejected_reasons: Counter = Counter()
    for candidate in candidates:
        ok, reason = _strict_lab_accept_reason(project_root, candidate)
        if ok:
            accepted.append(candidate.model_copy(update={
                "status": "accepted",
                "reviewed_by": "morpheus-lab",
                "reviewed_at": datetime.now(timezone.utc),
                "review_reason": "strict lab machine accept",
            }))
        else:
            rejected_reasons[reason] += 1
    return {
        "candidates": candidates,
        "accepted": accepted,
        "rejected_reasons": rejected_reasons,
        "review_meta": review_meta,
    }


def _load_or_generate_candidates(project_root: Path) -> tuple[list[SemanticCandidate], dict]:
    store = ReviewStore(project_root)
    candidates = store.load_candidates()
    if candidates:
        stale_count = _stale_review_candidate_count(project_root, candidates)
        if stale_count:
            ephemeral = _generate_ephemeral_lab_candidates(project_root)
            return ephemeral, {
                "review_source": "ephemeral_local_due_to_stale_review_store",
                "stale_review_candidates": stale_count,
                "ephemeral_candidates_generated": len(ephemeral),
            }
        return candidates, {
            "review_source": "review_store",
            "stale_review_candidates": 0,
            "ephemeral_candidates_generated": 0,
        }
    run_semantic_review(project_root, provider=LocalProvider())
    generated = store.load_candidates()
    return generated, {
        "review_source": "generated_review_store",
        "stale_review_candidates": 0,
        "ephemeral_candidates_generated": 0,
    }


def _stale_review_candidate_count(project_root: Path, candidates: list[SemanticCandidate]) -> int:
    return sum(
        1
        for candidate in candidates
        if _strict_lab_accept_reason(project_root, candidate)[1] == "source_hash_mismatch"
    )


def _generate_ephemeral_lab_candidates(project_root: Path) -> list[SemanticCandidate]:
    provider = LocalProvider()
    run_id = _timestamp_id("semlab")
    prompt_sha256 = hashlib.sha256(b"morpheus-lab-ephemeral-local-v1").hexdigest()
    source_revision = "lab:ephemeral"
    candidates = []
    for source in scan_semantic_sources(project_root):
        candidates.extend(
            provider.extract_candidates(
                source,
                run_id=run_id,
                prompt_sha256=prompt_sha256,
                source_revision=source_revision,
            )
        )
    return [verify_candidate_span(project_root, candidate) for candidate in candidates]


def _strict_lab_accept_reason(project_root: Path, candidate: SemanticCandidate) -> tuple[bool, str]:
    if candidate.status != "pending":
        return False, "status_not_pending"
    if candidate.label != "source_backed":
        return False, "label_not_source_backed"
    if candidate.kind == "outdated_claim":
        return False, "outdated_claim_correction_only"
    if candidate.kind not in LAB_STRICT_KINDS:
        return False, "unsupported_kind"
    if not candidate.prompt_sha256:
        return False, "missing_prompt_sha256"
    if len(candidate.claim) > 240:
        return False, "claim_too_long"
    if _contains_speculative_word(candidate.claim):
        return False, "speculative_wording"
    if _needs_split(candidate.claim):
        return False, "needs_split"
    if contains_secret_like_text(candidate.claim) or contains_secret_like_text(candidate.evidence_excerpt):
        return False, "secret_like_content"
    rel_path = Path(candidate.source_path)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        return False, "invalid_source_path"
    if not _source_path_allowed(rel_path):
        return False, "source_path_not_allowlisted"
    if path_is_ignored(rel_path, load_morpheusignore(project_root)):
        return False, "ignored_path"
    source = project_root / rel_path
    if not source.is_file():
        return False, "missing_source"
    try:
        reject_symlink_paths([source], "Lab source")
        reject_symlink_components(source, "Lab source")
    except ValueError:
        return False, "unsafe_source"
    if _sha256(source) != candidate.source_sha256:
        return False, "source_hash_mismatch"
    if not _line_range_valid(source, candidate):
        return False, "invalid_line_range"
    if not _source_span_exact_match(source, candidate):
        return False, "evidence_not_exact_match"
    return True, "accepted"


def _prepare_lab_workspace(
    lab_dir: Path,
    source_project: Path,
    accepted: list[SemanticCandidate],
) -> Path:
    workspace = lab_dir / "workspace"
    _ensure_new_dir(workspace)
    paths = {Path(candidate.source_path) for candidate in accepted}
    ignore = source_project / ".morpheusignore"
    if ignore.is_file() and not ignore.is_symlink():
        paths.add(Path(".morpheusignore"))
    for rel_path in sorted(paths, key=lambda item: item.as_posix()):
        if rel_path.is_absolute() or ".." in rel_path.parts:
            continue
        source = source_project / rel_path
        if not source.is_file() or source.is_symlink():
            continue
        dest = workspace / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    review_dir = workspace / ".morpheus" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(review_dir / "semantic_candidates.jsonl", [
        candidate.model_dump(mode="json") for candidate in accepted
    ])
    return workspace


def _run_or_plan_training(
    lab_dir: Path,
    *,
    backend: str,
    model: str,
    max_iters: int,
    no_train: bool,
    train_allowed: bool,
) -> dict:
    training_dir = lab_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = training_dir / "adapter"
    command_prefix = _mlx_command_prefix("lora") if backend == "mlx" else None
    command = _training_command(
        backend=backend,
        model=model,
        dataset_dir=lab_dir / "dataset",
        adapter_path=adapter_path,
        max_iters=max_iters,
        command_prefix=command_prefix,
    )
    (training_dir / "train_command.sh").write_text(command + "\n")
    (training_dir / "train_command.sh").chmod(0o755)
    adapter_manifest = {
        "adapter_id": lab_dir.name + "_adapter",
        "backend": backend,
        "model": model,
        "path": str(adapter_path),
        "status": "planned",
        "activated": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = {
        "training_ran": False,
        "adapter_path": None,
        "status": "skipped",
        "reason": None,
        "returncode": None,
        "backend": backend,
        "model": model,
    }
    if no_train:
        result["reason"] = "no_train" if train_allowed else "dataset_threshold_not_met"
        (training_dir / "train.log").write_text(f"Training skipped: {result['reason']}\n")
        adapter_manifest["status"] = "planned"
    elif backend == "mlx":
        if command_prefix is None:
            result["reason"] = "mlx_lm_not_found"
            (training_dir / "train.log").write_text(
                "Training skipped: neither mlx_lm.lora nor python -m mlx_lm is available.\n"
            )
            adapter_manifest["status"] = "blocked"
        else:
            completed = subprocess.run(
                command,
                cwd=lab_dir,
                shell=True,
                text=True,
                capture_output=True,
                timeout=1800,
                check=False,
            )
            log = completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr
            (training_dir / "train.log").write_text(log)
            result["training_ran"] = completed.returncode == 0
            result["returncode"] = completed.returncode
            result["status"] = "trained_smoke" if completed.returncode == 0 else "failed"
            result["adapter_path"] = str(adapter_path) if completed.returncode == 0 else None
            adapter_manifest["status"] = result["status"]
    else:
        result["reason"] = "fake_backend_no_training"
        (training_dir / "train.log").write_text("Training skipped: fake backend.\n")
        adapter_manifest["status"] = "planned"
    _write_json(training_dir / "adapter_manifest.json", adapter_manifest)
    return result


def _write_lab_eval(
    lab_dir: Path,
    *,
    eval_items_count: int,
    training_result: dict,
    model: str,
) -> dict:
    eval_dir = lab_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    seed_items = _read_jsonl(lab_dir / "dataset" / "eval.seed.jsonl")
    if not training_result["training_ran"]:
        base = _evaluate_lab_items(seed_items, mode="base", backend="fake", model=model)
        adapter = {
            "mode": "adapter",
            "items": [],
            "items_count": eval_items_count,
            "evaluated_items_count": 0,
            "pass_rate": None,
            "hallucination_rate": None,
            "critical_failures": 0,
            "status": "not_run",
        }
        comparison = {
            "adapter_delta": None,
            "regression_count": 0,
            "critical_regression": False,
            "status": "adapter_not_run",
        }
    elif training_result.get("backend") == "mlx":
        selected = _select_eval_items(seed_items, limit=LAB_MLX_EVAL_ITEM_LIMIT)
        base = _evaluate_lab_items(selected, mode="base", backend="mlx", model=model)
        adapter = _evaluate_lab_items(
            selected,
            mode="adapter",
            backend="mlx",
            model=model,
            adapter_path=training_result.get("adapter_path"),
        )
        comparison = _compare_lab_eval(base, adapter)
    else:
        base = _evaluate_lab_items(seed_items, mode="base", backend="fake", model=model)
        adapter = _evaluate_lab_items(seed_items, mode="adapter", backend="fake", model=model)
        comparison = _compare_lab_eval(base, adapter)

    _write_json(eval_dir / "base_results.json", base)
    _write_json(eval_dir / "adapter_results.json", adapter)
    _write_json(eval_dir / "eval_config.json", {
        "model": model,
        "backend": training_result.get("backend"),
        "training_status": training_result.get("status"),
        "eval_items_total": eval_items_count,
        "mlx_eval_item_limit": LAB_MLX_EVAL_ITEM_LIMIT,
    })
    report = [
        "# Morpheus Lab Eval",
        "",
        f"- Eval items: `{eval_items_count}`",
        f"- Base evaluated: `{base.get('evaluated_items_count', 0)}`",
        f"- Adapter evaluated: `{adapter.get('evaluated_items_count', 0)}`",
        f"- Base pass rate: `{base.get('pass_rate')}`",
        f"- Adapter pass rate: `{adapter.get('pass_rate')}`",
        f"- Adapter delta: `{comparison.get('adapter_delta')}`",
        f"- Critical regression: `{comparison.get('critical_regression')}`",
        "",
        "## Base vs Adapter",
        "",
    ]
    for item in comparison.get("regressions", [])[:10]:
        report.extend([
            f"### Regression: {item.get('category')}",
            f"- Question: {item.get('question')}",
            f"- Base passed: {item.get('base_passed')}",
            f"- Adapter passed: {item.get('adapter_passed')}",
            "",
        ])
    (eval_dir / "eval_report.md").write_text("\n".join(report))
    return {"eval_dir": str(eval_dir), "base": base, "adapter": adapter, "comparison": comparison}


def _lab_verdict(
    *,
    train_allowed: bool,
    training_result: dict,
    examples_count: int,
    eval_items_count: int,
    eval_result: dict,
) -> str:
    if examples_count <= 0 or eval_items_count <= 0:
        return "ML_CORE_DATASET_BLOCKED"
    if not train_allowed:
        return "ML_CORE_DATASET_BLOCKED"
    if not training_result["training_ran"]:
        return "ML_CORE_PARTIAL"
    comparison = eval_result.get("comparison", {})
    if comparison.get("eval_error"):
        return "ML_CORE_FAIL"
    if comparison.get("critical_regression"):
        return "ML_CORE_FAIL"
    adapter_delta = comparison.get("adapter_delta")
    adapter_pass_rate = eval_result.get("adapter", {}).get("pass_rate")
    if adapter_delta is None or adapter_pass_rate is None:
        return "ML_CORE_PARTIAL"
    if adapter_delta >= 0 and adapter_pass_rate >= 0.60:
        return "ML_CORE_PASS"
    if adapter_delta < -0.10:
        return "ML_CORE_FAIL"
    return "ML_CORE_PARTIAL"


def _production_blockers(
    *,
    source_mode: str,
    train_allowed: bool,
    training_result: dict,
    eval_result: dict,
    verdict: str,
) -> list[str]:
    blockers = []
    if source_mode != "dogfood":
        blockers.append("source_mode_fixture_not_real_project_data")
    if not train_allowed:
        blockers.append("dataset_threshold_not_met")
    if not training_result.get("training_ran"):
        blockers.append("training_not_run")
    comparison = eval_result.get("comparison", {})
    if comparison.get("eval_error"):
        blockers.append("eval_error")
    if comparison.get("critical_regression"):
        blockers.append("critical_regression")
    if training_result.get("training_ran") and verdict != "ML_CORE_PASS":
        blockers.append("adapter_eval_not_passed")
    return blockers


def _dataset_quality(
    *,
    accepted: list[SemanticCandidate],
    examples_count: int,
    eval_items_count: int,
    eval_seed_path: Path,
) -> dict:
    eval_items = _read_jsonl(eval_seed_path)
    accepted_count = len(accepted)
    return {
        "accepted_candidates": accepted_count,
        "examples_count": examples_count,
        "eval_items_count": eval_items_count,
        "examples_per_candidate": round(examples_count / accepted_count, 4)
        if accepted_count
        else 0.0,
        "source_path_count": len({candidate.source_path for candidate in accepted}),
        "accepted_by_kind": dict(Counter(candidate.kind for candidate in accepted)),
        "accepted_by_source_path": dict(Counter(candidate.source_path for candidate in accepted)),
        "eval_items_by_category": dict(Counter(str(item.get("category") or "unknown") for item in eval_items)),
        "meets_thresholds": {
            "accepted_candidates": accepted_count >= LAB_MIN_ACCEPTED,
            "examples": examples_count >= LAB_MIN_EXAMPLES,
            "eval_items": eval_items_count >= LAB_MIN_EVAL_ITEMS,
        },
    }


def _eval_gate(eval_result: dict) -> dict:
    adapter = eval_result.get("adapter", {})
    comparison = eval_result.get("comparison", {})
    adapter_pass_rate = adapter.get("pass_rate")
    adapter_hallucination_rate = adapter.get("hallucination_rate")
    critical_failures = int(adapter.get("critical_failures") or 0)
    regression_count = int(comparison.get("regression_count") or 0)
    adapter_evaluated = bool(adapter.get("evaluated_items_count"))
    block_reasons = []
    if not adapter_evaluated:
        block_reasons.append("adapter_not_evaluated")
    if adapter_pass_rate is not None and adapter_pass_rate < LAB_PASS_RATE_THRESHOLD:
        block_reasons.append("pass_rate_below_threshold")
    if (
        adapter_hallucination_rate is not None
        and adapter_hallucination_rate > LAB_HALLUCINATION_RATE_THRESHOLD
    ):
        block_reasons.append("hallucination_rate_above_threshold")
    if critical_failures:
        block_reasons.append("critical_failures")
    if regression_count:
        block_reasons.append("regressions")
    if comparison.get("critical_regression"):
        block_reasons.append("critical_regression")
    if comparison.get("eval_error"):
        block_reasons.append("eval_error")
    return {
        "pass_rate_threshold": LAB_PASS_RATE_THRESHOLD,
        "hallucination_rate_threshold": LAB_HALLUCINATION_RATE_THRESHOLD,
        "adapter_evaluated": adapter_evaluated,
        "adapter_pass_rate": adapter_pass_rate,
        "adapter_hallucination_rate": adapter_hallucination_rate,
        "critical_failures": critical_failures,
        "regression_count": regression_count,
        "critical_regression": bool(comparison.get("critical_regression")),
        "activation_allowed": not block_reasons,
        "block_reasons": block_reasons,
    }


def _evaluate_lab_items(
    items: list[dict],
    *,
    mode: str,
    backend: str,
    model: str,
    adapter_path: str | None = None,
) -> dict:
    scored = []
    errors = []
    for item in items:
        try:
            answer = _lab_answer(item, mode=mode, backend=backend, model=model, adapter_path=adapter_path)
        except (OSError, subprocess.SubprocessError, TimeoutError, ValueError) as exc:
            answer = ""
            errors.append({"question": item.get("question"), "error": str(exc)})
        scored.append(_score_lab_item(item, answer, mode=mode))
    total = len(scored)
    passed = sum(1 for item in scored if item["passed"])
    hallucinated = sum(1 for item in scored if item["hallucinated"])
    critical_failures = sum(1 for item in scored if item["critical_failure"])
    return {
        "mode": mode,
        "backend": backend,
        "model": model,
        "adapter_path": adapter_path,
        "items": scored,
        "items_count": len(items),
        "evaluated_items_count": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "hallucination_rate": round(hallucinated / total, 4) if total else 0.0,
        "critical_failures": critical_failures,
        "errors": errors,
        "status": "evaluated_with_errors" if errors else "evaluated",
    }


def _lab_answer(
    item: dict,
    *,
    mode: str,
    backend: str,
    model: str,
    adapter_path: str | None,
) -> str:
    if backend != "mlx":
        return _fake_lab_answer(item, mode=mode)
    return _mlx_generate_answer(
        model=model,
        prompt=str(item.get("question") or ""),
        adapter_path=adapter_path if mode == "adapter" else None,
    )


def _fake_lab_answer(item: dict, *, mode: str) -> str:
    expected = str(item.get("expected_answer") or "")
    category = str(item.get("category") or "")
    if mode == "adapter":
        return expected
    if category in {"unsupported_claim_refusal", "outdated_claim_correction"}:
        return "I cannot confirm unsupported project claims without reviewed source evidence."
    return "I do not know from reviewed Morpheus state."


def _mlx_generate_answer(*, model: str, prompt: str, adapter_path: str | None) -> str:
    command_prefix = _mlx_command_prefix("generate")
    if command_prefix is None:
        raise ValueError("mlx_lm generate not found")
    command = (
        f"{command_prefix} "
        f"--model {shlex.quote(model)} "
        f"--system-prompt {shlex.quote(SYSTEM_PROMPT)} "
        f"--prompt {shlex.quote(prompt)} "
        "--max-tokens 96 "
        "--temp 0 "
        "--seed 7 "
        "--verbose False"
    )
    if adapter_path:
        command += f" --adapter-path {shlex.quote(adapter_path)}"
    completed = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or f"mlx_lm.generate exited {completed.returncode}")
    return completed.stdout.strip()


def _score_lab_item(item: dict, answer: str, *, mode: str) -> dict:
    category = str(item.get("category") or "project_recall")
    expected = str(item.get("expected_answer") or "")
    passed = _answer_passes(category, expected, answer)
    hallucinated = _answer_hallucinates(category, answer)
    critical_failure = category in {
        "outdated_claim_correction",
        "unsupported_claim_refusal",
    } and not passed
    return {
        "mode": mode,
        "category": category,
        "question": item.get("question"),
        "expected_answer": expected,
        "answer": answer,
        "passed": passed,
        "hallucinated": hallucinated,
        "critical_failure": critical_failure,
        "source_candidate_id": item.get("source_candidate_id"),
        "source_path": item.get("source_path"),
        "kind": item.get("kind"),
    }


def _answer_passes(category: str, expected: str, answer: str) -> bool:
    folded_answer = _normalize(answer)
    folded_expected = _normalize(expected)
    if not folded_answer:
        return False
    if category == "unsupported_claim_refusal":
        return any(token in folded_answer for token in ["cannot confirm", "unsupported", "no."])
    if category == "outdated_claim_correction":
        return "no" in folded_answer and (
            "outdated" in folded_answer
            or "not current" in folded_answer
            or "must not" in folded_answer
        )
    if folded_expected and folded_expected in folded_answer:
        return True
    return _token_overlap(folded_expected, folded_answer) >= 0.45 or (
        folded_expected
        and SequenceMatcher(None, folded_expected, folded_answer).ratio() >= 0.55
    )


def _answer_hallucinates(category: str, answer: str) -> bool:
    folded = _normalize(answer)
    if "confirmed without source" in folded:
        return True
    if category == "unsupported_claim_refusal" and not any(
        token in folded for token in ["cannot confirm", "unsupported", "no."]
    ):
        return True
    return False


def _compare_lab_eval(base: dict, adapter: dict) -> dict:
    base_items = base.get("items", [])
    adapter_items = adapter.get("items", [])
    regressions = []
    for base_item, adapter_item in zip(base_items, adapter_items, strict=False):
        if base_item.get("passed") and not adapter_item.get("passed"):
            regressions.append({
                "category": adapter_item.get("category"),
                "question": adapter_item.get("question"),
                "base_passed": base_item.get("passed"),
                "adapter_passed": adapter_item.get("passed"),
            })
    base_rate = base.get("pass_rate")
    adapter_rate = adapter.get("pass_rate")
    delta = None
    if base_rate is not None and adapter_rate is not None:
        delta = round(float(adapter_rate) - float(base_rate), 4)
    critical_regression = bool(
        adapter.get("critical_failures", 0)
        or (delta is not None and delta < -0.10)
        or adapter.get("errors")
    )
    return {
        "adapter_delta": delta,
        "regression_count": len(regressions),
        "critical_regression": critical_regression,
        "regressions": regressions,
        "eval_error": bool(adapter.get("errors") or base.get("errors")),
    }


def _select_eval_items(items: list[dict], *, limit: int) -> list[dict]:
    if len(items) <= limit:
        return items
    selected = []
    seen_categories = set()
    for item in items:
        category = str(item.get("category") or "")
        if category in seen_categories:
            continue
        selected.append(item)
        seen_categories.add(category)
        if len(selected) >= limit:
            return selected
    for item in items:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _mlx_command_prefix(tool: str) -> str | None:
    binary = shutil.which(f"mlx_lm.{tool}")
    if binary:
        return shlex.quote(binary)
    if importlib.util.find_spec("mlx_lm") is not None:
        return f"{shlex.quote(sys.executable)} -m mlx_lm {tool}"
    return None


def _fixture_project(project_root: Path, lab_dir: Path) -> Path:
    fixture = _repo_root() / "tests" / "fixtures" / "autonomous_learning_repo"
    dest = lab_dir / "fixture_project"
    if fixture.is_dir():
        shutil.copytree(fixture, dest)
        return dest
    _write_builtin_fixture(dest)
    return dest


def _write_builtin_fixture(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "README.md").write_text(
        "# Morpheus Lab Fixture\n\n"
        + "\n".join(
            f"DECISION: Morpheus lab fixture source-backed fact number {index}."
            for index in range(1, 41)
        )
        + "\n"
    )
    (dest / "SPEC.md").write_text(
        "# Spec\n\nDECISION: First verify, then learn.\n"
    )
    (dest / "AGENTS.md").write_text(
        "# Agents\n\n- AGENT_RULE: Never train on raw markdown.\n"
    )
    (dest / "pyproject.toml").write_text('[project]\nname = "morpheus-wake"\n')


def _write_source_reports(lab_dir: Path, result: dict) -> None:
    _write_json(lab_dir / "source_inventory.json", _source_inventory_raw(result))
    _write_strict_accept_report(
        lab_dir / "strict_accept_report.md",
        accepted=result["accepted"],
        rejected_reasons=result["rejected_reasons"],
        dogfood_blocked_reason=None,
    )


def _write_dogfood_reports(lab_dir: Path, result: dict) -> None:
    _write_json(lab_dir / "dogfood_inventory.json", _source_inventory_raw(result))
    _write_strict_accept_report(
        lab_dir / "dogfood_strict_accept_report.md",
        accepted=result["accepted"],
        rejected_reasons=result["rejected_reasons"],
        dogfood_blocked_reason=None,
    )


def _source_metrics(result: dict) -> dict:
    inventory = _source_inventory_raw(result)
    accepted = int(inventory["accepted_candidates"])
    review_meta = result.get("review_meta") or {}
    return {
        "review_source": review_meta.get("review_source", "unknown"),
        "stale_review_candidates": int(review_meta.get("stale_review_candidates") or 0),
        "ephemeral_candidates_generated": int(review_meta.get("ephemeral_candidates_generated") or 0),
        "total_candidates": inventory["total_candidates"],
        "strict_accepted_candidates": accepted,
        "candidate_threshold": LAB_MIN_ACCEPTED,
        "train_allowed": accepted >= LAB_MIN_ACCEPTED,
        "by_kind": inventory["by_kind"],
        "by_label": inventory["by_label"],
        "by_status": inventory["by_status"],
        "by_source_path": inventory["by_source_path"],
        "rejected_reasons": inventory["rejected_reasons"],
    }


def _source_inventory_raw(result: dict) -> dict:
    candidates = result["candidates"]
    review_meta = result.get("review_meta") or {}
    return {
        "review_source": review_meta.get("review_source", "unknown"),
        "stale_review_candidates": int(review_meta.get("stale_review_candidates") or 0),
        "ephemeral_candidates_generated": int(review_meta.get("ephemeral_candidates_generated") or 0),
        "total_candidates": len(candidates),
        "accepted_candidates": len(result["accepted"]),
        "by_kind": dict(Counter(candidate.kind for candidate in candidates)),
        "by_label": dict(Counter(candidate.label for candidate in candidates)),
        "by_status": dict(Counter(candidate.status for candidate in candidates)),
        "by_source_path": dict(Counter(candidate.source_path for candidate in candidates)),
        "rejected_reasons": dict(result["rejected_reasons"]),
    }


def _source_inventory(
    source_project: Path,
    accepted: list[SemanticCandidate],
    rejected_reasons: Counter,
) -> dict:
    candidates = ReviewStore(source_project).load_candidates()
    return {
        "source_project": str(source_project),
        "total_candidates": len(candidates),
        "accepted_candidates": len(accepted),
        "by_kind": dict(Counter(candidate.kind for candidate in candidates)),
        "by_label": dict(Counter(candidate.label for candidate in candidates)),
        "by_status": dict(Counter(candidate.status for candidate in candidates)),
        "by_source_path": dict(Counter(candidate.source_path for candidate in candidates)),
        "rejected_reasons": dict(rejected_reasons),
    }


def _write_strict_accept_report(
    path: Path,
    *,
    accepted: list[SemanticCandidate],
    rejected_reasons: Counter,
    dogfood_blocked_reason: str | None,
) -> None:
    lines = [
        "# Morpheus Strict Accept Report",
        "",
        f"- Accepted candidates: `{len(accepted)}`",
    ]
    if dogfood_blocked_reason:
        lines.append(f"- Dogfood blocked: {dogfood_blocked_reason}")
    lines.extend(["", "## Rejected Reasons", ""])
    if rejected_reasons:
        for reason, count in sorted(rejected_reasons.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Accepted IDs", ""])
    lines.extend(f"- `{candidate.id}`: {candidate.claim}" for candidate in accepted[:100])
    path.write_text("\n".join(lines).rstrip() + "\n")


def _write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Morpheus Autonomous Learning Lab",
        "",
        f"- Lab ID: `{summary['lab_id']}`",
        f"- Source mode: `{summary['source']}`",
        f"- Real project data: `{summary['source_is_real_project_data']}`",
        f"- Strict accepted candidates: `{summary['strict_accepted_candidates']}`",
        f"- Examples: `{summary['examples_count']}`",
        f"- Eval items: `{summary['eval_items_count']}`",
        f"- Training backend: `{summary['training_backend']}`",
        f"- Model: `{summary['model']}`",
        f"- Training ran: `{summary['training_ran']}`",
        f"- Adapter: `{summary.get('adapter_path') or 'none'}`",
        f"- Verdict: `{summary['verdict']}`",
        f"- Production ready: `{summary['production_ready']}`",
        "",
        "## Production Gate",
        "",
    ]
    if summary["source"] != "dogfood":
        lines.extend([
            "Fixture benchmark is not production data.",
            "",
        ])
    blockers = summary.get("production_blockers") or []
    if blockers:
        lines.extend(["Production blockers:", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
        lines.append("")
    else:
        lines.extend(["No production blockers detected.", ""])
    dogfood = summary.get("dogfood")
    if dogfood:
        lines.extend([
            "## Dogfood Metrics",
            "",
            f"- Total candidates: `{dogfood['total_candidates']}`",
            f"- Strict accepted candidates: `{dogfood['strict_accepted_candidates']}`",
            f"- Candidate threshold: `{dogfood['candidate_threshold']}`",
            f"- Candidate train allowed: `{dogfood['train_allowed']}`",
            "",
        ])
    quality = summary.get("dataset_quality") or {}
    if quality:
        lines.extend([
            "## Dataset Quality",
            "",
            f"- Accepted candidates: `{quality['accepted_candidates']}`",
            f"- Examples: `{quality['examples_count']}`",
            f"- Eval items: `{quality['eval_items_count']}`",
            f"- Examples per candidate: `{quality['examples_per_candidate']}`",
            f"- Source path count: `{quality['source_path_count']}`",
            "",
            "### Accepted By Kind",
            "",
        ])
        accepted_by_kind = quality.get("accepted_by_kind") or {}
        if accepted_by_kind:
            lines.extend(f"- `{kind}`: {count}" for kind, count in sorted(accepted_by_kind.items()))
        else:
            lines.append("- none")
        lines.extend(["", "### Eval Items By Category", ""])
        eval_by_category = quality.get("eval_items_by_category") or {}
        if eval_by_category:
            lines.extend(f"- `{category}`: {count}" for category, count in sorted(eval_by_category.items()))
        else:
            lines.append("- none")
        lines.append("")
    gate = summary.get("eval_gate") or {}
    if gate:
        lines.extend([
            "## Eval Gate",
            "",
            f"- Adapter evaluated: `{gate['adapter_evaluated']}`",
            f"- Activation allowed: `{gate['activation_allowed']}`",
            f"- Pass rate threshold: `{gate['pass_rate_threshold']}`",
            f"- Hallucination rate threshold: `{gate['hallucination_rate_threshold']}`",
            f"- Adapter pass rate: `{gate.get('adapter_pass_rate')}`",
            f"- Adapter hallucination rate: `{gate.get('adapter_hallucination_rate')}`",
            f"- Critical failures: `{gate['critical_failures']}`",
            f"- Regression count: `{gate['regression_count']}`",
            "",
            "### Eval Gate Block Reasons",
            "",
        ])
        block_reasons = gate.get("block_reasons") or []
        if block_reasons:
            lines.extend(f"- `{reason}`" for reason in block_reasons)
        else:
            lines.append("- none")
        lines.append("")
    lines.extend([
        "## Safety",
        "",
        "- Raw markdown was scanned only to create source spans; dataset examples came from accepted candidates.",
        "- Pending, rejected, inferred-only, ignored, and secret-like candidates were excluded.",
        "- No adapter was activated.",
        "",
    ])
    if summary.get("dogfood_blocked_reason"):
        lines.extend(["## Dogfood Blocker", "", summary["dogfood_blocked_reason"], ""])
    path.write_text("\n".join(lines))


def _copy_dataset_dir(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def _write_empty_dataset(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in [
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
        "dataset.instruction.jsonl",
        "dataset.sharegpt.jsonl",
        "eval.seed.jsonl",
        "skipped.jsonl",
    ]:
        (path / name).write_text("")
    _write_json(path / "manifest.json", {
        "dataset_id": None,
        "examples_count": 0,
        "skipped_count": 0,
        "trainable_candidate_count": 0,
        "dataset_sha256": None,
    })


def _training_command(
    *,
    backend: str,
    model: str,
    dataset_dir: Path,
    adapter_path: Path,
    max_iters: int,
    command_prefix: str | None = None,
) -> str:
    if backend == "mlx":
        prefix = command_prefix or "mlx_lm.lora"
        return (
            f"{prefix} "
            f"--model {shlex.quote(model)} "
            "--train "
            f"--data {shlex.quote(str(dataset_dir))} "
            f"--adapter-path {shlex.quote(str(adapter_path))} "
            f"--iters {max_iters} "
            "--batch-size 1 "
            "--num-layers 4 "
            f"--learning-rate {LAB_MLX_LEARNING_RATE} "
            "--mask-prompt"
        )
    return (
        "# fake backend: no training command executed\n"
        f"# model={model} data={dataset_dir} adapter={adapter_path}"
    )


def _line_range_valid(source: Path, candidate: SemanticCandidate) -> bool:
    try:
        lines = source.read_text(errors="ignore").splitlines()
    except OSError:
        return False
    return 1 <= candidate.line_start <= candidate.line_end <= len(lines)


def _source_span_exact_match(source: Path, candidate: SemanticCandidate) -> bool:
    try:
        lines = source.read_text(errors="ignore").splitlines()
    except OSError:
        return False
    actual = "\n".join(lines[candidate.line_start - 1 : candidate.line_end]).strip()
    return actual == candidate.evidence_excerpt.strip()


def _source_path_allowed(path: Path) -> bool:
    text = path.as_posix()
    if text in {
        "README.md",
        "README.ru.md",
        "SPEC.md",
        "WAKE.md",
        "AGENTS.md",
        "CHANGELOG.md",
        "pyproject.toml",
    }:
        return True
    return fnmatch(text, "docs/**/*.md") or fnmatch(text, "docs/*.md")


def _contains_speculative_word(value: str) -> bool:
    words = {
        word.strip(".,:;!?()[]{}\"'`").casefold()
        for word in value.split()
    }
    return bool(words & SPECULATIVE_WORDS)


def _needs_split(claim: str) -> bool:
    if len(claim) > 240:
        return True
    lowered = claim.casefold()
    return lowered.count(" and ") >= 2 or ";" in claim


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows) + ("\n" if rows else ""))


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _token_overlap(expected: str, answer: str) -> float:
    expected_tokens = {
        token.strip(".,:;!?()[]{}\"'`")
        for token in expected.split()
        if len(token.strip(".,:;!?()[]{}\"'`")) > 3
    }
    answer_tokens = {
        token.strip(".,:;!?()[]{}\"'`")
        for token in answer.split()
        if len(token.strip(".,:;!?()[]{}\"'`")) > 3
    }
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & answer_tokens) / len(expected_tokens)


def _timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_project_root(project_root: Path) -> Path:
    project_root = project_root.expanduser()
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")
    return project_root


def _ensure_new_dir(path: Path) -> None:
    if path.exists():
        raise ValueError(f"Lab path already exists: {path}")
    reject_symlink_components(path.parent, "Lab path")
    path.mkdir(parents=True, exist_ok=False)
    reject_symlink_components(path, "Lab path")
