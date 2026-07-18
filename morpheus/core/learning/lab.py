"""Autonomous source-grounded learning lab for Morpheus."""
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from fnmatch import fnmatch
from pathlib import Path

from morpheus.core.command_contract import (
    canonical_command_answer_passes,
)
from morpheus.core.learning.categories import (
    CRITICAL_BENCHMARK_CATEGORIES as CRITICAL_EVAL_CATEGORIES,
)
from morpheus.core.learning.authority import learning_authority_transaction
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.dataset_validation import manifest_count, require_valid_dataset
from morpheus.core.learning.examples import SYSTEM_PROMPT
from morpheus.core.learning.training_runtime import (
    MLX_PINNED_LOADER_CONTRACT,
    RUNTIME_DATASET_DIR_PLACEHOLDER,
    RUNTIME_OUTPUT_DIR_PLACEHOLDER,
    RuntimeDatasetArgument,
    render_guarded_training_command,
    seal_dataset_snapshot,
    shell_quote_training_argument,
)
from morpheus.core.learning.safety import (
    contains_secret_like_text,
    load_morpheusignore,
    path_is_ignored,
)
from morpheus.core.learning.scoring import (
    critical_answer_hallucinates,
    critical_answer_passes,
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
DEFAULT_LAB_EVAL_LIMIT = LAB_MLX_EVAL_ITEM_LIMIT
LAB_PASS_RATE_THRESHOLD = 0.60
LAB_HALLUCINATION_RATE_THRESHOLD = 0.05
LAB_MLX_LEARNING_RATE = "1e-5"
DEFAULT_LAB_BACKEND = "fake"
DEFAULT_LAB_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_LAB_MAX_ITERS = 400
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
    max_iters: int = DEFAULT_LAB_MAX_ITERS,
    eval_limit: int = DEFAULT_LAB_EVAL_LIMIT,
) -> dict:
    """Run an autonomous benchmark or dogfood learning experiment."""
    project_root = _safe_project_root(project_root)
    lab_id = _timestamp_id("lab")
    lab_dir = project_root / ".morpheus" / "lab" / lab_id
    with learning_authority_transaction(project_root):
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
        "eval_limit": eval_limit,
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
                eval_limit=eval_limit,
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
            eval_limit=eval_limit,
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
        eval_limit=eval_limit,
        dogfood_blocked_reason=dogfood_blocked_reason,
        dogfood_metrics=dogfood_metrics,
    )


def run_autonomous_lab_stability(
    project_root: Path,
    *,
    repeat: int,
    backend: str = DEFAULT_LAB_BACKEND,
    model: str = DEFAULT_LAB_MODEL,
    no_train: bool = False,
    fixture_only: bool = False,
    dogfood: bool = False,
    max_iters: int = DEFAULT_LAB_MAX_ITERS,
    eval_limit: int = DEFAULT_LAB_EVAL_LIMIT,
) -> dict:
    """Run repeated autonomous labs and aggregate stability gate results."""
    project_root = _safe_project_root(project_root)
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    stability_id = _timestamp_id("stability")
    stability_dir = project_root / ".morpheus" / "lab" / "stability" / stability_id
    _ensure_new_dir(stability_dir)

    runs = []
    for index in range(1, repeat + 1):
        result = run_autonomous_lab(
            project_root,
            backend=backend,
            model=model,
            no_train=no_train,
            fixture_only=fixture_only,
            dogfood=dogfood,
            max_iters=max_iters,
            eval_limit=eval_limit,
        )
        runs.append(_stability_run_summary(index, result))

    blockers = _stability_blockers(runs)
    summary = {
        "stability_id": stability_id,
        "stability_dir": str(stability_dir),
        "repeat": repeat,
        "runs_count": len(runs),
        "backend": backend,
        "model": model,
        "no_train": no_train,
        "fixture_only": fixture_only,
        "dogfood": dogfood,
        "max_iters": max_iters,
        "eval_limit": eval_limit,
        "stability_passed": not blockers,
        "verdict": "ML_CORE_PASS" if not blockers else "ML_CORE_PARTIAL",
        "stability_blockers": blockers,
        "runs": runs,
    }
    _write_json(stability_dir / "stability_report.json", summary)
    _write_stability_report(stability_dir / "stability_report.md", summary)
    latest = project_root / ".morpheus" / "lab" / "LATEST_STABILITY_REPORT.md"
    latest.write_text((stability_dir / "stability_report.md").read_text())
    return summary


def strict_lab_accept_candidates(project_root: Path) -> dict:
    """Return strict machine-accepted candidates without mutating review state."""
    return _strict_accept_for_project(_safe_project_root(project_root))


def lab_auto_accept(project_root: Path, *, reviewed_by: str = "lab") -> dict:
    """Explicit lab-only status mutation for strict accepted candidates."""
    project_root = _safe_project_root(project_root)
    result = _strict_accept_for_project(project_root)
    accepted_ids = {candidate.id for candidate in result["accepted"]}
    store = ReviewStore(project_root)
    with store.transaction():
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
    eval_limit: int,
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
        seal_dataset_snapshot(lab_dataset_dir)
        dataset_manifest = _read_json(lab_dataset_dir / "manifest.json")
        dataset_id = str(dataset_manifest.get("dataset_id") or "")
        examples_count = manifest_count(dataset_manifest, "examples_count")
        eval_items_count = _count_jsonl(lab_dataset_dir / "eval.seed.jsonl")
        heldout_items_count = _count_jsonl(lab_dataset_dir / "eval.heldout.jsonl")
    else:
        _write_empty_dataset(lab_dir / "dataset")
        heldout_items_count = 0

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
        heldout_items_count=heldout_items_count,
        training_result=training_result,
        model=model,
        eval_limit=eval_limit,
    )
    verdict = _lab_verdict(
        train_allowed=train_allowed,
        training_result=training_result,
        examples_count=examples_count,
        eval_items_count=eval_items_count,
        eval_result=eval_result,
    )
    verdict = _apply_eval_readiness_to_verdict(
        verdict,
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
        "heldout_items_count": heldout_items_count,
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
        "eval_coverage": eval_result.get("coverage"),
        "eval": eval_result,
    }
    _write_json(lab_dir / "source_inventory.json", _source_inventory(source_project, accepted, rejected_reasons))
    _write_json(lab_dir / "lab_summary.json", _lab_status_summary(summary))
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


def _stability_run_summary(index: int, result: dict) -> dict:
    adapter = result.get("eval", {}).get("adapter", {})
    coverage = result.get("eval_coverage") or {}
    return {
        "index": index,
        "lab_id": result.get("lab_id"),
        "lab_dir": result.get("lab_dir"),
        "verdict": result.get("verdict"),
        "production_ready": bool(result.get("production_ready")),
        "production_blockers": result.get("production_blockers") or [],
        "full_eval_coverage": bool(coverage.get("full_eval_coverage")),
        "coverage_rate": coverage.get("coverage_rate"),
        "adapter_pass_rate": adapter.get("pass_rate"),
        "adapter_hallucination_rate": adapter.get("hallucination_rate"),
        "critical_failures": adapter.get("critical_failures"),
    }


def _lab_status_summary(summary: dict) -> dict:
    eval_gate = summary.get("eval_gate") or {}
    dataset_quality = summary.get("dataset_quality") or {}
    return {
        "lab_id": summary.get("lab_id"),
        "lab_dir": summary.get("lab_dir"),
        "source": summary.get("source"),
        "source_is_real_project_data": summary.get("source_is_real_project_data"),
        "verdict": summary.get("verdict"),
        "production_ready": bool(summary.get("production_ready")),
        "production_blockers": summary.get("production_blockers") or [],
        "strict_accepted_candidates": summary.get("strict_accepted_candidates"),
        "examples_count": summary.get("examples_count"),
        "eval_items_count": summary.get("eval_items_count"),
        "heldout_items_count": summary.get("heldout_items_count"),
        "dataset_id": summary.get("dataset_id"),
        "dataset_sha256": summary.get("dataset_sha256"),
        "training_backend": summary.get("training_backend"),
        "model": summary.get("model"),
        "training_ran": bool(summary.get("training_ran")),
        "adapter_path": summary.get("adapter_path"),
        "train_allowed": bool(summary.get("train_allowed")),
        "eval_gate": {
            "activation_allowed": bool(eval_gate.get("activation_allowed")),
            "adapter_evaluated": bool(eval_gate.get("adapter_evaluated")),
            "adapter_pass_rate": eval_gate.get("adapter_pass_rate"),
            "adapter_hallucination_rate": eval_gate.get("adapter_hallucination_rate"),
            "critical_failures": eval_gate.get("critical_failures"),
            "regression_count": eval_gate.get("regression_count"),
            "block_reasons": eval_gate.get("block_reasons") or [],
        },
        "dataset_quality": {
            "accepted_candidates": dataset_quality.get("accepted_candidates"),
            "examples_count": dataset_quality.get("examples_count"),
            "eval_items_count": dataset_quality.get("eval_items_count"),
            "examples_per_candidate": dataset_quality.get("examples_per_candidate"),
            "source_path_count": dataset_quality.get("source_path_count"),
        },
    }


def _stability_blockers(runs: list[dict]) -> list[str]:
    blockers = []
    for run in runs:
        index = run["index"]
        if run.get("verdict") != "ML_CORE_PASS":
            blockers.append(f"run_{index}_not_ml_core_pass")
        if not run.get("production_ready"):
            blockers.append(f"run_{index}_not_production_ready")
        if not run.get("full_eval_coverage"):
            blockers.append(f"run_{index}_eval_coverage_incomplete")
        if run.get("critical_failures"):
            blockers.append(f"run_{index}_critical_failures")
    return blockers


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
    if _looks_truncated_claim(candidate.claim):
        return False, "truncated_claim"
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
    validation = None
    if train_allowed:
        project_root = lab_dir.parent.parent.parent
        validation = require_valid_dataset(project_root, lab_dir / "dataset")
    training_dir = lab_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = training_dir / "adapter"
    output_identity = None
    if validation is not None and backend == "mlx":
        if adapter_path.exists() or adapter_path.is_symlink():
            raise ValueError(f"Training output already exists: {adapter_path}")
        adapter_path.mkdir(mode=0o700)
        adapter_stat = adapter_path.lstat()
        output_identity = (adapter_stat.st_dev, adapter_stat.st_ino)
    command_prefix = _mlx_command_prefix("lora") if backend == "mlx" else None
    backend_command = _training_command(
        backend=backend,
        model=model,
        dataset_dir=(
            RUNTIME_DATASET_DIR_PLACEHOLDER
            if validation is not None
            else lab_dir / "dataset"
        ),
        adapter_path=(
            RUNTIME_OUTPUT_DIR_PLACEHOLDER
            if output_identity is not None
            else adapter_path
        ),
        max_iters=max_iters,
        command_prefix=command_prefix,
    )
    command = (
        render_guarded_training_command(
            backend_command,
            project_root=project_root,
            source_dataset_dir=lab_dir / "dataset",
            snapshot_dir=lab_dir / "dataset",
            expected_binding_sha256=validation["dataset_binding_sha256"],
            trusted_loader=(
                MLX_PINNED_LOADER_CONTRACT if backend == "mlx" else None
            ),
            output_dir=adapter_path if output_identity is not None else None,
            expected_output_identity=output_identity,
        )
        if validation is not None
        else backend_command + "\n"
    )
    command_path = training_dir / "train_command.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)
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
                [str(command_path)],
                cwd=lab_dir,
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
    heldout_items_count: int = 0,
    training_result: dict,
    model: str,
    eval_limit: int = DEFAULT_LAB_EVAL_LIMIT,
) -> dict:
    eval_dir = lab_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    progress_path = eval_dir / "eval_progress.jsonl"
    progress_summary_path = eval_dir / "progress_summary.json"
    progress_path.unlink(missing_ok=True)
    seed_items = _read_jsonl(lab_dir / "dataset" / "eval.seed.jsonl")
    heldout_items = _read_jsonl(lab_dir / "dataset" / "eval.heldout.jsonl")
    all_items = [*seed_items, *heldout_items]
    _write_progress_event(progress_path, {
        "event": "eval_started",
        "total_items": len(all_items),
        "seed_items": len(seed_items),
        "heldout_items": len(heldout_items),
        "eval_limit": eval_limit,
        "backend": training_result.get("backend"),
        "training_status": training_result.get("status"),
    })
    if not training_result["training_ran"]:
        selected = all_items
        base = _evaluate_lab_items(
            all_items,
            mode="base",
            backend="fake",
            model=model,
            progress_path=progress_path,
        )
        adapter = {
            "mode": "adapter",
            "items": [],
            "items_count": eval_items_count + heldout_items_count,
            "evaluated_items_count": 0,
            "pass_rate": None,
            "hallucination_rate": None,
            "critical_failures": 0,
            "status": "not_run",
        }
        _write_progress_event(progress_path, {
            "event": "mode_skipped",
            "mode": "adapter",
            "reason": "training_not_run",
            "total_items": eval_items_count + heldout_items_count,
        })
        comparison = {
            "adapter_delta": None,
            "regression_count": 0,
            "critical_regression": False,
            "status": "adapter_not_run",
        }
    elif training_result.get("backend") == "mlx":
        selected = _select_eval_items(all_items, limit=eval_limit)
        base = _evaluate_lab_items(
            selected,
            mode="base",
            backend="mlx",
            model=model,
            progress_path=progress_path,
        )
        adapter = _evaluate_lab_items(
            selected,
            mode="adapter",
            backend="mlx",
            model=model,
            adapter_path=training_result.get("adapter_path"),
            progress_path=progress_path,
        )
        comparison = _compare_lab_eval(base, adapter)
    else:
        selected = all_items
        base = _evaluate_lab_items(
            all_items,
            mode="base",
            backend="fake",
            model=model,
            progress_path=progress_path,
        )
        adapter = _evaluate_lab_items(
            all_items,
            mode="adapter",
            backend="fake",
            model=model,
            progress_path=progress_path,
        )
        comparison = _compare_lab_eval(base, adapter)

    coverage = _eval_coverage(all_items, selected, eval_limit=eval_limit)
    progress_summary = _eval_progress_summary(
        progress_path=progress_path,
        base=base,
        adapter=adapter,
        comparison=comparison,
        coverage=coverage,
    )
    _write_json(progress_summary_path, progress_summary)
    _write_progress_event(progress_path, {
        "event": "eval_completed",
        "status": progress_summary["status"],
        "base_evaluated": progress_summary["base_evaluated"],
        "adapter_evaluated": progress_summary["adapter_evaluated"],
        "full_eval_coverage": progress_summary["full_eval_coverage"],
        "all_heldout_items_evaluated": progress_summary["all_heldout_items_evaluated"],
    })
    _write_json(eval_dir / "base_results.json", base)
    _write_json(eval_dir / "adapter_results.json", adapter)
    _write_json(eval_dir / "eval_config.json", {
        "model": model,
        "backend": training_result.get("backend"),
        "training_status": training_result.get("status"),
        "eval_items_total": eval_items_count,
        "heldout_items_total": heldout_items_count,
        "mlx_eval_item_limit": eval_limit,
        "coverage": coverage,
        "progress_path": str(progress_path),
        "progress_summary_path": str(progress_summary_path),
    })
    report = [
        "# Morpheus Lab Eval",
        "",
        f"- Eval items: `{eval_items_count}`",
        f"- Held-out eval items: `{heldout_items_count}`",
        f"- Base evaluated: `{base.get('evaluated_items_count', 0)}`",
        f"- Adapter evaluated: `{adapter.get('evaluated_items_count', 0)}`",
        f"- Base pass rate: `{base.get('pass_rate')}`",
        f"- Adapter pass rate: `{adapter.get('pass_rate')}`",
        f"- Adapter delta: `{comparison.get('adapter_delta')}`",
        f"- Critical regression: `{comparison.get('critical_regression')}`",
        f"- Progress log: `{progress_path}`",
        f"- Progress summary: `{progress_summary_path}`",
        "",
        "## Eval Coverage",
        "",
        f"- Evaluated items: `{coverage['evaluated_items_count']}`",
        f"- Eval item limit: `{coverage['eval_item_limit']}`",
        f"- Coverage rate: `{coverage['coverage_rate']}`",
        f"- Full eval coverage: `{coverage.get('full_eval_coverage')}`",
        f"- Held-out items total: `{coverage.get('heldout_items_total')}`",
        f"- Held-out items evaluated: `{coverage.get('heldout_items_evaluated')}`",
        f"- All held-out items evaluated: `{coverage.get('all_heldout_items_evaluated')}`",
        f"- Critical items total: `{coverage['critical_items_total']}`",
        f"- Critical items evaluated: `{coverage['critical_items_evaluated']}`",
        f"- All critical items evaluated: `{coverage['all_critical_items_evaluated']}`",
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
    return {
        "eval_dir": str(eval_dir),
        "base": base,
        "adapter": adapter,
        "comparison": comparison,
        "coverage": coverage,
        "progress": progress_summary,
    }


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


def _apply_eval_readiness_to_verdict(verdict: str, *, eval_result: dict) -> str:
    if verdict != "ML_CORE_PASS":
        return verdict
    comparison = eval_result.get("comparison") or {}
    if comparison.get("regression_count"):
        return "ML_CORE_PARTIAL"
    coverage = eval_result.get("coverage") or {}
    if not coverage.get("full_eval_coverage"):
        return "ML_CORE_PARTIAL"
    if not coverage.get("all_heldout_items_evaluated"):
        return "ML_CORE_PARTIAL"
    return verdict


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
    if comparison.get("regression_count"):
        blockers.append("regressions")
    coverage = eval_result.get("coverage") or {}
    if training_result.get("training_ran") and not coverage.get("full_eval_coverage"):
        blockers.append("eval_coverage_incomplete")
    if training_result.get("training_ran") and not coverage.get("all_heldout_items_evaluated"):
        blockers.append("heldout_eval_missing")
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
    coverage = eval_result.get("coverage") or {}
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
    if adapter_evaluated and not coverage.get("full_eval_coverage"):
        block_reasons.append("eval_coverage_incomplete")
    if adapter_evaluated and not coverage.get("all_heldout_items_evaluated"):
        block_reasons.append("heldout_eval_missing")
    return {
        "pass_rate_threshold": LAB_PASS_RATE_THRESHOLD,
        "hallucination_rate_threshold": LAB_HALLUCINATION_RATE_THRESHOLD,
        "adapter_evaluated": adapter_evaluated,
        "adapter_pass_rate": adapter_pass_rate,
        "adapter_hallucination_rate": adapter_hallucination_rate,
        "critical_failures": critical_failures,
        "regression_count": regression_count,
        "critical_regression": bool(comparison.get("critical_regression")),
        "eval_coverage_rate": coverage.get("coverage_rate"),
        "full_eval_coverage": bool(coverage.get("full_eval_coverage")),
        "heldout_items_total": coverage.get("heldout_items_total"),
        "heldout_items_evaluated": coverage.get("heldout_items_evaluated"),
        "all_heldout_items_evaluated": bool(coverage.get("all_heldout_items_evaluated")),
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
    progress_path: Path | None = None,
) -> dict:
    scored = []
    errors = []
    started_at = time.monotonic()
    _write_progress_event(progress_path, {
        "event": "mode_started",
        "mode": mode,
        "backend": backend,
        "model": model,
        "total_items": len(items),
        "adapter_path": adapter_path,
    })
    for index, item in enumerate(items, start=1):
        try:
            answer = _lab_answer(item, mode=mode, backend=backend, model=model, adapter_path=adapter_path)
        except (OSError, subprocess.SubprocessError, TimeoutError, ValueError) as exc:
            answer = ""
            errors.append({"question": item.get("question"), "error": str(exc)})
        scored_item = _score_lab_item(item, answer, mode=mode)
        scored.append(scored_item)
        _write_progress_event(progress_path, {
            "event": "item_evaluated",
            "mode": mode,
            "index": index,
            "total_items": len(items),
            "category": scored_item.get("category"),
            "source_candidate_id": scored_item.get("source_candidate_id"),
            "passed": scored_item.get("passed"),
            "hallucinated": scored_item.get("hallucinated"),
            "critical_failure": scored_item.get("critical_failure"),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        })
    total = len(scored)
    passed = sum(1 for item in scored if item["passed"])
    hallucinated = sum(1 for item in scored if item["hallucinated"])
    critical_failures = sum(1 for item in scored if item["critical_failure"])
    result = {
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
    _write_progress_event(progress_path, {
        "event": "mode_completed",
        "mode": mode,
        "backend": backend,
        "evaluated_items_count": total,
        "passed": passed,
        "errors_count": len(errors),
        "pass_rate": result["pass_rate"],
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    })
    return result


def _write_progress_event(progress_path: Path | None, payload: dict) -> None:
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with progress_path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        handle.flush()


def _eval_progress_summary(
    *,
    progress_path: Path,
    base: dict,
    adapter: dict,
    comparison: dict,
    coverage: dict,
) -> dict:
    adapter_evaluated = bool(adapter.get("evaluated_items_count"))
    errors_count = len(base.get("errors") or []) + len(adapter.get("errors") or [])
    if errors_count:
        status = "completed_with_errors"
    elif not adapter_evaluated:
        status = "adapter_not_run"
    else:
        status = "completed"
    return {
        "status": status,
        "progress_path": str(progress_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "base_evaluated": int(base.get("evaluated_items_count") or 0),
        "adapter_evaluated": int(adapter.get("evaluated_items_count") or 0),
        "base_pass_rate": base.get("pass_rate"),
        "adapter_pass_rate": adapter.get("pass_rate"),
        "adapter_delta": comparison.get("adapter_delta"),
        "regression_count": int(comparison.get("regression_count") or 0),
        "critical_regression": bool(comparison.get("critical_regression")),
        "full_eval_coverage": bool(coverage.get("full_eval_coverage")),
        "all_heldout_items_evaluated": bool(coverage.get("all_heldout_items_evaluated")),
        "errors_count": errors_count,
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
    if category in {"unsupported_claim_refusal", "stale_claim_correction"}:
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
    hallucinated = _answer_hallucinates(category, expected, answer)
    critical_failure = category in CRITICAL_EVAL_CATEGORIES and not passed
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
    command_decision = canonical_command_answer_passes(category, expected, answer)
    if command_decision is not None:
        return command_decision
    critical_decision = critical_answer_passes(category, expected, answer)
    if critical_decision is not None:
        return critical_decision
    if folded_expected and folded_expected in folded_answer:
        return True
    return _token_overlap(folded_expected, folded_answer) >= 0.45 or (
        folded_expected
        and SequenceMatcher(None, folded_expected, folded_answer).ratio() >= 0.55
    )


def _answer_hallucinates(category: str, expected: str, answer: str) -> bool:
    return critical_answer_hallucinates(category, expected, answer)


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
    limit = max(0, int(limit))
    if limit == 0:
        return items
    if len(items) <= limit:
        return items
    selected = []
    selected_keys = set()

    for item in items:
        if str(item.get("category") or "") not in CRITICAL_EVAL_CATEGORIES:
            continue
        key = _eval_item_key(item)
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)

    seen_categories = set()
    for item in items:
        key = _eval_item_key(item)
        if key in selected_keys:
            continue
        category = str(item.get("category") or "")
        if category in seen_categories:
            continue
        selected.append(item)
        selected_keys.add(key)
        seen_categories.add(category)
        if len(selected) >= limit:
            return selected
    for item in items:
        key = _eval_item_key(item)
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= limit:
            break
    return selected


def _eval_item_key(item: dict) -> tuple[str, str, str]:
    return (
        str(item.get("source_candidate_id") or ""),
        str(item.get("category") or ""),
        str(item.get("question") or ""),
    )


def _eval_coverage(seed_items: list[dict], selected_items: list[dict], *, eval_limit: int) -> dict:
    total = len(seed_items)
    selected_keys = {_eval_item_key(item) for item in selected_items}
    heldout_items = [
        item
        for item in seed_items
        if str(item.get("eval_split") or "") == "heldout"
    ]
    heldout_evaluated = sum(1 for item in heldout_items if _eval_item_key(item) in selected_keys)
    critical_items = [
        item
        for item in seed_items
        if str(item.get("category") or "") in CRITICAL_EVAL_CATEGORIES
    ]
    critical_evaluated = sum(1 for item in critical_items if _eval_item_key(item) in selected_keys)
    coverage_rate = round(len(selected_items) / total, 4) if total else 0.0
    return {
        "eval_items_total": total,
        "evaluated_items_count": len(selected_items),
        "eval_item_limit": eval_limit,
        "coverage_rate": coverage_rate,
        "full_eval_coverage": bool(total and len(selected_items) == total),
        "heldout_items_total": len(heldout_items),
        "heldout_items_evaluated": heldout_evaluated,
        "all_heldout_items_evaluated": bool(heldout_items and heldout_evaluated == len(heldout_items)),
        "critical_categories": sorted(CRITICAL_EVAL_CATEGORIES),
        "critical_items_total": len(critical_items),
        "critical_items_evaluated": critical_evaluated,
        "all_critical_items_evaluated": critical_evaluated == len(critical_items),
    }


def _mlx_command_prefix(tool: str) -> str | None:
    if tool == "lora":
        if importlib.util.find_spec("mlx_lm") is not None:
            return (
                f"{shlex.quote(sys.executable)} -m "
                "morpheus.core.learning.mlx_fd_loader"
            )
        return None
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
            f"- Eval coverage rate: `{gate.get('eval_coverage_rate')}`",
            f"- Full eval coverage: `{gate.get('full_eval_coverage')}`",
            f"- Held-out items total: `{gate.get('heldout_items_total')}`",
            f"- Held-out items evaluated: `{gate.get('heldout_items_evaluated')}`",
            f"- All held-out items evaluated: `{gate.get('all_heldout_items_evaluated')}`",
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
    coverage = summary.get("eval_coverage") or {}
    if coverage:
        lines.extend([
            "## Eval Coverage",
            "",
            f"- Eval items total: `{coverage['eval_items_total']}`",
            f"- Evaluated items: `{coverage['evaluated_items_count']}`",
            f"- Eval item limit: `{coverage['eval_item_limit']}`",
            f"- Coverage rate: `{coverage['coverage_rate']}`",
            f"- Full eval coverage: `{coverage.get('full_eval_coverage')}`",
            f"- Held-out items total: `{coverage.get('heldout_items_total')}`",
            f"- Held-out items evaluated: `{coverage.get('heldout_items_evaluated')}`",
            f"- All held-out items evaluated: `{coverage.get('all_heldout_items_evaluated')}`",
            f"- Critical items total: `{coverage['critical_items_total']}`",
            f"- Critical items evaluated: `{coverage['critical_items_evaluated']}`",
            f"- All critical items evaluated: `{coverage['all_critical_items_evaluated']}`",
            "",
        ])
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


def _write_stability_report(path: Path, summary: dict) -> None:
    lines = [
        "# Morpheus Lab Stability Report",
        "",
        f"- Stability ID: `{summary['stability_id']}`",
        f"- Runs: `{summary['runs_count']}`",
        f"- Backend: `{summary['backend']}`",
        f"- Model: `{summary['model']}`",
        f"- Eval limit: `{summary['eval_limit']}`",
        f"- Stability passed: `{summary['stability_passed']}`",
        f"- Verdict: `{summary['verdict']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = summary.get("stability_blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Runs", ""])
    for run in summary.get("runs", []):
        lines.extend([
            f"### Run {run['index']}",
            "",
            f"- Lab ID: `{run.get('lab_id')}`",
            f"- Verdict: `{run.get('verdict')}`",
            f"- Production ready: `{run.get('production_ready')}`",
            f"- Full eval coverage: `{run.get('full_eval_coverage')}`",
            f"- Coverage rate: `{run.get('coverage_rate')}`",
            f"- Adapter pass rate: `{run.get('adapter_pass_rate')}`",
            f"- Adapter hallucination rate: `{run.get('adapter_hallucination_rate')}`",
            f"- Critical failures: `{run.get('critical_failures')}`",
            f"- Lab dir: `{run.get('lab_dir')}`",
            "",
        ])
    path.write_text("\n".join(lines).rstrip() + "\n")


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
        "eval.heldout.jsonl",
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
    dataset_dir: Path | str | RuntimeDatasetArgument,
    adapter_path: Path | RuntimeDatasetArgument,
    max_iters: int,
    command_prefix: str | None = None,
) -> str:
    if backend == "mlx":
        prefix = command_prefix or "mlx_lm.lora"
        return (
            f"{prefix} "
            f"--model {shlex.quote(model)} "
            "--train "
            f"--data {shell_quote_training_argument(dataset_dir)} "
            f"--adapter-path {shell_quote_training_argument(adapter_path)} "
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


def _looks_truncated_claim(claim: str) -> bool:
    text = re.sub(r"^[-*]\s*", "", claim.strip())
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,")
    if not text:
        return True
    words = text.split()
    first = words[0].casefold().strip(".,:;!?()[]{}")
    last = words[-1].casefold().strip(".,:;!?()[]{}")
    if first in {
        "and",
        "or",
        "but",
        "into",
        "from",
        "with",
        "without",
        "while",
        "that",
        "which",
        "where",
    }:
        return True
    if last in {
        "and",
        "or",
        "but",
        "of",
        "to",
        "from",
        "with",
        "without",
        "while",
        "what",
        "that",
        "which",
        "where",
        "is",
        "are",
    }:
        return True
    return False


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
