"""Evaluation harness for reviewed Morpheus learning adapters."""
import base64
from collections import Counter
from contextlib import ExitStack, contextmanager
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from morpheus.core.command_contract import canonical_command_answer_passes
from morpheus.core.learning.adapter_artifacts import (
    validate_registered_adapter_artifact,
)
from morpheus.core.learning.authority import learning_authority_transaction
from morpheus.core.learning.categories import (
    BENCHMARK_CATEGORY_SCHEMA,
    CRITICAL_BENCHMARK_CATEGORIES,
)
from morpheus.core.learning.dataset_validation import (
    manifest_count,
    require_valid_dataset,
    validate_dataset,
)
from morpheus.core.learning.registry import latest_effective_dataset
from morpheus.core.learning.readiness import benchmark_readiness_gate
from morpheus.core.learning.scoring import (
    critical_answer_hallucinates,
    critical_answer_passes,
)
from morpheus.core.learning.train import DEFAULT_BASE_MODEL
from morpheus.core.portable_lock import portable_file_lock
from morpheus.core.provenance import receipt_signature_payload
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths
from morpheus.core.semantic.review import ReviewStore
from morpheus.core.state_authority import state_authority_transaction


DEFAULT_PASS_RATE_THRESHOLD = 0.8
DEFAULT_HALLUCINATION_RATE_THRESHOLD = 0.05
_BASE_EVAL_ARTIFACT_NAMES = frozenset({
    "eval_config.json",
    "eval_results.json",
    "eval_report.md",
})
_ACTIVATION_EVAL_RECEIPT_NAME = "activation_eval_receipt.json"
_ACTIVATION_EVAL_RECEIPT_SCHEMA = "morpheus-activation-eval-receipt/2"
_EVAL_PAIR_SCHEMA = "morpheus-eval-pair/1"
_ACTIVATION_EVALUATOR = {
    "name": "morpheus-learning-eval",
    "contract_version": 2,
}
_EVAL_ITEM_IDENTITY_FIELDS = (
    "category",
    "question",
    "expected_answer",
    "source_candidate_id",
    "source_path",
    "line_start",
    "line_end",
    "evidence_sha256",
    "kind",
)


@dataclass(frozen=True)
class FakeInferenceProvider:
    name: str
    quality: str = "passing"

    def infer(self, item: dict) -> str:
        if self.quality == "failing":
            return "Yes, confirmed without source. This stale project claim is current."
        category = str(item.get("category") or "")
        expected = str(item.get("expected_answer") or "")
        if category == "unsupported_claim_refusal":
            return "I cannot confirm unsupported project claims without reviewed source evidence."
        if category == "stale_claim_correction":
            return expected if expected else "No. That claim is outdated and must not be treated as active state."
        return expected


def run_learning_eval(
    project_root: Path,
    *,
    adapter_id: str | None = None,
    base_only: bool = False,
    dry_run: bool = True,
    fake_quality: str = "passing",
    dataset_id: str | None = None,
) -> dict:
    project_root = _safe_project_root(project_root)
    selected_dataset_dir = _dataset_dir_for_eval(project_root, dataset_id)
    if selected_dataset_dir is None:
        raise ValueError(
            "No trainable learning dataset manifest found. Run `morpheus learn dataset .` "
            "or `morpheus learn lab . --no-train` first."
        )
    selected_dataset_dir = selected_dataset_dir.resolve()
    review_roots = _review_authority_roots_for_dataset(
        project_root,
        selected_dataset_dir,
    )
    with state_authority_transaction(project_root):
        with _eval_review_authority_transaction(review_roots):
            current_dataset_dir = _dataset_dir_for_eval(project_root, dataset_id)
            if (
                current_dataset_dir is None
                or current_dataset_dir.resolve() != selected_dataset_dir
            ):
                raise ValueError("Dataset selection changed before eval authority lease")
            current_review_roots = _review_authority_roots_for_dataset(
                project_root,
                selected_dataset_dir,
            )
            if current_review_roots != review_roots:
                raise ValueError("Dataset review authority changed before evaluation")
            return _run_learning_eval_locked(
                project_root,
                dataset_dir=selected_dataset_dir,
                adapter_id=adapter_id,
                base_only=base_only,
                dry_run=dry_run,
                fake_quality=fake_quality,
            )


def _run_learning_eval_locked(
    project_root: Path,
    *,
    dataset_dir: Path,
    adapter_id: str | None,
    base_only: bool,
    dry_run: bool,
    fake_quality: str,
) -> dict:
    dataset_manifest = _read_json(dataset_dir / "manifest.json", "Dataset manifest")
    if manifest_count(dataset_manifest, "examples_count") <= 0:
        raise ValueError("Refusing to eval: dataset has zero examples.")
    validation = require_valid_dataset(project_root, dataset_dir, dataset_manifest)
    eval_seed_path = dataset_dir / "eval.seed.jsonl"
    if not eval_seed_path.is_file():
        raise ValueError("No eval.seed.jsonl found for latest dataset.")
    seed_items = _read_jsonl(eval_seed_path)
    if not seed_items:
        raise ValueError("Refusing to eval: eval seed is empty.")

    resolved_base_only = base_only
    resolved_adapter_id = None if base_only else adapter_id
    if not resolved_base_only and resolved_adapter_id is None:
        resolved_adapter_id = _latest_adapter_id(project_root)
        if resolved_adapter_id is None:
            resolved_base_only = True
    if resolved_adapter_id is not None:
        adapter_binding = _adapter_dataset_binding(
            project_root,
            resolved_adapter_id,
            dataset_id=dataset_manifest.get("dataset_id"),
            dataset_binding_sha256=validation["dataset_binding_sha256"],
        )
        if not adapter_binding["valid"]:
            raise ValueError(
                "Adapter dataset binding mismatch: "
                + ", ".join(adapter_binding["blockers"])
            )

    base_model = _base_model_for_eval(
        project_root,
        adapter_id=(adapter_id if resolved_base_only else resolved_adapter_id),
        dataset_id=dataset_manifest.get("dataset_id"),
        dataset_binding_sha256=validation["dataset_binding_sha256"],
    )

    provider = FakeInferenceProvider(
        name="diagnostic-fake",
        quality=fake_quality,
    )
    eval_id = _timestamp_id("eval")
    evals_root = _validated_evals_root(project_root)
    eval_dir = evals_root / eval_id
    current_manifest = _read_json(dataset_dir / "manifest.json", "Dataset manifest")
    current_validation = require_valid_dataset(
        project_root,
        dataset_dir,
        current_manifest,
    )
    if (
        current_validation["dataset_binding_sha256"]
        != validation["dataset_binding_sha256"]
    ):
        raise ValueError("Dataset binding changed while preparing evaluation.")
    benchmark_category_schema = dataset_manifest["format_versions"][
        "benchmark_categories"
    ]
    pair_config = _build_eval_pair_config(
        provider={"name": provider.name},
        evaluation_mode="diagnostic_fake",
        base_model=base_model,
    )
    pair_config_sha256 = _canonical_sha256(pair_config)
    config = {
        "eval_id": eval_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_manifest.get("dataset_id"),
        "dataset_binding_sha256": validation["dataset_binding_sha256"],
        "dataset_manifest_path": str(dataset_dir / "manifest.json"),
        "eval_seed_path": str(eval_seed_path),
        "adapter_id": resolved_adapter_id,
        "base_only": resolved_base_only,
        "dry_run": dry_run,
        "provider": {"name": provider.name},
        "diagnostic_quality": provider.quality,
        "evaluator": dict(_ACTIVATION_EVALUATOR),
        "evaluation_mode": "diagnostic_fake",
        "eval_pair_config": pair_config,
        "eval_pair_config_sha256": pair_config_sha256,
        "activation_eligible": False,
        "benchmark_category_schema": benchmark_category_schema,
        "categories": sorted({str(item["category"]) for item in seed_items}),
    }
    results_items = [_score_item(item, provider.infer(item)) for item in seed_items]
    metrics = _metrics(results_items)
    results = {
        "eval_id": eval_id,
        "created_at": config["created_at"],
        "adapter_id": resolved_adapter_id,
        "base_only": resolved_base_only,
        "dataset_id": dataset_manifest.get("dataset_id"),
        "dataset_binding_sha256": validation["dataset_binding_sha256"],
        "evaluation_mode": "diagnostic_fake",
        "eval_pair_config": pair_config,
        "eval_pair_config_sha256": pair_config_sha256,
        "activation_eligible": False,
        "benchmark_category_schema": benchmark_category_schema,
        "metrics": metrics,
        "items": results_items,
    }

    expected_contents = {
        "eval_config.json": json.dumps(config, indent=2, sort_keys=True) + "\n",
        "eval_results.json": json.dumps(results, indent=2, sort_keys=True) + "\n",
        "eval_report.md": _render_report(config, metrics, results_items),
    }
    staging_dir, staging_identity = _create_private_eval_staging(
        evals_root,
        eval_id,
    )
    for name, content in expected_contents.items():
        _write_private_text(staging_dir / name, content)
    _publish_staged_eval(
        staging_dir,
        eval_dir,
        staging_identity=staging_identity,
        expected_contents=expected_contents,
    )
    config_path = eval_dir / "eval_config.json"
    results_path = eval_dir / "eval_results.json"
    report_path = eval_dir / "eval_report.md"
    if resolved_adapter_id:
        adapter_eval_path = (
            project_root
            / ".morpheus"
            / "training"
            / "adapters"
            / resolved_adapter_id
            / "eval_results.json"
        )
        _write_adapter_eval_results(adapter_eval_path, results)
    return {
        "eval_id": eval_id,
        "eval_dir": str(eval_dir),
        "eval_config_path": str(config_path),
        "eval_results_path": str(results_path),
        "eval_report_path": str(report_path),
        "adapter_id": resolved_adapter_id,
        "base_only": resolved_base_only,
        "dry_run": dry_run,
        "metrics": metrics,
    }


def check_activation_gate(
    project_root: Path,
    adapter_id: str,
    *,
    pass_rate_threshold: float = DEFAULT_PASS_RATE_THRESHOLD,
    hallucination_rate_threshold: float = DEFAULT_HALLUCINATION_RATE_THRESHOLD,
    eval_id: str | None = None,
) -> dict:
    project_root = _safe_project_root(project_root)
    try:
        latest_eval = _latest_eval_for_adapter(
            project_root,
            adapter_id,
            eval_id=eval_id,
        )
    except (OSError, ValueError) as exc:
        return {
            "allowed": False,
            "reason": "eval_registry_invalid",
            "adapter_id": adapter_id,
            "eval_id": eval_id,
            "error": str(exc),
        }
    if latest_eval is None:
        return {"allowed": False, "reason": "missing_eval", "adapter_id": adapter_id}
    adapter_dir = (
        project_root / ".morpheus" / "training" / "adapters" / adapter_id
    )
    adapter_artifact = validate_registered_adapter_artifact(
        adapter_dir,
        expected_adapter_id=adapter_id,
    )
    if not adapter_artifact["valid"]:
        return {
            "allowed": False,
            "reason": "adapter_artifact_invalid",
            "adapter_id": adapter_id,
            "eval_id": latest_eval.name,
            "adapter_artifact_blockers": adapter_artifact["blockers"],
        }
    try:
        config = _read_json(latest_eval / "eval_config.json", "Eval config")
        results = _read_json(latest_eval / "eval_results.json", "Eval results")
    except (OSError, ValueError) as exc:
        return {
            "allowed": False,
            "reason": "eval_artifacts_invalid",
            "adapter_id": adapter_id,
            "eval_id": latest_eval.name,
            "eval_dir": str(latest_eval),
            "error": str(exc),
        }
    if not _eval_artifact_identity_is_valid(
        latest_eval,
        config,
        results,
        expected_adapter_id=adapter_id,
        expected_base_only=False,
        require_current_category_schema=True,
        require_current_pair_identity=True,
    ):
        return {
            "allowed": False,
            "reason": "eval_artifact_identity_mismatch",
            "adapter_id": adapter_id,
            "eval_id": latest_eval.name,
            "eval_dir": str(latest_eval),
        }
    dataset_id = results.get("dataset_id")
    dataset_binding = results.get("dataset_binding_sha256")
    current_dataset = _current_dataset_validation(
        project_root,
        dataset_id,
        dataset_binding,
    )
    if not current_dataset["valid"]:
        return {
            "allowed": False,
            "reason": "dataset_not_current",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "dataset_id": dataset_id,
            "dataset_binding_sha256": dataset_binding,
            "dataset_blockers": current_dataset["blockers"],
        }
    adapter_binding = _adapter_dataset_binding(
        project_root,
        adapter_id,
        dataset_id=dataset_id,
        dataset_binding_sha256=dataset_binding,
    )
    if not adapter_binding["valid"]:
        return {
            "allowed": False,
            "reason": "adapter_dataset_binding_mismatch",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "dataset_id": dataset_id,
            "dataset_binding_sha256": dataset_binding,
            "adapter_blockers": adapter_binding["blockers"],
        }
    eval_pair = _eval_pair_identity(config, results)
    pair_model = (
        eval_pair.get("config", {}).get("model")
        if isinstance(eval_pair.get("config"), dict)
        else {}
    )
    adapter_base_model = adapter_binding.get("base_model")
    if (
        not isinstance(pair_model, dict)
        or not isinstance(adapter_base_model, str)
        or not adapter_base_model.strip()
        or pair_model.get("base_model") != adapter_base_model.strip()
    ):
        return {
            "allowed": False,
            "reason": "adapter_eval_pair_model_mismatch",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "eval_pair_config_sha256": eval_pair.get("sha256"),
        }
    if not _eval_is_activation_eligible(config, results):
        provider = config.get("provider") if isinstance(config.get("provider"), dict) else {}
        return {
            "allowed": False,
            "reason": "diagnostic_eval_not_activation_eligible",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id") or config.get("eval_id"),
            "evaluation_mode": config.get("evaluation_mode") or "unknown",
            "provider": str(provider.get("name") or "") or None,
        }
    metrics = results.get("metrics") if isinstance(results.get("metrics"), dict) else {}
    if not _eval_metrics_are_valid(metrics):
        return {
            "allowed": False,
            "reason": "invalid_eval_metrics",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
        }
    if not _eval_matches_dataset_coverage(
        results,
        current_dataset.get("eval_coverage"),
    ):
        return {
            "allowed": False,
            "reason": "eval_dataset_coverage_mismatch",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "dataset_id": dataset_id,
        }
    if not _eval_metrics_match_result_items(results):
        return {
            "allowed": False,
            "reason": "invalid_eval_metrics",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
        }
    eval_receipt = _validate_activation_eval_receipt(
        project_root,
        latest_eval,
        config,
        results,
        current_dataset=current_dataset,
    )
    if not eval_receipt["valid"]:
        return {
            "allowed": False,
            "reason": "eval_activation_receipt_invalid",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "eval_receipt_blockers": eval_receipt["blockers"],
        }
    total_items = int(metrics["total_items"])
    passed_items = int(metrics["passed_items"])
    hallucinated_items = int(metrics["hallucinated_items"])
    critical_outdated_failures = int(metrics["critical_outdated_claim_failures"])
    metric_failure = None
    if _rate_below_threshold(
        passed_items,
        total_items,
        pass_rate_threshold,
    ):
        metric_failure = {
            "allowed": False,
            "reason": "pass_rate_below_threshold",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "metrics": metrics,
        }
    elif _rate_above_threshold(
        hallucinated_items,
        total_items,
        hallucination_rate_threshold,
    ):
        metric_failure = {
            "allowed": False,
            "reason": "hallucination_rate_above_threshold",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "metrics": metrics,
        }
    elif critical_outdated_failures:
        metric_failure = {
            "allowed": False,
            "reason": "critical_outdated_claim_failure",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "metrics": metrics,
        }

    def base_failure(failure: dict) -> dict:
        return metric_failure if metric_failure is not None else failure

    if not isinstance(dataset_id, str) or not dataset_id:
        return base_failure({
            "allowed": False,
            "reason": "missing_base_eval",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "dataset_id": dataset_id,
            "metrics": metrics,
        })
    try:
        base_eval = _latest_base_eval_for_dataset(
            project_root,
            dataset_id,
            dataset_binding_sha256=str(dataset_binding),
            eval_pair_config_sha256=str(eval_pair["sha256"]),
        )
    except (OSError, ValueError) as exc:
        return base_failure({
            "allowed": False,
            "reason": "base_eval_registry_invalid",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "dataset_id": dataset_id,
            "metrics": metrics,
            "error": str(exc),
        })
    if base_eval is None:
        missing_base_reason = (
            "missing_matching_base_eval"
            if _has_base_eval_for_dataset(
                project_root,
                dataset_id,
                dataset_binding_sha256=str(dataset_binding),
            )
            else "missing_base_eval"
        )
        return base_failure({
            "allowed": False,
            "reason": missing_base_reason,
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "dataset_id": dataset_id,
            "eval_pair_config_sha256": eval_pair["sha256"],
            "metrics": metrics,
        })
    try:
        base_config = _read_json(base_eval / "eval_config.json", "Base eval config")
        base_results = _read_json(base_eval / "eval_results.json", "Base eval results")
    except (OSError, ValueError) as exc:
        return base_failure({
            "allowed": False,
            "reason": "base_eval_artifacts_invalid",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_eval.name,
            "base_eval_dir": str(base_eval),
            "dataset_id": dataset_id,
            "metrics": metrics,
            "error": str(exc),
        })
    if not _eval_artifact_identity_is_valid(
        base_eval,
        base_config,
        base_results,
        expected_adapter_id=None,
        expected_base_only=True,
        require_current_category_schema=True,
        require_current_pair_identity=True,
    ):
        return base_failure({
            "allowed": False,
            "reason": "base_eval_artifact_identity_mismatch",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_eval.name,
            "dataset_id": dataset_id,
            "base_eval_dir": str(base_eval),
            "metrics": metrics,
        })
    base_pair = _eval_pair_identity(base_config, base_results)
    if base_pair.get("sha256") != eval_pair.get("sha256"):
        return base_failure({
            "allowed": False,
            "reason": "base_eval_pair_identity_mismatch",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_results.get("eval_id"),
            "dataset_id": dataset_id,
            "eval_pair_config_sha256": eval_pair.get("sha256"),
            "base_eval_pair_config_sha256": base_pair.get("sha256"),
            "metrics": metrics,
        })
    if (
        base_results.get("dataset_id") != dataset_id
        or base_results.get("dataset_binding_sha256") != dataset_binding
    ):
        return base_failure({
            "allowed": False,
            "reason": "base_eval_dataset_binding_mismatch",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_results.get("eval_id"),
            "dataset_id": dataset_id,
            "dataset_binding_sha256": dataset_binding,
            "metrics": metrics,
        })
    if not _eval_is_activation_eligible(base_config, base_results):
        return base_failure({
            "allowed": False,
            "reason": "diagnostic_base_eval_not_activation_eligible",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_results.get("eval_id") or base_config.get("eval_id"),
            "dataset_id": dataset_id,
            "metrics": metrics,
        })
    base_metrics = (
        base_results.get("metrics")
        if isinstance(base_results.get("metrics"), dict)
        else {}
    )
    if not _eval_metrics_are_valid(base_metrics):
        return base_failure({
            "allowed": False,
            "reason": "invalid_base_eval_metrics",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_results.get("eval_id"),
            "dataset_id": dataset_id,
            "metrics": metrics,
        })
    if not _eval_matches_dataset_coverage(
        base_results,
        current_dataset.get("eval_coverage"),
    ):
        return base_failure({
            "allowed": False,
            "reason": "base_eval_dataset_coverage_mismatch",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_results.get("eval_id"),
            "dataset_id": dataset_id,
            "metrics": metrics,
        })
    if not _eval_metrics_match_result_items(base_results):
        return base_failure({
            "allowed": False,
            "reason": "invalid_base_eval_metrics",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_results.get("eval_id"),
            "dataset_id": dataset_id,
            "metrics": metrics,
        })
    base_receipt = _validate_activation_eval_receipt(
        project_root,
        base_eval,
        base_config,
        base_results,
        current_dataset=current_dataset,
    )
    if not base_receipt["valid"]:
        return base_failure({
            "allowed": False,
            "reason": "base_eval_activation_receipt_invalid",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_results.get("eval_id"),
            "dataset_id": dataset_id,
            "metrics": metrics,
            "base_eval_receipt_blockers": base_receipt["blockers"],
        })
    comparison = _eval_category_comparison(base_results, results)
    if metric_failure is not None:
        return {
            **metric_failure,
            "base_eval_id": base_results.get("eval_id"),
            "dataset_binding_sha256": dataset_binding,
            "eval_pair_config_sha256": eval_pair["sha256"],
            "category_deltas": comparison["category_deltas"],
            "category_regressions": comparison["category_regressions"],
            "critical_regressions": comparison["critical_regressions"],
        }
    if comparison["critical_regressions"]:
        return {
            "allowed": False,
            "reason": "critical_category_regression",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_results.get("eval_id"),
            "dataset_id": dataset_id,
            "dataset_binding_sha256": dataset_binding,
            "eval_pair_config_sha256": eval_pair["sha256"],
            "metrics": metrics,
            "category_deltas": comparison["category_deltas"],
            "category_regressions": comparison["category_regressions"],
            "critical_regressions": comparison["critical_regressions"],
        }
    benchmark_gate = benchmark_readiness_gate(
        current_dataset.get("manifest"),
        current_dataset,
    )
    if not benchmark_gate["allowed"]:
        return {
            "allowed": False,
            "reason": "benchmark_blocked",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "base_eval_id": base_results.get("eval_id"),
            "dataset_id": dataset_id,
            "dataset_binding_sha256": dataset_binding,
            "eval_pair_config_sha256": eval_pair["sha256"],
            "dataset_dir": current_dataset.get("dataset_dir"),
            "metrics": metrics,
            "category_deltas": comparison["category_deltas"],
            "category_regressions": comparison["category_regressions"],
            "critical_regressions": [],
            "benchmark_blockers": benchmark_gate["blockers"],
            "benchmark_gate": benchmark_gate,
        }
    return {
        "allowed": True,
        "reason": "passed",
        "adapter_id": adapter_id,
        "eval_id": results.get("eval_id"),
        "base_eval_id": base_results.get("eval_id"),
        "dataset_id": dataset_id,
        "dataset_binding_sha256": dataset_binding,
        "eval_pair_config_sha256": eval_pair["sha256"],
        "dataset_dir": current_dataset.get("dataset_dir"),
        "eval_activation_receipt_sha256": eval_receipt["sha256"],
        "base_eval_activation_receipt_sha256": base_receipt["sha256"],
        "weight_artifact": adapter_artifact["artifact"],
        "metrics": metrics,
        "category_deltas": comparison["category_deltas"],
        "category_regressions": comparison["category_regressions"],
        "critical_regressions": [],
        "benchmark_gate": benchmark_gate,
    }


def _eval_artifact_identity_is_valid(
    eval_dir: Path,
    config: dict,
    results: dict,
    *,
    expected_adapter_id: str | None,
    expected_base_only: bool,
    require_current_category_schema: bool = False,
    require_current_pair_identity: bool = False,
) -> bool:
    eval_id = config.get("eval_id")
    dataset_id = config.get("dataset_id")
    dataset_binding = config.get("dataset_binding_sha256")
    adapter_id = config.get("adapter_id")
    base_only = config.get("base_only")
    category_schema = config.get("benchmark_category_schema")
    pair_identity = _eval_pair_identity(config, results)
    return bool(
        isinstance(eval_id, str)
        and eval_id
        and eval_id == eval_dir.name
        and results.get("eval_id") == eval_id
        and isinstance(dataset_id, str)
        and dataset_id
        and results.get("dataset_id") == dataset_id
        and _valid_sha256(dataset_binding)
        and results.get("dataset_binding_sha256") == dataset_binding
        and adapter_id == expected_adapter_id
        and results.get("adapter_id") == adapter_id
        and type(base_only) is bool
        and base_only is expected_base_only
        and type(results.get("base_only")) is bool
        and results.get("base_only") is base_only
        and results.get("benchmark_category_schema") == category_schema
        and pair_identity["valid"]
        and (
            not require_current_pair_identity
            or pair_identity["current"]
        )
        and (
            not require_current_category_schema
            or category_schema == BENCHMARK_CATEGORY_SCHEMA
        )
    )


def _build_eval_pair_config(
    *,
    provider: dict,
    evaluation_mode: str,
    base_model: str,
    inference_config: dict | None = None,
) -> dict:
    """Build the role-neutral configuration shared by a base/adapter eval pair."""
    provider_name = provider.get("name") if isinstance(provider, dict) else None
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ValueError("Eval pair provider identity is invalid")
    if not isinstance(evaluation_mode, str) or not evaluation_mode.strip():
        raise ValueError("Eval pair evaluation mode is invalid")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("Eval pair base model identity is invalid")
    resolved_inference_config = (
        {} if inference_config is None else inference_config
    )
    if not isinstance(resolved_inference_config, dict):
        raise ValueError("Eval pair inference config is invalid")
    pair_config = {
        "schema": _EVAL_PAIR_SCHEMA,
        "provider": provider,
        "evaluation_mode": evaluation_mode,
        "evaluator": dict(_ACTIVATION_EVALUATOR),
        "model": {
            "base_model": base_model,
            "inference_config": resolved_inference_config,
        },
    }
    try:
        _canonical_sha256(pair_config)
    except (TypeError, ValueError) as exc:
        raise ValueError("Eval pair config must be canonical JSON") from exc
    return pair_config


def _eval_pair_identity(config: dict, results: dict) -> dict:
    config_pair = config.get("eval_pair_config")
    results_pair = results.get("eval_pair_config")
    config_sha = config.get("eval_pair_config_sha256")
    results_sha = results.get("eval_pair_config_sha256")
    values = (config_pair, results_pair, config_sha, results_sha)
    if all(value is None for value in values):
        return {
            "valid": True,
            "current": False,
            "sha256": None,
            "config": None,
        }
    if not isinstance(config_pair, dict) or results_pair != config_pair:
        return {"valid": False, "current": False, "sha256": None, "config": None}
    try:
        expected_sha = _canonical_sha256(config_pair)
    except (TypeError, ValueError):
        return {"valid": False, "current": False, "sha256": None, "config": None}
    provider = config_pair.get("provider")
    model = config_pair.get("model")
    inference_config = (
        model.get("inference_config") if isinstance(model, dict) else None
    )
    base_model = model.get("base_model") if isinstance(model, dict) else None
    valid = bool(
        config_pair.get("schema") == _EVAL_PAIR_SCHEMA
        and isinstance(provider, dict)
        and isinstance(provider.get("name"), str)
        and provider.get("name").strip()
        and provider == config.get("provider")
        and isinstance(config_pair.get("evaluation_mode"), str)
        and config_pair.get("evaluation_mode")
        and config_pair.get("evaluation_mode") == config.get("evaluation_mode")
        and config_pair.get("evaluation_mode") == results.get("evaluation_mode")
        and config_pair.get("evaluator") == _ACTIVATION_EVALUATOR
        and config.get("evaluator") == _ACTIVATION_EVALUATOR
        and isinstance(base_model, str)
        and base_model.strip()
        and isinstance(inference_config, dict)
        and isinstance(config_sha, str)
        and config_sha == expected_sha
        and results_sha == expected_sha
    )
    return {
        "valid": valid,
        "current": valid,
        "sha256": expected_sha if valid else None,
        "config": config_pair if valid else None,
    }


def _config_eval_pair_sha256(config: object) -> str | None:
    """Return a trustworthy config-only pair id for registry relevance checks."""
    if not isinstance(config, dict):
        return None
    pair_config = config.get("eval_pair_config")
    claimed_sha = config.get("eval_pair_config_sha256")
    if not isinstance(pair_config, dict) or not isinstance(claimed_sha, str):
        return None
    try:
        expected_sha = _canonical_sha256(pair_config)
    except (TypeError, ValueError):
        return None
    provider = pair_config.get("provider")
    model = pair_config.get("model")
    return expected_sha if bool(
        claimed_sha == expected_sha
        and pair_config.get("schema") == _EVAL_PAIR_SCHEMA
        and isinstance(provider, dict)
        and isinstance(provider.get("name"), str)
        and provider.get("name").strip()
        and provider == config.get("provider")
        and pair_config.get("evaluation_mode") == config.get("evaluation_mode")
        and pair_config.get("evaluator") == _ACTIVATION_EVALUATOR
        and config.get("evaluator") == _ACTIVATION_EVALUATOR
        and isinstance(model, dict)
        and isinstance(model.get("base_model"), str)
        and model.get("base_model").strip()
        and isinstance(model.get("inference_config"), dict)
    ) else None


def _eval_is_activation_eligible(config: dict, results: dict) -> bool:
    provider = config.get("provider") if isinstance(config.get("provider"), dict) else {}
    provider_name = provider.get("name")
    evaluation_mode = config.get("evaluation_mode")
    return bool(
        config.get("activation_eligible") is True
        and results.get("activation_eligible") is True
        and config.get("dry_run") is False
        and isinstance(provider_name, str)
        and provider_name.strip()
        and not provider_name.casefold().startswith("fake-")
        and isinstance(evaluation_mode, str)
        and evaluation_mode.strip()
        and evaluation_mode != "diagnostic_fake"
        and results.get("evaluation_mode") == evaluation_mode
    )


def _build_activation_eval_receipt_bytes(
    project_root: Path,
    eval_dir: Path,
) -> bytes:
    """Build a signed receipt for an already completed non-diagnostic eval bundle.

    Callers publishing a real evaluator result include these returned bytes in
    the same private staging directory as the config, results, and report.
    Diagnostic fake evaluations never call this function.
    """
    config_bytes = _read_stable_regular_bytes(
        eval_dir / "eval_config.json",
        "Eval config",
    )
    results_bytes = _read_stable_regular_bytes(
        eval_dir / "eval_results.json",
        "Eval results",
    )
    config = _json_object_from_bytes(config_bytes, "Eval config")
    results = _json_object_from_bytes(results_bytes, "Eval results")
    dataset = _current_dataset_validation(
        project_root,
        results.get("dataset_id"),
        results.get("dataset_binding_sha256"),
    )
    if not dataset.get("valid"):
        raise ValueError("Cannot receipt an eval for a non-current dataset")
    seed_path = Path(str(dataset["dataset_dir"])) / "eval.seed.jsonl"
    seed_bytes = _read_stable_regular_bytes(seed_path, "Eval seed")
    seed_items = _jsonl_objects_from_bytes(seed_bytes, "Eval seed")
    claims = _activation_eval_receipt_claims(
        config,
        results,
        config_bytes=config_bytes,
        results_bytes=results_bytes,
        seed_bytes=seed_bytes,
        seed_items=seed_items,
    )
    receipt = {
        **claims,
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }
    private_key_path = project_root / ".morpheus" / "keys" / "local.key"
    private_bytes = _read_stable_regular_bytes(
        private_key_path,
        "Activation eval signing key",
        private=True,
    )
    try:
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
    except ValueError as exc:
        raise ValueError("Activation eval signing key is invalid") from exc
    signature = private_key.sign(receipt_signature_payload(receipt))
    receipt["signature"] = {
        "algo": "ed25519",
        "key_id": "local",
        "signature_b64": base64.b64encode(signature).decode(),
    }
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


def _validate_activation_eval_receipt(
    project_root: Path,
    eval_dir: Path,
    config: dict,
    results: dict,
    *,
    current_dataset: dict,
) -> dict:
    receipt_path = eval_dir / _ACTIVATION_EVAL_RECEIPT_NAME
    try:
        config_bytes = _read_stable_regular_bytes(
            eval_dir / "eval_config.json",
            "Eval config",
        )
        results_bytes = _read_stable_regular_bytes(
            eval_dir / "eval_results.json",
            "Eval results",
        )
        if (
            _json_object_from_bytes(config_bytes, "Eval config") != config
            or _json_object_from_bytes(results_bytes, "Eval results") != results
        ):
            raise ValueError("eval artifacts changed during receipt validation")
        receipt_bytes = _read_stable_regular_bytes(
            receipt_path,
            "Activation eval receipt",
            private=True,
        )
        receipt = _json_object_from_bytes(
            receipt_bytes,
            "Activation eval receipt",
        )
        dataset_dir = current_dataset.get("dataset_dir")
        if not isinstance(dataset_dir, str) or not dataset_dir:
            raise ValueError("current dataset path is missing")
        seed_path = Path(dataset_dir) / "eval.seed.jsonl"
        seed_bytes = _read_stable_regular_bytes(seed_path, "Eval seed")
        seed_items = _jsonl_objects_from_bytes(seed_bytes, "Eval seed")
        expected = _activation_eval_receipt_claims(
            config,
            results,
            config_bytes=config_bytes,
            results_bytes=results_bytes,
            seed_bytes=seed_bytes,
            seed_items=seed_items,
        )
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("receipt claims do not match eval artifacts")
        issued_at = receipt.get("issued_at")
        if not isinstance(issued_at, str) or not issued_at:
            raise ValueError("receipt issued_at is invalid")
        signature = receipt.get("signature")
        if not isinstance(signature, dict):
            raise ValueError("receipt signature is missing")
        if signature.get("algo") != "ed25519" or signature.get("key_id") != "local":
            raise ValueError("receipt signature identity is invalid")
        signature_b64 = signature.get("signature_b64")
        if not isinstance(signature_b64, str):
            raise ValueError("receipt signature is invalid")
        try:
            signature_bytes = base64.b64decode(signature_b64, validate=True)
        except ValueError as exc:
            raise ValueError("receipt signature encoding is invalid") from exc
        public_key = _activation_eval_public_key(project_root)
        try:
            public_key.verify(signature_bytes, receipt_signature_payload(receipt))
        except InvalidSignature as exc:
            raise ValueError("receipt signature verification failed") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"valid": False, "blockers": [str(exc)], "sha256": None}
    return {
        "valid": True,
        "blockers": [],
        "sha256": sha256(receipt_bytes).hexdigest(),
    }


def _activation_eval_receipt_claims(
    config: dict,
    results: dict,
    *,
    config_bytes: bytes,
    results_bytes: bytes,
    seed_bytes: bytes,
    seed_items: list[dict],
) -> dict:
    if not _eval_is_activation_eligible(config, results):
        raise ValueError("diagnostic evals cannot receive activation provenance")
    eval_id = config.get("eval_id")
    adapter_id = config.get("adapter_id")
    base_only = config.get("base_only")
    dataset_id = config.get("dataset_id")
    dataset_binding = config.get("dataset_binding_sha256")
    benchmark_category_schema = config.get("benchmark_category_schema")
    eval_pair = _eval_pair_identity(config, results)
    if (
        benchmark_category_schema != BENCHMARK_CATEGORY_SCHEMA
        or results.get("benchmark_category_schema")
        != BENCHMARK_CATEGORY_SCHEMA
    ):
        raise ValueError("activation benchmark category schema is invalid")
    if (
        not _is_canonical_eval_id(eval_id)
        or results.get("eval_id") != eval_id
        or not isinstance(dataset_id, str)
        or not dataset_id
        or results.get("dataset_id") != dataset_id
        or not _valid_sha256(dataset_binding)
        or results.get("dataset_binding_sha256") != dataset_binding
        or type(base_only) is not bool
        or results.get("base_only") is not base_only
        or results.get("adapter_id") != adapter_id
        or (
            (base_only is True and adapter_id is not None)
            or (
                base_only is False
                and (not isinstance(adapter_id, str) or not adapter_id)
            )
        )
    ):
        raise ValueError("activation eval artifact identity is invalid")
    if config.get("evaluator") != _ACTIVATION_EVALUATOR:
        raise ValueError("activation evaluator identity is invalid")
    if not eval_pair["valid"] or not eval_pair["current"]:
        raise ValueError("activation eval pair identity is invalid")
    provider = config.get("provider")
    if not isinstance(provider, dict):
        raise ValueError("activation eval provider identity is invalid")
    result_items = results.get("items")
    if not isinstance(result_items, list) or not result_items:
        raise ValueError("activation eval result items are invalid")
    if not _eval_metrics_match_result_items(results):
        raise ValueError(
            "activation eval results do not match canonical scoring"
        )
    result_identities, result_digest = _eval_item_identity_digest(result_items)
    seed_identities, seed_digest = _eval_item_identity_digest(seed_items)
    if result_identities != seed_identities or result_digest != seed_digest:
        raise ValueError("activation eval items do not match the bound eval seed")
    return {
        "schema": _ACTIVATION_EVAL_RECEIPT_SCHEMA,
        "eval_id": eval_id,
        "evaluator": dict(_ACTIVATION_EVALUATOR),
        "evaluation_mode": config.get("evaluation_mode"),
        "provider": provider,
        "adapter_id": adapter_id,
        "base_only": base_only,
        "dataset_id": dataset_id,
        "dataset_binding_sha256": dataset_binding,
        "benchmark_category_schema": benchmark_category_schema,
        "eval_pair_config": eval_pair["config"],
        "eval_pair_config_sha256": eval_pair["sha256"],
        "item_identities": result_identities,
        "items_sha256": result_digest,
        "eval_seed_sha256": sha256(seed_bytes).hexdigest(),
        "eval_config_sha256": sha256(config_bytes).hexdigest(),
        "eval_results_sha256": sha256(results_bytes).hexdigest(),
    }


def _eval_item_identity_digest(items: list[dict]) -> tuple[list[str], str]:
    projections = []
    identities = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("activation eval item must be a JSON object")
        projection = {field: item.get(field) for field in _EVAL_ITEM_IDENTITY_FIELDS}
        if any(
            not isinstance(projection[field], str) or not projection[field]
            for field in ("category", "question", "expected_answer", "kind")
        ):
            raise ValueError("activation eval item identity is incomplete")
        if any(
            projection[field] is not None
            and not isinstance(projection[field], str)
            for field in ("source_candidate_id", "source_path")
        ):
            raise ValueError("activation eval source identity is invalid")
        if projection["source_candidate_id"] is not None and (
            not projection["source_path"]
            or type(projection["line_start"]) is not int
            or projection["line_start"] < 1
            or type(projection["line_end"]) is not int
            or projection["line_end"] < projection["line_start"]
            or not _valid_sha256(projection["evidence_sha256"])
        ):
            raise ValueError("activation eval source span identity is invalid")
        projections.append(projection)
        identities.append(_canonical_sha256(projection))
    return identities, _canonical_sha256(projections)


def _activation_eval_public_key(project_root: Path) -> ed25519.Ed25519PublicKey:
    keys_dir = project_root / ".morpheus" / "keys"
    public_path = keys_dir / "local.pub"
    private_path = keys_dir / "local.key"
    try:
        if public_path.is_file():
            public_bytes = _read_stable_regular_bytes(
                public_path,
                "Activation eval public key",
            )
            return ed25519.Ed25519PublicKey.from_public_bytes(public_bytes)
        private_bytes = _read_stable_regular_bytes(
            private_path,
            "Activation eval signing key",
            private=True,
        )
        return ed25519.Ed25519PrivateKey.from_private_bytes(
            private_bytes
        ).public_key()
    except (OSError, ValueError) as exc:
        raise ValueError("Activation eval verification key is invalid") from exc


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json_object_from_bytes(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} invalid: expected JSON object")
    return value


def _jsonl_objects_from_bytes(payload: bytes, label: str) -> list[dict]:
    try:
        values = [
            json.loads(line)
            for line in payload.decode().splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} invalid: {exc}") from exc
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"{label} invalid: expected non-empty JSON objects")
    return values


def _read_stable_regular_bytes(
    path: Path,
    label: str,
    *,
    private: bool = False,
) -> bytes:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if private and os.name != "nt" and stat.S_IMODE(before.st_mode) != 0o600:
            raise ValueError(f"{label} permissions must be 0600")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or identity != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _eval_metrics_are_valid(metrics: dict) -> bool:
    total_items = metrics.get("total_items")
    passed_items = metrics.get("passed_items")
    hallucinated_items = metrics.get("hallucinated_items")
    critical_failures = metrics.get("critical_outdated_claim_failures")
    if not _is_positive_int(total_items):
        return False
    if not all(
        _count_fits_total(value, total_items)
        for value in (passed_items, hallucinated_items, critical_failures)
    ):
        return False
    if not _rate_matches_count(metrics.get("pass_rate"), passed_items, total_items):
        return False
    if not _rate_matches_count(
        metrics.get("hallucination_rate"),
        hallucinated_items,
        total_items,
    ):
        return False
    by_category = metrics.get("by_category")
    if not isinstance(by_category, dict) or not by_category:
        return False
    category_totals = {
        "total_items": 0,
        "passed_items": 0,
        "hallucinated_items": 0,
        "critical_failures": 0,
    }
    for category_metrics in by_category.values():
        if not isinstance(category_metrics, dict):
            return False
        category_total = category_metrics.get("total_items")
        category_passed = category_metrics.get("passed_items")
        category_hallucinated = category_metrics.get("hallucinated_items")
        category_critical = category_metrics.get("critical_failures")
        if not _is_positive_int(category_total):
            return False
        if not all(
            _count_fits_total(value, category_total)
            for value in (
                category_passed,
                category_hallucinated,
                category_critical,
            )
        ):
            return False
        if not _rate_matches_count(
            category_metrics.get("pass_rate"),
            category_passed,
            category_total,
        ):
            return False
        if not _rate_matches_count(
            category_metrics.get("hallucination_rate"),
            category_hallucinated,
            category_total,
        ):
            return False
        category_totals["total_items"] += category_total
        category_totals["passed_items"] += category_passed
        category_totals["hallucinated_items"] += category_hallucinated
        category_totals["critical_failures"] += category_critical
    return bool(
        category_totals["total_items"] == total_items
        and category_totals["passed_items"] == passed_items
        and category_totals["hallucinated_items"] == hallucinated_items
        and category_totals["critical_failures"] == critical_failures
    )


def _eval_matches_dataset_coverage(results: dict, expected: object) -> bool:
    if not isinstance(expected, dict):
        return False
    expected_total = expected.get("total_items")
    expected_categories = expected.get("by_category")
    if (
        not _is_positive_int(expected_total)
        or not isinstance(expected_categories, dict)
        or not expected_categories
        or any(
            not isinstance(category, str)
            or not category
            or not _is_positive_int(count)
            for category, count in expected_categories.items()
        )
        or sum(expected_categories.values()) != expected_total
    ):
        return False
    metrics = results.get("metrics")
    items = results.get("items")
    if not isinstance(metrics, dict) or not isinstance(items, list):
        return False
    by_category = metrics.get("by_category")
    if not isinstance(by_category, dict):
        return False
    metric_categories = {}
    for category, category_metrics in by_category.items():
        if not isinstance(category_metrics, dict):
            return False
        metric_categories[category] = category_metrics.get("total_items")
    item_categories = Counter()
    for item in items:
        if not isinstance(item, dict):
            return False
        category = item.get("category")
        if not isinstance(category, str) or not category:
            return False
        item_categories[category] += 1
    return bool(
        metrics.get("total_items") == expected_total
        and metric_categories == expected_categories
        and len(items) == expected_total
        and dict(sorted(item_categories.items())) == expected_categories
    )


def _eval_metrics_match_result_items(results: dict) -> bool:
    metrics = results.get("metrics")
    items = results.get("items")
    if not isinstance(metrics, dict) or not isinstance(items, list) or not items:
        return False
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("category"), str)
            or not item["category"].strip()
            or not isinstance(item.get("expected_answer"), str)
            or not item["expected_answer"].strip()
            or not isinstance(item.get("answer"), str)
            or type(item.get("passed")) is not bool
            or type(item.get("hallucinated")) is not bool
            or type(item.get("critical_outdated_claim_failure")) is not bool
        ):
            return False
        canonical = _score_item(item, item["answer"])
        if any(
            item[field] != canonical[field]
            for field in (
                "passed",
                "hallucinated",
                "critical_outdated_claim_failure",
            )
        ):
            return False
    return metrics == _metrics(items)


def _is_bounded_rate(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _rate_matches_count(rate: object, count: int, total: int) -> bool:
    return bool(
        _is_bounded_rate(rate)
        and math.isclose(
            float(rate),
            round(count / total, 4),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )


def _rate_below_threshold(count: int, total: int, threshold: float) -> bool:
    threshold_fraction = Fraction(str(float(threshold)))
    return (
        count * threshold_fraction.denominator
        < threshold_fraction.numerator * total
    )


def _rate_above_threshold(count: int, total: int, threshold: float) -> bool:
    threshold_fraction = Fraction(str(float(threshold)))
    return (
        count * threshold_fraction.denominator
        > threshold_fraction.numerator * total
    )


def _count_fits_total(value: object, total: int) -> bool:
    return _is_non_negative_int(value) and value <= total


def _is_positive_int(value: object) -> bool:
    return _is_non_negative_int(value) and value > 0


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def latest_eval_category_comparison(
    project_root: Path,
    *,
    dataset_id: str,
    dataset_binding_sha256: str | None,
    adapter_id: str | None = None,
) -> dict:
    """Compare the latest base and adapter evals for one exact dataset."""
    base_eval, adapter_eval = _latest_eval_results_for_dataset(
        project_root,
        dataset_id=dataset_id,
        dataset_binding_sha256=dataset_binding_sha256,
        adapter_id=adapter_id,
    )
    return _eval_category_comparison(base_eval, adapter_eval)


def latest_adapter_eval_status(project_root: Path, adapter_id: str) -> dict:
    """Return the canonical newest eval status without falling back on corruption."""
    project_root = _safe_project_root(project_root)
    try:
        eval_dir = _latest_eval_for_adapter(project_root, adapter_id)
    except (OSError, ValueError) as exc:
        return {
            "status": "invalid",
            "eval_id": None,
            "metrics": {},
            "blocker": str(exc),
        }
    if eval_dir is None:
        return {
            "status": "missing",
            "eval_id": None,
            "metrics": {},
            "blocker": None,
        }
    inspection = _inspect_eval_entry(eval_dir)
    if inspection.get("valid") is not True:
        return {
            "status": "invalid",
            "eval_id": eval_dir.name,
            "metrics": {},
            "blocker": str(
                inspection.get("error") or "eval artifacts invalid"
            ),
        }
    results = inspection["results"]
    metrics = results.get("metrics")
    return {
        "status": "valid",
        "eval_id": eval_dir.name,
        "metrics": metrics if isinstance(metrics, dict) else {},
        "blocker": None,
    }


def _eval_category_comparison(
    base_eval: dict | None,
    adapter_eval: dict | None,
) -> dict:
    base_categories = _category_metrics(base_eval)
    adapter_categories = _category_metrics(adapter_eval)
    category_deltas = {}
    if (
        base_eval is not None
        and adapter_eval is not None
        and not base_eval.get("_registry_invalid")
        and not adapter_eval.get("_registry_invalid")
    ):
        for category in sorted(set(base_categories) | set(adapter_categories)):
            base = base_categories.get(category) or {}
            adapter = adapter_categories.get(category) or {}
            base_rate = float(base.get("pass_rate") or 0.0)
            adapter_rate = float(adapter.get("pass_rate") or 0.0)
            base_hallucination_rate = float(
                base.get("hallucination_rate") or 0.0
            )
            adapter_hallucination_rate = float(
                adapter.get("hallucination_rate") or 0.0
            )
            category_deltas[category] = {
                "base_pass_rate": round(base_rate, 4),
                "adapter_pass_rate": round(adapter_rate, 4),
                "pass_rate_delta": round(adapter_rate - base_rate, 4),
                "base_hallucination_rate": round(
                    base_hallucination_rate,
                    4,
                ),
                "adapter_hallucination_rate": round(
                    adapter_hallucination_rate,
                    4,
                ),
                "hallucination_rate_delta": round(
                    adapter_hallucination_rate - base_hallucination_rate,
                    4,
                ),
                "base_total_items": int(base.get("total_items") or 0),
                "adapter_total_items": int(adapter.get("total_items") or 0),
            }
    category_regressions = _category_regressions(
        category_deltas,
        base_categories,
        adapter_categories,
    )
    critical_regressions = [
        regression
        for regression in category_regressions
        if regression["category"] in CRITICAL_BENCHMARK_CATEGORIES
    ]
    return {
        "base_eval": _eval_summary(base_eval),
        "adapter_eval": _eval_summary(adapter_eval),
        "category_deltas": category_deltas,
        "category_regressions": category_regressions,
        "critical_regressions": critical_regressions,
    }


def _latest_eval_results_for_dataset(
    project_root: Path,
    *,
    dataset_id: str,
    dataset_binding_sha256: str | None,
    adapter_id: str | None,
) -> tuple[dict | None, dict | None]:
    if not _valid_sha256(dataset_binding_sha256):
        return None, None
    evals_root = _existing_evals_root(project_root)
    if evals_root is None:
        return None, None
    base_result = None
    adapter_result = None
    for entry in reversed(_canonical_eval_entries(evals_root)):
        if base_result is not None and adapter_result is not None:
            break
        inspection = _inspect_eval_entry(entry)
        config = inspection.get("config")
        if inspection.get("valid") is True:
            result = inspection["results"]
            if (
                result.get("dataset_id") != dataset_id
                or result.get("dataset_binding_sha256")
                != dataset_binding_sha256
            ):
                continue
            if result.get("base_only") is True:
                if base_result is None:
                    base_result = result
                continue
            if adapter_id is None or result.get("adapter_id") == adapter_id:
                if adapter_result is None:
                    adapter_result = result
            continue

        marker = _invalid_eval_marker(entry, inspection)
        invalid_role = _invalid_eval_role_for_dataset(
            entry,
            config,
            dataset_id=dataset_id,
            dataset_binding_sha256=dataset_binding_sha256,
            adapter_id=adapter_id,
        )
        if invalid_role == "base":
            if base_result is None:
                base_result = marker
        elif invalid_role == "adapter":
            if adapter_result is None:
                adapter_result = marker
        elif invalid_role == "unrelated":
            continue
        else:
            # The corrupt canonical entry cannot prove its dataset/role, so it
            # blocks fallback for every unresolved side of the comparison.
            if base_result is None:
                base_result = marker
            if adapter_result is None:
                adapter_result = marker
    adapter_pair_sha256 = (
        adapter_result.get("eval_pair_config_sha256")
        if isinstance(adapter_result, dict)
        and not adapter_result.get("_registry_invalid")
        else None
    )
    if isinstance(adapter_pair_sha256, str) and _valid_sha256(
        adapter_pair_sha256
    ):
        exact_base = _latest_base_eval_for_dataset(
            project_root,
            dataset_id,
            dataset_binding_sha256=dataset_binding_sha256,
            eval_pair_config_sha256=adapter_pair_sha256,
        )
        if exact_base is None:
            base_result = None
        else:
            exact_inspection = _inspect_eval_entry(exact_base)
            base_result = (
                exact_inspection["results"]
                if exact_inspection.get("valid") is True
                else _invalid_eval_marker(exact_base, exact_inspection)
            )
    return base_result, adapter_result


def _inspect_eval_entry(eval_dir: Path) -> dict:
    """Read one canonical eval entry without letting corruption escape."""
    try:
        entry_stat = eval_dir.stat(follow_symlinks=False)
    except OSError as exc:
        return {"valid": False, "config": None, "error": str(exc)}
    if not stat.S_ISDIR(entry_stat.st_mode) or eval_dir.is_symlink():
        return {
            "valid": False,
            "config": None,
            "error": "canonical eval entry is not a regular directory",
        }
    try:
        config = _read_json(eval_dir / "eval_config.json", "Eval config")
    except (OSError, ValueError) as exc:
        return {"valid": False, "config": None, "error": str(exc)}

    base_only = config.get("base_only")
    adapter_id = config.get("adapter_id")
    if base_only is True and adapter_id is None:
        expected_adapter_id = None
    elif base_only is False and isinstance(adapter_id, str) and adapter_id:
        expected_adapter_id = adapter_id
    else:
        return {
            "valid": False,
            "config": config,
            "error": "eval config role is invalid",
        }
    try:
        results = _read_json(eval_dir / "eval_results.json", "Eval results")
    except (OSError, ValueError) as exc:
        return {"valid": False, "config": config, "error": str(exc)}
    if not _eval_artifact_identity_is_valid(
        eval_dir,
        config,
        results,
        expected_adapter_id=expected_adapter_id,
        expected_base_only=base_only,
    ):
        return {
            "valid": False,
            "config": config,
            "results": results,
            "error": "eval artifact identity mismatch",
        }
    results = _normalize_eval_results_for_reporting(results)
    metrics = results.get("metrics")
    if (
        not isinstance(metrics, dict)
        or not _eval_metrics_are_valid(metrics)
        or not _eval_metrics_match_result_items(results)
    ):
        return {
            "valid": False,
            "config": config,
            "results": results,
            "error": "eval metrics invalid",
        }
    return {"valid": True, "config": config, "results": results, "error": None}


def _invalid_eval_role_for_dataset(
    eval_dir: Path,
    config: object,
    *,
    dataset_id: str,
    dataset_binding_sha256: str,
    adapter_id: str | None,
) -> str | None:
    if not _config_has_complete_eval_identity(eval_dir, config):
        return "ambiguous"
    assert isinstance(config, dict)
    config_dataset = config.get("dataset_id")
    config_binding = config.get("dataset_binding_sha256")
    if (
        config_dataset != dataset_id
        and config_binding != dataset_binding_sha256
    ):
        return "unrelated"
    if config.get("base_only") is True:
        return "base"
    config_adapter = config.get("adapter_id")
    if adapter_id is not None and config_adapter != adapter_id:
        return "unrelated"
    return "adapter"


def _config_has_complete_eval_identity(eval_dir: Path, config: object) -> bool:
    if not isinstance(config, dict):
        return False
    adapter_id = config.get("adapter_id")
    base_only = config.get("base_only")
    role_valid = bool(
        (base_only is True and adapter_id is None)
        or (
            base_only is False
            and isinstance(adapter_id, str)
            and adapter_id
        )
    )
    return bool(
        config.get("eval_id") == eval_dir.name
        and isinstance(config.get("dataset_id"), str)
        and config.get("dataset_id")
        and _valid_sha256(config.get("dataset_binding_sha256"))
        and role_valid
    )


def _invalid_eval_marker(eval_dir: Path, inspection: dict) -> dict:
    config = inspection.get("config")
    config = config if isinstance(config, dict) else {}
    return {
        "eval_id": eval_dir.name,
        "adapter_id": config.get("adapter_id"),
        "base_only": config.get("base_only") is True,
        "dataset_id": config.get("dataset_id"),
        "dataset_binding_sha256": config.get("dataset_binding_sha256"),
        "_registry_invalid": True,
        "registry_blocker": str(
            inspection.get("error") or "eval artifacts invalid"
        ),
    }


def _normalize_eval_results_for_reporting(result: dict) -> dict:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or "hallucinated_items" in metrics:
        return result
    by_category = metrics.get("by_category")
    if not isinstance(by_category, dict) or not by_category:
        return result
    hallucinated_counts = []
    for category_metrics in by_category.values():
        if not isinstance(category_metrics, dict):
            return result
        hallucinated_count = category_metrics.get("hallucinated_items")
        if not _is_non_negative_int(hallucinated_count):
            return result
        hallucinated_counts.append(hallucinated_count)
    return {
        **result,
        "metrics": {
            **metrics,
            "hallucinated_items": sum(hallucinated_counts),
        },
    }


def _category_regressions(
    category_deltas: dict,
    base_categories: dict,
    adapter_categories: dict,
) -> list[dict]:
    regressions = []
    for category, delta in sorted(category_deltas.items()):
        base = base_categories.get(category) or {}
        adapter = adapter_categories.get(category) or {}
        base_total = _category_metric_count(base, "total_items")
        adapter_total = _category_metric_count(adapter, "total_items")
        reasons = []
        if adapter_total < base_total:
            reasons.append("coverage_decreased")
        if _rate_decreased(
            adapter_count=_category_metric_count(adapter, "passed_items"),
            adapter_total=adapter_total,
            base_count=_category_metric_count(base, "passed_items"),
            base_total=base_total,
        ):
            reasons.append("pass_rate_decreased")
        if _rate_increased(
            adapter_count=_category_metric_count(adapter, "hallucinated_items"),
            adapter_total=adapter_total,
            base_count=_category_metric_count(base, "hallucinated_items"),
            base_total=base_total,
        ):
            reasons.append("hallucination_rate_increased")
        if reasons:
            regressions.append({
                "category": category,
                **delta,
                "reasons": reasons,
            })
    return regressions


def _category_metric_count(metrics: dict, field: str) -> int:
    value = metrics.get(field)
    return value if type(value) is int and value >= 0 else 0


def _rate_decreased(
    *,
    adapter_count: int,
    adapter_total: int,
    base_count: int,
    base_total: int,
) -> bool:
    if adapter_total == 0:
        return base_total > 0 and base_count > 0
    if base_total == 0:
        return False
    return adapter_count * base_total < base_count * adapter_total


def _rate_increased(
    *,
    adapter_count: int,
    adapter_total: int,
    base_count: int,
    base_total: int,
) -> bool:
    if adapter_total == 0:
        return False
    if base_total == 0:
        return adapter_count > 0
    return adapter_count * base_total > base_count * adapter_total


def _category_metrics(eval_results: dict | None) -> dict:
    if not eval_results or eval_results.get("_registry_invalid"):
        return {}
    metrics = eval_results.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    by_category = metrics.get("by_category")
    return by_category if isinstance(by_category, dict) else {}


def _eval_summary(eval_results: dict | None) -> dict | None:
    if not eval_results:
        return None
    summary = {
        "eval_id": eval_results.get("eval_id"),
        "adapter_id": eval_results.get("adapter_id"),
        "base_only": bool(eval_results.get("base_only")),
        "dataset_id": eval_results.get("dataset_id"),
        "dataset_binding_sha256": eval_results.get("dataset_binding_sha256"),
        "eval_pair_config_sha256": eval_results.get(
            "eval_pair_config_sha256"
        ),
        "pass_rate": (eval_results.get("metrics") or {}).get("pass_rate"),
    }
    if eval_results.get("_registry_invalid"):
        summary.update({
            "valid": False,
            "blocker": eval_results.get("registry_blocker")
            or "eval_artifacts_invalid",
        })
    return summary


def _score_item(item: dict, answer: str) -> dict:
    category = str(item.get("category") or "project_recall")
    expected = str(item.get("expected_answer") or "")
    passed = _passes(category, expected, answer)
    hallucinated = _hallucinated(category, expected, answer)
    critical_failure = category == "stale_claim_correction" and not passed
    return {
        "category": category,
        "question": item.get("question"),
        "expected_answer": expected,
        "answer": answer,
        "passed": passed,
        "hallucinated": hallucinated,
        "critical_outdated_claim_failure": critical_failure,
        "source_candidate_id": item.get("source_candidate_id"),
        "source_path": item.get("source_path"),
        "line_start": item.get("line_start"),
        "line_end": item.get("line_end"),
        "evidence_sha256": item.get("evidence_sha256"),
        "kind": item.get("kind"),
    }


def _passes(category: str, expected: str, answer: str) -> bool:
    folded_answer = answer.casefold()
    folded_expected = expected.casefold()
    command_decision = canonical_command_answer_passes(category, expected, answer)
    if command_decision is not None:
        return command_decision
    critical_decision = critical_answer_passes(category, expected, answer)
    if critical_decision is not None:
        return critical_decision
    return bool(folded_expected) and folded_expected in folded_answer


def _hallucinated(category: str, expected: str, answer: str) -> bool:
    return critical_answer_hallucinates(category, expected, answer)


def _metrics(items: list[dict]) -> dict:
    total = len(items)
    passed = sum(1 for item in items if item["passed"])
    hallucinated = sum(1 for item in items if item["hallucinated"])
    outdated = [item for item in items if item["category"] == "stale_claim_correction"]
    unsupported = [item for item in items if item["category"] == "unsupported_claim_refusal"]
    outdated_failures = sum(1 for item in outdated if not item["passed"])
    unsupported_passed = sum(1 for item in unsupported if item["passed"])
    pass_rate = passed / total if total else 0.0
    hallucination_rate = hallucinated / total if total else 0.0
    return {
        "pass_rate": round(pass_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "outdated_claim_failure_rate": round(
            outdated_failures / len(outdated), 4
        ) if outdated else 0.0,
        "unsupported_claim_refusal_rate": round(
            unsupported_passed / len(unsupported), 4
        ) if unsupported else 0.0,
        "regression_score": round(pass_rate * (1 - hallucination_rate), 4),
        "critical_outdated_claim_failures": outdated_failures,
        "total_items": total,
        "passed_items": passed,
        "hallucinated_items": hallucinated,
        "by_category": _metrics_by_category(items),
    }


def _metrics_by_category(items: list[dict]) -> dict:
    categories = sorted({str(item.get("category") or "project_recall") for item in items})
    metrics = {}
    for category in categories:
        category_items = [item for item in items if item["category"] == category]
        total = len(category_items)
        passed = sum(1 for item in category_items if item["passed"])
        hallucinated = sum(1 for item in category_items if item["hallucinated"])
        critical_failures = sum(
            1 for item in category_items if item["critical_outdated_claim_failure"]
        )
        metrics[category] = {
            "total_items": total,
            "passed_items": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "hallucinated_items": hallucinated,
            "hallucination_rate": round(hallucinated / total, 4) if total else 0.0,
            "critical_failures": critical_failures,
        }
    return metrics


def _render_report(config: dict, metrics: dict, items: list[dict]) -> str:
    lines = [
        "# Morpheus Learning Eval",
        "",
        f"- Eval ID: `{config['eval_id']}`",
        f"- Adapter: `{config.get('adapter_id') or 'base-only'}`",
        f"- Provider: `{config['provider']['name']}`",
        f"- Pass rate: `{metrics['pass_rate']}`",
        f"- Hallucination rate: `{metrics['hallucination_rate']}`",
        f"- Outdated claim failure rate: `{metrics['outdated_claim_failure_rate']}`",
        f"- Unsupported claim refusal rate: `{metrics['unsupported_claim_refusal_rate']}`",
        f"- Regression score: `{metrics['regression_score']}`",
        "",
        "## Category Metrics",
        "",
        *_render_category_metrics(metrics.get("by_category") or {}),
        "",
        "## Items",
        "",
    ]
    for item in items:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- `{status}` {item['category']}: {item['question']}")
    return "\n".join(lines).rstrip() + "\n"


def _render_category_metrics(by_category: dict) -> list[str]:
    if not by_category:
        return ["- No category metrics"]
    return [
        (
            f"- `{category}`: {item.get('passed_items', 0)}/"
            f"{item.get('total_items', 0)} pass, "
            f"hallucination_rate `{item.get('hallucination_rate', 0.0)}`"
        )
        for category, item in sorted(by_category.items())
    ]


def _latest_adapter_id(project_root: Path) -> str | None:
    runs_root = project_root / ".morpheus" / "training" / "runs"
    if runs_root.is_symlink():
        raise ValueError(f"Training runs path must not be a symlink: {runs_root}")
    reject_symlink_components(runs_root, "Training runs path")
    if not runs_root.is_dir():
        return None
    manifests = sorted(runs_root.glob("*/adapter_manifest.json"), key=lambda item: item.as_posix())
    for manifest_path in reversed(manifests):
        manifest = _read_json(manifest_path, "Adapter manifest")
        adapter_id = manifest.get("adapter_id")
        if isinstance(adapter_id, str) and adapter_id:
            return adapter_id
    return None


def _adapter_dataset_binding(
    project_root: Path,
    adapter_id: str,
    *,
    dataset_id: object,
    dataset_binding_sha256: object,
) -> dict:
    adapter_name = Path(adapter_id)
    if (
        adapter_name.is_absolute()
        or len(adapter_name.parts) != 1
        or adapter_id in {"", ".", ".."}
        or "/" in adapter_id
        or "\\" in adapter_id
    ):
        return {"valid": False, "blockers": ["adapter_id_invalid"]}
    adapter_dir = project_root / ".morpheus" / "training" / "adapters" / adapter_id
    manifest_path = adapter_dir / "adapter_manifest.json"
    try:
        if adapter_dir.is_symlink():
            raise ValueError("adapter directory is a symlink")
        reject_symlink_components(adapter_dir, "Adapter path")
        manifest = _read_json(manifest_path, "Adapter manifest")
    except (OSError, ValueError) as exc:
        return {
            "valid": False,
            "blockers": [f"adapter_manifest_invalid:{exc}"],
        }
    blockers = []
    if manifest.get("adapter_id") != adapter_id:
        blockers.append("adapter_id_mismatch")
    if not isinstance(dataset_id, str) or manifest.get("dataset_id") != dataset_id:
        blockers.append("adapter_dataset_id_mismatch")
    if (
        not _valid_sha256(dataset_binding_sha256)
        or manifest.get("dataset_binding_sha256") != dataset_binding_sha256
    ):
        blockers.append("adapter_dataset_binding_mismatch")
    return {
        "valid": not blockers,
        "blockers": blockers,
        "manifest_path": str(manifest_path),
        "base_model": manifest.get("base_model"),
    }


def _base_model_for_eval(
    project_root: Path,
    *,
    adapter_id: str | None,
    dataset_id: object,
    dataset_binding_sha256: object,
) -> str:
    """Resolve the untreated model identity shared by both sides of an eval."""
    if adapter_id is not None:
        binding = _adapter_dataset_binding(
            project_root,
            adapter_id,
            dataset_id=dataset_id,
            dataset_binding_sha256=dataset_binding_sha256,
        )
        if not binding["valid"]:
            raise ValueError(
                "Adapter dataset binding mismatch: "
                + ", ".join(binding["blockers"])
            )
        base_model = binding.get("base_model")
        return (
            base_model.strip()
            if isinstance(base_model, str) and base_model.strip()
            else DEFAULT_BASE_MODEL
        )

    adapters_root = project_root / ".morpheus" / "training" / "adapters"
    if adapters_root.is_symlink():
        raise ValueError(f"Adapter registry must not be a symlink: {adapters_root}")
    reject_symlink_components(adapters_root, "Adapter registry")
    if adapters_root.is_dir():
        manifests = sorted(
            adapters_root.glob("*/adapter_manifest.json"),
            key=lambda path: path.parent.name,
            reverse=True,
        )
        for manifest_path in manifests:
            if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
                continue
            manifest = _read_json(manifest_path, "Adapter manifest")
            base_model = manifest.get("base_model")
            if (
                manifest.get("dataset_id") == dataset_id
                and manifest.get("dataset_binding_sha256")
                == dataset_binding_sha256
                and isinstance(base_model, str)
                and base_model.strip()
            ):
                return base_model.strip()
    return DEFAULT_BASE_MODEL


def _dataset_dir_for_eval(project_root: Path, dataset_id: str | None) -> Path | None:
    if dataset_id is None:
        effective = latest_effective_dataset(project_root)
        return Path(str(effective["dataset_dir"])) if effective is not None else None
    manifest_paths = [
        *project_root.glob(".morpheus/training/datasets/*/manifest.json"),
        *project_root.glob(".morpheus/lab/*/dataset/manifest.json"),
    ]
    matches = []
    for manifest_path in sorted(manifest_paths, key=lambda item: item.as_posix()):
        if (
            manifest_path.is_symlink()
            or manifest_path.parent.is_symlink()
            or manifest_path.parent.name.startswith(".")
        ):
            continue
        manifest = _read_json(manifest_path, "Dataset manifest")
        if manifest.get("dataset_id") != dataset_id:
            continue
        if manifest_count(manifest, "examples_count") <= 0:
            continue
        matches.append(manifest_path.parent)
    if not matches:
        raise ValueError(f"No trainable learning dataset found for dataset_id={dataset_id}.")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous learning dataset_id={dataset_id}.")
    return matches[0]


def _existing_evals_root(project_root: Path) -> Path | None:
    evals_root = project_root / ".morpheus" / "training" / "evals"
    if evals_root.is_symlink():
        raise ValueError(f"Eval registry must not be a symlink: {evals_root}")
    reject_symlink_components(evals_root, "Eval registry")
    if not evals_root.exists():
        return None
    if not evals_root.is_dir():
        raise ValueError(f"Eval registry must be a directory: {evals_root}")
    return evals_root


def _canonical_eval_entries(evals_root: Path) -> list[Path]:
    return sorted(
        (
            entry
            for entry in evals_root.iterdir()
            if _is_canonical_eval_id(entry.name)
        ),
        key=lambda entry: entry.name,
    )


def _is_canonical_eval_id(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("eval_"):
        return False
    path = Path(value)
    return bool(
        len(value) > len("eval_")
        and not path.is_absolute()
        and len(path.parts) == 1
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
    )


def _latest_eval_for_adapter(
    project_root: Path,
    adapter_id: str,
    *,
    eval_id: str | None = None,
) -> Path | None:
    evals_root = _existing_evals_root(project_root)
    if evals_root is None:
        return None
    entries = _canonical_eval_entries(evals_root)
    if eval_id is not None:
        if not _is_canonical_eval_id(eval_id):
            raise ValueError(f"Eval id is unsafe: {eval_id!r}")
        return next((entry for entry in entries if entry.name == eval_id), None)

    for entry in reversed(entries):
        inspection = _inspect_eval_entry(entry)
        config = inspection.get("config")
        if isinstance(config, dict) and config.get("adapter_id") == adapter_id:
            return entry
        if _eval_entry_routing_identity_is_trusted(
            project_root,
            entry,
            inspection,
        ):
            continue
        # Unsigned role/id fields cannot prove that this newer entry is
        # unrelated; returning it blocks fallback to older activation state.
        return entry
    return None


def _latest_base_eval_for_dataset(
    project_root: Path,
    dataset_id: str,
    *,
    dataset_binding_sha256: str,
    eval_pair_config_sha256: str,
) -> Path | None:
    evals_root = _existing_evals_root(project_root)
    if evals_root is None:
        return None
    for entry in reversed(_canonical_eval_entries(evals_root)):
        inspection = _inspect_eval_entry(entry)
        config = inspection.get("config")
        routing_identity_is_trusted = _eval_entry_routing_identity_is_trusted(
            project_root,
            entry,
            inspection,
        )
        if isinstance(config, dict) and config.get("base_only") is True:
            config_dataset = config.get("dataset_id")
            config_binding = config.get("dataset_binding_sha256")
            if (
                config_dataset == dataset_id
                or config_binding == dataset_binding_sha256
            ):
                config_pair_sha256 = _config_eval_pair_sha256(config)
                if config_pair_sha256 == eval_pair_config_sha256:
                    return entry
                if routing_identity_is_trusted:
                    continue
                return entry
        if routing_identity_is_trusted:
            continue
        # A role or dataset relabel without trusted provenance must not hide a
        # newer base candidate from activation authority.
        return entry
    return None


def _eval_entry_routing_identity_is_trusted(
    project_root: Path,
    eval_dir: Path,
    inspection: dict,
) -> bool:
    """Trust routing fields only for exact diagnostics or signed evals."""
    if inspection.get("valid") is not True:
        return False
    config = inspection.get("config")
    results = inspection.get("results")
    if not isinstance(config, dict) or not isinstance(results, dict):
        return False
    if _is_exact_builtin_diagnostic_eval(eval_dir, config, results):
        return True
    if not _eval_is_activation_eligible(config, results):
        return False
    return _activation_eval_routing_receipt_is_valid(
        project_root,
        eval_dir,
        config,
        results,
    )


def _activation_eval_routing_receipt_is_valid(
    project_root: Path,
    eval_dir: Path,
    config: dict,
    results: dict,
) -> bool:
    """Verify signed routing claims without requiring live dataset authority."""
    try:
        config_bytes = _read_stable_regular_bytes(
            eval_dir / "eval_config.json",
            "Eval config",
        )
        results_bytes = _read_stable_regular_bytes(
            eval_dir / "eval_results.json",
            "Eval results",
        )
        if (
            _json_object_from_bytes(config_bytes, "Eval config") != config
            or _json_object_from_bytes(results_bytes, "Eval results") != results
        ):
            return False
        receipt_bytes = _read_stable_regular_bytes(
            eval_dir / _ACTIVATION_EVAL_RECEIPT_NAME,
            "Activation eval receipt",
            private=True,
        )
        receipt = _json_object_from_bytes(
            receipt_bytes,
            "Activation eval receipt",
        )
        pair = _eval_pair_identity(config, results)
        expected = {
            "schema": _ACTIVATION_EVAL_RECEIPT_SCHEMA,
            "eval_id": config.get("eval_id"),
            "evaluator": dict(_ACTIVATION_EVALUATOR),
            "evaluation_mode": config.get("evaluation_mode"),
            "provider": config.get("provider"),
            "adapter_id": config.get("adapter_id"),
            "base_only": config.get("base_only"),
            "dataset_id": config.get("dataset_id"),
            "dataset_binding_sha256": config.get(
                "dataset_binding_sha256"
            ),
            "benchmark_category_schema": BENCHMARK_CATEGORY_SCHEMA,
            "eval_pair_config": pair.get("config"),
            "eval_pair_config_sha256": pair.get("sha256"),
            "eval_config_sha256": sha256(config_bytes).hexdigest(),
            "eval_results_sha256": sha256(results_bytes).hexdigest(),
        }
        if (
            pair.get("current") is not True
            or any(receipt.get(key) != value for key, value in expected.items())
            or not isinstance(receipt.get("issued_at"), str)
            or not receipt.get("issued_at")
        ):
            return False
        signature = receipt.get("signature")
        if not isinstance(signature, dict):
            return False
        if signature.get("algo") != "ed25519" or signature.get("key_id") != "local":
            return False
        signature_b64 = signature.get("signature_b64")
        if not isinstance(signature_b64, str):
            return False
        signature_bytes = base64.b64decode(signature_b64, validate=True)
        _activation_eval_public_key(project_root).verify(
            signature_bytes,
            receipt_signature_payload(receipt),
        )
    except (OSError, ValueError, json.JSONDecodeError, InvalidSignature):
        return False
    return True


def _is_exact_builtin_diagnostic_eval(
    eval_dir: Path,
    config: dict,
    results: dict,
) -> bool:
    pair = _eval_pair_identity(config, results)
    pair_config = pair.get("config")
    model = (
        pair_config.get("model")
        if isinstance(pair_config, dict)
        and isinstance(pair_config.get("model"), dict)
        else {}
    )
    try:
        artifact_names = {entry.name for entry in eval_dir.iterdir()}
    except OSError:
        return False
    return bool(
        artifact_names == _BASE_EVAL_ARTIFACT_NAMES
        and config.get("provider") == {"name": "diagnostic-fake"}
        and config.get("evaluation_mode") == "diagnostic_fake"
        and results.get("evaluation_mode") == "diagnostic_fake"
        and config.get("activation_eligible") is False
        and results.get("activation_eligible") is False
        and config.get("dry_run") is True
        and config.get("diagnostic_quality") in {"passing", "failing"}
        and config.get("evaluator") == _ACTIVATION_EVALUATOR
        and pair.get("current") is True
        and pair_config.get("provider") == {"name": "diagnostic-fake"}
        and pair_config.get("evaluation_mode") == "diagnostic_fake"
        and pair_config.get("evaluator") == _ACTIVATION_EVALUATOR
        and isinstance(model.get("base_model"), str)
        and model.get("base_model").strip()
        and model.get("inference_config") == {}
    )


def _has_base_eval_for_dataset(
    project_root: Path,
    dataset_id: str,
    *,
    dataset_binding_sha256: str,
) -> bool:
    evals_root = _existing_evals_root(project_root)
    if evals_root is None:
        return False
    for entry in reversed(_canonical_eval_entries(evals_root)):
        inspection = _inspect_eval_entry(entry)
        config = inspection.get("config")
        if not isinstance(config, dict) or config.get("base_only") is not True:
            continue
        if (
            config.get("dataset_id") == dataset_id
            or config.get("dataset_binding_sha256") == dataset_binding_sha256
        ):
            return True
    return False


def _current_dataset_validation(
    project_root: Path,
    dataset_id: object,
    dataset_binding_sha256: object,
) -> dict:
    if not isinstance(dataset_id, str) or not dataset_id:
        return {"valid": False, "blockers": ["dataset_id_invalid"]}
    if not _valid_sha256(dataset_binding_sha256):
        return {"valid": False, "blockers": ["dataset_binding_invalid"]}
    try:
        effective = latest_effective_dataset(project_root)
        if not isinstance(effective, dict):
            return {
                "valid": False,
                "blockers": ["latest_effective_dataset_missing"],
            }
        dataset_dir = _dataset_dir_for_eval(project_root, dataset_id)
        if dataset_dir is None:
            return {"valid": False, "blockers": ["dataset_missing"]}
        effective_dir_value = effective.get("dataset_dir")
        effective_validation = effective.get("validation")
        effective_binding = (
            effective_validation.get("dataset_binding_sha256")
            if isinstance(effective_validation, dict)
            else None
        )
        if (
            effective.get("dataset_id") != dataset_id
            or effective_binding != dataset_binding_sha256
            or not isinstance(effective_dir_value, str)
            or Path(effective_dir_value) != dataset_dir
        ):
            return {
                "valid": False,
                "blockers": ["dataset_not_latest_effective"],
                "dataset_dir": str(dataset_dir),
                "latest_effective_dataset_id": effective.get("dataset_id"),
                "latest_effective_dataset_dir": effective_dir_value,
            }
        manifest = _read_json(dataset_dir / "manifest.json", "Dataset manifest")
        validation = validate_dataset(project_root, dataset_dir, manifest)
    except (OSError, ValueError) as exc:
        return {
            "valid": False,
            "blockers": [f"dataset_resolution_failed:{exc}"],
        }
    if validation.get("dataset_binding_sha256") != dataset_binding_sha256:
        blockers = list(validation.get("blockers") or [])
        if "dataset_binding_changed" not in blockers:
            blockers.append("dataset_binding_changed")
        return {
            **validation,
            "valid": False,
            "blockers": blockers,
            "dataset_dir": str(dataset_dir),
            "manifest": manifest,
        }
    return {
        **validation,
        "dataset_dir": str(dataset_dir),
        "manifest": manifest,
    }


def _review_authority_roots_for_dataset(
    project_root: Path,
    dataset_dir: Path,
) -> tuple[Path, ...]:
    manifest = _read_json(dataset_dir / "manifest.json", "Dataset manifest")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Dataset provenance is missing")
    source_root_value = provenance.get("source_root")
    if not isinstance(source_root_value, str) or not source_root_value:
        raise ValueError("Dataset review authority root is missing")
    source_root = Path(source_root_value).expanduser()
    if source_root.is_symlink():
        raise ValueError("Dataset review authority root must not be a symlink")
    reject_symlink_components(source_root, "Dataset review authority")
    source_root = source_root.resolve()
    roots = [project_root]
    if source_root != project_root:
        lab_root = project_root / ".morpheus" / "lab"
        try:
            relative = source_root.relative_to(lab_root)
        except ValueError as exc:
            raise ValueError("Dataset review authority is outside the project") from exc
        if (
            len(relative.parts) != 2
            or not relative.parts[0].startswith("lab_")
            or relative.parts[1] != "workspace"
            or not source_root.is_dir()
        ):
            raise ValueError("Dataset lab review authority is invalid")
        roots.append(source_root)
    return tuple(roots)


@contextmanager
def _eval_review_authority_transaction(review_roots: tuple[Path, ...]):
    """Hold the exact selected dataset review authorities in stable order."""
    with ExitStack() as stack:
        for review_root in review_roots:
            stack.enter_context(ReviewStore(review_root).transaction())
        yield


def _safe_project_root(project_root: Path) -> Path:
    project_root = project_root.expanduser()
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")
    return project_root


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _validated_evals_root(project_root: Path) -> Path:
    evals_root = project_root / ".morpheus" / "training" / "evals"
    if evals_root.is_symlink():
        raise ValueError(f"Eval registry must not be a symlink: {evals_root}")
    reject_symlink_components(evals_root.parent, "Eval registry")
    evals_root.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(evals_root, "Eval registry")
    if evals_root.is_symlink() or not evals_root.is_dir():
        raise ValueError(f"Eval registry must be a directory: {evals_root}")
    return evals_root.resolve()


def _create_private_eval_staging(
    evals_root: Path,
    eval_id: str,
) -> tuple[Path, tuple[int, int]]:
    if not _is_canonical_eval_id(eval_id):
        raise ValueError(f"Eval identity is unsafe: {eval_id!r}")
    reject_symlink_components(evals_root, "Eval registry")
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f".{eval_id}.",
        suffix=".staging",
        dir=evals_root,
    ))
    os.chmod(staging_dir, 0o700)
    staging_stat = staging_dir.stat(follow_symlinks=False)
    if not stat.S_ISDIR(staging_stat.st_mode):
        raise ValueError(f"Eval staging must be a directory: {staging_dir}")
    return staging_dir, (staging_stat.st_dev, staging_stat.st_ino)


def _publish_staged_activation_eval(
    project_root: Path,
    staging_dir: Path,
    eval_dir: Path,
    *,
    staging_identity: tuple[int, int],
    expected_contents: dict[str, str],
) -> None:
    """Sign and atomically publish one genuine local held-out eval bundle."""
    if frozenset(expected_contents) != _BASE_EVAL_ARTIFACT_NAMES:
        raise ValueError("Activation eval publication requires exact base artifacts")
    config = _json_object_from_bytes(
        _read_stable_regular_bytes(
            staging_dir / "eval_config.json",
            "Eval config",
        ),
        "Eval config",
    )
    selected_dataset = _dataset_dir_for_eval(project_root, config.get("dataset_id"))
    if selected_dataset is None:
        raise ValueError("Activation eval dataset is missing")
    selected_dataset = selected_dataset.resolve()
    review_roots = _review_authority_roots_for_dataset(
        project_root,
        selected_dataset,
    )
    with state_authority_transaction(project_root):
        with _eval_review_authority_transaction(review_roots):
            current_dataset = _dataset_dir_for_eval(
                project_root,
                config.get("dataset_id"),
            )
            if current_dataset is None or current_dataset.resolve() != selected_dataset:
                raise ValueError("Activation eval dataset selection changed")
            config = _json_object_from_bytes(
                _read_stable_regular_bytes(
                    staging_dir / "eval_config.json",
                    "Eval config",
                ),
                "Eval config",
            )
            if config.get("eval_id") != eval_dir.name:
                raise ValueError("Activation eval destination identity mismatch")
            receipt_bytes = _build_activation_eval_receipt_bytes(
                project_root,
                staging_dir,
            )
            receipt_text = receipt_bytes.decode()
            _write_private_text(
                staging_dir / _ACTIVATION_EVAL_RECEIPT_NAME,
                receipt_text,
            )
            _publish_staged_eval(
                staging_dir,
                eval_dir,
                staging_identity=staging_identity,
                expected_contents={
                    **expected_contents,
                    _ACTIVATION_EVAL_RECEIPT_NAME: receipt_text,
                },
            )


def _publish_staged_eval(
    staging_dir: Path,
    eval_dir: Path,
    *,
    staging_identity: tuple[int, int],
    expected_contents: dict[str, str],
) -> None:
    evals_root = eval_dir.parent
    if staging_dir.parent != evals_root:
        raise ValueError("Eval staging is outside its registry")
    expected_names = frozenset(expected_contents)
    if expected_names not in {
        _BASE_EVAL_ARTIFACT_NAMES,
        _BASE_EVAL_ARTIFACT_NAMES | {_ACTIVATION_EVAL_RECEIPT_NAME},
    }:
        raise ValueError("Eval publication contract has unexpected artifacts")
    reject_symlink_components(evals_root, "Eval registry")
    lock_path = evals_root / ".registry.lock"
    reject_symlink_paths([lock_path], "Eval registry lock")
    reject_symlink_components(lock_path, "Eval registry lock")
    authority_root = evals_root.parents[2]
    with learning_authority_transaction(authority_root):
        with portable_file_lock(lock_path):
            if _descriptor_publish_supported():
                _publish_staged_eval_with_descriptors(
                    evals_root,
                    staging_dir,
                    eval_dir,
                    staging_identity=staging_identity,
                    expected_contents=expected_contents,
                )
            else:  # pragma: no cover - POSIX supplies descriptor-relative APIs.
                _publish_staged_eval_with_paths(
                    staging_dir,
                    eval_dir,
                    staging_identity=staging_identity,
                    expected_contents=expected_contents,
                )


def _publish_staged_eval_with_descriptors(
    evals_root: Path,
    staging_dir: Path,
    eval_dir: Path,
    *,
    staging_identity: tuple[int, int],
    expected_contents: dict[str, str],
) -> None:
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    registry_descriptor = os.open(evals_root, directory_flags)
    staging_descriptor = -1
    try:
        try:
            staging_descriptor = os.open(
                staging_dir.name,
                directory_flags,
                dir_fd=registry_descriptor,
            )
        except OSError as exc:
            raise ValueError(
                "Eval staging identity changed before publication"
            ) from exc
        _verify_staged_eval_descriptor(
            staging_descriptor,
            staging_identity=staging_identity,
            expected_contents=expected_contents,
        )
        current = os.stat(
            staging_dir.name,
            dir_fd=registry_descriptor,
            follow_symlinks=False,
        )
        if (current.st_dev, current.st_ino) != staging_identity:
            raise ValueError("Eval staging identity changed before publication")
        try:
            os.stat(
                eval_dir.name,
                dir_fd=registry_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"Eval output already exists: {eval_dir}")
        _fsync_descriptor(staging_descriptor)
        os.rename(
            staging_dir.name,
            eval_dir.name,
            src_dir_fd=registry_descriptor,
            dst_dir_fd=registry_descriptor,
        )
        published = os.stat(
            eval_dir.name,
            dir_fd=registry_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(published.st_mode)
            or (published.st_dev, published.st_ino) != staging_identity
        ):
            raise ValueError("Published eval identity changed during publication")
        _fsync_descriptor(registry_descriptor)
    finally:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        os.close(registry_descriptor)


def _publish_staged_eval_with_paths(
    staging_dir: Path,
    eval_dir: Path,
    *,
    staging_identity: tuple[int, int],
    expected_contents: dict[str, str],
) -> None:
    _verify_staged_eval_path(
        staging_dir,
        staging_identity=staging_identity,
        expected_contents=expected_contents,
    )
    reject_symlink_paths([eval_dir], "Eval output")
    if eval_dir.exists() or eval_dir.is_symlink():
        raise ValueError(f"Eval output already exists: {eval_dir}")
    current = staging_dir.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino) != staging_identity:
        raise ValueError("Eval staging identity changed before publication")
    _fsync_directory_path(staging_dir)
    staging_dir.rename(eval_dir)
    published = eval_dir.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(published.st_mode)
        or (published.st_dev, published.st_ino) != staging_identity
    ):
        raise ValueError("Published eval identity changed during publication")
    _fsync_directory_path(eval_dir.parent)


def _verify_staged_eval_descriptor(
    staging_descriptor: int,
    *,
    staging_identity: tuple[int, int],
    expected_contents: dict[str, str],
) -> None:
    staging_stat = os.fstat(staging_descriptor)
    if (
        not stat.S_ISDIR(staging_stat.st_mode)
        or (staging_stat.st_dev, staging_stat.st_ino) != staging_identity
    ):
        raise ValueError("Eval staging identity changed before publication")
    if os.name != "nt" and stat.S_IMODE(staging_stat.st_mode) != 0o700:
        raise ValueError("Eval staging permissions changed before publication")
    actual_names = set(os.listdir(staging_descriptor))
    if actual_names != set(expected_contents):
        raise ValueError("Eval staging has unexpected entries; publication refused")
    for name, expected in expected_contents.items():
        actual = _read_private_regular_file(name, dir_fd=staging_descriptor)
        if actual != expected.encode():
            raise ValueError(f"Eval staging artifact changed: {name}")


def _verify_staged_eval_path(
    staging_dir: Path,
    *,
    staging_identity: tuple[int, int],
    expected_contents: dict[str, str],
) -> None:
    reject_symlink_components(staging_dir, "Eval staging")
    staging_stat = staging_dir.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(staging_stat.st_mode)
        or (staging_stat.st_dev, staging_stat.st_ino) != staging_identity
    ):
        raise ValueError("Eval staging identity changed before publication")
    if os.name != "nt" and stat.S_IMODE(staging_stat.st_mode) != 0o700:
        raise ValueError("Eval staging permissions changed before publication")
    actual_names = {entry.name for entry in staging_dir.iterdir()}
    if actual_names != set(expected_contents):
        raise ValueError("Eval staging has unexpected entries; publication refused")
    for name, expected in expected_contents.items():
        actual = _read_private_regular_file(staging_dir / name)
        if actual != expected.encode():
            raise ValueError(f"Eval staging artifact changed: {name}")


def _read_private_regular_file(
    path: str | Path,
    *,
    dir_fd: int | None = None,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, dir_fd=dir_fd)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"Eval artifact must be a regular file: {path}")
        if os.name != "nt" and stat.S_IMODE(file_stat.st_mode) != 0o600:
            raise ValueError(f"Eval artifact permissions changed: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_private_text(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        data = content.encode()
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _descriptor_publish_supported() -> bool:
    return bool(
        os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
    )


def _fsync_descriptor(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError:
        pass


def _fsync_directory_path(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        _fsync_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _write_adapter_eval_results(path: Path, results: dict) -> None:
    reject_symlink_components(path.parent, "Adapter eval output")
    if not path.parent.is_dir():
        raise ValueError(f"Adapter not found: {path.parent.name}")
    reject_symlink_paths([path], "Adapter eval output")
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path, label: str) -> dict:
    reject_symlink_paths([path], label)
    reject_symlink_components(path, label)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} invalid: expected JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict]:
    reject_symlink_paths([path], "Eval seed")
    reject_symlink_components(path, "Eval seed")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
