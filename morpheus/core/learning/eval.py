"""Evaluation harness for reviewed Morpheus learning adapters."""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from morpheus.core.learning.registry import latest_dataset_dir
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths


EVAL_CATEGORIES = {
    "project_recall",
    "active_decision_recall",
    "outdated_claim_correction",
    "unsupported_claim_refusal",
    "agent_rule_adherence",
}
DEFAULT_PASS_RATE_THRESHOLD = 0.8
DEFAULT_HALLUCINATION_RATE_THRESHOLD = 0.05


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
        if category == "outdated_claim_correction":
            return expected if expected else "No. That claim is outdated and must not be treated as active state."
        return expected


def run_learning_eval(
    project_root: Path,
    *,
    adapter_id: str | None = None,
    base_only: bool = False,
    dry_run: bool = True,
    fake_quality: str = "passing",
) -> dict:
    project_root = _safe_project_root(project_root)
    dataset_dir = latest_dataset_dir(project_root)
    if dataset_dir is None:
        raise ValueError("No learning dataset manifest found. Run `morpheus learn dataset .` first.")
    dataset_manifest = _read_json(dataset_dir / "manifest.json", "Dataset manifest")
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

    provider = FakeInferenceProvider(
        name="fake-base" if resolved_base_only else "fake-adapter",
        quality=fake_quality,
    )
    eval_id = _timestamp_id("eval")
    eval_dir = project_root / ".morpheus" / "training" / "evals" / eval_id
    _ensure_eval_dir(eval_dir)

    config = {
        "eval_id": eval_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_manifest.get("dataset_id"),
        "dataset_manifest_path": str(dataset_dir / "manifest.json"),
        "eval_seed_path": str(eval_seed_path),
        "adapter_id": resolved_adapter_id,
        "base_only": resolved_base_only,
        "dry_run": dry_run,
        "provider": {"name": provider.name, "quality": provider.quality},
        "categories": sorted(EVAL_CATEGORIES),
    }
    results_items = [_score_item(item, provider.infer(item)) for item in seed_items]
    metrics = _metrics(results_items)
    results = {
        "eval_id": eval_id,
        "created_at": config["created_at"],
        "adapter_id": resolved_adapter_id,
        "base_only": resolved_base_only,
        "dataset_id": dataset_manifest.get("dataset_id"),
        "metrics": metrics,
        "items": results_items,
    }

    config_path = eval_dir / "eval_config.json"
    results_path = eval_dir / "eval_results.json"
    report_path = eval_dir / "eval_report.md"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    report_path.write_text(_render_report(config, metrics, results_items))
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
) -> dict:
    project_root = _safe_project_root(project_root)
    latest_eval = _latest_eval_for_adapter(project_root, adapter_id)
    if latest_eval is None:
        return {"allowed": False, "reason": "missing_eval", "adapter_id": adapter_id}
    results = _read_json(latest_eval / "eval_results.json", "Eval results")
    metrics = results.get("metrics") if isinstance(results.get("metrics"), dict) else {}
    pass_rate = float(metrics.get("pass_rate") or 0)
    hallucination_rate = float(metrics.get("hallucination_rate") or 0)
    critical_outdated_failures = int(metrics.get("critical_outdated_claim_failures") or 0)
    if pass_rate < pass_rate_threshold:
        return {
            "allowed": False,
            "reason": "pass_rate_below_threshold",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "metrics": metrics,
        }
    if hallucination_rate > hallucination_rate_threshold:
        return {
            "allowed": False,
            "reason": "hallucination_rate_above_threshold",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "metrics": metrics,
        }
    if critical_outdated_failures:
        return {
            "allowed": False,
            "reason": "critical_outdated_claim_failure",
            "adapter_id": adapter_id,
            "eval_id": results.get("eval_id"),
            "metrics": metrics,
        }
    return {
        "allowed": True,
        "reason": "passed",
        "adapter_id": adapter_id,
        "eval_id": results.get("eval_id"),
        "metrics": metrics,
    }


def _score_item(item: dict, answer: str) -> dict:
    category = str(item.get("category") or "project_recall")
    expected = str(item.get("expected_answer") or "")
    passed = _passes(category, expected, answer)
    hallucinated = _hallucinated(category, answer)
    critical_failure = category == "outdated_claim_correction" and not passed
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
        "kind": item.get("kind"),
    }


def _passes(category: str, expected: str, answer: str) -> bool:
    folded_answer = answer.casefold()
    folded_expected = expected.casefold()
    if category == "unsupported_claim_refusal":
        return "cannot confirm" in folded_answer or "unsupported" in folded_answer
    if category == "outdated_claim_correction":
        return folded_answer.startswith("no") and "outdated" in folded_answer
    return bool(folded_expected) and folded_expected in folded_answer


def _hallucinated(category: str, answer: str) -> bool:
    folded = answer.casefold()
    if "confirmed without source" in folded:
        return True
    if category == "unsupported_claim_refusal" and not (
        "cannot confirm" in folded or "unsupported" in folded
    ):
        return True
    return False


def _metrics(items: list[dict]) -> dict:
    total = len(items)
    passed = sum(1 for item in items if item["passed"])
    hallucinated = sum(1 for item in items if item["hallucinated"])
    outdated = [item for item in items if item["category"] == "outdated_claim_correction"]
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
    }


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
        "## Items",
        "",
    ]
    for item in items:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- `{status}` {item['category']}: {item['question']}")
    return "\n".join(lines).rstrip() + "\n"


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


def _latest_eval_for_adapter(project_root: Path, adapter_id: str) -> Path | None:
    evals_root = project_root / ".morpheus" / "training" / "evals"
    if evals_root.is_symlink():
        raise ValueError(f"Eval registry must not be a symlink: {evals_root}")
    reject_symlink_components(evals_root, "Eval registry")
    if not evals_root.is_dir():
        return None
    matches = []
    for config_path in sorted(evals_root.glob("*/eval_config.json"), key=lambda item: item.as_posix()):
        config = _read_json(config_path, "Eval config")
        if config.get("adapter_id") == adapter_id and not config.get("base_only"):
            matches.append(config_path.parent)
    return matches[-1] if matches else None


def _safe_project_root(project_root: Path) -> Path:
    project_root = project_root.expanduser()
    if project_root.is_symlink():
        raise ValueError(f"Project root must not be a symlink: {project_root}")
    reject_symlink_components(project_root, "Project root")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")
    return project_root


def _ensure_eval_dir(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Eval output must not be a symlink: {path}")
    reject_symlink_components(path.parent, "Eval output root")
    path.mkdir(parents=True, exist_ok=False)
    reject_symlink_components(path, "Eval output")


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
