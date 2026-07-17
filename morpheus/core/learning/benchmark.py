"""Benchmark readiness reports for Morpheus learning datasets."""
from datetime import datetime, timezone
import json
from pathlib import Path

from morpheus.core.learning.quality import build_quality_report
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths


DEFAULT_BENCHMARK_BACKEND = "mlx"
DEFAULT_BENCHMARK_MAX_ITERS = 50


def write_benchmark_report(
    project_root: Path,
    *,
    dry_run: bool = True,
    backend: str = DEFAULT_BENCHMARK_BACKEND,
    max_iters: int = DEFAULT_BENCHMARK_MAX_ITERS,
) -> dict:
    """Write a benchmark readiness report without training or activating adapters."""
    project_root = project_root.expanduser().resolve()
    created_at = datetime.now(timezone.utc)
    benchmark_id = f"bench_{created_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    quality = build_quality_report(project_root)
    manifest = (quality.get("dataset", {}).get("latest_manifest") or {})
    benchmark_allowed = bool(quality.get("benchmark_allowed"))
    benchmark_blockers = list(quality.get("benchmark_blockers") or [])
    eval_comparison = latest_eval_category_comparison(project_root)
    next_command = (
        f"morpheus learn lab . --backend {backend} --max-iters {max_iters}"
        if benchmark_allowed
        else "morpheus learn quality ."
    )

    benchmark_dir = project_root / ".morpheus" / "training" / "benchmarks" / benchmark_id
    reject_symlink_components(benchmark_dir, "Learning benchmark directory")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    config_path = benchmark_dir / "benchmark_config.json"
    report_path = benchmark_dir / "benchmark_report.json"
    markdown_path = benchmark_dir / "benchmark_report.md"
    reject_symlink_paths(
        [config_path, report_path, markdown_path],
        "Learning benchmark report",
    )

    config = {
        "benchmark_id": benchmark_id,
        "backend": backend,
        "dry_run": dry_run,
        "max_iters": max_iters,
        "project_root": str(project_root),
    }
    result = {
        "benchmark_id": benchmark_id,
        "created_at": created_at.isoformat(),
        "dry_run": dry_run,
        "backend": backend,
        "max_iters": max_iters,
        "dataset_id": manifest.get("dataset_id"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "examples_count": int(manifest.get("examples_count") or 0),
        "eval_items_count": int(manifest.get("eval_items_count") or 0),
        "trainable_candidate_count": int(manifest.get("trainable_candidate_count") or 0),
        "benchmark_allowed": benchmark_allowed,
        "benchmark_blockers": benchmark_blockers,
        "benchmark_gate": quality.get("benchmark_gate") or {},
        "latest_base_eval": eval_comparison["base_eval"],
        "latest_adapter_eval": eval_comparison["adapter_eval"],
        "category_deltas": eval_comparison["category_deltas"],
        "quality_report": quality,
        "next_command": next_command,
        "benchmark_config_path": str(config_path),
        "benchmark_report_path": str(report_path),
        "benchmark_report_md_path": str(markdown_path),
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_benchmark_report(result))
    return result


def render_benchmark_report(report: dict) -> str:
    """Render the benchmark readiness report for human review."""
    status_line = "Benchmark ready" if report["benchmark_allowed"] else "Benchmark blocked"
    lines = [
        "# Morpheus Learning Benchmark",
        "",
        f"## {status_line}",
        "",
        f"- Benchmark id: `{report['benchmark_id']}`",
        f"- Dataset id: `{report.get('dataset_id') or 'none'}`",
        f"- Examples: {report['examples_count']}",
        f"- Eval items: {report['eval_items_count']}",
        f"- Trainable candidates: {report['trainable_candidate_count']}",
        f"- Backend: `{report['backend']}`",
        f"- Dry run: {report['dry_run']}",
        "",
        "## Gate",
        "",
        f"- Allowed: {report['benchmark_allowed']}",
    ]
    for blocker in report["benchmark_blockers"]:
        lines.append(f"- Blocker: {blocker}")

    gate = report.get("benchmark_gate") or {}
    eval_counts = gate.get("eval_category_counts") or {}
    if eval_counts:
        lines.extend(["", "## Eval Categories", ""])
        for category, count in sorted(eval_counts.items()):
            lines.append(f"- `{category}`: {count}")

    requirements = gate.get("requirements") or {}
    class_counts = requirements.get("class_counts") or {}
    if class_counts:
        lines.extend(["", "## Class Requirements", ""])
        for class_name, item in sorted(class_counts.items()):
            lines.append(
                f"- `{class_name}`: {item.get('count', 0)}/{item.get('minimum', 0)}"
            )

    deltas = report.get("category_deltas") or {}
    if deltas:
        lines.extend(["", "## Category Deltas", ""])
        for category, item in sorted(deltas.items()):
            lines.append(
                f"- `{category}`: base `{item['base_pass_rate']}`, "
                f"adapter `{item['adapter_pass_rate']}`, "
                f"delta `{item['pass_rate_delta']}`"
            )

    lines.extend([
        "",
        "## Next Command",
        "",
        f"`{report['next_command']}`",
    ])
    return "\n".join(lines).rstrip() + "\n"


def latest_eval_category_comparison(project_root: Path) -> dict:
    """Return latest base/adaptor category metrics and per-category deltas."""
    base_eval, adapter_eval = _latest_eval_results(project_root)
    base_categories = _category_metrics(base_eval)
    adapter_categories = _category_metrics(adapter_eval)
    category_deltas = {}
    for category in sorted(set(base_categories) | set(adapter_categories)):
        base_rate = float((base_categories.get(category) or {}).get("pass_rate") or 0.0)
        adapter_rate = float((adapter_categories.get(category) or {}).get("pass_rate") or 0.0)
        category_deltas[category] = {
            "base_pass_rate": round(base_rate, 4),
            "adapter_pass_rate": round(adapter_rate, 4),
            "pass_rate_delta": round(adapter_rate - base_rate, 4),
            "base_total_items": int((base_categories.get(category) or {}).get("total_items") or 0),
            "adapter_total_items": int(
                (adapter_categories.get(category) or {}).get("total_items") or 0
            ),
        }
    return {
        "base_eval": _eval_summary(base_eval),
        "adapter_eval": _eval_summary(adapter_eval),
        "category_deltas": category_deltas,
    }


def _latest_eval_results(project_root: Path) -> tuple[dict | None, dict | None]:
    evals_root = project_root / ".morpheus" / "training" / "evals"
    if evals_root.is_symlink():
        raise ValueError(f"Eval registry must not be a symlink: {evals_root}")
    reject_symlink_components(evals_root, "Eval registry")
    if not evals_root.is_dir():
        return None, None
    base_results = []
    adapter_results = []
    for results_path in sorted(evals_root.glob("*/eval_results.json"), key=lambda item: item.as_posix()):
        if results_path.is_symlink():
            continue
        reject_symlink_components(results_path, "Eval results")
        result = _read_json(results_path, "Eval results")
        if result.get("base_only"):
            base_results.append(result)
        elif result.get("adapter_id"):
            adapter_results.append(result)
    return (
        base_results[-1] if base_results else None,
        adapter_results[-1] if adapter_results else None,
    )


def _category_metrics(eval_results: dict | None) -> dict:
    if not eval_results:
        return {}
    metrics = eval_results.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    by_category = metrics.get("by_category")
    return by_category if isinstance(by_category, dict) else {}


def _eval_summary(eval_results: dict | None) -> dict | None:
    if not eval_results:
        return None
    return {
        "eval_id": eval_results.get("eval_id"),
        "adapter_id": eval_results.get("adapter_id"),
        "base_only": bool(eval_results.get("base_only")),
        "dataset_id": eval_results.get("dataset_id"),
        "pass_rate": (eval_results.get("metrics") or {}).get("pass_rate"),
    }


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
