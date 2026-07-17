"""Benchmark readiness reports for Morpheus learning datasets."""
from datetime import datetime, timezone
import json
from pathlib import Path

from morpheus.core.learning.eval import (
    check_activation_gate,
    latest_eval_category_comparison,
    run_learning_eval,
)
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
    dataset_id = manifest.get("dataset_id")
    benchmark_allowed = bool(quality.get("benchmark_allowed"))
    benchmark_blockers = list(quality.get("benchmark_blockers") or [])
    eval_comparison = latest_eval_category_comparison(
        project_root,
        dataset_id=str(dataset_id or ""),
    )
    if benchmark_allowed and dataset_id and eval_comparison["base_eval"] is None:
        run_learning_eval(
            project_root,
            base_only=True,
            dry_run=True,
            dataset_id=str(dataset_id),
        )
        eval_comparison = latest_eval_category_comparison(
            project_root,
            dataset_id=str(dataset_id),
        )
    latest_adapter_eval = eval_comparison["adapter_eval"]
    if not benchmark_allowed:
        activation_gate = {"allowed": False, "reason": "benchmark_blocked"}
    elif latest_adapter_eval is None:
        activation_gate = {"allowed": False, "reason": "missing_adapter_eval"}
    else:
        activation_gate = check_activation_gate(
            project_root,
            str(latest_adapter_eval["adapter_id"]),
            eval_id=str(latest_adapter_eval["eval_id"]),
        )
    activation_ready = bool(benchmark_allowed and activation_gate["allowed"])
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
        "dataset_id": dataset_id,
        "dataset_sha256": manifest.get("dataset_sha256"),
        "examples_count": int(manifest.get("examples_count") or 0),
        "eval_items_count": int(manifest.get("eval_items_count") or 0),
        "trainable_candidate_count": int(manifest.get("trainable_candidate_count") or 0),
        "benchmark_allowed": benchmark_allowed,
        "benchmark_blockers": benchmark_blockers,
        "benchmark_gate": quality.get("benchmark_gate") or {},
        "latest_base_eval": eval_comparison["base_eval"],
        "latest_adapter_eval": latest_adapter_eval,
        "category_deltas": eval_comparison["category_deltas"],
        "critical_regressions": eval_comparison["critical_regressions"],
        "activation_gate": activation_gate,
        "activation_ready": activation_ready,
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
        f"- Base eval: `{(report.get('latest_base_eval') or {}).get('eval_id') or 'none'}`",
        f"- Adapter eval: `{(report.get('latest_adapter_eval') or {}).get('eval_id') or 'none'}`",
        f"- Activation ready: {report.get('activation_ready', False)}",
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

    regressions = report.get("critical_regressions") or []
    if regressions:
        lines.extend(["", "## Critical Regressions", ""])
        for item in regressions:
            lines.append(
                f"- `{item['category']}`: "
                + ", ".join(item.get("reasons") or [])
            )

    lines.extend([
        "",
        "## Next Command",
        "",
        f"`{report['next_command']}`",
    ])
    return "\n".join(lines).rstrip() + "\n"
