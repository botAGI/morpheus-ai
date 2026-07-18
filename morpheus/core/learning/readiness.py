"""Pure benchmark readiness policy for Morpheus learning datasets."""

from morpheus.core.learning.categories import CANONICAL_BENCHMARK_CATEGORIES
from morpheus.core.learning.dataset_validation import (
    manifest_count,
    validation_blocker_messages,
)


BENCHMARK_MIN_TRAINABLE = 20
BENCHMARK_MIN_EXAMPLES = 100
BENCHMARK_MIN_EVAL_ITEMS = 30
BENCHMARK_MIN_SOURCE_PATHS = 3
BENCHMARK_CLASS_MINIMUMS = {
    "product": 1,
    "command": 2,
    "architecture": 1,
    "security": 1,
    "convention": 1,
}
BENCHMARK_EVAL_CATEGORY_MINIMUMS = {
    category: 1 for category in sorted(CANONICAL_BENCHMARK_CATEGORIES)
}


def benchmark_readiness_gate(manifest: dict, validation: dict) -> dict:
    """Evaluate benchmark readiness from one validated dataset snapshot."""
    manifest = manifest if isinstance(manifest, dict) else {}
    validation = validation if isinstance(validation, dict) else {}
    raw_class_counts = manifest.get("class_counts")
    class_counts = raw_class_counts if isinstance(raw_class_counts, dict) else {}
    raw_route_counts = manifest.get("route_counts")
    route_counts = raw_route_counts if isinstance(raw_route_counts, dict) else {}
    raw_source_paths = manifest.get("source_paths")
    source_paths = raw_source_paths if isinstance(raw_source_paths, list) else []
    raw_eval_coverage = validation.get("eval_coverage")
    eval_coverage = raw_eval_coverage if isinstance(raw_eval_coverage, dict) else {}
    raw_eval_category_counts = eval_coverage.get("by_category")
    eval_category_counts = (
        raw_eval_category_counts
        if isinstance(raw_eval_category_counts, dict)
        else {}
    )

    trainable_count = manifest_count(manifest, "trainable_candidate_count")
    examples_count = manifest_count(manifest, "examples_count")
    eval_items_count = manifest_count(manifest, "eval_items_count")
    blockers = []
    if trainable_count < BENCHMARK_MIN_TRAINABLE:
        blockers.append(
            f"trainable_candidate_count < {BENCHMARK_MIN_TRAINABLE}"
        )
    if examples_count < BENCHMARK_MIN_EXAMPLES:
        blockers.append(f"examples < {BENCHMARK_MIN_EXAMPLES}")
    if eval_items_count < BENCHMARK_MIN_EVAL_ITEMS:
        blockers.append(f"eval_items < {BENCHMARK_MIN_EVAL_ITEMS}")
    if len(source_paths) < BENCHMARK_MIN_SOURCE_PATHS:
        blockers.append(f"source_paths < {BENCHMARK_MIN_SOURCE_PATHS}")
    if validation.get("valid") is not True:
        validation_messages = validation_blocker_messages(validation)
        blockers.extend(
            validation_messages or ["dataset provenance invalid"]
        )

    class_requirements = {}
    for class_name, minimum in BENCHMARK_CLASS_MINIMUMS.items():
        count = manifest_count(class_counts, class_name)
        class_requirements[class_name] = {"count": count, "minimum": minimum}
        if count < minimum:
            blockers.append(f"class {class_name} < {minimum}")

    eval_requirements = {}
    for category, minimum in BENCHMARK_EVAL_CATEGORY_MINIMUMS.items():
        count = manifest_count(eval_category_counts, category)
        eval_requirements[category] = {"count": count, "minimum": minimum}
        if count < minimum:
            blockers.append(f"eval_category {category} < {minimum}")

    blockers = sorted(set(blockers))
    return {
        "allowed": not blockers,
        "blockers": blockers,
        "eval_category_counts": dict(sorted(eval_category_counts.items())),
        "requirements": {
            "class_counts": class_requirements,
            "class_groups": {},
            "eval_categories": eval_requirements,
            "route_counts": {
                "adapter_training": {
                    "count": manifest_count(route_counts, "adapter_training"),
                    "minimum": BENCHMARK_MIN_TRAINABLE,
                },
            },
            "source_paths": {
                "count": len(source_paths),
                "minimum": BENCHMARK_MIN_SOURCE_PATHS,
            },
        },
    }
