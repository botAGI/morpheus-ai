# Changelog

## [0.2.0-alpha]

DECISION: morpheus learn lab runs an autonomous local learning experiment.
DECISION: The lab writes strict_accept_report.md for audit.
DECISION: The lab writes accepted_candidates.jsonl before dataset generation.
DECISION: The lab writes train.jsonl, valid.jsonl, and test.jsonl.
DECISION: The lab writes eval.seed.jsonl for base-vs-adapter checks.
DECISION: The lab returns ML_CORE_PARTIAL when training is skipped.
DECISION: The lab returns ML_CORE_DATASET_BLOCKED when data thresholds fail.
DECISION: The lab never activates adapters automatically.

