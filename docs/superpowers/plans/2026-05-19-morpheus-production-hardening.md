# Morpheus Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn v0.2a1 from an impressive alpha into a source-grounded learning system that reports real readiness honestly and can run repeatable real-dogfood experiments without stale review data.

**Architecture:** Keep the truth layer as the data-quality gate and treat weights as an experimental artifact until eval passes. The immediate work is in the learning lab and review freshness path: lab runs must be reproducible, avoid stale candidate state, preserve human review files, and expose production blockers as metrics.

**Tech Stack:** Python, Typer CLI, Pydantic semantic candidates, local JSONL state under `.morpheus/`, pytest, ruff, optional MLX runtime outside CI.

---

## Research Findings

- Real dogfood no-train currently meets data thresholds: 34 strict accepted candidates, 102 examples, 40 eval items.
- The real review store still contains stale candidates: latest dogfood metrics reported `source_hash_mismatch=10`.
- `morpheus learn lab` can fallback to fixture data, which is useful for testing but must never be mistaken for production readiness.
- Training is still intentionally not run in the latest dogfood result, so `production_ready=false` is correct.
- Next production blocker is data freshness, not more docs or marketing.

## Task 1: Lab Candidate Freshness Isolation

**Files:**
- Modify: `morpheus/core/learning/lab.py`
- Test: `tests/test_learning_lab.py`

- [ ] **Step 1: Write failing test for stale review store isolation**

Add a test that creates a stale `.morpheus/review/semantic_candidates.jsonl`, then runs `morpheus learn lab --dogfood --no-train`. Expected behavior: lab uses ephemeral regenerated candidates, writes `review_source` metadata, and preserves the stale review file unchanged.

- [ ] **Step 2: Run targeted test and verify red**

Run: `.venv/bin/pytest tests/test_learning_lab.py::test_learn_lab_regenerates_ephemeral_candidates_when_review_store_is_stale -q`

Expected: fail because `review_source` metadata does not exist and stale candidates are used directly.

- [ ] **Step 3: Implement ephemeral candidate generation**

Add a lab-only candidate loader that:
- loads review store candidates,
- evaluates strict accept rejection reasons,
- if any candidate fails with `source_hash_mismatch`, generates fresh candidates in memory with `LocalProvider` and `scan_semantic_sources`,
- does not overwrite `.morpheus/review/semantic_candidates.jsonl`,
- returns metadata: `review_source`, `stale_review_candidates`, `ephemeral_candidates_generated`.

- [ ] **Step 4: Verify targeted tests green**

Run: `.venv/bin/pytest tests/test_learning_lab.py::test_learn_lab_regenerates_ephemeral_candidates_when_review_store_is_stale tests/test_learning_lab.py::test_learn_lab_dogfood_no_train_reports_real_data_metrics -q`

Expected: pass.

- [ ] **Step 5: Commit**

Run: `git add morpheus/core/learning/lab.py tests/test_learning_lab.py && git commit -m "fix: isolate lab from stale review candidates"`

## Task 2: Dogfood Dataset Quality Report

**Files:**
- Modify: `morpheus/core/learning/lab.py`
- Test: `tests/test_learning_lab.py`

- [ ] **Step 1: Write failing test for dataset quality metrics**

Require lab summary and `REPORT.md` to include category coverage, source path coverage, examples per candidate, and eval item count by category.

- [ ] **Step 2: Implement quality metrics**

Compute quality metrics from `accepted_candidates.jsonl`, dataset manifest, and `eval.seed.jsonl`. Add `dataset_quality` to JSON summary and report.

- [ ] **Step 3: Verify targeted tests green**

Run: `.venv/bin/pytest tests/test_learning_lab.py::test_learn_lab_reports_dataset_quality_metrics -q`

Expected: pass.

- [ ] **Step 4: Commit**

Run: `git add morpheus/core/learning/lab.py tests/test_learning_lab.py && git commit -m "feat: report lab dataset quality metrics"`

## Task 3: Eval Gate Calibration

**Files:**
- Modify: `morpheus/core/learning/lab.py`
- Test: `tests/test_learning_lab.py`

- [ ] **Step 1: Write failing test for explicit eval gate reasons**

Require `eval_gate` to include pass rate threshold, hallucination threshold, critical failures, regression count, and whether activation would be blocked.

- [ ] **Step 2: Implement eval gate object**

Add `eval_gate` to lab summary and report. Keep adapter activation impossible in lab.

- [ ] **Step 3: Verify targeted tests green**

Run: `.venv/bin/pytest tests/test_learning_lab.py::test_learn_lab_reports_eval_gate_reasons -q`

Expected: pass.

- [ ] **Step 4: Commit**

Run: `git add morpheus/core/learning/lab.py tests/test_learning_lab.py && git commit -m "feat: add explicit lab eval gate"`

## Task 4: Real Dogfood MLX Smoke Run

**Files:**
- No source changes unless a bug appears.
- Output: `.morpheus/lab/<lab_id>/REPORT.md`

- [ ] **Step 1: Verify MLX availability**

Run: `which mlx_lm.lora || true && which mlx_lm.generate || true`

- [ ] **Step 2: Run dogfood no-train again**

Run: `.venv/bin/morpheus learn lab . --dogfood --no-train`

Expected: real source mode, fresh/ephemeral candidates, train_allowed true.

- [ ] **Step 3: If MLX exists, run bounded smoke training**

Run: `.venv/bin/morpheus learn lab . --dogfood --backend mlx --max-iters 20`

Expected: no activation, report with base vs adapter eval. If model/runtime is unavailable, report exact blocker and do not fake pass.

## Final Gates

Run after each committed task:

```bash
.venv/bin/ruff check .
.venv/bin/pytest tests/ -q
.venv/bin/morpheus wake . --private
.venv/bin/morpheus verify --all
.venv/bin/morpheus check --input tests/fixtures/check_stale_input.txt --local || true
.venv/bin/morpheus check --input tests/fixtures/check_correct_input.txt --local || true
.venv/bin/morpheus learn lab . --dogfood --no-train
```

Do not tag, release, publish, or push unless explicitly instructed.
