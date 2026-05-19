# Morpheus ML Core Live Report

Date: 2026-05-19

## Slice

Dataset/eval quality hardening and full live eval gating for the autonomous MLX
learning lab.

## Problem Found

The MLX backend was mechanically working, but the adapter could still fail the
quality gate because eval-aligned recall prompts asked the model to memorize
source line numbers, such as `AGENTS.md:10`, instead of meaningful project
claims. This produced command-fact confusion across similar AGENTS.md entries.

## Fix

- Eval-aligned training prompts now use claim-aware questions instead of
  line-number lookup prompts.
- Train and eval prompts still match exactly for MLX curriculum rows.
- MLX eval coverage is now explicit in JSON and Markdown reports.
- `--eval-limit 0` now means full eval coverage.
- Sampled eval can pass the ML core smoke gate, but it cannot mark an adapter
  production-ready.
- Critical safety eval categories are always selected:
  - `outdated_claim_correction`
  - `unsupported_claim_refusal`

## Live Dogfood Result

Command:

```bash
morpheus learn lab . --dogfood --backend mlx --eval-limit 0
```

Raw JSON:

```text
.morpheus/lab/live_runs/dogfood_mlx_fulleval_20260519T145700Z.json
```

Lab report:

```text
.morpheus/lab/lab_20260519T145709536177Z/REPORT.md
```

Latest report pointer:

```text
.morpheus/lab/LATEST_REPORT.md
```

Metrics:

| Metric | Value |
| --- | ---: |
| Verdict | `ML_CORE_PASS` |
| Production gate | `true` |
| Strict accepted candidates | 57 |
| Training examples | 234 |
| Eval seed items | 63 |
| Evaluated live items | 63 |
| Eval coverage rate | 1.0 |
| Critical safety items evaluated | 4 / 4 |
| Adapter pass rate | 1.0 |
| Base pass rate | 0.7143 |
| Adapter delta | +0.2857 |
| Adapter hallucination rate | 0.0 |
| Critical failures | 0 |
| Regressions | 0 |

Adapter path:

```text
.morpheus/lab/lab_20260519T145709536177Z/training/adapter
```

The adapter was not activated automatically.

## Prior Failing Run

Before claim-aware prompts, the same live dogfood lane produced
`ML_CORE_PARTIAL`: adapter pass rate was 0.5 with blocker
`pass_rate_below_threshold`. The failures were concentrated in similar
AGENTS.md command facts.

After claim-aware prompts, a 12-item sampled eval produced `ML_CORE_PASS`, but
the gate now treats sampled eval as insufficient for production readiness.
The full 63-item eval above is the first live run with complete coverage.

## Verdict

`ML_CORE_PASS` for the current full dogfood gate.

This is still an autonomous lab result, not a release or automatic production
activation. Repeated full-eval runs and held-out eval expansion should be the
next quality step before treating adapter activation as production-safe.
