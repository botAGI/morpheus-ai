# Morpheus ML Core Live Report

Date: 2026-05-20

## Slice

Autonomous MLX learning-lab quality loop on real dogfood data.

The goal was to move from "training mechanically works" to "the dataset,
held-out eval, and production gate catch real adapter regressions and can be
polished until the adapter passes the full gate."

## Fixes In This Loop

- Preserved code-only claim topics in held-out prompts instead of collapsing
  them to `this claim`.
- Added safety-rule training paraphrases without leaking held-out eval prompts.
- Classified `morpheus compile` and `morpheus stale` as command facts.
- Made command scoring require expected command flags, so
  `morpheus compile --review` does not satisfy
  `morpheus compile --semantic --review`.
- Rejected truncated Markdown/source-line fragments during strict lab accept.
- Normalized dataset/eval answer text by removing leading Markdown list
  markers, so the adapter learns the fact instead of a bare `-`.
- Strengthened hard-negative training examples and the lab system prompt for
  the critical positioning/safety claims: Morpheus is not mainly a LoRA
  trainer, raw markdown is never training data, `morpheus check` stays local by
  default, and adapters require accepted source-backed state plus eval.

## Live Run Progression

All runs used:

```bash
morpheus learn lab . --dogfood --backend mlx --eval-limit 0
```

| Run | Raw JSON | Verdict | Adapter pass | Base pass | Delta | Regressions | Gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Held-out expansion | `.morpheus/lab/live_runs/dogfood_mlx_heldout_fulleval_20260519T153630Z.json` | `ML_CORE_PARTIAL` | 0.9758 | 0.7742 | +0.2016 | 2 | blocked |
| Prompt polish | `.morpheus/lab/live_runs/dogfood_mlx_promptpolish_fulleval_20260519T155925Z.json` | `ML_CORE_PARTIAL` | 0.9677 | 0.8226 | +0.1451 | 2 | blocked |
| Prompt/scoring polish | `.morpheus/lab/live_runs/dogfood_mlx_promptscore_fulleval_20260519T161710Z.json` | `ML_CORE_PARTIAL` | 0.9597 | 0.8065 | +0.1532 | 2 | blocked |
| Strict data gate | `.morpheus/lab/live_runs/dogfood_mlx_datagate_fulleval_20260519T163715Z.json` | `ML_CORE_PARTIAL` | 0.9732 | 0.8125 | +0.1607 | 1 | blocked |
| Answer normalization | `.morpheus/lab/live_runs/dogfood_mlx_answerclean_fulleval_20260519T165319Z.json` | `ML_CORE_PASS` | 1.0 | 0.8125 | +0.1875 | 0 | allowed |
| Current main revalidation | `.morpheus/lab/live_runs/dogfood_mlx_prod_20260520T142318Z.json` | `ML_CORE_PASS` | 1.0 | 0.8125 | +0.1875 | 0 | allowed |
| Repeat stability, run 1 | `.morpheus/lab/live_runs/dogfood_mlx_stability_fixed_20260520T161542Z.json` | `ML_CORE_PASS` | 0.9932 | 0.7973 | +0.1959 | 0 | allowed |
| Repeat stability, run 2 | `.morpheus/lab/live_runs/dogfood_mlx_stability_fixed_20260520T161542Z.json` | `ML_CORE_PASS` | 0.9932 | 0.7973 | +0.1959 | 0 | allowed |

## Current Passing Dogfood Result

Repeat-2 stability JSON:

```text
.morpheus/lab/live_runs/dogfood_mlx_stability_fixed_20260520T161542Z.json
```

Stability report:

```text
.morpheus/lab/stability/stability_20260520T161542494930Z/stability_report.md
```

Run reports:

```text
.morpheus/lab/lab_20260520T161542495419Z/REPORT.md
.morpheus/lab/lab_20260520T163443316809Z/REPORT.md
```

Metrics:

| Metric | Value |
| --- | ---: |
| Stability verdict | `ML_CORE_PASS` |
| Stability passed | `true` |
| Stability blockers | `[]` |
| Runs | `2` |
| Strict accepted candidates per run | 69 |
| Training examples per run | 290 |
| Eval seed items per run | 75 |
| Held-out eval items per run | 73 |
| Full eval coverage | 1.0 |
| All held-out items evaluated | `true` |
| Adapter pass rate | 0.9932 |
| Base pass rate | 0.7973 |
| Adapter delta | +0.1959 |
| Adapter hallucination rate | 0.0 |
| Critical failures | 0 |
| Regressions | 0 |

The adapter was not activated automatically.

## Data Gate Result

The stricter dogfood no-train preflight produced:

```text
.morpheus/lab/live_runs/dogfood_notrain_answerclean_20260519T165532Z.json
```

Summary:

- Strict accepted candidates: `51`
- Training examples: `214`
- Eval seed items: `57`
- Held-out eval items: `55`
- Training allowed: `true`
- Rejected dogfood reasons included `truncated_claim: 6`, `needs_split: 1`,
  and `source_path_not_allowlisted: 7`.

This confirms the lab can filter weak source-line fragments without collapsing
the real dogfood dataset below training thresholds.

## Reliability Revalidation

The previous live full-eval command wrote final JSON only after completion.
During long MLX runs the output file stayed empty because stdout was buffered.
That reliability gap is now addressed and revalidated on the current main
branch with incremental eval progress artifacts:

- `eval/eval_progress.jsonl` records `eval_started`, `mode_started`,
  `item_evaluated`, `mode_completed`, `mode_skipped`, and `eval_completed`
  events.
- Each item event records mode, index, total, category, source candidate ID,
  pass/fail flags, critical-failure flag, and elapsed seconds.
- `eval/progress_summary.json` records final base/adapter evaluated counts,
  pass rates, regression count, coverage, held-out coverage, and status.
- Final JSON output remains stable for automation.

Dogfood progress preflight:

```text
.morpheus/lab/live_runs/dogfood_progress_notrain_20260519T195506Z.json
```

Progress artifacts:

```text
.morpheus/lab/lab_20260519T195510721491Z/eval/eval_progress.jsonl
.morpheus/lab/lab_20260519T195510721491Z/eval/progress_summary.json
```

Observed progress metrics:

- Progress events: `117`
- Base evaluated: `112`
- Adapter evaluated: `0` (`--no-train`)
- Status: `adapter_not_run`
- All held-out items evaluated: `true`

Current full MLX revalidation:

- Run: `.morpheus/lab/live_runs/dogfood_mlx_prod_20260520T142318Z.json`
- Progress log:
  `.morpheus/lab/lab_20260520T142318839346Z/eval/eval_progress.jsonl`
- Progress summary:
  `.morpheus/lab/lab_20260520T142318839346Z/eval/progress_summary.json`
- Base evaluated: `112`
- Adapter evaluated: `112`
- Full eval coverage: `true`
- All held-out items evaluated: `true`
- Progress artifacts updated during both `base` and `adapter` modes.

## Verdict

`ML_CORE_PASS` for the current autonomous dogfood MLX learning-lab gate.

This is still a lab result, not an automatic adapter activation or release.
The source spans/check layer remains the source of truth; adapter output is
allowed only after eval and rollback controls.
