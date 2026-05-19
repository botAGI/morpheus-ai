# Morpheus ML Core Live Report

Date: 2026-05-19

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

## Current Passing Dogfood Result

Raw JSON:

```text
.morpheus/lab/live_runs/dogfood_mlx_answerclean_fulleval_20260519T165319Z.json
```

Lab report:

```text
.morpheus/lab/lab_20260519T165324095828Z/REPORT.md
```

Adapter path:

```text
.morpheus/lab/lab_20260519T165324095828Z/training/adapter
```

Metrics:

| Metric | Value |
| --- | ---: |
| Verdict | `ML_CORE_PASS` |
| Production ready | `true` |
| Eval gate activation allowed | `true` |
| Strict accepted candidates | 51 |
| Training examples | 214 |
| Eval seed items | 57 |
| Held-out eval items | 55 |
| Full eval coverage | 1.0 |
| All held-out items evaluated | `true` |
| Adapter pass rate | 1.0 |
| Base pass rate | 0.8125 |
| Adapter delta | +0.1875 |
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

## Remaining Reliability Work

The live full-eval command writes final JSON only after completion. During
long MLX runs the output file stays empty because stdout is buffered.

Next reliability slice:

- Add progress logging or incremental progress artifacts for MLX lab eval.
- Include current eval item index, total items, mode (`base`/`adapter`), and
  elapsed time.
- Keep final JSON stable for automation.

## Verdict

`ML_CORE_PASS` for the current autonomous dogfood MLX learning-lab gate.

This is still a lab result, not an automatic adapter activation or release.
The source spans/check layer remains the source of truth; adapter output is
allowed only after eval and rollback controls.
