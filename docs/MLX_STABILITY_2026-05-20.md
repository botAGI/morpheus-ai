# Morpheus MLX Stability Report - 2026-05-20

## Command

```bash
morpheus learn lab . --dogfood --backend mlx --eval-limit 0 --repeat 2
```

## Result

- Stability ID: `stability_20260520T161542494930Z`
- Raw output: `.morpheus/lab/live_runs/dogfood_mlx_stability_fixed_20260520T161542Z.json`
- Stability report: `.morpheus/lab/stability/stability_20260520T161542494930Z/stability_report.md`
- Verdict: `ML_CORE_PASS`
- Stability passed: `true`
- Stability blockers: `[]`

## Runs

| Run | Lab ID | Accepted | Examples | Eval items | Held-out | Base pass | Adapter pass | Delta | Critical failures | Regressions | Production blockers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `lab_20260520T161542495419Z` | 69 | 290 | 75 | 73 | 0.7973 | 0.9932 | +0.1959 | 0 | 0 | `[]` |
| 2 | `lab_20260520T163443316809Z` | 69 | 290 | 75 | 73 | 0.7973 | 0.9932 | +0.1959 | 0 | 0 | `[]` |

## Interpretation

The previous live issue was not an MLX backend failure. It was a data-quality
problem: the adapter could answer "not a LoRA trainer" while drifting into an
old weights-only framing. The fix strengthened hard-negative examples and the
lab system prompt around the product rule:

```text
First verify. Then learn.
```

The repeated live run now shows that the adapter can preserve the truth-layer
safety rules across the full eval set without critical failures or regressions.

## Safety

- No raw markdown training was used.
- Strict machine-accepted source-backed candidates were the only positive
  training source.
- Hard-negative safety examples covered LoRA framing, raw markdown training,
  cloud defaults, and adapter activation gates.
- No adapter was activated automatically.
- Adapter output remains downstream of source spans, `morpheus check`, eval,
  and rollback controls.
