# Reproduction Delta Table (Supervisor Share)

Reference headline targets are from:
`Feature Repository/REPRODUCING_HEADLINE_SCORES.md` (table rows 1-9).

Latest reproduced run used for comparison:
`tmp/runs/teacher_student_distillation_v1/2026-04-13_204746_041859_success_reproduction/reproduction_results.csv`

`delta = reproduced_test_f0.5 - headline_target_f0.5`

| # | Experiment | Headline Target F0.5 | Reproduced Test F0.5 | Delta |
|---|---|---:|---:|---:|
| 1 | HQ only | 0.2730 | 0.2726 | -0.0004 |
| 2 | LLM Engineering only | 0.2840 | 0.2843 | +0.0003 |
| 3 | Policy Induction only (v25) | 0.2900 | 0.2905 | +0.0005 |
| 4 | HQ + top-30 lambda policy | 0.2840 | 0.2711 | -0.0129 |
| 5 | HQ + top-40 lambda policy | 0.2930 | 0.2991 | +0.0061 |
| 6 | HQ + Policy Induction (v25) | 0.3000 | 0.3005 | +0.0005 |
| 7 | LLM Engineering + top-30 lambda policy | 0.2680 | 0.2858 | +0.0178 |
| 8 | LLM Engineering + top-40 lambda policy | 0.2830 | 0.2790 | -0.0040 |
| 9 | LLM Engineering + Policy Induction (v25) | 0.3340 | 0.3344 | +0.0004 |

## Quick Readout

- 6/9 experiments are very close (absolute delta <= 0.001).
- Largest disagreement is in the lambda top-k experiments (4, 5, 7, 8), especially:
  - Exp 4: `-0.0129`
  - Exp 7: `+0.0178`
