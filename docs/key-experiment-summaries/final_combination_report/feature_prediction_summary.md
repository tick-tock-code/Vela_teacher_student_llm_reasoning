# Feature Prediction Combination Summary

This is the paper-facing Stage A summary for predicting `v25_policies` reasoning targets from richer input-feature combinations. The selected model for the downstream success-transfer experiment is:

`data/saved_model_configs/2026-04-21_132505_959940_model_testing::v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007`

The selection uses the stability-adjusted screen score rather than raw R2 alone.

## Run Metadata

- Run ID: `2026-04-22_184529_729730_model_testing`
- Mode: `model_testing_mode` (train-only Stage A screening)
- Target family: `v25_policies` (16 regression targets)
- Feature sets: 19
- Model family: `linear_l2` (`ridge`, single-target)
- CV protocol: stratified 3-fold, repeats = 16 (seeds: `42 + 10000*k`)
- Nested tuning: `off`
- Held-out/test usage: `none`

## Full 19-Set Ranking (R²)

| Rank | Feature set | R² mean | R² std | Screen score | RMSE mean | MAE mean | Recommended |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle` | 0.4283 | 0.0013 | 0.4276 | 0.2638 | 0.1757 | True |
| 2 | `hq_plus_lambda_policies_plus_sentence_bundle` | 0.4286 | 0.0030 | 0.4271 | 0.2635 | 0.1759 | True |
| 3 | `llm_engineering_plus_lambda_policies_plus_sentence_bundle` | 0.4218 | 0.0015 | 0.4210 | 0.2651 | 0.1770 | False |
| 4 | `hq_plus_llm_engineering_plus_sentence_bundle` | 0.4207 | 0.0014 | 0.4200 | 0.2657 | 0.1807 | False |
| 5 | `lambda_policies_plus_sentence_bundle` | 0.4211 | 0.0029 | 0.4196 | 0.2651 | 0.1774 | False |
| 6 | `hq_plus_sentence_bundle` | 0.4186 | 0.0015 | 0.4178 | 0.2659 | 0.1814 | False |
| 7 | `lambda_policies_plus_sentence_prose` | 0.4086 | 0.0027 | 0.4072 | 0.2682 | 0.1806 | False |
| 8 | `hq_plus_sentence_prose` | 0.4057 | 0.0013 | 0.4051 | 0.2691 | 0.1849 | False |
| 9 | `lambda_policies_plus_sentence_structured` | 0.4015 | 0.0027 | 0.4002 | 0.2697 | 0.1815 | False |
| 10 | `hq_plus_sentence_structured` | 0.3953 | 0.0011 | 0.3947 | 0.2714 | 0.1864 | False |
| 11 | `llm_engineering_plus_sentence_bundle` | 0.3898 | 0.0014 | 0.3891 | 0.2736 | 0.1891 | False |
| 12 | `llm_engineering_plus_sentence_prose` | 0.3715 | 0.0014 | 0.3708 | 0.2782 | 0.1942 | False |
| 13 | `sentence_bundle` | 0.3647 | 0.0011 | 0.3642 | 0.2782 | 0.1945 | False |
| 14 | `llm_engineering_plus_sentence_structured` | 0.3500 | 0.0015 | 0.3493 | 0.2830 | 0.1984 | False |
| 15 | `sentence_prose` | 0.3403 | 0.0009 | 0.3398 | 0.2839 | 0.2013 | False |
| 16 | `sentence_structured` | 0.3119 | 0.0008 | 0.3114 | 0.2900 | 0.2064 | False |
| 17 | `lambda_policies` | 0.2741 | 0.0025 | 0.2729 | 0.2990 | 0.2077 | False |
| 18 | `hq_baseline` | 0.2507 | 0.0011 | 0.2502 | 0.3048 | 0.2180 | False |
| 19 | `llm_engineering` | 0.1383 | 0.0011 | 0.1378 | 0.3303 | 0.2487 | False |

## Stability Notes

- Highest raw mean R²: `hq_plus_lambda_policies_plus_sentence_bundle` (0.4286)
- Best stability-adjusted score: `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle` (screen 0.4276)
- Highest variance among top contenders: `hq_plus_lambda_policies_plus_sentence_bundle` (std 0.0030)

## Per-Target Highlights (Top 2 Feature Sets)

### `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle`

- Best 5 targets by mean R²:
  - `v25_p112`: 0.5142
  - `v25_p58`: 0.5056
  - `v25_p80`: 0.5023
  - `v25_p116`: 0.5005
  - `v25_p52`: 0.4971
- Hardest 5 targets by mean R²:
  - `v25_p161`: 0.2663
  - `v25_p11`: 0.3143
  - `v25_p72`: 0.3249
  - `v25_p143`: 0.3606
  - `v25_p121`: 0.3656

### `hq_plus_lambda_policies_plus_sentence_bundle`

- Best 5 targets by mean R²:
  - `v25_p112`: 0.5134
  - `v25_p58`: 0.5078
  - `v25_p116`: 0.5007
  - `v25_p80`: 0.5001
  - `v25_p52`: 0.4966
- Hardest 5 targets by mean R²:
  - `v25_p161`: 0.2568
  - `v25_p11`: 0.3183
  - `v25_p72`: 0.3249
  - `v25_p143`: 0.3727
  - `v25_p121`: 0.3765

## Source Artifacts

- `tmp/runs/teacher_student_distillation_v1/2026-04-22_184529_729730_model_testing/feature_set_screening.csv`
- `tmp/runs/teacher_student_distillation_v1/2026-04-22_184529_729730_model_testing/feature_set_screening_repeat_summary.csv`
- `tmp/runs/teacher_student_distillation_v1/2026-04-22_184529_729730_model_testing/feature_set_screening_per_target.csv`
