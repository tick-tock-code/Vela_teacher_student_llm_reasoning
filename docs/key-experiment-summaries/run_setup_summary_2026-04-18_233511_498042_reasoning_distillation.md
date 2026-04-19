# Run Setup Summary

- Mode: `reasoning_distillation_mode`
- Source run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\ut\c_da569d44\runs\smoke_feature_repository_pipeline\2026-04-18_233511_498042_reasoning_distillation`

## Run Summary

# Run Summary

- Run mode: `reasoning_distillation_mode`
- Target family: `v25_policies`
- Target task kind: `regression`
- Target count: 2
- Repository feature banks loaded: 1
- Intermediary banks prepared: 0
- Feature-set comparisons run: 1
- Distillation models run per target: 1
- Max parallel workers: 2
- Public CV strategy: stratified on quantile buckets of the row-wise mean selected target score.
- CV seed repeats: 1 (enabled=False)
- Nested hyperparameter sweep requested: False
- Nested hyperparameter sweep effective: False
- Distillation outer folds used: 3
- Distillation inner folds used for sweep: 0
- Save reasoning prediction CSVs: True
- OOF prediction columns written: 2
- Held-out prediction columns written: 2


## Reasoning Metrics

# Reasoning Metrics Summary

- Target family: `v25_policies`
- Task kind: `regression`

## Public OOF (Averaged Across Targets)

| feature_set_id | model_id | mean_r2 | mean_rmse | mean_mae | mean_pearson | mean_spearman |
|---|---:|---:|---:|---:|---:|---:|
| hq_baseline | ridge | 0.9999 | 0.0009 | 0.0008 | 1.0000 | 1.0000 |

## Held-Out (Averaged Across Targets)

| feature_set_id | model_id | mean_r2 | mean_rmse | mean_mae | mean_pearson | mean_spearman |
|---|---:|---:|---:|---:|---:|---:|
| hq_baseline | ridge | 1.0000 | 0.0002 | 0.0002 | 1.0000 | 1.0000 |
