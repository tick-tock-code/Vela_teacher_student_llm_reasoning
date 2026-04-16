# Run Setup Summary

- Mode: `reasoning_distillation_mode`
- Source run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-16_020531_592697_reasoning_distillation`

## Run Summary

# Run Summary

- Run mode: `reasoning_distillation_mode`
- Target family: `v25_policies`
- Target task kind: `regression`
- Target count: 16
- Repository feature banks loaded: 1
- Intermediary banks prepared: 0
- Feature-set comparisons run: 1
- Distillation models run per target: 1
- Max parallel workers: 1
- Public CV strategy: stratified on quantile buckets of the row-wise mean selected target score.
- CV seed repeats: 1 (enabled=False)
- Nested hyperparameter sweep requested: True
- Nested hyperparameter sweep effective: False
- Distillation outer folds used: 3
- Distillation inner folds used for sweep: 0
- Save reasoning prediction CSVs: False
- OOF prediction columns written: 16
- Held-out prediction columns written: 0
- Held-out evaluation was skipped because it was not requested.


## Reasoning Metrics

# Reasoning Metrics Summary

- Target family: `v25_policies`
- Task kind: `regression`

## Public OOF (Averaged Across Targets)

| feature_set_id | model_id | mean_r2 | mean_rmse | mean_mae | mean_pearson | mean_spearman |
|---|---:|---:|---:|---:|---:|---:|
| hq_baseline | mlp_regressor | 0.1461 | 0.3232 | 0.2252 | 0.4329 | 0.4020 |
