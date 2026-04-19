# Run Setup Summary

- Mode: `reasoning_distillation_mode`
- Source run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\ut\c_0de546db\runs\smoke_feature_repository_pipeline\2026-04-19_012706_052461_reasoning_distillation`

## Run Summary

# Run Summary

- Run mode: `reasoning_distillation_mode`
- Target family: `taste_policies`
- Target task kind: `classification`
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

- Target family: `taste_policies`
- Task kind: `classification`

## Public OOF (Averaged Across Targets)

| feature_set_id | model_id | mean_f0_5 | mean_roc_auc | mean_pr_auc | mean_precision | mean_recall |
|---|---:|---:|---:|---:|---:|---:|
| hq_baseline | logreg_classifier | 0.8581 | 0.8565 | 0.8316 | 0.8636 | 0.8611 |

## Held-Out (Averaged Across Targets)

| feature_set_id | model_id | mean_f0_5 | mean_roc_auc | mean_pr_auc | mean_precision | mean_recall |
|---|---:|---:|---:|---:|---:|---:|
| hq_baseline | logreg_classifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
