# Most Recent Model Testing Run

- Mode: `model_testing_mode`
- Source run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-16_140747_859648_model_testing`
- Docs artifacts dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\docs\key-experiment-summaries\most-recent-model-testing-run`
- Copied top-level artifacts: 9
- Copied child-run artifact files: 10

## Artifact Snapshot

- `feature_set_screening.csv` rows: 2
- `feature_set_screening_repeat_metrics.csv` rows: 2
- `model_testing_results.csv` rows: 0
- `feature_set_screening_child_runs.json` rows: 2
- `model_testing_child_runs.json` rows: 0

### Copied Top-Level Files

- `feature_set_screening.csv`
- `feature_set_screening_child_runs.json`
- `feature_set_screening_repeat_metrics.csv`
- `feature_set_screening_report.md`
- `model_testing_report.md`
- `model_testing_results.csv`
- `resolved_config.json`
- `resolved_run_options.json`
- `run_summary.md`

## Run Summary

# Model Testing Summary

- Candidate feature sets: 1
- Repeats: 1
- Estimated Stage A outer fits: 108
- Max parallel workers: 2
- Stage A model families: xgb1
- Output modes: single_target
- Model-family output modes: {'xgb1': ['single_target']}
- Nested requested: False
- Nested effective (Stage A): {'v25_policies::single_target': False, 'taste_policies::single_target': False}
- Nested effective (Stage B): {}
- Stage B enabled: False
- Use latest xgb calibration: False
- Use latest rf calibration: False
- Use latest mlp calibration: False
- Held-out/test features or targets are not used in this mode.

## Screening Summary

# Feature-Set Screening Report

- Repeats: 1
- Stage A models: `xgb1`
- Held-out features/targets: not used
- Recommendation rule: top score + any within `best - 0.005` (max 3).

## v25_policies | single_target

| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---:|---|---:|---:|---:|---:|
| 1 | lambda_policies_plus_sentence_prose | 0.3415 | 0.0000 | 0.3415 | True |

## taste_policies | single_target

| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---:|---|---:|---:|---:|---:|
| 1 | lambda_policies_plus_sentence_prose | 0.7534 | 0.0000 | 0.7534 | True |

## Advanced Stage Summary

# Model Testing Report

- Repeats: 1
- This report compares shortlisted feature sets by model family.

Advanced model stage was skipped or produced no rows.

## Data Tables

### feature_set_screening.csv

| target_family | output_mode | feature_set_id | rank | primary_metric | primary_mean | primary_std | screen_score | recommended_take_forward | r2_mean | rmse_mean | mae_mean | f0_5_mean | roc_auc_mean | pr_auc_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v25_policies | single_target | lambda_policies_plus_sentence_prose | 1 | r2 | 0.341550 | 0.000000 | 0.341550 | True | 0.341550 | 0.284035 | 0.199542 |  |  |  |
| taste_policies | single_target | lambda_policies_plus_sentence_prose | 1 | f0_5 | 0.753359 | 0.000000 | 0.753359 | True |  |  |  | 0.753359 | 0.924356 | 0.769128 |

### feature_set_screening_repeat_metrics.csv (preview)

| feature_set_id | model_id | output_mode | pearson | spearman | mae | rmse | r2 | cv_seed_repeat_count | repeat_index | repeat_seed | target_family | stage | roc_auc | pr_auc | precision | recall | f0_5 | brier | precision_at_01 | precision_at_05 | precision_at_10 | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lambda_policies_plus_sentence_prose | xgb1_regressor | single_target | 0.590224 | 0.586402 | 0.199542 | 0.284035 | 0.341550 | 1.000000 | 0 | 42 | v25_policies | screening |  |  |  |  |  |  |  |  |  |  |
| lambda_policies_plus_sentence_prose | xgb1_classifier | single_target |  |  |  |  |  | 1.000000 | 0 | 42 | taste_policies | screening | 0.924356 | 0.769128 | 0.785164 | 0.666555 | 0.753359 | 0.135040 | 0.863333 | 0.763111 | 0.698556 | 0.797000 |

### model_testing_results.csv

_No rows were produced for this artifact._

### feature_set_screening_child_runs.json

| target_family | repeat_index | repeat_seed | run_dir | output_mode |
|---|---|---|---|---|
| v25_policies | 0 | 42 | C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-16_140748_076929_reasoning_distillation | single_target |
| taste_policies | 0 | 42 | C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-16_140836_152708_reasoning_distillation | single_target |
