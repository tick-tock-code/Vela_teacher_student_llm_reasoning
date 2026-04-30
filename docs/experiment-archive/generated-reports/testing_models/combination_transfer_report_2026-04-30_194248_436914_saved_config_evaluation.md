# Combination Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_194248_436914_saved_config_evaluation`
- Selected combo ref: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\ut\c_a6c615ac\bundle::combo_selected`
- Target family: `v25_policies`

## Source CV Validation (Selected Combo)

These metrics come from the source Stage-A model-testing run (train-only CV).

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| combo_selected | hq_plus_sentence_bundle | ridge | single_target | 0.3000 | 0.0000 | 0.2000 | 0.1000 |

## Held-out Test Reasoning Agreement

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| combo_selected | hq_plus_sentence_bundle | ridge | single_target | 0.2500 | 0.0000 | 0.2200 | 0.1100 |

## Success Transfer (Predicted Reasoning Appended)

| success_branch_id | model_variant | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | test_f0_5 | train_cv_roc_auc | test_roc_auc | train_cv_pr_auc | test_pr_auc | selected_c_final | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hq_baseline__with_override | single_model | hq_baseline | with_override | 0.3000 | 0.3100 | 0.6000 | 0.6100 | 0.4000 | 0.4100 | 5.0000 | 0.5000 |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `combination_transfer_success_train_threshold_sweep.csv`.

| success_branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Detailed Data Tables (CSV Artifacts)

- `combination_transfer_source_cv_per_target.csv`
- `combination_transfer_source_cv_summary.csv`
- `combination_transfer_reasoning_per_target.csv`
- `combination_transfer_reasoning_summary.csv`
- `combination_transfer_success_metrics.csv`
- `combination_transfer_success_train_threshold_sweep.csv`

## Per-Target Source CV Metrics

| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.3000 | 0.2000 | 0.1000 |

## Per-Target Held-out Test Reasoning Metrics

| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.2500 | 0.2200 | 0.1100 |
