# Combination Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_191214_462272_saved_config_evaluation`
- Selected combo ref: `data/saved_model_configs/2026-04-21_132505_959940_model_testing::v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007`
- Target family: `v25_policies`

## Source CV Validation (Selected Combo)

These metrics come from the source Stage-A model-testing run (train-only CV).

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007 | hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4286 | 0.0794 | 0.2638 | 0.1758 |

## Held-out Test Reasoning Agreement

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007 | hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4023 | 0.0973 | 0.2659 | 0.1927 |

## Success Transfer (Predicted Reasoning Appended)

| success_branch_id | model_variant | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | test_f0_5 | train_cv_roc_auc | test_roc_auc | train_cv_pr_auc | test_pr_auc | selected_c_final | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_engineering__without_override | single_model | llm_engineering | without_override | 0.3362 | 0.3060 | 0.7653 | 0.7314 | 0.2760 | 0.2614 | 1.0000 | 0.3100 |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `combination_transfer_success_train_threshold_sweep.csv`.

| success_branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| llm_engineering__without_override | single_model | without_override | 0.3000 | 0.3291 | 0.3784 | 0.2165 | 0.3100 | 0.3362 | 0.3060 |
| llm_engineering__without_override | single_model | without_override | 0.3100 | 0.3280 | 0.3860 | 0.2049 | 0.3100 | 0.3362 | 0.3060 |
| llm_engineering__without_override | single_model | without_override | 0.2900 | 0.3272 | 0.3673 | 0.2277 | 0.3100 | 0.3362 | 0.3060 |

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
| v25_p1 | 0.3801 | 0.3327 | 0.2520 |
| v25_p11 | 0.3141 | 0.2500 | 0.1523 |
| v25_p112 | 0.5123 | 0.3335 | 0.2578 |
| v25_p116 | 0.4982 | 0.3150 | 0.2265 |
| v25_p121 | 0.3630 | 0.1701 | 0.0793 |
| v25_p135 | 0.4061 | 0.3055 | 0.2247 |
| v25_p143 | 0.3638 | 0.1366 | 0.0546 |
| v25_p150 | 0.4900 | 0.3295 | 0.2559 |
| v25_p157 | 0.4927 | 0.2225 | 0.1257 |
| v25_p161 | 0.2666 | 0.2095 | 0.1093 |
| v25_p38 | 0.4548 | 0.1700 | 0.0782 |
| v25_p52 | 0.4978 | 0.3408 | 0.2602 |
| v25_p55 | 0.4877 | 0.2994 | 0.2154 |
| v25_p58 | 0.5043 | 0.2700 | 0.1700 |
| v25_p72 | 0.3260 | 0.2575 | 0.1639 |
| v25_p80 | 0.4998 | 0.2778 | 0.1864 |

## Per-Target Held-out Test Reasoning Metrics

| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.3488 | 0.3390 | 0.2699 |
| v25_p11 | 0.2757 | 0.2552 | 0.1756 |
| v25_p112 | 0.4977 | 0.3384 | 0.2735 |
| v25_p116 | 0.4830 | 0.3201 | 0.2508 |
| v25_p121 | 0.3074 | 0.1667 | 0.0906 |
| v25_p135 | 0.3928 | 0.3054 | 0.2396 |
| v25_p143 | 0.2256 | 0.1423 | 0.0694 |
| v25_p150 | 0.5204 | 0.3178 | 0.2569 |
| v25_p157 | 0.4768 | 0.2187 | 0.1382 |
| v25_p161 | 0.2705 | 0.2132 | 0.1265 |
| v25_p38 | 0.4003 | 0.1732 | 0.0933 |
| v25_p52 | 0.4967 | 0.3431 | 0.2770 |
| v25_p55 | 0.4586 | 0.3070 | 0.2384 |
| v25_p58 | 0.4858 | 0.2727 | 0.1895 |
| v25_p72 | 0.2970 | 0.2665 | 0.1897 |
| v25_p80 | 0.4999 | 0.2756 | 0.2043 |
