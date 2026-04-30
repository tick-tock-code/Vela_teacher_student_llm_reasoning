# Combination Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_191111_782723_saved_config_evaluation`
- Selected combo ref: `data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003`
- Target family: `v25_policies`

## Source CV Validation (Selected Combo)

These metrics come from the source Stage-A model-testing run (train-only CV).

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4229 | 0.0800 | 0.2647 | 0.1774 |

## Held-out Test Reasoning Agreement

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4008 | 0.0941 | 0.2666 | 0.1929 |

## Success Transfer (Predicted Reasoning Appended)

| success_branch_id | model_variant | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | test_f0_5 | train_cv_roc_auc | test_roc_auc | train_cv_pr_auc | test_pr_auc | selected_c_final | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_engineering__without_override | single_model | llm_engineering | without_override | 0.3463 | 0.3099 | 0.7683 | 0.7318 | 0.2827 | 0.2610 | 1.0000 | 0.3050 |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `combination_transfer_success_train_threshold_sweep.csv`.

| success_branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| llm_engineering__without_override | single_model | without_override | 0.3100 | 0.3379 | 0.3978 | 0.2109 | 0.3050 | 0.3463 | 0.3099 |
| llm_engineering__without_override | single_model | without_override | 0.3000 | 0.3375 | 0.3884 | 0.2216 | 0.3050 | 0.3463 | 0.3099 |
| llm_engineering__without_override | single_model | without_override | 0.2900 | 0.3371 | 0.3799 | 0.2325 | 0.3050 | 0.3463 | 0.3099 |

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
| v25_p1 | 0.3710 | 0.3345 | 0.2550 |
| v25_p11 | 0.3140 | 0.2492 | 0.1518 |
| v25_p112 | 0.5022 | 0.3370 | 0.2621 |
| v25_p116 | 0.5006 | 0.3142 | 0.2272 |
| v25_p121 | 0.3530 | 0.1704 | 0.0788 |
| v25_p135 | 0.3946 | 0.3081 | 0.2288 |
| v25_p143 | 0.3652 | 0.1357 | 0.0533 |
| v25_p150 | 0.4709 | 0.3356 | 0.2659 |
| v25_p157 | 0.4885 | 0.2225 | 0.1259 |
| v25_p161 | 0.2670 | 0.2082 | 0.1074 |
| v25_p38 | 0.4366 | 0.1726 | 0.0790 |
| v25_p52 | 0.4947 | 0.3417 | 0.2628 |
| v25_p55 | 0.4844 | 0.3001 | 0.2181 |
| v25_p58 | 0.5080 | 0.2687 | 0.1701 |
| v25_p72 | 0.3108 | 0.2601 | 0.1661 |
| v25_p80 | 0.5044 | 0.2761 | 0.1861 |

## Per-Target Held-out Test Reasoning Metrics

| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.3418 | 0.3408 | 0.2720 |
| v25_p11 | 0.2734 | 0.2556 | 0.1747 |
| v25_p112 | 0.4972 | 0.3385 | 0.2744 |
| v25_p116 | 0.4829 | 0.3202 | 0.2490 |
| v25_p121 | 0.3073 | 0.1667 | 0.0889 |
| v25_p135 | 0.3891 | 0.3064 | 0.2403 |
| v25_p143 | 0.2410 | 0.1409 | 0.0678 |
| v25_p150 | 0.4951 | 0.3261 | 0.2661 |
| v25_p157 | 0.4862 | 0.2168 | 0.1366 |
| v25_p161 | 0.2851 | 0.2110 | 0.1244 |
| v25_p38 | 0.3854 | 0.1753 | 0.0931 |
| v25_p52 | 0.4916 | 0.3448 | 0.2781 |
| v25_p55 | 0.4650 | 0.3052 | 0.2362 |
| v25_p58 | 0.4831 | 0.2734 | 0.1895 |
| v25_p72 | 0.2887 | 0.2681 | 0.1908 |
| v25_p80 | 0.5004 | 0.2755 | 0.2039 |
