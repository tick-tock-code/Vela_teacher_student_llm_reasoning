# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_024204_856747_saved_config_evaluation`
- Target family: `v25_policies`
- Feature set: `lambda_policies_plus_sentence_bundle`
- Combo refs: ['data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003', 'data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004', 'data/saved_model_configs/2026-04-20_010135_130404_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002']
- Reproduction reference run: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_024252_015048_success_reproduction`

## CV Validation Performance (Source Runs)

These are CV metrics taken from the source model-testing runs used to build the saved model bundles.

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| ridge | 0.4229 | 0.0800 | 0.2647 | 0.1774 |
| xgb3_regressor | 0.4202 | 0.0789 | 0.2656 | 0.1734 |
| mlp_regressor | 0.4172 | 0.0733 | 0.2660 | 0.1710 |
| combined_best | 0.4326 | 0.0777 | 0.2626 | 0.1706 |

`combined_best` is defined per target by highest source CV R² across ridge/xgb/mlp (tie-break: ridge > xgb3_regressor > mlp_regressor).

## Held-out Test Performance (Reasoning Agreement)

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| xgb3_regressor | 0.4201 | 0.0835 | 0.2631 | 0.1776 |
| combined_best | 0.4166 | 0.0888 | 0.2633 | 0.1834 |
| ridge | 0.4008 | 0.0941 | 0.2666 | 0.1929 |
| mlp_regressor | 0.4000 | 0.0964 | 0.2663 | 0.1874 |

## Held-out Test Performance (Success Transfer)

### reasoning_pred-only

| model_set_id | f0_5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---:|---:|---:|---:|---:|---:|
| combined_best | 0.2821 | 0.7268 | 0.2480 | 0.3913 | 0.1333 | 0.2700 |
| mlp_regressor | 0.2797 | 0.7221 | 0.2430 | 0.3671 | 0.1432 | 0.2700 |
| ridge | 0.2756 | 0.7210 | 0.2452 | 0.3386 | 0.1580 | 0.2600 |
| xgb3_regressor | 0.2570 | 0.7271 | 0.2475 | 0.5072 | 0.0864 | 0.3000 |

### HQ + reasoning_pred

| model_set_id | f0_5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---:|---:|---:|---:|---:|---:|
| xgb3_regressor | 0.3260 | 0.7304 | 0.2333 | 0.3969 | 0.1901 | 0.3000 |
| mlp_regressor | 0.3226 | 0.7311 | 0.2361 | 0.3469 | 0.2519 | 0.2700 |
| ridge | 0.3222 | 0.7281 | 0.2368 | 0.3513 | 0.2420 | 0.2700 |
| combined_best | 0.3202 | 0.7318 | 0.2359 | 0.3767 | 0.2000 | 0.3000 |

### LLM-eng + reasoning_pred

| model_set_id | f0_5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---:|---:|---:|---:|---:|---:|
| mlp_regressor | 0.3299 | 0.7385 | 0.2640 | 0.3871 | 0.2074 | 0.2800 |
| ridge | 0.3100 | 0.7343 | 0.2630 | 0.3838 | 0.1753 | 0.3000 |
| combined_best | 0.3076 | 0.7432 | 0.2734 | 0.3918 | 0.1654 | 0.3000 |
| xgb3_regressor | 0.3058 | 0.7381 | 0.2670 | 0.4167 | 0.1481 | 0.3000 |

## Reproduction Consistency Check

- Tolerance: ±0.005 F0.5

| experiment_id | headline_target_f0_5 | reproduced_test_f0_5 | delta_f0_5 | abs_delta_f0_5 | within_tolerance |
|---|---:|---:|---:|---:|---|
| hq_only | 0.2730 | 0.2726 | -0.0004 | 0.0004 | True |
| hq_plus_policy_induction | 0.3000 | 0.3005 | +0.0005 | 0.0005 | True |
| llm_engineering_only | 0.2840 | 0.2843 | +0.0003 | 0.0003 | True |
| llm_engineering_plus_policy_induction | 0.3340 | 0.3344 | +0.0004 | 0.0004 | True |

## Combined Best Assignment (CV-R2 Source)

| target_id | selected_model_id | selected_combo_id | cv_r2 |
|---|---|---|---:|
| v25_p1 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3770 |
| v25_p11 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3262 |
| v25_p112 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.5022 |
| v25_p116 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.5125 |
| v25_p121 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3633 |
| v25_p135 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3983 |
| v25_p143 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.3901 |
| v25_p150 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.4709 |
| v25_p157 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.4885 |
| v25_p161 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.2838 |
| v25_p38 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.4366 |
| v25_p52 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.4947 |
| v25_p55 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.5072 |
| v25_p58 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.5374 |
| v25_p72 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3282 |
| v25_p80 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.5044 |

## Detailed Data Tables (CSV Artifacts)

- `reasoning_transfer_cv_summary.csv`
- `reasoning_transfer_cv_per_target.csv`
- `reasoning_transfer_per_target.csv`
- `reasoning_transfer_summary.csv`
- `success_transfer_metrics.csv`
- `combined_best_assignment.csv`
- `reproduction_consistency_check.csv`
