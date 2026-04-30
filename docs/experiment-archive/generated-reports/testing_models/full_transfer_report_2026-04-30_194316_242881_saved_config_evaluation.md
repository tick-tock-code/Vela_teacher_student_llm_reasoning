# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_194316_242881_saved_config_evaluation`
- Target family: `v25_policies`
- Feature set: `lambda_policies_plus_sentence_bundle`
- Combo refs: ['data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003']
- Reproduction reference run: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_024252_015048_success_reproduction`

- Success CV repeats: `16` (enabled=True)

## CV Validation Performance (Source Runs)

These are CV metrics taken from the source model-testing runs used to build the saved model bundles.

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| ridge | 0.4229 | 0.0800 | 0.2647 | 0.1774 |

## Held-out Test Performance (Reasoning Agreement)

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| ridge | 0.4008 | 0.0941 | 0.2666 | 0.1929 |

## Held-out Test Performance (Success Transfer)

### reasoning_pred-only

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | with_override | 0.3131 +/- 0.0061 | 0.2980 | 0.7282 | 0.2323 | 0.3104 | 0.2568 | 0.2500 |
| ridge | single_model | without_override | 0.3330 +/- 0.0092 | 0.2749 | 0.7206 | 0.2446 | 0.3226 | 0.1728 | 0.2600 |

### HQ + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | with_override | 0.3067 +/- 0.0062 | 0.3211 | 0.7281 | 0.2368 | 0.3472 | 0.2469 | 0.2650 |
| ridge | single_model | without_override | 0.3275 +/- 0.0087 | 0.3016 | 0.7247 | 0.2622 | 0.3679 | 0.1753 | 0.2650 |

### LLM-eng + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | with_override | 0.3207 +/- 0.0056 | 0.3260 | 0.7378 | 0.2410 | 0.3463 | 0.2642 | 0.2800 |
| ridge | single_model | without_override | 0.3434 +/- 0.0068 | 0.3081 | 0.7343 | 0.2630 | 0.3719 | 0.1827 | 0.2900 |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `success_transfer_train_threshold_sweep.csv`.

| branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hq_plus_pred_reasoning | single_model | with_override | 0.2700 | 0.3020 | 0.3265 | 0.2327 | 0.2650 | 0.3067 | 0.3211 |
| hq_plus_pred_reasoning | single_model | with_override | 0.2500 | 0.3005 | 0.3137 | 0.2574 | 0.2650 | 0.3067 | 0.3211 |
| hq_plus_pred_reasoning | single_model | with_override | 0.2800 | 0.2999 | 0.3295 | 0.2210 | 0.2650 | 0.3067 | 0.3211 |
| hq_plus_pred_reasoning | single_model | without_override | 0.2700 | 0.3217 | 0.3904 | 0.1892 | 0.2650 | 0.3275 | 0.3016 |
| hq_plus_pred_reasoning | single_model | without_override | 0.2800 | 0.3188 | 0.4010 | 0.1755 | 0.2650 | 0.3275 | 0.3016 |
| hq_plus_pred_reasoning | single_model | without_override | 0.2500 | 0.3177 | 0.3595 | 0.2171 | 0.2650 | 0.3275 | 0.3016 |
| llm_engineering_plus_pred_reasoning | single_model | with_override | 0.2900 | 0.3152 | 0.3325 | 0.2610 | 0.2800 | 0.3207 | 0.3260 |
| llm_engineering_plus_pred_reasoning | single_model | with_override | 0.2800 | 0.3150 | 0.3283 | 0.2713 | 0.2800 | 0.3207 | 0.3260 |
| llm_engineering_plus_pred_reasoning | single_model | with_override | 0.3000 | 0.3146 | 0.3361 | 0.2508 | 0.2800 | 0.3207 | 0.3260 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.3000 | 0.3353 | 0.4040 | 0.1998 | 0.2900 | 0.3434 | 0.3081 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.2800 | 0.3351 | 0.3829 | 0.2238 | 0.2900 | 0.3434 | 0.3081 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.2700 | 0.3345 | 0.3736 | 0.2361 | 0.2900 | 0.3434 | 0.3081 |
| pred_reasoning_only | single_model | with_override | 0.2600 | 0.3075 | 0.3271 | 0.2483 | 0.2500 | 0.3131 | 0.2980 |
| pred_reasoning_only | single_model | with_override | 0.2700 | 0.3065 | 0.3310 | 0.2367 | 0.2500 | 0.3131 | 0.2980 |
| pred_reasoning_only | single_model | with_override | 0.2500 | 0.3065 | 0.3213 | 0.2591 | 0.2500 | 0.3131 | 0.2980 |
| pred_reasoning_only | single_model | without_override | 0.2600 | 0.3258 | 0.3868 | 0.2002 | 0.2600 | 0.3330 | 0.2749 |
| pred_reasoning_only | single_model | without_override | 0.2700 | 0.3256 | 0.3997 | 0.1873 | 0.2600 | 0.3330 | 0.2749 |
| pred_reasoning_only | single_model | without_override | 0.2500 | 0.3230 | 0.3722 | 0.2117 | 0.2600 | 0.3330 | 0.2749 |

## Reproduction Consistency Check

- Tolerance: ±0.005 F0.5

| experiment_id | headline_target_f0_5 | reproduced_test_f0_5 | delta_f0_5 | abs_delta_f0_5 | within_tolerance |
|---|---:|---:|---:|---:|---|
| hq_only | 0.2730 | 0.2726 | -0.0004 | 0.0004 | True |
| hq_plus_policy_induction | 0.3000 | 0.3005 | +0.0005 | 0.0005 | True |
| llm_engineering_only | 0.2840 | 0.2843 | +0.0003 | 0.0003 | True |
| llm_engineering_plus_policy_induction | 0.3340 | 0.3344 | +0.0004 | 0.0004 | True |

## Detailed Data Tables (CSV Artifacts)

- `reasoning_transfer_cv_summary.csv`
- `reasoning_transfer_cv_per_target.csv`
- `reasoning_transfer_per_target.csv`
- `reasoning_transfer_summary.csv`
- `success_transfer_metrics.csv`
- `success_transfer_train_threshold_sweep.csv`
- `reproduction_consistency_check.csv`
