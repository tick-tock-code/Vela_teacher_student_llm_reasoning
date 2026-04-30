# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_165820_897200_saved_config_evaluation`
- Target family: `v25_policies`
- Feature set: `lambda_policies_plus_sentence_bundle`
- Combo refs: ['2026-04-21_132505_959940_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003']
- Reproduction reference run: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_024252_015048_success_reproduction`

- Success CV repeats: `2` (enabled=True)

## CV Validation Performance (Source Runs)

These are CV metrics taken from the source model-testing runs used to build the saved model bundles.

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| ridge | 0.4211 | 0.0804 | 0.2651 | 0.1775 |

## Held-out Test Performance (Reasoning Agreement)

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| ridge | 0.4008 | 0.0941 | 0.2666 | 0.1929 |

## Held-out Test Performance (Success Transfer)

### reasoning_pred-only

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | without_override | 0.3330 +/- 0.0114 | 0.2682 | 0.7209 | 0.2446 | 0.3333 | 0.1506 | 0.2750 |
| ridge | soft_avg_model | without_override | 0.3511 +/- 0.0063 | 0.2793 | 0.7212 | 0.2455 | 0.3258 | 0.1778 | 0.2500 |
| ridge | soft_avg_weighted_model | without_override | 0.3525 +/- 0.0049 | 0.2784 | 0.7211 | 0.2453 | 0.3243 | 0.1778 | 0.2500 |

### HQ + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | with_override | 0.3099 +/- 0.0024 | 0.3241 | 0.7281 | 0.2368 | 0.3480 | 0.2543 | 0.2600 |
| ridge | soft_avg_model | with_override | 0.3280 +/- 0.0028 | 0.3213 | 0.7281 | 0.2372 | 0.3500 | 0.2420 | 0.2700 |
| ridge | soft_avg_weighted_model | with_override | 0.3296 +/- 0.0021 | 0.3213 | 0.7280 | 0.2371 | 0.3500 | 0.2420 | 0.2700 |

### LLM-eng + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | without_override | 0.3464 +/- 0.0018 | 0.3048 | 0.7343 | 0.2630 | 0.3711 | 0.1778 | 0.2950 |
| ridge | soft_avg_model | without_override | 0.3744 +/- 0.0019 | 0.3055 | 0.7349 | 0.2634 | 0.3842 | 0.1679 | 0.3000 |
| ridge | soft_avg_weighted_model | without_override | 0.3735 +/- 0.0011 | 0.3089 | 0.7348 | 0.2634 | 0.3876 | 0.1704 | 0.3000 |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `success_transfer_train_threshold_sweep.csv`.

| branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hq_plus_pred_reasoning | single_model | with_override | 0.2500 | 0.3080 | 0.3204 | 0.2667 | 0.2600 | 0.3099 | 0.3241 |
| hq_plus_pred_reasoning | single_model | with_override | 0.2600 | 0.3074 | 0.3244 | 0.2543 | 0.2600 | 0.3099 | 0.3241 |
| hq_plus_pred_reasoning | single_model | with_override | 0.2700 | 0.3040 | 0.3271 | 0.2370 | 0.2600 | 0.3099 | 0.3241 |
| hq_plus_pred_reasoning | soft_avg_model | with_override | 0.2700 | 0.3249 | 0.3492 | 0.2543 | 0.2700 | 0.3249 | 0.3213 |
| hq_plus_pred_reasoning | soft_avg_model | with_override | 0.2600 | 0.3243 | 0.3429 | 0.2667 | 0.2700 | 0.3249 | 0.3213 |
| hq_plus_pred_reasoning | soft_avg_model | with_override | 0.2500 | 0.3229 | 0.3353 | 0.2815 | 0.2700 | 0.3249 | 0.3213 |
| hq_plus_pred_reasoning | soft_avg_weighted_model | with_override | 0.2700 | 0.3241 | 0.3480 | 0.2543 | 0.2700 | 0.3241 | 0.3213 |
| hq_plus_pred_reasoning | soft_avg_weighted_model | with_override | 0.2600 | 0.3228 | 0.3407 | 0.2667 | 0.2700 | 0.3241 | 0.3213 |
| hq_plus_pred_reasoning | soft_avg_weighted_model | with_override | 0.2500 | 0.3222 | 0.3343 | 0.2815 | 0.2700 | 0.3241 | 0.3213 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.3000 | 0.3461 | 0.4146 | 0.2083 | 0.2950 | 0.3464 | 0.3048 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.2900 | 0.3431 | 0.4023 | 0.2159 | 0.2950 | 0.3464 | 0.3048 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.3100 | 0.3415 | 0.4227 | 0.1932 | 0.2950 | 0.3464 | 0.3048 |
| llm_engineering_plus_pred_reasoning | soft_avg_model | without_override | 0.3000 | 0.3733 | 0.4550 | 0.2172 | 0.3000 | 0.3733 | 0.3055 |
| llm_engineering_plus_pred_reasoning | soft_avg_model | without_override | 0.2900 | 0.3701 | 0.4390 | 0.2273 | 0.3000 | 0.3733 | 0.3055 |
| llm_engineering_plus_pred_reasoning | soft_avg_model | without_override | 0.3100 | 0.3674 | 0.4556 | 0.2071 | 0.3000 | 0.3733 | 0.3055 |
| llm_engineering_plus_pred_reasoning | soft_avg_weighted_model | without_override | 0.3000 | 0.3733 | 0.4550 | 0.2172 | 0.3000 | 0.3733 | 0.3089 |
| llm_engineering_plus_pred_reasoning | soft_avg_weighted_model | without_override | 0.2900 | 0.3689 | 0.4369 | 0.2273 | 0.3000 | 0.3733 | 0.3089 |
| llm_engineering_plus_pred_reasoning | soft_avg_weighted_model | without_override | 0.3100 | 0.3687 | 0.4581 | 0.2071 | 0.3000 | 0.3733 | 0.3089 |
| pred_reasoning_only | single_model | without_override | 0.2500 | 0.3275 | 0.3779 | 0.2136 | 0.2750 | 0.3330 | 0.2682 |
| pred_reasoning_only | single_model | without_override | 0.2700 | 0.3267 | 0.4029 | 0.1864 | 0.2750 | 0.3330 | 0.2682 |
| pred_reasoning_only | single_model | without_override | 0.2800 | 0.3267 | 0.4155 | 0.1765 | 0.2750 | 0.3330 | 0.2682 |
| pred_reasoning_only | soft_avg_model | without_override | 0.2500 | 0.3520 | 0.4061 | 0.2296 | 0.2500 | 0.3520 | 0.2793 |
| pred_reasoning_only | soft_avg_model | without_override | 0.2600 | 0.3483 | 0.4123 | 0.2148 | 0.2500 | 0.3520 | 0.2793 |
| pred_reasoning_only | soft_avg_model | without_override | 0.2400 | 0.3436 | 0.3871 | 0.2370 | 0.2500 | 0.3520 | 0.2793 |
| pred_reasoning_only | soft_avg_weighted_model | without_override | 0.2500 | 0.3531 | 0.4079 | 0.2296 | 0.2500 | 0.3531 | 0.2784 |
| pred_reasoning_only | soft_avg_weighted_model | without_override | 0.2600 | 0.3465 | 0.4115 | 0.2123 | 0.2500 | 0.3531 | 0.2784 |
| pred_reasoning_only | soft_avg_weighted_model | without_override | 0.2400 | 0.3436 | 0.3871 | 0.2370 | 0.2500 | 0.3531 | 0.2784 |

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
