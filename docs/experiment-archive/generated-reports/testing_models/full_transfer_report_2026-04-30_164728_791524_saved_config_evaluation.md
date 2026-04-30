# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_164728_791524_saved_config_evaluation`
- Target family: `v25_policies`
- Feature set: `lambda_policies_plus_sentence_bundle`
- Combo refs: ['data\\saved_model_configs\\2026-04-21_132505_959940_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003']
- Reproduction reference run: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_024252_015048_success_reproduction`

- Success CV repeats: `16` (enabled=True)

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
| ridge | single_model | with_override | 0.3149 +/- 0.0050 | 0.3086 | 0.7278 | 0.2320 | 0.3259 | 0.2543 | 0.2500 |
| ridge | soft_avg_model | with_override | 0.3228 +/- 0.0019 | 0.3078 | 0.7277 | 0.2318 | 0.3249 | 0.2543 | 0.2500 |
| ridge | soft_avg_weighted_model | with_override | 0.3227 +/- 0.0019 | 0.3078 | 0.7277 | 0.2318 | 0.3249 | 0.2543 | 0.2500 |
| ridge | single_model | without_override | 0.3340 +/- 0.0073 | 0.2832 | 0.7210 | 0.2452 | 0.3303 | 0.1802 | 0.2500 |
| ridge | soft_avg_model | without_override | 0.3464 +/- 0.0038 | 0.2823 | 0.7210 | 0.2451 | 0.3288 | 0.1802 | 0.2500 |
| ridge | soft_avg_weighted_model | without_override | 0.3470 +/- 0.0040 | 0.2823 | 0.7210 | 0.2450 | 0.3288 | 0.1802 | 0.2500 |

### HQ + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | with_override | 0.3061 +/- 0.0045 | 0.3211 | 0.7281 | 0.2368 | 0.3472 | 0.2469 | 0.2650 |
| ridge | soft_avg_model | with_override | 0.3267 +/- 0.0032 | 0.3205 | 0.7281 | 0.2371 | 0.3488 | 0.2420 | 0.2700 |
| ridge | soft_avg_weighted_model | with_override | 0.3230 +/- 0.0035 | 0.3264 | 0.7281 | 0.2371 | 0.3502 | 0.2568 | 0.2600 |
| ridge | single_model | without_override | 0.3267 +/- 0.0067 | 0.2923 | 0.7247 | 0.2622 | 0.3646 | 0.1630 | 0.2700 |
| ridge | soft_avg_model | without_override | 0.3625 +/- 0.0056 | 0.2969 | 0.7247 | 0.2633 | 0.3676 | 0.1679 | 0.2700 |
| ridge | soft_avg_weighted_model | without_override | 0.3584 +/- 0.0050 | 0.3122 | 0.7247 | 0.2633 | 0.3744 | 0.1877 | 0.2600 |

### LLM-eng + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | with_override | 0.3203 +/- 0.0054 | 0.3260 | 0.7378 | 0.2410 | 0.3463 | 0.2642 | 0.2800 |
| ridge | soft_avg_model | with_override | 0.3379 +/- 0.0033 | 0.3230 | 0.7382 | 0.2412 | 0.3525 | 0.2420 | 0.3000 |
| ridge | soft_avg_weighted_model | with_override | 0.3375 +/- 0.0033 | 0.3239 | 0.7381 | 0.2412 | 0.3538 | 0.2420 | 0.3000 |
| ridge | single_model | without_override | 0.3430 +/- 0.0074 | 0.3100 | 0.7343 | 0.2630 | 0.3838 | 0.1753 | 0.3000 |
| ridge | soft_avg_model | without_override | 0.3717 +/- 0.0044 | 0.3056 | 0.7348 | 0.2637 | 0.3812 | 0.1704 | 0.3000 |
| ridge | soft_avg_weighted_model | without_override | 0.3718 +/- 0.0037 | 0.3067 | 0.7348 | 0.2636 | 0.3833 | 0.1704 | 0.3000 |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `success_transfer_train_threshold_sweep.csv`.

| branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hq_plus_pred_reasoning | single_model | with_override | 0.2700 | 0.3017 | 0.3255 | 0.2335 | 0.2650 | 0.3061 | 0.3211 |
| hq_plus_pred_reasoning | single_model | with_override | 0.2600 | 0.3006 | 0.3187 | 0.2452 | 0.2650 | 0.3061 | 0.3211 |
| hq_plus_pred_reasoning | single_model | with_override | 0.2500 | 0.3001 | 0.3136 | 0.2563 | 0.2650 | 0.3061 | 0.3211 |
| hq_plus_pred_reasoning | soft_avg_model | with_override | 0.2700 | 0.3250 | 0.3505 | 0.2519 | 0.2700 | 0.3250 | 0.3205 |
| hq_plus_pred_reasoning | soft_avg_model | with_override | 0.2500 | 0.3237 | 0.3363 | 0.2815 | 0.2700 | 0.3250 | 0.3205 |
| hq_plus_pred_reasoning | soft_avg_model | with_override | 0.2600 | 0.3235 | 0.3418 | 0.2667 | 0.2700 | 0.3250 | 0.3205 |
| hq_plus_pred_reasoning | soft_avg_weighted_model | with_override | 0.2600 | 0.3259 | 0.3450 | 0.2667 | 0.2600 | 0.3259 | 0.3264 |
| hq_plus_pred_reasoning | soft_avg_weighted_model | with_override | 0.2700 | 0.3242 | 0.3493 | 0.2519 | 0.2600 | 0.3259 | 0.3264 |
| hq_plus_pred_reasoning | soft_avg_weighted_model | with_override | 0.2500 | 0.3237 | 0.3363 | 0.2815 | 0.2600 | 0.3259 | 0.3264 |
| hq_plus_pred_reasoning | single_model | without_override | 0.2700 | 0.3206 | 0.3880 | 0.1895 | 0.2700 | 0.3267 | 0.2923 |
| hq_plus_pred_reasoning | single_model | without_override | 0.2600 | 0.3187 | 0.3719 | 0.2029 | 0.2700 | 0.3267 | 0.2923 |
| hq_plus_pred_reasoning | single_model | without_override | 0.2800 | 0.3178 | 0.3988 | 0.1756 | 0.2700 | 0.3267 | 0.2923 |
| hq_plus_pred_reasoning | soft_avg_model | without_override | 0.2700 | 0.3622 | 0.4372 | 0.2148 | 0.2700 | 0.3622 | 0.2969 |
| hq_plus_pred_reasoning | soft_avg_model | without_override | 0.2600 | 0.3607 | 0.4167 | 0.2346 | 0.2700 | 0.3622 | 0.2969 |
| hq_plus_pred_reasoning | soft_avg_model | without_override | 0.2900 | 0.3589 | 0.4771 | 0.1802 | 0.2700 | 0.3622 | 0.2969 |
| hq_plus_pred_reasoning | soft_avg_weighted_model | without_override | 0.2600 | 0.3640 | 0.4222 | 0.2346 | 0.2600 | 0.3640 | 0.3122 |
| hq_plus_pred_reasoning | soft_avg_weighted_model | without_override | 0.2700 | 0.3622 | 0.4372 | 0.2148 | 0.2600 | 0.3640 | 0.3122 |
| hq_plus_pred_reasoning | soft_avg_weighted_model | without_override | 0.2500 | 0.3554 | 0.3976 | 0.2494 | 0.2600 | 0.3640 | 0.3122 |
| llm_engineering_plus_pred_reasoning | single_model | with_override | 0.2900 | 0.3152 | 0.3324 | 0.2612 | 0.2800 | 0.3203 | 0.3260 |
| llm_engineering_plus_pred_reasoning | single_model | with_override | 0.3000 | 0.3146 | 0.3357 | 0.2516 | 0.2800 | 0.3203 | 0.3260 |
| llm_engineering_plus_pred_reasoning | single_model | with_override | 0.2800 | 0.3145 | 0.3275 | 0.2715 | 0.2800 | 0.3203 | 0.3260 |
| llm_engineering_plus_pred_reasoning | soft_avg_model | with_override | 0.3000 | 0.3397 | 0.3643 | 0.2677 | 0.3000 | 0.3397 | 0.3230 |
| llm_engineering_plus_pred_reasoning | soft_avg_model | with_override | 0.3100 | 0.3377 | 0.3636 | 0.2626 | 0.3000 | 0.3397 | 0.3230 |
| llm_engineering_plus_pred_reasoning | soft_avg_model | with_override | 0.2900 | 0.3339 | 0.3528 | 0.2753 | 0.3000 | 0.3397 | 0.3230 |
| llm_engineering_plus_pred_reasoning | soft_avg_weighted_model | with_override | 0.3000 | 0.3397 | 0.3643 | 0.2677 | 0.3000 | 0.3397 | 0.3239 |
| llm_engineering_plus_pred_reasoning | soft_avg_weighted_model | with_override | 0.2900 | 0.3378 | 0.3571 | 0.2778 | 0.3000 | 0.3397 | 0.3239 |
| llm_engineering_plus_pred_reasoning | soft_avg_weighted_model | with_override | 0.3100 | 0.3377 | 0.3636 | 0.2626 | 0.3000 | 0.3397 | 0.3239 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.3000 | 0.3366 | 0.4046 | 0.2015 | 0.3000 | 0.3430 | 0.3100 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.2900 | 0.3361 | 0.3936 | 0.2123 | 0.3000 | 0.3430 | 0.3100 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.2800 | 0.3353 | 0.3826 | 0.2247 | 0.3000 | 0.3430 | 0.3100 |
| llm_engineering_plus_pred_reasoning | soft_avg_model | without_override | 0.3000 | 0.3720 | 0.4526 | 0.2172 | 0.3000 | 0.3720 | 0.3056 |
| llm_engineering_plus_pred_reasoning | soft_avg_model | without_override | 0.3100 | 0.3710 | 0.4565 | 0.2121 | 0.3000 | 0.3720 | 0.3056 |
| llm_engineering_plus_pred_reasoning | soft_avg_model | without_override | 0.3200 | 0.3640 | 0.4691 | 0.1919 | 0.3000 | 0.3720 | 0.3056 |
| llm_engineering_plus_pred_reasoning | soft_avg_weighted_model | without_override | 0.3000 | 0.3720 | 0.4526 | 0.2172 | 0.3000 | 0.3720 | 0.3067 |
| llm_engineering_plus_pred_reasoning | soft_avg_weighted_model | without_override | 0.3100 | 0.3710 | 0.4565 | 0.2121 | 0.3000 | 0.3720 | 0.3067 |
| llm_engineering_plus_pred_reasoning | soft_avg_weighted_model | without_override | 0.2900 | 0.3653 | 0.4306 | 0.2273 | 0.3000 | 0.3720 | 0.3067 |
| pred_reasoning_only | single_model | with_override | 0.2500 | 0.3106 | 0.3261 | 0.2613 | 0.2500 | 0.3149 | 0.3086 |
| pred_reasoning_only | single_model | with_override | 0.2400 | 0.3090 | 0.3197 | 0.2728 | 0.2500 | 0.3149 | 0.3086 |
| pred_reasoning_only | single_model | with_override | 0.2600 | 0.3072 | 0.3274 | 0.2465 | 0.2500 | 0.3149 | 0.3086 |
| pred_reasoning_only | soft_avg_model | with_override | 0.2500 | 0.3241 | 0.3406 | 0.2716 | 0.2500 | 0.3241 | 0.3078 |
| pred_reasoning_only | soft_avg_model | with_override | 0.2300 | 0.3216 | 0.3279 | 0.2988 | 0.2500 | 0.3241 | 0.3078 |
| pred_reasoning_only | soft_avg_model | with_override | 0.2400 | 0.3200 | 0.3314 | 0.2815 | 0.2500 | 0.3241 | 0.3078 |
| pred_reasoning_only | soft_avg_weighted_model | with_override | 0.2500 | 0.3241 | 0.3406 | 0.2716 | 0.2500 | 0.3241 | 0.3078 |
| pred_reasoning_only | soft_avg_weighted_model | with_override | 0.2300 | 0.3216 | 0.3279 | 0.2988 | 0.2500 | 0.3241 | 0.3078 |
| pred_reasoning_only | soft_avg_weighted_model | with_override | 0.2400 | 0.3208 | 0.3324 | 0.2815 | 0.2500 | 0.3241 | 0.3078 |
| pred_reasoning_only | single_model | without_override | 0.2500 | 0.3285 | 0.3796 | 0.2137 | 0.2500 | 0.3340 | 0.2832 |
| pred_reasoning_only | single_model | without_override | 0.2600 | 0.3255 | 0.3879 | 0.1981 | 0.2500 | 0.3340 | 0.2832 |
| pred_reasoning_only | single_model | without_override | 0.2400 | 0.3243 | 0.3640 | 0.2259 | 0.2500 | 0.3340 | 0.2832 |
| pred_reasoning_only | soft_avg_model | without_override | 0.2500 | 0.3476 | 0.4027 | 0.2247 | 0.2500 | 0.3476 | 0.2823 |
| pred_reasoning_only | soft_avg_model | without_override | 0.2600 | 0.3432 | 0.4057 | 0.2123 | 0.2500 | 0.3476 | 0.2823 |
| pred_reasoning_only | soft_avg_model | without_override | 0.2300 | 0.3428 | 0.3741 | 0.2568 | 0.2500 | 0.3476 | 0.2823 |
| pred_reasoning_only | soft_avg_weighted_model | without_override | 0.2500 | 0.3476 | 0.4027 | 0.2247 | 0.2500 | 0.3476 | 0.2823 |
| pred_reasoning_only | soft_avg_weighted_model | without_override | 0.2400 | 0.3436 | 0.3871 | 0.2370 | 0.2500 | 0.3476 | 0.2823 |
| pred_reasoning_only | soft_avg_weighted_model | without_override | 0.2300 | 0.3404 | 0.3718 | 0.2543 | 0.2500 | 0.3476 | 0.2823 |

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
