# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_194838_115531_saved_config_evaluation`
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
| ridge | single_model | with_override | 0.3161 +/- 0.0049 | 0.2953 | 0.7270 | 0.2311 | 0.3106 | 0.2469 | 0.2700 |
| ridge | single_model | without_override | 0.3384 +/- 0.0063 | 0.2599 | 0.7187 | 0.2420 | 0.3032 | 0.1654 | 0.2700 |

### HQ + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | with_override | 0.3034 +/- 0.0058 | 0.3089 | 0.7235 | 0.2356 | 0.3174 | 0.2790 | 0.2700 |
| ridge | single_model | without_override | 0.3198 +/- 0.0071 | 0.3044 | 0.7187 | 0.2632 | 0.3398 | 0.2148 | 0.2750 |

### LLM-eng + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | with_override | 0.3224 +/- 0.0046 | 0.3231 | 0.7350 | 0.2398 | 0.3443 | 0.2593 | 0.2950 |
| ridge | single_model | without_override | 0.3463 +/- 0.0082 | 0.3122 | 0.7318 | 0.2610 | 0.3744 | 0.1877 | 0.3050 |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `success_transfer_train_threshold_sweep.csv`.

| branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hq_plus_pred_reasoning | single_model | with_override | 0.2700 | 0.2988 | 0.3050 | 0.2764 | 0.2700 | 0.3034 | 0.3089 |
| hq_plus_pred_reasoning | single_model | with_override | 0.2800 | 0.2976 | 0.3069 | 0.2654 | 0.2700 | 0.3034 | 0.3089 |
| hq_plus_pred_reasoning | single_model | with_override | 0.2600 | 0.2973 | 0.2997 | 0.2883 | 0.2700 | 0.3034 | 0.3089 |
| hq_plus_pred_reasoning | single_model | without_override | 0.2800 | 0.3139 | 0.3497 | 0.2227 | 0.2750 | 0.3198 | 0.3044 |
| hq_plus_pred_reasoning | single_model | without_override | 0.2700 | 0.3137 | 0.3429 | 0.2341 | 0.2750 | 0.3198 | 0.3044 |
| hq_plus_pred_reasoning | single_model | without_override | 0.2900 | 0.3125 | 0.3559 | 0.2100 | 0.2750 | 0.3198 | 0.3044 |
| llm_engineering_plus_pred_reasoning | single_model | with_override | 0.2800 | 0.3182 | 0.3262 | 0.2896 | 0.2950 | 0.3224 | 0.3231 |
| llm_engineering_plus_pred_reasoning | single_model | with_override | 0.2900 | 0.3177 | 0.3290 | 0.2792 | 0.2950 | 0.3224 | 0.3231 |
| llm_engineering_plus_pred_reasoning | single_model | with_override | 0.2700 | 0.3168 | 0.3211 | 0.3007 | 0.2950 | 0.3224 | 0.3231 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.3100 | 0.3379 | 0.3978 | 0.2109 | 0.3050 | 0.3463 | 0.3122 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.3000 | 0.3375 | 0.3884 | 0.2216 | 0.3050 | 0.3463 | 0.3122 |
| llm_engineering_plus_pred_reasoning | single_model | without_override | 0.2900 | 0.3371 | 0.3799 | 0.2325 | 0.3050 | 0.3463 | 0.3122 |
| pred_reasoning_only | single_model | with_override | 0.2600 | 0.3140 | 0.3233 | 0.2816 | 0.2700 | 0.3161 | 0.2953 |
| pred_reasoning_only | single_model | with_override | 0.2700 | 0.3126 | 0.3260 | 0.2685 | 0.2700 | 0.3161 | 0.2953 |
| pred_reasoning_only | single_model | with_override | 0.2800 | 0.3102 | 0.3277 | 0.2556 | 0.2700 | 0.3161 | 0.2953 |
| pred_reasoning_only | single_model | without_override | 0.2600 | 0.3343 | 0.3741 | 0.2347 | 0.2700 | 0.3384 | 0.2599 |
| pred_reasoning_only | single_model | without_override | 0.2700 | 0.3338 | 0.3829 | 0.2205 | 0.2700 | 0.3384 | 0.2599 |
| pred_reasoning_only | single_model | without_override | 0.2800 | 0.3307 | 0.3900 | 0.2057 | 0.2700 | 0.3384 | 0.2599 |

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
