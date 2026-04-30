# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-22_193057_453568_saved_config_evaluation`
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
| ridge | democratic_model | with_override | 0.2261 +/- 0.0042 | 0.2786 | 0.5625 | 0.1326 | 0.3830 | 0.1333 | 0.3000 |
| ridge | single_model | with_override | 0.3149 +/- 0.0050 | 0.3086 | 0.7278 | 0.2320 | 0.3259 | 0.2543 | 0.2500 |
| ridge | democratic_model | without_override | 0.1576 +/- 0.0134 | 0.1329 | 0.5262 | 0.1221 | 0.6190 | 0.0321 | 0.3000 |
| ridge | single_model | without_override | 0.3340 +/- 0.0073 | 0.2832 | 0.7210 | 0.2452 | 0.3303 | 0.1802 | 0.2500 |

### HQ + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | democratic_model | with_override | 0.2198 +/- 0.0033 | 0.2803 | 0.5708 | 0.1387 | 0.3819 | 0.1358 | 0.3000 |
| ridge | single_model | with_override | 0.3061 +/- 0.0045 | 0.3211 | 0.7281 | 0.2368 | 0.3472 | 0.2469 | 0.2650 |
| ridge | democratic_model | without_override | 0.1602 +/- 0.0094 | 0.1572 | 0.5345 | 0.1338 | 0.6154 | 0.0395 | 0.3000 |
| ridge | single_model | without_override | 0.3267 +/- 0.0067 | 0.2923 | 0.7247 | 0.2622 | 0.3646 | 0.1630 | 0.2700 |

### LLM-eng + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | democratic_model | with_override | 0.2389 +/- 0.0068 | 0.3073 | 0.5773 | 0.1452 | 0.4065 | 0.1556 | 0.3500 |
| ridge | single_model | with_override | 0.3203 +/- 0.0054 | 0.3260 | 0.7378 | 0.2410 | 0.3463 | 0.2642 | 0.2800 |
| ridge | democratic_model | without_override | 0.2126 +/- 0.0182 | 0.2004 | 0.5465 | 0.1473 | 0.6111 | 0.0543 | 0.4000 |
| ridge | single_model | without_override | 0.3430 +/- 0.0074 | 0.3100 | 0.7343 | 0.2630 | 0.3838 | 0.1753 | 0.3000 |

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
- `reproduction_consistency_check.csv`
