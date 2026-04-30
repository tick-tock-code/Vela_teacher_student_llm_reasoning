# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-22_190451_188920_saved_config_evaluation`
- Target family: `v25_policies`
- Feature set: `lambda_policies_plus_sentence_bundle`
- Combo refs: ['data/saved_model_configs/2026-04-21_132505_959940_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003']
- Reproduction reference run: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_024252_015048_success_reproduction`

- Success CV repeats: `1` (enabled=False)

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
| ridge | democratic_model | without_override | 0.1524 +/- 0.0000 | 0.1329 | 0.5161 | 0.1095 | 0.6190 | 0.0321 | 0.3000 |
| ridge | single_model | without_override | 0.3444 +/- 0.0000 | 0.2756 | 0.7210 | 0.2452 | 0.3386 | 0.1580 | 0.2600 |

### HQ + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | democratic_model | with_override | 0.2207 +/- 0.0000 | 0.2803 | 0.5615 | 0.1323 | 0.3819 | 0.1358 | 0.3000 |
| ridge | single_model | with_override | 0.3075 +/- 0.0000 | 0.3222 | 0.7281 | 0.2368 | 0.3513 | 0.2420 | 0.2700 |

### LLM-eng + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | democratic_model | without_override | 0.2007 +/- 0.0000 | 0.2065 | 0.5327 | 0.1292 | 0.5455 | 0.0593 | 0.3000 |
| ridge | single_model | without_override | 0.3482 +/- 0.0000 | 0.3100 | 0.7343 | 0.2630 | 0.3838 | 0.1753 | 0.3000 |

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
