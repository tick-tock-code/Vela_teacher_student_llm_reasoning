# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-21_120320_019540_saved_config_evaluation`
- Target family: `v25_policies`
- Feature set: `lambda_policies_plus_sentence_bundle`
- Combo refs: ['data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003']
- Reproduction reference run: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_024252_015048_success_reproduction`

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

| model_set_id | hq_exit_override_branch | f0_5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| ridge | with_override | 0.3042 | 0.7278 | 0.2320 | 0.3298 | 0.2321 | 0.2600 |
| ridge | without_override | 0.2756 | 0.7210 | 0.2452 | 0.3386 | 0.1580 | 0.2600 |

### HQ + reasoning_pred

| model_set_id | hq_exit_override_branch | f0_5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| ridge | with_override | 0.3222 | 0.7281 | 0.2368 | 0.3513 | 0.2420 | 0.2700 |
| ridge | without_override | 0.3029 | 0.7247 | 0.2622 | 0.4254 | 0.1407 | 0.3000 |

### LLM-eng + reasoning_pred

| model_set_id | hq_exit_override_branch | f0_5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| ridge | with_override | 0.3262 | 0.7378 | 0.2410 | 0.3546 | 0.2469 | 0.3000 |
| ridge | without_override | 0.3100 | 0.7343 | 0.2630 | 0.3838 | 0.1753 | 0.3000 |

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
