# Combination Success Screening Report (Train CV Only)

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-21_132957_957691_saved_config_evaluation`
- Selected combo ref: `data/saved_model_configs/2026-04-21_132505_959940_model_testing::v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007`
- Target family: `v25_policies`
- Held-out evaluation: disabled

## Source CV Validation (Selected Combo)

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007 | hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4286 | 0.0794 | 0.2638 | 0.1758 |

## Success Screening (Nested L2, CV only)

| success_branch_id | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | train_cv_roc_auc | train_cv_pr_auc | selected_c_final | threshold |
|---|---|---|---:|---:|---:|---:|---:|
| hq_plus_llm_engineering__without_override | hq_plus_llm_engineering | without_override | 0.3510 | 0.7665 | 0.2859 | 0.0600 | 0.3000 |
| hq_plus_llm_engineering_plus_lambda_policies__without_override | hq_plus_llm_engineering_plus_lambda_policies | without_override | 0.3418 | 0.7590 | 0.2749 | 0.0230 | 0.3200 |
| hq_baseline__without_override | hq_baseline | without_override | 0.3388 | 0.7620 | 0.2763 | 0.0440 | 0.3000 |
| hq_plus_lambda_policies__without_override | hq_plus_lambda_policies | without_override | 0.3378 | 0.7576 | 0.2743 | 0.0210 | 0.3300 |
| llm_engineering__without_override | llm_engineering | without_override | 0.3367 | 0.7644 | 0.2759 | 0.1840 | 0.2700 |
| llm_engineering_plus_lambda_policies__without_override | llm_engineering_plus_lambda_policies | without_override | 0.3359 | 0.7560 | 0.2710 | 0.0340 | 0.2300 |
| lambda_policies__without_override | lambda_policies | without_override | 0.3277 | 0.7552 | 0.2687 | 0.0220 | 0.2700 |
| llm_engineering_plus_lambda_policies__with_override | llm_engineering_plus_lambda_policies | with_override | 0.3218 | 0.7575 | 0.2290 | 0.0340 | 0.2300 |
| hq_plus_llm_engineering__with_override | hq_plus_llm_engineering | with_override | 0.3197 | 0.7668 | 0.2339 | 0.0600 | 0.3000 |
| llm_engineering__with_override | llm_engineering | with_override | 0.3177 | 0.7638 | 0.2329 | 0.1840 | 0.2800 |
| hq_baseline__with_override | hq_baseline | with_override | 0.3125 | 0.7615 | 0.2289 | 0.0440 | 0.3000 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | hq_plus_llm_engineering_plus_lambda_policies | with_override | 0.3100 | 0.7607 | 0.2299 | 0.0230 | 0.2300 |
| lambda_policies__with_override | lambda_policies | with_override | 0.3086 | 0.7554 | 0.2265 | 0.0220 | 0.2300 |
| hq_plus_lambda_policies__with_override | hq_plus_lambda_policies | with_override | 0.3057 | 0.7585 | 0.2284 | 0.0210 | 0.3300 |
