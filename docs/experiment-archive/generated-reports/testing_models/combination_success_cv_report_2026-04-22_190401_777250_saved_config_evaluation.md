# Combination Success Screening Report (Train CV Only)

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-22_190401_777250_saved_config_evaluation`
- Selected combo ref: `data/saved_model_configs/2026-04-21_132505_959940_model_testing::v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007`
- Target family: `v25_policies`
- Held-out evaluation: disabled
- Repeat CV with new seeds: False (n_runs=1)

## Source CV Validation (Selected Combo)

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007 | hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4286 | 0.0794 | 0.2638 | 0.1758 |

## Success Screening (Nested L2, CV only)

| success_branch_id | model_variant | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | train_cv_f0_5_std | train_cv_roc_auc | train_cv_pr_auc | selected_c_final | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| llm_engineering__without_override | democratic_model | llm_engineering | without_override | 0.2273 | 0.0000 | 0.5383 | 0.1344 | 0.1840 | 0.3000 |
| llm_engineering__without_override | single_model | llm_engineering | without_override | 0.3367 | 0.0000 | 0.7644 | 0.2759 | 0.1840 | 0.2700 |
