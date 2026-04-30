# Combination Success Screening Report (Train CV Only)

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-21_133521_705091_saved_config_evaluation`
- Selected combo ref: `data/saved_model_configs/2026-04-21_132505_959940_model_testing::v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007`
- Target family: `v25_policies`
- Held-out evaluation: disabled
- Repeat CV with new seeds: True (n_runs=4)

## Source CV Validation (Selected Combo)

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007 | hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4286 | 0.0794 | 0.2638 | 0.1758 |

## Success Screening (Nested L2, CV only)

| success_branch_id | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | train_cv_f0_5_std | train_cv_roc_auc | train_cv_pr_auc | selected_c_final | threshold |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hq_plus_llm_engineering__without_override | hq_plus_llm_engineering | without_override | 0.3396 | 0.0067 | 0.7643 | 0.2735 | 0.0510 | 0.2950 |
| llm_engineering__without_override | llm_engineering | without_override | 0.3357 | 0.0048 | 0.7644 | 0.2731 | 0.1585 | 0.2925 |
| hq_plus_llm_engineering_plus_lambda_policies__without_override | hq_plus_llm_engineering_plus_lambda_policies | without_override | 0.3344 | 0.0050 | 0.7610 | 0.2698 | 0.0193 | 0.3050 |
| hq_baseline__without_override | hq_baseline | without_override | 0.3319 | 0.0059 | 0.7582 | 0.2641 | 0.0305 | 0.2850 |
| llm_engineering_plus_lambda_policies__without_override | llm_engineering_plus_lambda_policies | without_override | 0.3315 | 0.0081 | 0.7619 | 0.2693 | 0.0215 | 0.2175 |
| hq_plus_lambda_policies__without_override | hq_plus_lambda_policies | without_override | 0.3232 | 0.0085 | 0.7585 | 0.2655 | 0.0160 | 0.2700 |
| lambda_policies__without_override | lambda_policies | without_override | 0.3211 | 0.0062 | 0.7584 | 0.2628 | 0.0160 | 0.2425 |
| llm_engineering_plus_lambda_policies__with_override | llm_engineering_plus_lambda_policies | with_override | 0.3172 | 0.0066 | 0.7629 | 0.2300 | 0.0215 | 0.2175 |
| llm_engineering__with_override | llm_engineering | with_override | 0.3139 | 0.0038 | 0.7641 | 0.2319 | 0.1585 | 0.2950 |
| hq_plus_llm_engineering__with_override | hq_plus_llm_engineering | with_override | 0.3118 | 0.0050 | 0.7654 | 0.2317 | 0.0510 | 0.2925 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | hq_plus_llm_engineering_plus_lambda_policies | with_override | 0.3095 | 0.0009 | 0.7630 | 0.2301 | 0.0193 | 0.2450 |
| hq_baseline__with_override | hq_baseline | with_override | 0.3086 | 0.0056 | 0.7584 | 0.2262 | 0.0305 | 0.2850 |
| lambda_policies__with_override | lambda_policies | with_override | 0.3070 | 0.0039 | 0.7590 | 0.2267 | 0.0160 | 0.2075 |
| hq_plus_lambda_policies__with_override | hq_plus_lambda_policies | with_override | 0.3048 | 0.0006 | 0.7598 | 0.2272 | 0.0160 | 0.2300 |
