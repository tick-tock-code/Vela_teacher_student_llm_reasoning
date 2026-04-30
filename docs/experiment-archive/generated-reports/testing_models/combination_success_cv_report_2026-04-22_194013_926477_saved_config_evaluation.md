# Combination Success Screening Report (Train CV Only)

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-22_194013_926477_saved_config_evaluation`
- Selected combo ref: `data\saved_model_configs\2026-04-21_132505_959940_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003`
- Target family: `v25_policies`
- Held-out evaluation: disabled
- Repeat CV with new seeds: False (n_runs=1)

## Source CV Validation (Selected Combo)

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4211 | 0.0804 | 0.2651 | 0.1775 |

## Success Screening (Nested L2, CV only)

| success_branch_id | model_variant | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | train_cv_f0_5_std | train_cv_roc_auc | train_cv_pr_auc | selected_c_final | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| hq_baseline__with_override | democratic_model | hq_baseline | with_override | 0.2207 | 0.0000 | 0.5423 | 0.1122 | 0.0150 | 0.3000 |
| hq_baseline__with_override | single_model | hq_baseline | with_override | 0.3075 | 0.0000 | 0.7618 | 0.2306 | 0.0150 | 0.2700 |
| hq_plus_lambda_policies__with_override | democratic_model | hq_plus_lambda_policies | with_override | 0.2329 | 0.0000 | 0.5505 | 0.1167 | 0.0110 | 0.3000 |
| hq_plus_lambda_policies__with_override | single_model | hq_plus_lambda_policies | with_override | 0.3028 | 0.0000 | 0.7621 | 0.2300 | 0.0110 | 0.2400 |
| hq_plus_llm_engineering__with_override | democratic_model | hq_plus_llm_engineering | with_override | 0.2558 | 0.0000 | 0.5566 | 0.1183 | 0.0480 | 0.3000 |
| hq_plus_llm_engineering__with_override | single_model | hq_plus_llm_engineering | with_override | 0.3140 | 0.0000 | 0.7657 | 0.2334 | 0.0480 | 0.2800 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | democratic_model | hq_plus_llm_engineering_plus_lambda_policies | with_override | 0.2461 | 0.0000 | 0.5601 | 0.1206 | 0.0230 | 0.4500 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | single_model | hq_plus_llm_engineering_plus_lambda_policies | with_override | 0.3076 | 0.0000 | 0.7604 | 0.2289 | 0.0230 | 0.2500 |
| lambda_policies__without_override | democratic_model | lambda_policies | without_override | 0.1821 | 0.0000 | 0.5347 | 0.1273 | 0.0130 | 0.3000 |
| lambda_policies__without_override | single_model | lambda_policies | without_override | 0.3232 | 0.0000 | 0.7606 | 0.2660 | 0.0130 | 0.2000 |
| llm_engineering__without_override | democratic_model | llm_engineering | without_override | 0.2007 | 0.0000 | 0.5403 | 0.1389 | 0.1700 | 0.3000 |
| llm_engineering__without_override | single_model | llm_engineering | without_override | 0.3482 | 0.0000 | 0.7673 | 0.2840 | 0.1700 | 0.3000 |
| llm_engineering_plus_lambda_policies__without_override | democratic_model | llm_engineering_plus_lambda_policies | without_override | 0.2068 | 0.0000 | 0.5444 | 0.1482 | 0.0320 | 0.3000 |
| llm_engineering_plus_lambda_policies__without_override | single_model | llm_engineering_plus_lambda_policies | without_override | 0.3393 | 0.0000 | 0.7569 | 0.2705 | 0.0320 | 0.2500 |
