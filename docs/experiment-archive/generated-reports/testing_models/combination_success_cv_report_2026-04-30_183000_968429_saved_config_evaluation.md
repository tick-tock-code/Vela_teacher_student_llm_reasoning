# Combination Success Screening Report (Train CV Only)

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_183000_968429_saved_config_evaluation`
- Selected combo ref: `data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003`
- Target family: `v25_policies`
- Held-out evaluation: disabled
- Repeat CV with new seeds: True (n_runs=16)
- Success LR nested C CV: False
- Success LR fixed C when nested CV disabled: `1`

## Source CV Validation (Selected Combo)

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4229 | 0.0800 | 0.2647 | 0.1774 |

## Success Screening (L2 Logistic Regression, CV only)

| rank | success_branch_id | model_variant | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | train_cv_f0_5_std | train_cv_roc_auc | train_cv_pr_auc | selected_c_final | threshold |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | llm_engineering__without_override | single_model | llm_engineering | without_override | 0.3463 | 0.0082 | 0.7683 | 0.2827 | 1.0000 | 0.3050 |
| 2 | llm_engineering_plus_lambda_policies__without_override | single_model | llm_engineering_plus_lambda_policies | without_override | 0.3393 | 0.0134 | 0.7523 | 0.2642 | 1.0000 | 0.3325 |
| 3 | hq_plus_llm_engineering_plus_lambda_policies__without_override | single_model | hq_plus_llm_engineering_plus_lambda_policies | without_override | 0.3266 | 0.0107 | 0.7464 | 0.2576 | 1.0000 | 0.3063 |
| 4 | hq_plus_llm_engineering__without_override | single_model | hq_plus_llm_engineering | without_override | 0.3263 | 0.0074 | 0.7614 | 0.2755 | 1.0000 | 0.2925 |
| 5 | llm_engineering__with_override | single_model | llm_engineering | with_override | 0.3224 | 0.0046 | 0.7684 | 0.2360 | 1.0000 | 0.2950 |
| 6 | hq_baseline__without_override | single_model | hq_baseline | without_override | 0.3198 | 0.0071 | 0.7552 | 0.2638 | 1.0000 | 0.2800 |
| 7 | llm_engineering_plus_lambda_policies__with_override | single_model | llm_engineering_plus_lambda_policies | with_override | 0.3196 | 0.0075 | 0.7567 | 0.2287 | 1.0000 | 0.3225 |
| 8 | hq_plus_llm_engineering_plus_lambda_policies__with_override | single_model | hq_plus_llm_engineering_plus_lambda_policies | with_override | 0.3120 | 0.0076 | 0.7523 | 0.2255 | 1.0000 | 0.3044 |
| 9 | hq_plus_llm_engineering__with_override | single_model | hq_plus_llm_engineering | with_override | 0.3086 | 0.0050 | 0.7642 | 0.2317 | 1.0000 | 0.2594 |
| 10 | lambda_policies__without_override | single_model | lambda_policies | without_override | 0.3069 | 0.0094 | 0.7452 | 0.2531 | 1.0000 | 0.2775 |
| 11 | hq_plus_lambda_policies__without_override | single_model | hq_plus_lambda_policies | without_override | 0.3057 | 0.0089 | 0.7417 | 0.2505 | 1.0000 | 0.2869 |
| 12 | hq_baseline__with_override | single_model | hq_baseline | with_override | 0.3034 | 0.0058 | 0.7564 | 0.2269 | 1.0000 | 0.2731 |
| 13 | lambda_policies__with_override | single_model | lambda_policies | with_override | 0.2958 | 0.0080 | 0.7484 | 0.2206 | 1.0000 | 0.2744 |
| 14 | hq_plus_lambda_policies__with_override | single_model | hq_plus_lambda_policies | with_override | 0.2940 | 0.0065 | 0.7460 | 0.2195 | 1.0000 | 0.2831 |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `combination_success_cv_train_threshold_sweep.csv`.

| success_branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hq_baseline__with_override | single_model | with_override | 0.2700 | 0.2988 | 0.3050 | 0.2764 | 0.2731 | 0.3034 |
| hq_baseline__with_override | single_model | with_override | 0.2800 | 0.2976 | 0.3069 | 0.2654 | 0.2731 | 0.3034 |
| hq_baseline__with_override | single_model | with_override | 0.2600 | 0.2973 | 0.2997 | 0.2883 | 0.2731 | 0.3034 |
| hq_baseline__without_override | single_model | without_override | 0.2800 | 0.3139 | 0.3497 | 0.2227 | 0.2800 | 0.3198 |
| hq_baseline__without_override | single_model | without_override | 0.2700 | 0.3137 | 0.3429 | 0.2341 | 0.2800 | 0.3198 |
| hq_baseline__without_override | single_model | without_override | 0.2900 | 0.3125 | 0.3559 | 0.2100 | 0.2800 | 0.3198 |
| hq_plus_lambda_policies__with_override | single_model | with_override | 0.2700 | 0.2878 | 0.2861 | 0.2949 | 0.2831 | 0.2940 |
| hq_plus_lambda_policies__with_override | single_model | with_override | 0.2600 | 0.2873 | 0.2827 | 0.3074 | 0.2831 | 0.2940 |
| hq_plus_lambda_policies__with_override | single_model | with_override | 0.2900 | 0.2872 | 0.2910 | 0.2728 | 0.2831 | 0.2940 |
| hq_plus_lambda_policies__without_override | single_model | without_override | 0.2700 | 0.2980 | 0.3118 | 0.2534 | 0.2869 | 0.3057 |
| hq_plus_lambda_policies__without_override | single_model | without_override | 0.2900 | 0.2966 | 0.3210 | 0.2278 | 0.2869 | 0.3057 |
| hq_plus_lambda_policies__without_override | single_model | without_override | 0.2600 | 0.2964 | 0.3051 | 0.2665 | 0.2869 | 0.3057 |
| hq_plus_llm_engineering__with_override | single_model | with_override | 0.2700 | 0.3026 | 0.3049 | 0.2937 | 0.2594 | 0.3086 |
| hq_plus_llm_engineering__with_override | single_model | with_override | 0.2600 | 0.3018 | 0.3010 | 0.3051 | 0.2594 | 0.3086 |
| hq_plus_llm_engineering__with_override | single_model | with_override | 0.2800 | 0.3017 | 0.3072 | 0.2819 | 0.2594 | 0.3086 |
| hq_plus_llm_engineering__without_override | single_model | without_override | 0.2900 | 0.3182 | 0.3527 | 0.2289 | 0.2925 | 0.3263 |
| hq_plus_llm_engineering__without_override | single_model | without_override | 0.2800 | 0.3177 | 0.3465 | 0.2386 | 0.2925 | 0.3263 |
| hq_plus_llm_engineering__without_override | single_model | without_override | 0.2700 | 0.3175 | 0.3401 | 0.2509 | 0.2925 | 0.3263 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.2900 | 0.3058 | 0.3061 | 0.3048 | 0.3044 | 0.3120 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.2700 | 0.3051 | 0.3004 | 0.3259 | 0.3044 | 0.3120 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.2800 | 0.3050 | 0.3027 | 0.3149 | 0.3044 | 0.3120 |
| hq_plus_llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.2900 | 0.3178 | 0.3377 | 0.2576 | 0.3063 | 0.3266 |
| hq_plus_llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.3000 | 0.3168 | 0.3415 | 0.2461 | 0.3063 | 0.3266 |
| hq_plus_llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.2800 | 0.3167 | 0.3316 | 0.2686 | 0.3063 | 0.3266 |
| lambda_policies__with_override | single_model | with_override | 0.2400 | 0.2887 | 0.2804 | 0.3272 | 0.2744 | 0.2958 |
| lambda_policies__with_override | single_model | with_override | 0.2800 | 0.2880 | 0.2900 | 0.2802 | 0.2744 | 0.2958 |
| lambda_policies__with_override | single_model | with_override | 0.2300 | 0.2879 | 0.2773 | 0.3403 | 0.2744 | 0.2958 |
| lambda_policies__without_override | single_model | without_override | 0.2400 | 0.2977 | 0.2999 | 0.2895 | 0.2775 | 0.3069 |
| lambda_policies__without_override | single_model | without_override | 0.2800 | 0.2977 | 0.3184 | 0.2364 | 0.2775 | 0.3069 |
| lambda_policies__without_override | single_model | without_override | 0.3200 | 0.2976 | 0.3457 | 0.1914 | 0.2775 | 0.3069 |
| llm_engineering__with_override | single_model | with_override | 0.2800 | 0.3182 | 0.3262 | 0.2896 | 0.2950 | 0.3224 |
| llm_engineering__with_override | single_model | with_override | 0.2900 | 0.3177 | 0.3290 | 0.2792 | 0.2950 | 0.3224 |
| llm_engineering__with_override | single_model | with_override | 0.2700 | 0.3168 | 0.3211 | 0.3007 | 0.2950 | 0.3224 |
| llm_engineering__without_override | single_model | without_override | 0.3100 | 0.3379 | 0.3978 | 0.2109 | 0.3050 | 0.3463 |
| llm_engineering__without_override | single_model | without_override | 0.3000 | 0.3375 | 0.3884 | 0.2216 | 0.3050 | 0.3463 |
| llm_engineering__without_override | single_model | without_override | 0.2900 | 0.3371 | 0.3799 | 0.2325 | 0.3050 | 0.3463 |
| llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.3100 | 0.3151 | 0.3229 | 0.2872 | 0.3225 | 0.3196 |
| llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.3200 | 0.3151 | 0.3257 | 0.2787 | 0.3225 | 0.3196 |
| llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.3300 | 0.3149 | 0.3286 | 0.2700 | 0.3225 | 0.3196 |
| llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.3400 | 0.3326 | 0.3889 | 0.2109 | 0.3325 | 0.3393 |
| llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.3300 | 0.3322 | 0.3813 | 0.2194 | 0.3325 | 0.3393 |
| llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.3200 | 0.3315 | 0.3734 | 0.2289 | 0.3325 | 0.3393 |
