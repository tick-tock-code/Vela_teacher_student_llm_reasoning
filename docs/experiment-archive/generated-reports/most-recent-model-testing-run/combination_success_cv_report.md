# Combination Success Screening Report (Train CV Only)

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_183723_343974_saved_config_evaluation`
- Selected combo ref: `data/saved_model_configs/2026-04-21_132505_959940_model_testing::v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007`
- Target family: `v25_policies`
- Held-out evaluation: disabled
- Repeat CV with new seeds: True (n_runs=16)
- Success LR nested C CV: False
- Success LR fixed C when nested CV disabled: `1`

## Source CV Validation (Selected Combo)

| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---|---|---|---:|---:|---:|---:|
| v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007 | hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle | ridge | single_target | 0.4286 | 0.0794 | 0.2638 | 0.1758 |

## Success Screening (L2 Logistic Regression, CV only)

| rank | success_branch_id | model_variant | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | train_cv_f0_5_std | train_cv_roc_auc | train_cv_pr_auc | selected_c_final | threshold |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | llm_engineering__without_override | single_model | llm_engineering | without_override | 0.3362 | 0.0090 | 0.7653 | 0.2760 | 1.0000 | 0.3100 |
| 2 | llm_engineering_plus_lambda_policies__without_override | single_model | llm_engineering_plus_lambda_policies | without_override | 0.3335 | 0.0112 | 0.7494 | 0.2578 | 1.0000 | 0.3212 |
| 3 | hq_plus_llm_engineering_plus_lambda_policies__without_override | single_model | hq_plus_llm_engineering_plus_lambda_policies | without_override | 0.3280 | 0.0103 | 0.7471 | 0.2566 | 1.0000 | 0.3256 |
| 4 | hq_plus_llm_engineering__without_override | single_model | hq_plus_llm_engineering | without_override | 0.3254 | 0.0059 | 0.7622 | 0.2737 | 1.0000 | 0.2806 |
| 5 | llm_engineering_plus_lambda_policies__with_override | single_model | llm_engineering_plus_lambda_policies | with_override | 0.3188 | 0.0065 | 0.7535 | 0.2268 | 1.0000 | 0.3137 |
| 6 | hq_baseline__without_override | single_model | hq_baseline | without_override | 0.3182 | 0.0057 | 0.7561 | 0.2649 | 1.0000 | 0.2744 |
| 7 | llm_engineering__with_override | single_model | llm_engineering | with_override | 0.3168 | 0.0061 | 0.7650 | 0.2331 | 1.0000 | 0.2944 |
| 8 | hq_plus_llm_engineering_plus_lambda_policies__with_override | single_model | hq_plus_llm_engineering_plus_lambda_policies | with_override | 0.3139 | 0.0081 | 0.7531 | 0.2262 | 1.0000 | 0.3069 |
| 9 | hq_plus_llm_engineering__with_override | single_model | hq_plus_llm_engineering | with_override | 0.3079 | 0.0042 | 0.7649 | 0.2319 | 1.0000 | 0.2594 |
| 10 | lambda_policies__without_override | single_model | lambda_policies | without_override | 0.3078 | 0.0134 | 0.7435 | 0.2472 | 1.0000 | 0.2831 |
| 11 | hq_plus_lambda_policies__without_override | single_model | hq_plus_lambda_policies | without_override | 0.3027 | 0.0105 | 0.7441 | 0.2474 | 1.0000 | 0.2656 |
| 12 | hq_baseline__with_override | single_model | hq_baseline | with_override | 0.3021 | 0.0035 | 0.7575 | 0.2262 | 1.0000 | 0.2600 |
| 13 | lambda_policies__with_override | single_model | lambda_policies | with_override | 0.3010 | 0.0109 | 0.7468 | 0.2198 | 1.0000 | 0.2738 |
| 14 | hq_plus_lambda_policies__with_override | single_model | hq_plus_lambda_policies | with_override | 0.2964 | 0.0083 | 0.7492 | 0.2209 | 1.0000 | 0.2662 |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `combination_success_cv_train_threshold_sweep.csv`.

| success_branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hq_baseline__with_override | single_model | with_override | 0.2500 | 0.2961 | 0.2947 | 0.3022 | 0.2600 | 0.3021 |
| hq_baseline__with_override | single_model | with_override | 0.2300 | 0.2959 | 0.2879 | 0.3333 | 0.2600 | 0.3021 |
| hq_baseline__with_override | single_model | with_override | 0.2400 | 0.2955 | 0.2907 | 0.3161 | 0.2600 | 0.3021 |
| hq_baseline__without_override | single_model | without_override | 0.2900 | 0.3109 | 0.3541 | 0.2091 | 0.2744 | 0.3182 |
| hq_baseline__without_override | single_model | without_override | 0.2800 | 0.3106 | 0.3461 | 0.2202 | 0.2744 | 0.3182 |
| hq_baseline__without_override | single_model | without_override | 0.2700 | 0.3099 | 0.3379 | 0.2330 | 0.2744 | 0.3182 |
| hq_plus_lambda_policies__with_override | single_model | with_override | 0.2600 | 0.2897 | 0.2835 | 0.3179 | 0.2662 | 0.2964 |
| hq_plus_lambda_policies__with_override | single_model | with_override | 0.2500 | 0.2895 | 0.2809 | 0.3299 | 0.2662 | 0.2964 |
| hq_plus_lambda_policies__with_override | single_model | with_override | 0.2900 | 0.2888 | 0.2907 | 0.2814 | 0.2662 | 0.2964 |
| hq_plus_lambda_policies__without_override | single_model | without_override | 0.2500 | 0.2953 | 0.2977 | 0.2861 | 0.2656 | 0.3027 |
| hq_plus_lambda_policies__without_override | single_model | without_override | 0.2900 | 0.2951 | 0.3162 | 0.2330 | 0.2656 | 0.3027 |
| hq_plus_lambda_policies__without_override | single_model | without_override | 0.2600 | 0.2949 | 0.3013 | 0.2719 | 0.2656 | 0.3027 |
| hq_plus_llm_engineering__with_override | single_model | with_override | 0.2700 | 0.3033 | 0.3058 | 0.2939 | 0.2594 | 0.3079 |
| hq_plus_llm_engineering__with_override | single_model | with_override | 0.2800 | 0.3033 | 0.3088 | 0.2831 | 0.2594 | 0.3079 |
| hq_plus_llm_engineering__with_override | single_model | with_override | 0.2500 | 0.3030 | 0.2991 | 0.3196 | 0.2594 | 0.3079 |
| hq_plus_llm_engineering__without_override | single_model | without_override | 0.2800 | 0.3206 | 0.3498 | 0.2405 | 0.2806 | 0.3254 |
| hq_plus_llm_engineering__without_override | single_model | without_override | 0.2900 | 0.3197 | 0.3557 | 0.2277 | 0.2806 | 0.3254 |
| hq_plus_llm_engineering__without_override | single_model | without_override | 0.3000 | 0.3192 | 0.3624 | 0.2162 | 0.2806 | 0.3254 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.2600 | 0.3060 | 0.2987 | 0.3395 | 0.3069 | 0.3139 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.2700 | 0.3055 | 0.3006 | 0.3270 | 0.3069 | 0.3139 |
| hq_plus_llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.2900 | 0.3054 | 0.3055 | 0.3049 | 0.3069 | 0.3139 |
| hq_plus_llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.2800 | 0.3166 | 0.3314 | 0.2689 | 0.3256 | 0.3280 |
| hq_plus_llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.2700 | 0.3166 | 0.3266 | 0.2822 | 0.3256 | 0.3280 |
| hq_plus_llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.3000 | 0.3166 | 0.3412 | 0.2459 | 0.3256 | 0.3280 |
| lambda_policies__with_override | single_model | with_override | 0.2800 | 0.2934 | 0.2953 | 0.2858 | 0.2738 | 0.3010 |
| lambda_policies__with_override | single_model | with_override | 0.2600 | 0.2933 | 0.2901 | 0.3071 | 0.2738 | 0.3010 |
| lambda_policies__with_override | single_model | with_override | 0.2700 | 0.2932 | 0.2927 | 0.2958 | 0.2738 | 0.3010 |
| lambda_policies__without_override | single_model | without_override | 0.2800 | 0.2997 | 0.3207 | 0.2378 | 0.2831 | 0.3078 |
| lambda_policies__without_override | single_model | without_override | 0.2600 | 0.2995 | 0.3109 | 0.2615 | 0.2831 | 0.3078 |
| lambda_policies__without_override | single_model | without_override | 0.2700 | 0.2992 | 0.3152 | 0.2487 | 0.2831 | 0.3078 |
| llm_engineering__with_override | single_model | with_override | 0.3000 | 0.3121 | 0.3273 | 0.2634 | 0.2944 | 0.3168 |
| llm_engineering__with_override | single_model | with_override | 0.2900 | 0.3109 | 0.3221 | 0.2730 | 0.2944 | 0.3168 |
| llm_engineering__with_override | single_model | with_override | 0.2800 | 0.3108 | 0.3184 | 0.2836 | 0.2944 | 0.3168 |
| llm_engineering__without_override | single_model | without_override | 0.3000 | 0.3291 | 0.3784 | 0.2165 | 0.3100 | 0.3362 |
| llm_engineering__without_override | single_model | without_override | 0.3100 | 0.3280 | 0.3860 | 0.2049 | 0.3100 | 0.3362 |
| llm_engineering__without_override | single_model | without_override | 0.2900 | 0.3272 | 0.3673 | 0.2277 | 0.3100 | 0.3362 |
| llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.3200 | 0.3148 | 0.3261 | 0.2768 | 0.3137 | 0.3188 |
| llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.3300 | 0.3131 | 0.3273 | 0.2670 | 0.3137 | 0.3188 |
| llm_engineering_plus_lambda_policies__with_override | single_model | with_override | 0.2900 | 0.3128 | 0.3154 | 0.3030 | 0.3137 | 0.3188 |
| llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.3200 | 0.3278 | 0.3696 | 0.2260 | 0.3212 | 0.3335 |
| llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.3300 | 0.3259 | 0.3744 | 0.2150 | 0.3212 | 0.3335 |
| llm_engineering_plus_lambda_policies__without_override | single_model | without_override | 0.3100 | 0.3245 | 0.3595 | 0.2336 | 0.3212 | 0.3335 |
