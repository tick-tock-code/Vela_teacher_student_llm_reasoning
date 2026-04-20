# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_024204_856747_saved_config_evaluation`
- Target family: `v25_policies`
- Feature set: `lambda_policies_plus_sentence_bundle`
- Combo refs: ['data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003', 'data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004', 'data/saved_model_configs/2026-04-20_010135_130404_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002']
- Reproduction reference run: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_024252_015048_success_reproduction`

## Reasoning Transfer Summary

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| xgb3_regressor | 0.4201 | 0.0835 | 0.2631 | 0.1776 |
| combined_best | 0.4166 | 0.0888 | 0.2633 | 0.1834 |
| ridge | 0.4008 | 0.0941 | 0.2666 | 0.1929 |
| mlp_regressor | 0.4000 | 0.0964 | 0.2663 | 0.1874 |

## Success Transfer Metrics

| model_set_id | branch_id | f0_5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| mlp_regressor | llm_engineering_plus_pred_reasoning | 0.3299 | 0.7385 | 0.2640 | 0.3871 | 0.2074 | 0.2800 |
| xgb3_regressor | hq_plus_pred_reasoning | 0.3260 | 0.7304 | 0.2333 | 0.3969 | 0.1901 | 0.3000 |
| mlp_regressor | hq_plus_pred_reasoning | 0.3226 | 0.7311 | 0.2361 | 0.3469 | 0.2519 | 0.2700 |
| ridge | hq_plus_pred_reasoning | 0.3222 | 0.7281 | 0.2368 | 0.3513 | 0.2420 | 0.2700 |
| combined_best | hq_plus_pred_reasoning | 0.3202 | 0.7318 | 0.2359 | 0.3767 | 0.2000 | 0.3000 |
| ridge | llm_engineering_plus_pred_reasoning | 0.3100 | 0.7343 | 0.2630 | 0.3838 | 0.1753 | 0.3000 |
| combined_best | llm_engineering_plus_pred_reasoning | 0.3076 | 0.7432 | 0.2734 | 0.3918 | 0.1654 | 0.3000 |
| xgb3_regressor | llm_engineering_plus_pred_reasoning | 0.3058 | 0.7381 | 0.2670 | 0.4167 | 0.1481 | 0.3000 |
| combined_best | pred_reasoning_only | 0.2821 | 0.7268 | 0.2480 | 0.3913 | 0.1333 | 0.2700 |
| mlp_regressor | pred_reasoning_only | 0.2797 | 0.7221 | 0.2430 | 0.3671 | 0.1432 | 0.2700 |
| ridge | pred_reasoning_only | 0.2756 | 0.7210 | 0.2452 | 0.3386 | 0.1580 | 0.2600 |
| xgb3_regressor | pred_reasoning_only | 0.2570 | 0.7271 | 0.2475 | 0.5072 | 0.0864 | 0.3000 |

## Reproduction Consistency Check

- Tolerance: ±0.005 F0.5

| experiment_id | headline_target_f0_5 | reproduced_test_f0_5 | delta_f0_5 | abs_delta_f0_5 | within_tolerance |
|---|---:|---:|---:|---:|---|
| hq_only | 0.2730 | 0.2726 | -0.0004 | 0.0004 | True |
| hq_plus_policy_induction | 0.3000 | 0.3005 | +0.0005 | 0.0005 | True |
| llm_engineering_only | 0.2840 | 0.2843 | +0.0003 | 0.0003 | True |
| llm_engineering_plus_policy_induction | 0.3340 | 0.3344 | +0.0004 | 0.0004 | True |

## Combined Best Assignment (CV-R2 Source)

| target_id | selected_model_id | selected_combo_id | cv_r2 |
|---|---|---|---:|
| v25_p1 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3770 |
| v25_p11 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3262 |
| v25_p112 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.5022 |
| v25_p116 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.5125 |
| v25_p121 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3633 |
| v25_p135 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3983 |
| v25_p143 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.3901 |
| v25_p150 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.4709 |
| v25_p157 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.4885 |
| v25_p161 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.2838 |
| v25_p38 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.4366 |
| v25_p52 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.4947 |
| v25_p55 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.5072 |
| v25_p58 | xgb3_regressor | v25_policies__lambda_policies_plus_sentence_bundle__xgb3_regressor__single_target__0004 | 0.5374 |
| v25_p72 | mlp_regressor | v25_policies__lambda_policies_plus_sentence_bundle__mlp_regressor__multi_output__0002 | 0.3282 |
| v25_p80 | ridge | v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.5044 |

## Reasoning Per-Target Metrics

| model_set_id | target_id | r2 | rmse | mae |
|---|---|---:|---:|---:|
| combined_best | v25_p1 | 0.3479 | 0.3392 | 0.2646 |
| combined_best | v25_p11 | 0.2976 | 0.2513 | 0.1652 |
| combined_best | v25_p112 | 0.4972 | 0.3385 | 0.2744 |
| combined_best | v25_p116 | 0.5265 | 0.3064 | 0.2230 |
| combined_best | v25_p121 | 0.2849 | 0.1694 | 0.1027 |
| combined_best | v25_p135 | 0.3961 | 0.3046 | 0.2172 |
| combined_best | v25_p143 | 0.3124 | 0.1341 | 0.0474 |
| combined_best | v25_p150 | 0.4951 | 0.3261 | 0.2661 |
| combined_best | v25_p157 | 0.4862 | 0.2168 | 0.1366 |
| combined_best | v25_p161 | 0.3262 | 0.2049 | 0.0991 |
| combined_best | v25_p38 | 0.3854 | 0.1753 | 0.0931 |
| combined_best | v25_p52 | 0.4916 | 0.3448 | 0.2781 |
| combined_best | v25_p55 | 0.4937 | 0.2969 | 0.2151 |
| combined_best | v25_p58 | 0.5159 | 0.2646 | 0.1697 |
| combined_best | v25_p72 | 0.3092 | 0.2642 | 0.1779 |
| combined_best | v25_p80 | 0.5004 | 0.2755 | 0.2039 |
| mlp_regressor | v25_p1 | 0.3479 | 0.3392 | 0.2646 |
| mlp_regressor | v25_p11 | 0.2976 | 0.2513 | 0.1652 |
| mlp_regressor | v25_p112 | 0.4757 | 0.3457 | 0.2778 |
| mlp_regressor | v25_p116 | 0.5031 | 0.3138 | 0.2302 |
| mlp_regressor | v25_p121 | 0.2849 | 0.1694 | 0.1027 |
| mlp_regressor | v25_p135 | 0.3961 | 0.3046 | 0.2172 |
| mlp_regressor | v25_p143 | 0.2038 | 0.1443 | 0.0790 |
| mlp_regressor | v25_p150 | 0.4849 | 0.3294 | 0.2595 |
| mlp_regressor | v25_p157 | 0.4315 | 0.2280 | 0.1504 |
| mlp_regressor | v25_p161 | 0.2927 | 0.2099 | 0.1261 |
| mlp_regressor | v25_p38 | 0.3817 | 0.1758 | 0.0954 |
| mlp_regressor | v25_p52 | 0.4852 | 0.3470 | 0.2714 |
| mlp_regressor | v25_p55 | 0.4815 | 0.3005 | 0.2158 |
| mlp_regressor | v25_p58 | 0.5103 | 0.2661 | 0.1750 |
| mlp_regressor | v25_p72 | 0.3092 | 0.2642 | 0.1779 |
| mlp_regressor | v25_p80 | 0.5147 | 0.2716 | 0.1898 |
| ridge | v25_p1 | 0.3418 | 0.3408 | 0.2720 |
| ridge | v25_p11 | 0.2734 | 0.2556 | 0.1747 |
| ridge | v25_p112 | 0.4972 | 0.3385 | 0.2744 |
| ridge | v25_p116 | 0.4829 | 0.3202 | 0.2490 |
| ridge | v25_p121 | 0.3073 | 0.1667 | 0.0889 |
| ridge | v25_p135 | 0.3891 | 0.3064 | 0.2403 |
| ridge | v25_p143 | 0.2410 | 0.1409 | 0.0678 |
| ridge | v25_p150 | 0.4951 | 0.3261 | 0.2661 |
| ridge | v25_p157 | 0.4862 | 0.2168 | 0.1366 |
| ridge | v25_p161 | 0.2851 | 0.2110 | 0.1244 |
| ridge | v25_p38 | 0.3854 | 0.1753 | 0.0931 |
| ridge | v25_p52 | 0.4916 | 0.3448 | 0.2781 |
| ridge | v25_p55 | 0.4650 | 0.3052 | 0.2362 |
| ridge | v25_p58 | 0.4831 | 0.2734 | 0.1895 |
| ridge | v25_p72 | 0.2887 | 0.2681 | 0.1908 |
| ridge | v25_p80 | 0.5004 | 0.2755 | 0.2039 |
| xgb3_regressor | v25_p1 | 0.3546 | 0.3375 | 0.2630 |
| xgb3_regressor | v25_p11 | 0.2816 | 0.2541 | 0.1571 |
| xgb3_regressor | v25_p112 | 0.4822 | 0.3436 | 0.2748 |
| xgb3_regressor | v25_p116 | 0.5265 | 0.3064 | 0.2230 |
| xgb3_regressor | v25_p121 | 0.3570 | 0.1606 | 0.0740 |
| xgb3_regressor | v25_p135 | 0.3921 | 0.3056 | 0.2273 |
| xgb3_regressor | v25_p143 | 0.3124 | 0.1341 | 0.0474 |
| xgb3_regressor | v25_p150 | 0.4995 | 0.3247 | 0.2604 |
| xgb3_regressor | v25_p157 | 0.4539 | 0.2235 | 0.1255 |
| xgb3_regressor | v25_p161 | 0.3262 | 0.2049 | 0.0991 |
| xgb3_regressor | v25_p38 | 0.4044 | 0.1726 | 0.0770 |
| xgb3_regressor | v25_p52 | 0.4852 | 0.3469 | 0.2748 |
| xgb3_regressor | v25_p55 | 0.4937 | 0.2969 | 0.2151 |
| xgb3_regressor | v25_p58 | 0.5159 | 0.2646 | 0.1697 |
| xgb3_regressor | v25_p72 | 0.3149 | 0.2631 | 0.1714 |
| xgb3_regressor | v25_p80 | 0.5208 | 0.2698 | 0.1814 |
