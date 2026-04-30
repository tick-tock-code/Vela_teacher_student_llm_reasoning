# Stage 2 CV Success Screen: hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle

## Summary

This is the train-CV-only Stage 2 success screen for the larger reasoning-feature predictor. It uses the saved Ridge model trained on `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle` to predict the 16 reasoning features, then evaluates single logistic regression success models with different HQ/LLM/lambda base-success feature combinations.

- Selected reasoning combo: `data/saved_model_configs/2026-04-21_132505_959940_model_testing::v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007`
- Run dir: `tmp/runs/teacher_student_distillation_v1/2026-04-30_183723_343974_saved_config_evaluation`
- Generated report: `docs/experiment-archive/generated-reports/testing_models/combination_success_cv_report_2026-04-30_183723_343974_saved_config_evaluation.md`
- Reasoning model: Ridge, single target per reasoning feature
- Reasoning Ridge alpha: `1.0`
- Reasoning input features: `985`
- Reasoning targets: `16`
- Success model: single logistic regression only
- Success LR nested C CV: disabled
- Success LR fixed C: `1.0`
- CV repeats: `16`
- Held-out/test evaluation: not run

## Reasoning Source CV

| combo_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007 | 0.4286 | 0.0794 | 0.2638 | 0.1758 |

## Top Candidates For Later Held-Out Test

The top three by the locked ranking rule are:

| rank | success_branch_id | train_cv_f0_5 | train_cv_roc_auc | train_cv_pr_auc | C | threshold |
|---:|---|---:|---:|---:|---:|---:|
| 1 | llm_engineering__without_override | 0.3362 | 0.7653 | 0.2760 | 1.0000 | 0.3100 |
| 2 | llm_engineering_plus_lambda_policies__without_override | 0.3335 | 0.7494 | 0.2578 | 1.0000 | 0.3212 |
| 3 | hq_plus_llm_engineering_plus_lambda_policies__without_override | 0.3280 | 0.7471 | 0.2566 | 1.0000 | 0.3256 |

## All Train-CV Branches

Rows are ordered by `train_cv_f0_5` descending, then `train_cv_roc_auc` descending, then `train_cv_pr_auc` descending, then `success_branch_id` ascending.

| rank | success_branch_id | train_cv_f0_5 | train_cv_f0_5_std | train_cv_roc_auc | train_cv_pr_auc | train_cv_precision | train_cv_recall | C | threshold |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | llm_engineering__without_override | 0.3362 | 0.0090 | 0.7653 | 0.2760 | 0.3994 | 0.2113 | 1.0000 | 0.3100 |
| 2 | llm_engineering_plus_lambda_policies__without_override | 0.3335 | 0.0112 | 0.7494 | 0.2578 | 0.3789 | 0.2306 | 1.0000 | 0.3212 |
| 3 | hq_plus_llm_engineering_plus_lambda_policies__without_override | 0.3280 | 0.0103 | 0.7471 | 0.2566 | 0.3712 | 0.2336 | 1.0000 | 0.3256 |
| 4 | hq_plus_llm_engineering__without_override | 0.3254 | 0.0059 | 0.7622 | 0.2737 | 0.3570 | 0.2457 | 1.0000 | 0.2806 |
| 5 | llm_engineering_plus_lambda_policies__with_override | 0.3188 | 0.0065 | 0.7535 | 0.2268 | 0.3286 | 0.2879 | 1.0000 | 0.3137 |
| 6 | hq_baseline__without_override | 0.3182 | 0.0057 | 0.7561 | 0.2649 | 0.3534 | 0.2378 | 1.0000 | 0.2744 |
| 7 | llm_engineering__with_override | 0.3168 | 0.0061 | 0.7650 | 0.2331 | 0.3303 | 0.2753 | 1.0000 | 0.2944 |
| 8 | hq_plus_llm_engineering_plus_lambda_policies__with_override | 0.3139 | 0.0081 | 0.7531 | 0.2262 | 0.3188 | 0.3015 | 1.0000 | 0.3069 |
| 9 | hq_plus_llm_engineering__with_override | 0.3079 | 0.0042 | 0.7649 | 0.2319 | 0.3071 | 0.3172 | 1.0000 | 0.2594 |
| 10 | lambda_policies__without_override | 0.3078 | 0.0134 | 0.7435 | 0.2472 | 0.3356 | 0.2429 | 1.0000 | 0.2831 |
| 11 | hq_plus_lambda_policies__without_override | 0.3027 | 0.0105 | 0.7441 | 0.2474 | 0.3146 | 0.2745 | 1.0000 | 0.2656 |
| 12 | hq_baseline__with_override | 0.3021 | 0.0035 | 0.7575 | 0.2262 | 0.3044 | 0.2991 | 1.0000 | 0.2600 |
| 13 | lambda_policies__with_override | 0.3010 | 0.0109 | 0.7468 | 0.2198 | 0.3022 | 0.3045 | 1.0000 | 0.2737 |
| 14 | hq_plus_lambda_policies__with_override | 0.2964 | 0.0083 | 0.7492 | 0.2209 | 0.2926 | 0.3217 | 1.0000 | 0.2662 |

## Interpretation

The larger reasoning predictor slightly improves reasoning-feature CV (`r2_mean=0.4286` versus `0.4229` for lambda+bundle), but its best Stage 2 success train-CV F0.5 is lower in this screen. The strongest branch is still `llm_engineering__without_override`. The top three are candidate branches for a later held-out run, but no held-out/test result is included in this report.
