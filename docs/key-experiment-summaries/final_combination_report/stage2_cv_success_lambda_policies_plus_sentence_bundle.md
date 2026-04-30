# Stage 2 CV Success Screen: lambda_policies_plus_sentence_bundle

## Summary

This is the train-CV-only Stage 2 success screen for the minimal reasoning-feature predictor. It uses the saved Ridge model trained on `lambda_policies_plus_sentence_bundle` to predict the 16 reasoning features, then evaluates single logistic regression success models with different HQ/LLM/lambda base-success feature combinations.

- Selected reasoning combo: `data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003`
- Run dir: `tmp/runs/teacher_student_distillation_v1/2026-04-30_183000_968429_saved_config_evaluation`
- Generated report: `docs/experiment-archive/generated-reports/testing_models/combination_success_cv_report_2026-04-30_183000_968429_saved_config_evaluation.md`
- Reasoning model: Ridge, single target per reasoning feature
- Reasoning Ridge alpha: `1.0`
- Reasoning input features: `940`
- Reasoning targets: `16`
- Success model: single logistic regression only
- Success LR nested C CV: disabled
- Success LR fixed C: `1.0`
- CV repeats: `16`
- Held-out/test evaluation: not run

## Reasoning Source CV

| combo_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003 | 0.4229 | 0.0800 | 0.2647 | 0.1774 |

## Top Candidates For Later Held-Out Test

The top three by the locked ranking rule are:

| rank | success_branch_id | train_cv_f0_5 | train_cv_roc_auc | train_cv_pr_auc | C | threshold |
|---:|---|---:|---:|---:|---:|---:|
| 1 | llm_engineering__without_override | 0.3463 | 0.7683 | 0.2827 | 1.0000 | 0.3050 |
| 2 | llm_engineering_plus_lambda_policies__without_override | 0.3393 | 0.7523 | 0.2642 | 1.0000 | 0.3325 |
| 3 | hq_plus_llm_engineering_plus_lambda_policies__without_override | 0.3266 | 0.7464 | 0.2576 | 1.0000 | 0.3063 |

## All Train-CV Branches

Rows are ordered by `train_cv_f0_5` descending, then `train_cv_roc_auc` descending, then `train_cv_pr_auc` descending, then `success_branch_id` ascending.

| rank | success_branch_id | train_cv_f0_5 | train_cv_f0_5_std | train_cv_roc_auc | train_cv_pr_auc | train_cv_precision | train_cv_recall | C | threshold |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | llm_engineering__without_override | 0.3463 | 0.0082 | 0.7683 | 0.2827 | 0.4062 | 0.2216 | 1.0000 | 0.3050 |
| 2 | llm_engineering_plus_lambda_policies__without_override | 0.3393 | 0.0134 | 0.7523 | 0.2642 | 0.3928 | 0.2235 | 1.0000 | 0.3325 |
| 3 | hq_plus_llm_engineering_plus_lambda_policies__without_override | 0.3266 | 0.0107 | 0.7464 | 0.2576 | 0.3596 | 0.2511 | 1.0000 | 0.3063 |
| 4 | hq_plus_llm_engineering__without_override | 0.3263 | 0.0074 | 0.7614 | 0.2755 | 0.3670 | 0.2360 | 1.0000 | 0.2925 |
| 5 | llm_engineering__with_override | 0.3224 | 0.0046 | 0.7684 | 0.2360 | 0.3358 | 0.2787 | 1.0000 | 0.2950 |
| 6 | hq_baseline__without_override | 0.3198 | 0.0071 | 0.7552 | 0.2638 | 0.3588 | 0.2285 | 1.0000 | 0.2800 |
| 7 | llm_engineering_plus_lambda_policies__with_override | 0.3196 | 0.0075 | 0.7567 | 0.2287 | 0.3315 | 0.2830 | 1.0000 | 0.3225 |
| 8 | hq_plus_llm_engineering_plus_lambda_policies__with_override | 0.3120 | 0.0076 | 0.7523 | 0.2255 | 0.3167 | 0.3013 | 1.0000 | 0.3044 |
| 9 | hq_plus_llm_engineering__with_override | 0.3086 | 0.0050 | 0.7642 | 0.2317 | 0.3077 | 0.3215 | 1.0000 | 0.2594 |
| 10 | lambda_policies__without_override | 0.3069 | 0.0094 | 0.7452 | 0.2531 | 0.3312 | 0.2535 | 1.0000 | 0.2775 |
| 11 | hq_plus_lambda_policies__without_override | 0.3057 | 0.0089 | 0.7417 | 0.2505 | 0.3322 | 0.2434 | 1.0000 | 0.2869 |
| 12 | hq_baseline__with_override | 0.3034 | 0.0058 | 0.7564 | 0.2269 | 0.3109 | 0.2796 | 1.0000 | 0.2731 |
| 13 | lambda_policies__with_override | 0.2958 | 0.0080 | 0.7484 | 0.2206 | 0.2977 | 0.3006 | 1.0000 | 0.2744 |
| 14 | hq_plus_lambda_policies__with_override | 0.2940 | 0.0065 | 0.7460 | 0.2195 | 0.2967 | 0.2917 | 1.0000 | 0.2831 |

## Interpretation

For the minimal reasoning predictor, the strongest train-CV branch is `llm_engineering__without_override`. The top three all avoid the HQ forced-success override, and all use the fixed success LR `C=1.0`. These are candidate branches for a later held-out run, but no held-out/test result is included in this report.
