# Stage 2 Held-Out Success Test: llm_engineering__without_override

## Summary

This held-out test evaluates the top train-CV success branch, `llm_engineering__without_override`, for both selected reasoning-feature predictors. Both runs use single logistic regression only, fixed success LR `C=1.0`, no nested success C tuning, and 16 CV seed repeats for train-threshold selection.

- Success branch: `llm_engineering__without_override`
- Success model: single logistic regression
- Success LR nested C CV: disabled
- Success LR fixed C: `1.0`
- CV repeats: `16`
- Held-out/test evaluation: enabled

## Held-Out Results

| reasoning predictor | run dir | reasoning held-out r2_mean | train_cv_f0_5 | test_f0_5 | test_roc_auc | test_pr_auc | test_precision | test_recall | threshold | C |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lambda_policies_plus_sentence_bundle | `tmp/runs/teacher_student_distillation_v1/2026-04-30_191111_782723_saved_config_evaluation` | 0.4008 | 0.3463 | 0.3099 | 0.7318 | 0.2610 | 0.3729 | 0.1872 | 0.3050 | 1.0000 |
| hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle | `tmp/runs/teacher_student_distillation_v1/2026-04-30_191214_462272_saved_config_evaluation` | 0.4023 | 0.3362 | 0.3060 | 0.7314 | 0.2614 | 0.3733 | 0.1815 | 0.3100 | 1.0000 |

## Interpretation

The minimal `lambda_policies_plus_sentence_bundle` reasoning predictor is slightly better on held-out success F0.5 for the selected `llm_engineering__without_override` branch: `0.3099` versus `0.3060`. The larger reasoning predictor has marginally higher held-out reasoning-feature agreement, but that does not translate into better held-out success performance in this branch.

No democratic or soft-ensemble success variants were included.
