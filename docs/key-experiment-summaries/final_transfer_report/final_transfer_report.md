# Final Transfer Report

## Summary

This report is the paper-facing working summary for the simple LR-only transfer story. Stage 1 uses a fixed saved Ridge model to predict the 16 `v25_policies` reasoning features from `lambda_policies_plus_sentence_bundle`. Stage 2 appends those predicted reasoning features to the success-prediction pipeline and evaluates single logistic regression only.

The headline narrative is:

1. Among the simple `X + sentence_bundle` candidates, `lambda_policies_plus_sentence_bundle` is the strongest reasoning-feature predictor.
2. The saved Ridge reasoning predictor transfers cleanly to held-out data: source CV mean R2 is 0.4229 and held-out mean R2 is 0.4008.
3. On held-out success prediction, `reasoning_pred < HQ + reasoning_pred < LLM-eng + reasoning_pred` under both HQ override conditions.
4. HQ override improves held-out F0.5 for every success feature branch in the nested-C run, and also improves every branch in the fixed-C run, with only a small HQ-branch margin.
5. The result is robust to either nested LR tuning over `C = [0.01, 0.05, 0.1, 0.5, 1.0]` or fixed non-nested `C = 1.0`.

Complex reasoning feature combinations, non-Ridge reasoning models, and democratic/soft ensemble success models are not used in the headline narrative because they did not improve clarity or held-out performance enough to justify the extra moving parts.

## Source Artifacts

- Saved reasoning combo: `data/saved_model_configs/2026-04-19_195413_226932_model_testing::v25_policies__lambda_policies_plus_sentence_bundle__ridge__single_target__0003`
- Reasoning screen run: `tmp/runs/teacher_student_distillation_v1/2026-04-22_184529_729730_model_testing`
- Nested-C transfer run: `tmp/runs/teacher_student_distillation_v1/2026-04-30_194316_242881_saved_config_evaluation`
- Fixed-C transfer run: `tmp/runs/teacher_student_distillation_v1/2026-04-30_194838_115531_saved_config_evaluation`
- Archived generated reports: `docs/experiment-archive/generated-reports/testing_models/`

## Stage 1: Reasoning Feature Prediction

These are the simple bundle candidates used for the paper narrative. The screen score is `mean - 0.5 * std`, so the ranking is intentionally conservative.

| Rank | Feature set | Output mode | Mean R2 | R2 std | Screen score | RMSE | MAE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `lambda_policies_plus_sentence_bundle` | `single_target` | 0.4211 | 0.0029 | 0.4196 | 0.2651 | 0.1774 |
| 2 | `hq_plus_sentence_bundle` | `single_target` | 0.4186 | 0.0015 | 0.4178 | 0.2659 | 0.1814 |
| 3 | `llm_engineering_plus_sentence_bundle` | `single_target` | 0.3898 | 0.0014 | 0.3891 | 0.2736 | 0.1891 |
| 4 | `sentence_bundle` | `single_target` | 0.3647 | 0.0011 | 0.3642 | 0.2782 | 0.1945 |

Ridge was used for the selected reasoning predictor. The saved model config uses `alpha = 1.0` for all 16 reasoning targets.

## Reasoning Agreement

The selected `lambda_policies_plus_sentence_bundle` Ridge model has similar source-CV and held-out reasoning agreement. This supports using the predicted reasoning features as a practical approximation of the teacher policy features rather than treating the transfer stage as a distribution-shift failure.

| Evaluation | Model | Mean R2 | R2 std | RMSE | MAE |
|---|---|---:|---:|---:|---:|
| Source CV | Ridge | 0.4229 | 0.0800 | 0.2647 | 0.1774 |
| Held-out test | Ridge | 0.4008 | 0.0941 | 0.2666 | 0.1929 |

### Held-out Reasoning Agreement By Target

| Target | R2 | RMSE | MAE |
|---|---:|---:|---:|
| `v25_p1` | 0.3418 | 0.3408 | 0.2720 |
| `v25_p11` | 0.2734 | 0.2556 | 0.1747 |
| `v25_p38` | 0.3854 | 0.1753 | 0.0931 |
| `v25_p52` | 0.4916 | 0.3448 | 0.2781 |
| `v25_p55` | 0.4650 | 0.3052 | 0.2362 |
| `v25_p58` | 0.4831 | 0.2734 | 0.1895 |
| `v25_p72` | 0.2887 | 0.2681 | 0.1908 |
| `v25_p80` | 0.5004 | 0.2755 | 0.2039 |
| `v25_p112` | 0.4972 | 0.3385 | 0.2744 |
| `v25_p116` | 0.4829 | 0.3202 | 0.2490 |
| `v25_p121` | 0.3073 | 0.1667 | 0.0889 |
| `v25_p135` | 0.3891 | 0.3064 | 0.2403 |
| `v25_p143` | 0.2410 | 0.1409 | 0.0678 |
| `v25_p150` | 0.4951 | 0.3261 | 0.2661 |
| `v25_p157` | 0.4862 | 0.2168 | 0.1366 |
| `v25_p161` | 0.2851 | 0.2110 | 0.1244 |

## Stage 2: Success Transfer

Both reruns used:

- Success model: single logistic regression only.
- CV repeats: 16.
- Held-out evaluation: enabled.
- Success feature branches: `reasoning_pred`, `HQ + reasoning_pred`, `LLM-eng + reasoning_pred`.
- HQ override: evaluated separately as `with_override` and `without_override`.

The nested run tunes success LR `C` inside each outer fold over `[0.01, 0.05, 0.1, 0.5, 1.0]`. The fixed run disables nested tuning and uses `C = 1.0`.

### Nested-C LR: HQ Override On

| Success features | Train CV F0.5 | CV std | Held-out F0.5 | ROC AUC | PR AUC | Precision | Recall | C final | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reasoning_pred` | 0.3131 | 0.0061 | 0.2980 | 0.7282 | 0.2323 | 0.3104 | 0.2568 | 0.05 | 0.250 |
| `HQ + reasoning_pred` | 0.3067 | 0.0062 | 0.3211 | 0.7281 | 0.2368 | 0.3472 | 0.2469 | 0.01 | 0.265 |
| `LLM-eng + reasoning_pred` | 0.3207 | 0.0056 | 0.3260 | 0.7378 | 0.2410 | 0.3463 | 0.2642 | 0.10 | 0.280 |

### Nested-C LR: HQ Override Off

| Success features | Train CV F0.5 | CV std | Held-out F0.5 | ROC AUC | PR AUC | Precision | Recall | C final | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reasoning_pred` | 0.3330 | 0.0092 | 0.2749 | 0.7206 | 0.2446 | 0.3226 | 0.1728 | 0.05 | 0.260 |
| `HQ + reasoning_pred` | 0.3275 | 0.0087 | 0.3016 | 0.7247 | 0.2622 | 0.3679 | 0.1753 | 0.01 | 0.265 |
| `LLM-eng + reasoning_pred` | 0.3434 | 0.0068 | 0.3081 | 0.7343 | 0.2630 | 0.3719 | 0.1827 | 0.10 | 0.290 |

### Fixed-C LR: HQ Override On

| Success features | Train CV F0.5 | CV std | Held-out F0.5 | ROC AUC | PR AUC | Precision | Recall | C final | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reasoning_pred` | 0.3161 | 0.0049 | 0.2953 | 0.7270 | 0.2311 | 0.3106 | 0.2469 | 1.00 | 0.270 |
| `HQ + reasoning_pred` | 0.3034 | 0.0058 | 0.3089 | 0.7235 | 0.2356 | 0.3174 | 0.2790 | 1.00 | 0.270 |
| `LLM-eng + reasoning_pred` | 0.3224 | 0.0046 | 0.3231 | 0.7350 | 0.2398 | 0.3443 | 0.2593 | 1.00 | 0.295 |

### Fixed-C LR: HQ Override Off

| Success features | Train CV F0.5 | CV std | Held-out F0.5 | ROC AUC | PR AUC | Precision | Recall | C final | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reasoning_pred` | 0.3384 | 0.0063 | 0.2599 | 0.7187 | 0.2420 | 0.3032 | 0.1654 | 1.00 | 0.270 |
| `HQ + reasoning_pred` | 0.3198 | 0.0071 | 0.3044 | 0.7187 | 0.2632 | 0.3398 | 0.2148 | 1.00 | 0.275 |
| `LLM-eng + reasoning_pred` | 0.3463 | 0.0082 | 0.3122 | 0.7318 | 0.2610 | 0.3744 | 0.1877 | 1.00 | 0.305 |

## Robustness Summary

The core ordering is stable across nested and fixed regularisation. Held-out F0.5 improves as more grounded success features are added: reasoning-only is weakest, adding HQ features improves it, and adding LLM-engineering features is strongest.

| Regularisation | HQ override | `reasoning_pred` | `HQ + reasoning_pred` | `LLM-eng + reasoning_pred` |
|---|---|---:|---:|---:|
| Nested C grid | On | 0.2980 | 0.3211 | 0.3260 |
| Nested C grid | Off | 0.2749 | 0.3016 | 0.3081 |
| Fixed C=1.0 | On | 0.2953 | 0.3089 | 0.3231 |
| Fixed C=1.0 | Off | 0.2599 | 0.3044 | 0.3122 |

The fixed-C run is a useful robustness check because it removes inner hyperparameter tuning from the success stage. The best held-out result is still `LLM-eng + reasoning_pred`, and the fixed result with HQ override on is close to the nested result: 0.3231 versus 0.3260.

Train CV F0.5 is consistently higher without the HQ override in these reruns, while held-out F0.5 is consistently better with the override. For the paper narrative, the clean claim should be about held-out performance and should present the override-on and override-off conditions separately rather than implying train-CV and test move identically.

## Reproduction Reference

The success pipeline reproduction check remains within the prior tolerance.

| Experiment | Headline F0.5 | Reproduced held-out F0.5 | Delta | Within tolerance |
|---|---:|---:|---:|---|
| `hq_only` | 0.2730 | 0.2726 | -0.0004 | True |
| `hq_plus_policy_induction` | 0.3000 | 0.3005 | +0.0005 | True |
| `llm_engineering_only` | 0.2840 | 0.2843 | +0.0003 | True |
| `llm_engineering_plus_policy_induction` | 0.3340 | 0.3344 | +0.0004 | True |

## Interpretation

The publishable story should stay simple: lambda policies plus sentence embeddings are enough to predict the policy-reasoning layer well; the predicted reasoning layer transfers to success prediction; and the strongest held-out success result comes from combining predicted reasoning with LLM-engineering features. HQ override should be reported as an explicit condition, not folded into the feature-set definition.

The result does not claim that the student fully replaces the teacher policy features. The reproduced `LLM Engineering + Policy Induction` benchmark is still higher at 0.3344. The cleaner claim is that the student reasoning layer is compact, reproducible, and preserves useful success-prediction signal with held-out F0.5 around 0.323-0.326 when paired with LLM-engineering features.

## Excluded From Headline

- Richer reasoning feature predictors such as `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle` slightly improve reasoning R2 in some screens, but did not improve the held-out success story enough to justify the extra feature-dependence.
- Non-Ridge reasoning models made the model-comparison report harder to explain and did not provide a clean held-out success gain.
- Democratic/soft ensemble success variants did not outperform the single LR story and should remain appendix material only.

## Validation Notes

- `experiments/teacher_student_distillation_v1.json` now uses success LR nested grid `[0.01, 0.05, 0.1, 0.5, 1.0]`.
- Nested run validation: `distillation_nested_sweep=true`; all selected success LR C values are in the new grid.
- Fixed run validation: `distillation_nested_sweep=false`; `success_lr_fixed_c=1.0`; all selected success LR C values are 1.0.
- Both reruns contain only `single_model` success rows and cover all three success branches under both HQ override conditions.
- Targeted tests passed with the default Anaconda Python: `python -m pytest tests/test_components.py` reported 72 passed. The requested `vela_TRL` interpreter currently lacks `pytest`, so `C:\Users\joelb\.conda\envs\vela_TRL\python.exe -m pytest tests/test_components.py` could not run until `pytest` is installed in that environment.
