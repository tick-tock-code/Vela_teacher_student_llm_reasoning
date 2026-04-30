# Ensemble And Democratic Appendix

This appendix keeps the democratic/soft-ensemble work as a negative or secondary result. It should not be used in headline tables for the paper-facing combination report.

## Soft Ensemble Full Transfer Summary

Run date: 2026-04-30

Source run: `tmp/runs/teacher_student_distillation_v1/2026-04-30_164728_791524_saved_config_evaluation`

Saved model bundle: `data/saved_model_configs/2026-04-21_132505_959940_model_testing`

Evaluation setup:

- Saved eval mode: `full_transfer_report`
- Reasoning combo default: ridge / `lambda_policies_plus_sentence_bundle` / single-target regression
- Success CV repeats: 16
- HQ override mode: `both_force_off_and_on_all_branches`
- Success variants: `single_model`, `soft_avg_model`, `soft_avg_weighted_model`
- Branches: `pred_reasoning_only`, `hq_plus_pred_reasoning`, `llm_engineering_plus_pred_reasoning`

## Main Takeaway

The soft ensemble variants improve train-side F0.5, but most of that lift does not generalise to the held-out test set. This is clearest with HQ override off, where the soft ensembles show larger train-test gaps than the single fitted model.

The weighted soft ensemble is still worth keeping as an experimental row: it was the best mean test performer overall and gave the best held-out result for `hq_plus_pred_reasoning` with HQ override both on and off. However, the current train F0.5 for soft ensembles should be interpreted as threshold-selection/train-fit performance, not a clean generalisation estimate.

## Mean Performance By Variant

| model_variant | mean train F0.5 | mean test F0.5 | test - train gap |
|---|---:|---:|---:|
| `soft_avg_weighted_model` | 0.3434 | 0.3099 | -0.0335 |
| `single_model` | 0.3242 | 0.3069 | -0.0173 |
| `soft_avg_model` | 0.3447 | 0.3060 | -0.0387 |

## HQ Override Off

| model_variant | mean train F0.5 | mean test F0.5 | test - train gap |
|---|---:|---:|---:|
| `soft_avg_weighted_model` | 0.3591 | 0.3004 | -0.0587 |
| `single_model` | 0.3346 | 0.2952 | -0.0394 |
| `soft_avg_model` | 0.3602 | 0.2949 | -0.0653 |

Without HQ override, the ensembles fit the training distribution more strongly than the single model. The weighted ensemble has the best mean test F0.5, but it also has a substantially larger train-test gap than `single_model`.

## HQ Override On

| model_variant | mean train F0.5 | mean test F0.5 | test - train gap |
|---|---:|---:|---:|
| `soft_avg_weighted_model` | 0.3278 | 0.3194 | -0.0084 |
| `single_model` | 0.3138 | 0.3186 | +0.0048 |
| `soft_avg_model` | 0.3291 | 0.3171 | -0.0120 |

With HQ override on, the train-test gap is much smaller. This suggests the override is stabilising the success signal and reducing the extent to which threshold tuning overfits to train-side probability structure.

## Best Row Per Branch And Override

| branch | HQ override | best model_variant | train F0.5 | train std | test F0.5 | test - train gap | ROC AUC | PR AUC | precision | recall | threshold |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hq_plus_pred_reasoning` | on | `soft_avg_weighted_model` | 0.3230 | 0.0035 | 0.3264 | +0.0034 | 0.7281 | 0.2371 | 0.3502 | 0.2568 | 0.2600 |
| `hq_plus_pred_reasoning` | off | `soft_avg_weighted_model` | 0.3584 | 0.0050 | 0.3122 | -0.0462 | 0.7247 | 0.2633 | 0.3744 | 0.1877 | 0.2600 |
| `llm_engineering_plus_pred_reasoning` | on | `single_model` | 0.3203 | 0.0054 | 0.3260 | +0.0057 | 0.7378 | 0.2410 | 0.3463 | 0.2642 | 0.2800 |
| `llm_engineering_plus_pred_reasoning` | off | `single_model` | 0.3430 | 0.0074 | 0.3100 | -0.0329 | 0.7343 | 0.2630 | 0.3838 | 0.1753 | 0.3000 |
| `pred_reasoning_only` | on | `single_model` | 0.3149 | 0.0050 | 0.3086 | -0.0063 | 0.7278 | 0.2320 | 0.3259 | 0.2543 | 0.2500 |
| `pred_reasoning_only` | off | `single_model` | 0.3340 | 0.0073 | 0.2832 | -0.0509 | 0.7210 | 0.2452 | 0.3303 | 0.1802 | 0.2500 |

## Interpretation

The current soft ensemble train sweep is optimistic because each train founder is scored by an ensemble containing many models that were trained with that founder in their training fold. That makes the threshold sweep different from a pure out-of-fold CV estimate.

This explains the pattern: soft ensembles increase train F0.5 by smoothing and strengthening the fitted probability signal, but the held-out test lift is small or inconsistent. The effect is most visible when HQ override is off:

- `pred_reasoning_only`: soft average train F0.5 rises to 0.3476, but test F0.5 is 0.2823, slightly below the single model.
- `llm_engineering_plus_pred_reasoning`: soft average train F0.5 rises to 0.3720, but test F0.5 is 0.3056, below the single model.
- `hq_plus_pred_reasoning`: weighted soft average is useful, improving test F0.5 from 0.2923 to 0.3122, but still with a large train-test gap.

## Recommended Next Step

Keep the three output rows for now, but relabel the soft ensemble train metric in future reports as threshold-selection/train-fit F0.5 unless the implementation is changed to use strict out-of-fold ensemble scores for threshold tuning.

A stricter version would tune the soft ensemble threshold using only voters that did not train on each founder. That would make the reported train F0.5 more comparable to the `single_model` CV estimate and give a cleaner read on whether soft averaging genuinely improves generalisation.

## Artifacts

- `full_transfer_report.md`
- `success_transfer_metrics.csv`
- `success_transfer_train_threshold_sweep.csv`
