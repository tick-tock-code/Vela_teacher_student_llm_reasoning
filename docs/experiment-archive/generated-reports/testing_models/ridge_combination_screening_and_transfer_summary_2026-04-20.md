# Ridge Combination Screening And Transfer Summary (2026-04-20)

## Stage A Combination Screening (v25, train-only CV)

- Run: `2026-04-20_224403_128673_model_testing`
- Runtime: `242.7s` (~`4m03s`)
- Protocol: stratified 3-fold CV, 4 repeats, no nested tuning, no held-out evaluation
- Quick best readout:
  - Single-target best: `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle` with `R²=0.4286` (`std=0.00094`)
  - Multi-output best: `lambda_policies_plus_sentence_bundle` with `R²=0.4181` (`std=0.00113`)

### Full Stage A Results Across 7 Combinations

#### Single-target (Linear L2 / Ridge)

| Rank | Feature set | R² mean | R² std | RMSE | MAE | Screen score |
|---|---|---:|---:|---:|---:|---:|
| 1 | `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle` | 0.4286 | 0.0009 | 0.2638 | 0.1758 | 0.4281 |
| 2 | `hq_plus_lambda_policies_plus_sentence_bundle` | 0.4284 | 0.0013 | 0.2636 | 0.1760 | 0.4277 |
| 3 | `llm_engineering_plus_lambda_policies_plus_sentence_bundle` | 0.4218 | 0.0020 | 0.2651 | 0.1770 | 0.4208 |
| 4 | `lambda_policies_plus_sentence_bundle` | 0.4211 | 0.0015 | 0.2651 | 0.1775 | 0.4203 |
| 5 | `hq_plus_llm_engineering_plus_sentence_bundle` | 0.4204 | 0.0009 | 0.2658 | 0.1808 | 0.4199 |
| 6 | `hq_plus_sentence_bundle` | 0.4187 | 0.0012 | 0.2659 | 0.1815 | 0.4181 |
| 7 | `llm_engineering_plus_sentence_bundle` | 0.3894 | 0.0007 | 0.2737 | 0.1892 | 0.3891 |

#### Multi-output (MLP)

| Rank | Feature set | R² mean | R² std | RMSE | MAE | Screen score |
|---|---|---:|---:|---:|---:|---:|
| 1 | `lambda_policies_plus_sentence_bundle` | 0.4181 | 0.0011 | 0.2658 | 0.1721 | 0.4175 |
| 2 | `llm_engineering_plus_lambda_policies_plus_sentence_bundle` | 0.4138 | 0.0046 | 0.2667 | 0.1737 | 0.4115 |
| 3 | `llm_engineering_plus_sentence_bundle` | 0.3939 | 0.0015 | 0.2727 | 0.1842 | 0.3932 |
| 4 | `hq_plus_lambda_policies_plus_sentence_bundle` | 0.3510 | 0.0026 | 0.2792 | 0.1869 | 0.3497 |
| 5 | `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle` | 0.3524 | 0.0086 | 0.2789 | 0.1862 | 0.3481 |
| 6 | `hq_plus_llm_engineering_plus_sentence_bundle` | 0.3504 | 0.0075 | 0.2798 | 0.1899 | 0.3467 |
| 7 | `hq_plus_sentence_bundle` | 0.3452 | 0.0041 | 0.2810 | 0.1918 | 0.3431 |

## Train-Only Success CV (3 repeats)

Using predicted reasoning from the ridge reasoning model
`v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0001`.

Protocol:
- 3-fold CV
- repeated over seeds `42`, `10042`, `20042`
- train-only (optimistic by construction)

| Base combo | Train CV F0.5 mean | Train CV F0.5 std |
|---|---:|---:|
| `llm_engineering` | 0.3244 | 0.0084 |
| `llm_engineering_plus_lambda_policies` | 0.3098 | 0.0164 |
| `hq_plus_llm_engineering` | 0.3049 | 0.0064 |
| `hq_plus_llm_engineering_plus_lambda_policies` | 0.3006 | 0.0087 |
| `hq_baseline` | 0.2988 | 0.0038 |
| `hq_plus_lambda_policies` | 0.2944 | 0.0086 |
| `lambda_policies` | 0.2866 | 0.0139 |

## Held-out Test Success (Nested L2 Regularisation)

Pred reasoning source for all branches below:
- `data/saved_model_configs/2026-04-20_230125_954084_model_testing::v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0001`

All held-out numbers below use the nested L2 protocol (outer 5-fold, inner 3-fold, threshold selected on OOF train scores), not the fixed-`C=5` shortcut.

| Branch | F0.5 | ROC-AUC | PR-AUC | Precision | Recall | Threshold | Selected C (final) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hq_plus_llm_engineering + pred_reasoning` (HQ override ON) | 0.337187 | 0.736125 | 0.241784 | 0.364583 | 0.259259 | 0.300 | 0.050 |
| `llm_engineering + pred_reasoning` | 0.313167 | 0.732708 | 0.262207 | 0.352000 | 0.217284 | 0.270 | 0.200 |
| `llm_engineering_plus_lambda_policies + pred_reasoning` | 0.327553 | 0.732605 | 0.267552 | 0.354167 | 0.251852 | 0.230 | 0.010 |

## Train-Only Success CV (3-fold, HQ Override Forced ON For All 7 Combos)

Protocol:
- train-only, 3-fold CV
- fixed L2 (`C=5.0`), no nested tuning
- HQ override forced ON for every base-combo branch

| Base combo | Train CV F0.5 |
|---|---:|
| `llm_engineering_plus_lambda_policies` | 0.3200 |
| `llm_engineering` | 0.3146 |
| `hq_plus_llm_engineering` | 0.3139 |
| `hq_plus_llm_engineering_plus_lambda_policies` | 0.3104 |
| `hq_baseline` | 0.3019 |
| `hq_plus_lambda_policies` | 0.2970 |
| `lambda_policies` | 0.2920 |


## Held-out Test Success (Nested L2, HQ Override Forced ON For All Three Branches)

This section applies the HQ override rule (`exit_count > 0 => probability=1`) to all three branches for a like-for-like sensitivity check.

| Branch | F0.5 | ROC-AUC | PR-AUC | Precision | Recall | Threshold | Selected C (final) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hq_plus_llm_engineering + pred_reasoning` (HQ override ON) | 0.337187 | 0.736125 | 0.241784 | 0.364583 | 0.259259 | 0.300 | 0.050 |
| `llm_engineering + pred_reasoning` (forced HQ override) | 0.324985 | 0.736185 | 0.237745 | 0.342767 | 0.269136 | 0.280 | 0.200 |
| `llm_engineering_plus_lambda_policies + pred_reasoning` (forced HQ override) | 0.329908 | 0.734250 | 0.239726 | 0.337950 | 0.301235 | 0.230 | 0.010 |

### Notes

- The strongest held-out branch among the three tested here is `hq_plus_llm_engineering + pred_reasoning` (`F0.5=0.337187`).
- With HQ override forced ON for non-HQ branches, both `llm_engineering + pred_reasoning` and `llm_engineering_plus_lambda_policies + pred_reasoning` increase their held-out `F0.5`.
