# Model Types For Reasoning Reconstruction

## Recommended Order
1. Linear L2 (`ridge` for `v25_*`, `logreg_classifier` for `taste_*`)
2. XGB1 (`xgb1_regressor` / `xgb1_classifier`)
3. ElasticNet (`elasticnet_regressor` / `elasticnet_logreg_classifier`)
4. RandomForest (`randomforest_regressor` / `randomforest_classifier`)
5. MLP (`mlp_regressor` / `mlp_classifier`)

## Why / When To Use Each
- Linear L2: strongest baseline for stability, fast iteration, easy diagnostics, low overfit risk.
- XGB1: stronger non-linearity with compact trees; usually best early uplift after linear.
- ElasticNet: useful when many correlated features exist and sparse-ish weighting helps.
- RandomForest: non-linear interactions with less tuning friction; strong fallback for tabular heterogeneity.
- MLP: flexible function approximator; most useful when signal is distributed across many dimensions.

## Risks And Expected Wins
- Linear L2:
  - Risk: underfits interaction-heavy relationships.
  - Expected win: stable, interpretable baseline and robust screening signal.
- XGB1:
  - Risk: can overfit if depth/learning-rate/estimators drift upward.
  - Expected win: better fit on non-linear feature-target mappings.
- ElasticNet:
  - Risk: sensitive to alpha/l1-ratio scaling and feature standardization quality.
  - Expected win: improved generalization when redundant features are common.
- RandomForest:
  - Risk: larger compute/memory footprint; less calibrated probabilities by default.
  - Expected win: captures interaction structure with low manual feature engineering.
- MLP:
  - Risk: highest variance across seeds; easiest to overfit, longest runtime.
  - Expected win: potential gains where target signal is smooth but high-dimensional.

## Practical Guidance
- Use repeated CV and stability metrics (`mean`, `std`, `screen_score`) before promoting any feature set/model pair.
- Keep nested CV on when sweeping hyperparameters; otherwise prefer fixed defaults for quick screening throughput.
- Promote feature sets first, then tune heavier models on shortlisted sets only.
