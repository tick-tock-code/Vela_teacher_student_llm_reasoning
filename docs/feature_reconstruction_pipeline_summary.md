# Feature Reconstruction Pipeline Summary

## Active Aim

The active distillation pipeline reconstructs policy targets from feature banks:
- `v25_policies` (continuous, regression) uses automatic `ridge` (linear L2) modeling
- `taste_policies` (binary, classification) uses automatic `logreg_classifier` (logistic L2) modeling
- `v25_and_taste` runs both families in one launch (as two child runs)

## Feature-Set Comparisons

Current configured feature-set matrix includes:
- `hq_baseline`
- `llm_engineering`
- `lambda_policies`
- `sentence_prose`
- `sentence_structured`
- `sentence_bundle`
- `hq_plus_sentence_prose`
- `hq_plus_sentence_structured`
- `hq_plus_sentence_bundle`
- `llm_engineering_plus_sentence_prose`
- `llm_engineering_plus_sentence_structured`
- `llm_engineering_plus_sentence_bundle`
- `lambda_policies_plus_sentence_prose`
- `lambda_policies_plus_sentence_structured`
- `lambda_policies_plus_sentence_bundle`
- `full_repository_plus_sentence_bundle`

## Where To Read R² / Metrics

Every distillation run now writes:
- `reasoning_metrics.csv` (full per-target metrics)
- `reasoning_metrics_summary.md` (human-readable feature-set summary)

For `v25_policies`, `reasoning_metrics_summary.md` includes mean R² by feature set/model on `oof_overall`.
For `taste_policies`, the same summary file reports F0.5 / ROC-AUC / PR-AUC / precision / recall.

If held-out evaluation is enabled and targets are available, the summary also includes held-out averages.

## Prediction Artifact Toggle

You can now disable writing prediction tables (`reasoning_oof_predictions.csv` and held-out prediction CSVs) to save disk space:
- GUI checkbox: `Save reasoning predictions to tmp`
- CLI flags: `--save-reasoning-predictions` / `--no-save-reasoning-predictions`
