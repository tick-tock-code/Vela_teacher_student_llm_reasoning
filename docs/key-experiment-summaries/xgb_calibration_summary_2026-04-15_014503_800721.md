# XGB Calibration Summary

- Run artifacts: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-15_014503_800721_xgb_calibration`
- Calibration type: training CV only, no held-out/test usage.
- Early stopping: disabled (no early stopping rounds/eval-set callbacks used).
- n_estimators sweep: [40]
- Outer CV: 5-fold stratified (random_state=42)
- Parallel target workers: 7

## Selected Defaults

- `v25_policies`: `n_estimators=40`

## Top Feature Sets At Selected n_estimators

- `v25_policies`: `hq_plus_sentence_bundle`

## Step 2 (Default Routine Run)

Run model testing on all selected feature sets with `use_latest_xgb_calibration=true` and nested CV off.

## Step 3 (Confirmatory Run)

Run nested CV only on top-2 feature sets per target family, then compare tuned-vs-fixed deltas and stability.

## Calibration Table (feature_set x n_estimators)

| target_family | feature_set_id | n_estimators | primary_metric | primary_mean | primary_std |
|---|---|---:|---|---:|---:|
| v25_policies | hq_plus_sentence_bundle | 40 | r2 | 0.2163 | 0.0667 |
