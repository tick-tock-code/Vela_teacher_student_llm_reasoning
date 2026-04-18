# Random Forest Calibration Summary

- Run artifacts: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-16_051953_929644_rf_calibration`
- Calibration type: training CV only, no held-out/test usage.
- Fixed params: n_estimators=500, bootstrap=True
- Outer CV: 5-fold stratified (random_state=42)
- Parallel target workers: 2

## Selected Defaults

- `v25_policies`: `min_samples_leaf=2`, `max_depth=None`, `max_features=sqrt`

## Top Feature Sets At Selected Params

- `v25_policies`: `hq_plus_sentence_prose`

## Calibration Table (feature_set x parameter combo)

| target_family | feature_set_id | min_samples_leaf | max_depth | max_features | primary_metric | primary_mean | primary_std |
|---|---|---:|---|---|---|---:|---:|
| v25_policies | hq_plus_sentence_prose | 2 | None | sqrt | r2 | 0.3298 | 0.0612 |
