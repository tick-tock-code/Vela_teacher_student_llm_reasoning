# MLP Calibration Summary

- Run artifacts: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\ut\c_9b9c65d3\runs\smoke_feature_repository_pipeline\2026-04-15_170516_567561_mlp_calibration`
- Calibration type: training CV only, no held-out/test usage.
- Training form: one native multi-output MLP per fold/parameter combo.
- Outer CV: 3-fold stratified (random_state=42)
- Parallel target workers: 1

## Selected Defaults

- `v25_policies`: `hidden_layer_sizes=(8,)`, `alpha=0.1`

## Top Feature Sets At Selected Params

- `v25_policies`: `llm_engineering`, `hq_baseline`

## Calibration Table (feature_set x parameter combo)

| target_family | feature_set_id | hidden_layer_sizes | alpha | primary_metric | primary_mean | primary_std |
|---|---|---|---:|---|---:|---:|
| v25_policies | hq_baseline | (8,) | 0.1000 | r2 | -13.8295 | 7.8341 |
| v25_policies | llm_engineering | (8,) | 0.1000 | r2 | -4.1230 | 2.6076 |
