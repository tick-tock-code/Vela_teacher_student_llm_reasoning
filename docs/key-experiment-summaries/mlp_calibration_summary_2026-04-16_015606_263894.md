# MLP Calibration Summary

- Run artifacts: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\ut\c_0a18caea\runs\smoke_feature_repository_pipeline\2026-04-16_015606_263894_mlp_calibration`
- Calibration type: training CV only, no held-out/test usage.
- Training form: one native multi-output MLP per fold/parameter combo.
- Outer CV: 3-fold stratified (random_state=42)
- Parallel target workers: 2

## Selected Defaults

- `v25_policies`: `hidden_layer_sizes=(8,)`, `alpha=0.1`

## Top Feature Sets At Selected Params

- `v25_policies`: `llm_engineering`, `hq_baseline`

## Calibration Table (feature_set x parameter combo)

| target_family | feature_set_id | hidden_layer_sizes | alpha | primary_metric | primary_mean | primary_std |
|---|---|---|---:|---|---:|---:|
| v25_policies | hq_baseline | (8,) | 0.1000 | r2 | -12.5787 | 7.1477 |
| v25_policies | llm_engineering | (8,) | 0.1000 | r2 | -1.2289 | 0.7012 |
