# MLP Calibration Summary

- Run artifacts: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-16_001949_134642_mlp_calibration`
- Calibration type: training CV only, no held-out/test usage.
- Training form: one native multi-output MLP per fold/parameter combo.
- Outer CV: 5-fold stratified (random_state=42)
- Parallel target workers: 7

## Selected Defaults

- `v25_policies`: `hidden_layer_sizes=(32,)`, `alpha=0.1`
- `taste_policies`: `hidden_layer_sizes=(32,)`, `alpha=0.1`

## Top Feature Sets At Selected Params

- `v25_policies`: `lambda_policies_plus_sentence_bundle`, `lambda_policies_plus_sentence_prose`
- `taste_policies`: `hq_plus_sentence_bundle`, `lambda_policies_plus_sentence_bundle`

## Calibration Table (feature_set x parameter combo)

| target_family | feature_set_id | hidden_layer_sizes | alpha | primary_metric | primary_mean | primary_std |
|---|---|---|---:|---|---:|---:|
| taste_policies | hq_plus_sentence_bundle | (32,) | 0.1000 | f0_5 | 0.7857 | 0.1698 |
| taste_policies | hq_plus_sentence_prose | (32,) | 0.1000 | f0_5 | 0.7754 | 0.1759 |
| taste_policies | lambda_policies_plus_sentence_bundle | (32,) | 0.1000 | f0_5 | 0.7839 | 0.1694 |
| taste_policies | lambda_policies_plus_sentence_prose | (32,) | 0.1000 | f0_5 | 0.7761 | 0.1727 |
| taste_policies | llm_engineering_plus_sentence_bundle | (32,) | 0.1000 | f0_5 | 0.7578 | 0.1747 |
| taste_policies | llm_engineering_plus_sentence_prose | (32,) | 0.1000 | f0_5 | 0.7478 | 0.1731 |
| v25_policies | hq_plus_sentence_bundle | (32,) | 0.1000 | r2 | 0.3520 | 0.0930 |
| v25_policies | hq_plus_sentence_prose | (32,) | 0.1000 | r2 | 0.3182 | 0.0947 |
| v25_policies | lambda_policies_plus_sentence_bundle | (32,) | 0.1000 | r2 | 0.4038 | 0.0805 |
| v25_policies | lambda_policies_plus_sentence_prose | (32,) | 0.1000 | r2 | 0.3839 | 0.0826 |
| v25_policies | llm_engineering_plus_sentence_bundle | (32,) | 0.1000 | r2 | 0.3577 | 0.0743 |
| v25_policies | llm_engineering_plus_sentence_prose | (32,) | 0.1000 | r2 | 0.3306 | 0.0714 |
