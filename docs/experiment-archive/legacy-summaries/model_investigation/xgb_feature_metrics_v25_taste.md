# XGB Calibration Summary

- Run artifacts: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-15_014658_924597_xgb_calibration`
- Calibration type: training CV only, no held-out/test usage.
- Early stopping: disabled (no early stopping rounds/eval-set callbacks used).
- n_estimators sweep: [40, 80, 120, 180, 240, 320]
- Outer CV: 5-fold stratified (random_state=42)
- Parallel target workers: 7

## Selected Defaults

- `v25_policies`: `n_estimators=320`
- `taste_policies`: `n_estimators=320`

## Top Feature Sets At Selected n_estimators

- `v25_policies`: `lambda_policies_plus_sentence_bundle`, `hq_plus_sentence_bundle`
- `taste_policies`: `hq_plus_sentence_bundle`, `lambda_policies_plus_sentence_bundle`

## Step 2 (Default Routine Run)

Run model testing on all selected feature sets with `use_latest_xgb_calibration=true` and nested CV off.

## Step 3 (Confirmatory Run)

Run nested CV only on top-2 feature sets per target family, then compare tuned-vs-fixed deltas and stability.

## Calibration Table (feature_set x n_estimators)

| target_family | feature_set_id | n_estimators | primary_metric | primary_mean | primary_std |
|---|---|---:|---|---:|---:|
| taste_policies | hq_plus_sentence_bundle | 40 | f0_5 | 0.7257 | 0.2171 |
| taste_policies | lambda_policies_plus_sentence_bundle | 40 | f0_5 | 0.7196 | 0.2109 |
| taste_policies | llm_engineering_plus_sentence_bundle | 40 | f0_5 | 0.6414 | 0.2089 |
| taste_policies | sentence_bundle | 40 | f0_5 | 0.5395 | 0.1958 |
| taste_policies | hq_plus_sentence_bundle | 80 | f0_5 | 0.7440 | 0.2032 |
| taste_policies | lambda_policies_plus_sentence_bundle | 80 | f0_5 | 0.7359 | 0.1994 |
| taste_policies | llm_engineering_plus_sentence_bundle | 80 | f0_5 | 0.6629 | 0.2022 |
| taste_policies | sentence_bundle | 80 | f0_5 | 0.5683 | 0.1963 |
| taste_policies | hq_plus_sentence_bundle | 120 | f0_5 | 0.7542 | 0.1967 |
| taste_policies | lambda_policies_plus_sentence_bundle | 120 | f0_5 | 0.7453 | 0.1926 |
| taste_policies | llm_engineering_plus_sentence_bundle | 120 | f0_5 | 0.6741 | 0.1967 |
| taste_policies | sentence_bundle | 120 | f0_5 | 0.5843 | 0.1947 |
| taste_policies | hq_plus_sentence_bundle | 180 | f0_5 | 0.7627 | 0.1908 |
| taste_policies | lambda_policies_plus_sentence_bundle | 180 | f0_5 | 0.7545 | 0.1864 |
| taste_policies | llm_engineering_plus_sentence_bundle | 180 | f0_5 | 0.6864 | 0.1901 |
| taste_policies | sentence_bundle | 180 | f0_5 | 0.6008 | 0.1993 |
| taste_policies | hq_plus_sentence_bundle | 240 | f0_5 | 0.7691 | 0.1853 |
| taste_policies | lambda_policies_plus_sentence_bundle | 240 | f0_5 | 0.7602 | 0.1819 |
| taste_policies | llm_engineering_plus_sentence_bundle | 240 | f0_5 | 0.6931 | 0.1869 |
| taste_policies | sentence_bundle | 240 | f0_5 | 0.6097 | 0.1981 |
| taste_policies | hq_plus_sentence_bundle | 320 | f0_5 | 0.7733 | 0.1841 |
| taste_policies | lambda_policies_plus_sentence_bundle | 320 | f0_5 | 0.7650 | 0.1793 |
| taste_policies | llm_engineering_plus_sentence_bundle | 320 | f0_5 | 0.7000 | 0.1826 |
| taste_policies | sentence_bundle | 320 | f0_5 | 0.6193 | 0.1953 |
| v25_policies | hq_plus_sentence_bundle | 40 | r2 | 0.2163 | 0.0667 |
| v25_policies | lambda_policies_plus_sentence_bundle | 40 | r2 | 0.2304 | 0.0772 |
| v25_policies | llm_engineering_plus_sentence_bundle | 40 | r2 | 0.1821 | 0.0542 |
| v25_policies | sentence_bundle | 40 | r2 | 0.1575 | 0.0484 |
| v25_policies | hq_plus_sentence_bundle | 80 | r2 | 0.2756 | 0.0689 |
| v25_policies | lambda_policies_plus_sentence_bundle | 80 | r2 | 0.2879 | 0.0783 |
| v25_policies | llm_engineering_plus_sentence_bundle | 80 | r2 | 0.2352 | 0.0556 |
| v25_policies | sentence_bundle | 80 | r2 | 0.2055 | 0.0561 |
| v25_policies | hq_plus_sentence_bundle | 120 | r2 | 0.3065 | 0.0690 |
| v25_policies | lambda_policies_plus_sentence_bundle | 120 | r2 | 0.3167 | 0.0775 |
| v25_policies | llm_engineering_plus_sentence_bundle | 120 | r2 | 0.2633 | 0.0566 |
| v25_policies | sentence_bundle | 120 | r2 | 0.2307 | 0.0599 |
| v25_policies | hq_plus_sentence_bundle | 180 | r2 | 0.3321 | 0.0700 |
| v25_policies | lambda_policies_plus_sentence_bundle | 180 | r2 | 0.3415 | 0.0768 |
| v25_policies | llm_engineering_plus_sentence_bundle | 180 | r2 | 0.2877 | 0.0586 |
| v25_policies | sentence_bundle | 180 | r2 | 0.2532 | 0.0638 |
| v25_policies | hq_plus_sentence_bundle | 240 | r2 | 0.3471 | 0.0708 |
| v25_policies | lambda_policies_plus_sentence_bundle | 240 | r2 | 0.3559 | 0.0767 |
| v25_policies | llm_engineering_plus_sentence_bundle | 240 | r2 | 0.3026 | 0.0606 |
| v25_policies | sentence_bundle | 240 | r2 | 0.2677 | 0.0662 |
| v25_policies | hq_plus_sentence_bundle | 320 | r2 | 0.3593 | 0.0724 |
| v25_policies | lambda_policies_plus_sentence_bundle | 320 | r2 | 0.3678 | 0.0768 |
| v25_policies | llm_engineering_plus_sentence_bundle | 320 | r2 | 0.3153 | 0.0628 |
| v25_policies | sentence_bundle | 320 | r2 | 0.2805 | 0.0682 |
