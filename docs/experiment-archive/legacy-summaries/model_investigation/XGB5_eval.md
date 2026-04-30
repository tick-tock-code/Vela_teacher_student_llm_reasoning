# XGB depth=5 Evaluation (Reduced 4-Combo Run)

- Run dir: `tmp/runs/teacher_student_distillation_v1/2026-04-18_222820_701928_model_testing`
- Scope: depth=5, XGB only, bundle only (`hq_plus_sentence_bundle`, `lambda_policies_plus_sentence_bundle`), regression+classification, repeats=1.
- Aggregation: `split_id = oof_overall` per target; `oof_mean` is average across targets and `oof_std` is std across targets.

| family | metric | feature_set_id | n_targets | oof_mean | oof_std | min | max |
|---|---:|---|---:|---:|---:|---:|---:|
| v25_policies | R2 | lambda_policies_plus_sentence_bundle | 16 | 0.4201 | 0.0814 | 0.2856 | 0.5351 |
| v25_policies | R2 | hq_plus_sentence_bundle | 16 | 0.4134 | 0.0801 | 0.2803 | 0.5045 |
| taste_policies | F0.5 | hq_plus_sentence_bundle | 20 | 0.7887 | 0.1845 | 0.3303 | 1.0000 |
| taste_policies | F0.5 | lambda_policies_plus_sentence_bundle | 20 | 0.7812 | 0.1784 | 0.3544 | 1.0000 |
