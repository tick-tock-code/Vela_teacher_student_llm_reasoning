# Base Feature Selection Summary

Run source: `tmp/runs/teacher_student_distillation_v1/2026-04-14_044026_387723_model_testing`  
Run type: model testing, single-target models, `Linear L2` only, both `v25_policies` and `taste_policies`, training-CV screening only (no held-out usage).

## Winner Summary

The two strongest feature sets are:

1. `hq_plus_sentence_bundle`
2. `lambda_policies_plus_sentence_bundle`

They ranked top-2 for both target families and were both marked `recommended_take_forward=true`.

## Key Metrics

| Target family | Feature set | Rank | Primary metric | Primary value | Secondary metrics |
|---|---|---:|---|---:|---|
| v25_policies | hq_plus_sentence_bundle | 1 | R² | 0.4236 | RMSE 0.2647, MAE 0.1802 |
| v25_policies | lambda_policies_plus_sentence_bundle | 2 | R² | 0.4225 | RMSE 0.2644, MAE 0.1767 |
| taste_policies | hq_plus_sentence_bundle | 1 | F0.5 | 0.7888 | ROC-AUC 0.9400, PR-AUC 0.8071 |
| taste_policies | lambda_policies_plus_sentence_bundle | 2 | F0.5 | 0.7885 | ROC-AUC 0.9424, PR-AUC 0.8109 |

## Selection Decision

- Carry forward `hq_plus_sentence_bundle` and `lambda_policies_plus_sentence_bundle` as the base feature-set shortlist.
- Keep both for next-stage model-family testing because they are effectively tied at the top, with minor trade-offs by metric family.
