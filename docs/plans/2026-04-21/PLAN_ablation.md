## Recalculate Original Ablation (Train-Only Reasoning, 19-Set Matrix)

### Summary
Run a dedicated `model_testing_mode` ablation rerun on `v25_policies` using the **19-set extended feature matrix** and **Linear L2 only**, with standard stratified 3-fold CV, **no repeats**, **no nested tuning**, and **no held-out/test evaluation**.  
This is execution-only with current pipeline capabilities (no core logic changes required), plus a reproducible run profile and clear artifact checks.

### Implementation Changes
- Add an ablation run profile (CLI preset + optional GUI preset label) named `ablation_v25_19set_linear`.
- Profile must hard-lock:
  - `run_mode=model_testing_mode`
  - `target_family=v25_policies`
  - `model_families=[linear_l2]`
  - `output_modes=[single_target]`
  - `repeat_cv_with_new_seeds=false` (repeat count resolves to 1)
  - `distillation_nested_sweep=false`
  - `heldout_evaluation=false`
  - `save_model_configs_after_training=false`
  - `use_latest_xgb_calibration=false`, `use_latest_rf_calibration=false`, `use_latest_mlp_calibration=false`
- Candidate feature sets must be exactly these 19:
  1. `hq_baseline`
  2. `llm_engineering`
  3. `lambda_policies`
  4. `sentence_prose`
  5. `sentence_structured`
  6. `sentence_bundle`
  7. `hq_plus_sentence_prose`
  8. `hq_plus_sentence_structured`
  9. `hq_plus_sentence_bundle`
  10. `llm_engineering_plus_sentence_prose`
  11. `llm_engineering_plus_sentence_structured`
  12. `llm_engineering_plus_sentence_bundle`
  13. `lambda_policies_plus_sentence_prose`
  14. `lambda_policies_plus_sentence_structured`
  15. `lambda_policies_plus_sentence_bundle`
  16. `hq_plus_llm_engineering_plus_sentence_bundle`
  17. `hq_plus_lambda_policies_plus_sentence_bundle`
  18. `llm_engineering_plus_lambda_policies_plus_sentence_bundle`
  19. `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle`
- CLI-equivalent execution contract:
  - `python -m src.pipeline.run_distillation --config experiments/teacher_student_distillation_v1.json --run-mode model_testing_mode --target-family v25_policies --candidate-feature-sets <19 ids above> --model-families linear_l2 --output-modes single_target --no-repeat-cv-with-new-seeds --no-distillation-nested-sweep --no-heldout-evaluation --no-save-model-configs-after-training --no-use-latest-xgb-calibration --no-use-latest-rf-calibration --no-use-latest-mlp-calibration`
- Reporting/output expectations:
  - Use existing Stage A artifacts only: `feature_set_screening.csv`, `feature_set_screening_by_architecture.csv`, `feature_set_screening_repeat_summary.csv`, `feature_set_screening_repeat_metrics.csv`, `feature_set_screening_per_target.csv`, `feature_set_screening_report.md`, `run_summary.md`.
  - No saved-eval/test artifacts should be generated.

### Test Plan
- Preflight validation:
  - `resolved_run_options.json` shows `run_mode=model_testing_mode`, `target_family=v25_policies`, `model_families=["linear_l2"]`, `heldout_evaluation=false`, `distillation_nested_sweep=false`, repeat count `1`.
  - Candidate feature set count is exactly `19`.
- Runtime validation:
  - Stage A starts and completes for all 19 sets with no Stage B/saved-eval dispatch.
  - `feature_set_screening_report.md` includes train-only statement and per-target v25 section.
- Post-run consistency checks:
  - `feature_set_screening.csv` has one row per feature set for the active target/output/model combination.
  - `feature_set_screening_per_target.csv` has `19 x 16 = 304` per-target rows (v25 only).
  - No held-out metrics files are present.

### Assumptions and Defaults
- This rerun is strictly train-only reasoning reconstruction; no success-prediction or held-out evaluation path is allowed.
- CV protocol is fixed to the project’s distillation default (stratified 3-fold, one pass).
- Linear L2 here maps to `ridge` for `v25_policies`.
- Existing model-testing code path is reused directly; this is a controlled rerun profile, not a new pipeline branch.
