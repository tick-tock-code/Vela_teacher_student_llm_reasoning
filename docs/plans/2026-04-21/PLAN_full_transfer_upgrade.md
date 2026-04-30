### Full-Transfer Robustness Upgrade (Default 16 CV Repeats + Train-CV/Test Reporting)

### Summary
Implement a full-transfer-focused robustness upgrade so the pipeline defaults to `n_runs=16` for CV in:
1. Reasoning model training (`model_testing_mode` used to produce saved bundles), and  
2. Success model training inside `saved_eval_mode=full_transfer_report`.

Also upgrade the full-transfer success section to explicitly show `Avg Train CV F0.5 +/- std | Test F0.5` so held-out results are accompanied by CV stability evidence.

### Implementation Changes
- **Repeat defaulting for full-transfer flow**
  - Update run-option resolution so repeat behavior can distinguish “unset” from explicit false.
  - Set default repeat policy to:
    - `model_testing_mode`: `repeat_cv_with_new_seeds=true`, `cv_seed_repeat_count=16` by default.
    - `saved_config_evaluation_mode` + `saved_eval_mode=full_transfer_report`: same default (`true`, `16`).
    - Other modes remain unchanged unless explicitly overridden.
  - Keep explicit overrides authoritative (`--no-repeat-cv-with-new-seeds`, explicit repeat count, GUI explicit values).

- **CLI/override interface adjustments**
  - Change repeat flag parsing to tri-state (`None/True/False`) so mode-specific defaults can be applied safely.
  - Preserve backward compatibility for existing invocations that already pass repeat flags/counts.

- **Full-transfer success evaluation metrics**
  - Extend `_evaluate_success_transfer(...)` to support repeated CV runs (seed offsets per repeat) for train CV robustness statistics.
  - Persist per-row metrics including:
    - `train_cv_f0_5_mean`, `train_cv_f0_5_std`
    - existing held-out test metrics (`f0_5`, `roc_auc`, `pr_auc`, `precision`, `recall`, `threshold`)
    - repeat metadata (`repeat_cv_with_new_seeds`, `cv_seed_repeat_count`)
  - For your selected report format, keep **single** `Test F0.5` per row and add train CV mean/std alongside it.

- **Full-transfer report rendering**
  - In the “Held-out Test Performance (Success Transfer)” tables, replace/augment columns to show:
    - `Avg Train CV F0.5 +/- std`
    - `Test F0.5`
    - existing test quality columns as currently shown
  - Add a short run metadata line in the report header indicating success CV repeats used (`n_runs=16` by default).

- **Config default**
  - Set `model_testing.screening_repeat_cv_count` default to `16` (config + loader fallback) so reasoning-side default repeat count aligns with the new policy.

### Test Plan
- **Run-option behavior**
  - Verify default resolution:
    - `model_testing_mode` defaults to repeat enabled and `16`.
    - `saved_eval_mode=full_transfer_report` defaults to repeat enabled and `16`.
    - non-target modes keep prior defaults.
  - Verify explicit overrides:
    - `--no-repeat-cv-with-new-seeds` forces single run.
    - explicit `--cv-seed-repeat-count` respected when repeat enabled.

- **Full-transfer output correctness**
  - Run one full-transfer evaluation and confirm:
    - `success_transfer_metrics.csv` includes train CV mean/std + repeat metadata + test metrics.
    - Markdown success tables include `Avg Train CV F0.5 +/- std | Test F0.5`.

- **Regression checks**
  - Existing report sections (source CV reasoning metrics, held-out reasoning agreement, reproduction consistency) remain unchanged in meaning and ordering.
  - No breakage in combination-report paths from the new full-transfer changes.

### Assumptions
- “Full-transfer flow” scope means: `model_testing_mode` (reasoning training CV) + `saved_eval_mode=full_transfer_report` (success training CV), not every CV-enabled mode.
- In held-out success reporting, `Test F0.5` remains a single value per row (not averaged/std across repeats), while train CV robustness is shown as mean±std.
