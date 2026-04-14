# Pipeline Architecture

## Overview

The repo now has one shared configuration surface and two execution tracks:

- `reproduction_mode` for reproducing the benchmark success-prediction matrix from [Feature Repository](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository)
- `reasoning_distillation_mode` for reconstructing LLM-derived policy targets from feature banks and new intermediary features

Both tracks use the same config file and CLI/GUI override layer, but they intentionally diverge in targets, CV design, and evaluation outputs.

## Active Modules

- `src/data/loading.py`: table readers and identifier helpers
- `src/data/raw_datasets.py`: public/private VCBench loading for generated intermediary banks
- `src/data/feature_repository.py`: canonical split loading plus repository-backed feature-bank loaders
- `src/data/targets.py`: target-family extraction for `v25_policies` and `taste_policies`
- `src/data/splits.py`: split builders for benchmark reproduction and distillation
- `src/intermediary_features/mirror.py`: deterministic numeric raw-VCBench features
- `src/intermediary_features/structured_text.py`: deterministic structured founder rendering
- `src/intermediary_features/sentence_transformer.py`: text embedding builders
- `src/intermediary_features/storage.py`: cached bank storage and manifests
- `src/intermediary_features/registry.py`: build, load, and feature-set assembly across repository and intermediary banks
- `src/student/models.py`: reproduction and distillation model builders
- `src/student/reasoning_regression.py`: per-target regression training loops
- `src/student/reasoning_classification.py`: per-target classification training loops for Taste
- `src/pipeline/config.py`: experiment schema
- `src/pipeline/run_options.py`: CLI and GUI override resolution
- `src/pipeline/reproduction.py`: benchmark reproduction runner
- `src/pipeline/distillation.py`: mode dispatch and reasoning-distillation runner
- `src/pipeline/run_distillation.py`: CLI entrypoint
- `src/gui/run_launcher.py`: Tkinter launcher

## Source-Of-Truth Boundaries

Feature Repository owned inputs:

- split order and labels
- HQ baseline features
- LLM-engineering features
- lambda policy features
- policy teacher targets

`data/` owned inputs:

- raw VCBench CSVs
- generated intermediary banks such as sentence-transformer embeddings

Guardrail:

- `policy_v25` may be used in reproduction mode for the success benchmark, but it is blocked as a student input bank when the target family is `v25_policies`

## Reproduction Mode Data Flow

1. Load the canonical train/test founder ordering and success labels from `Feature Repository/splits/`.
2. Load the enabled repository feature banks.
3. Recompute lambda rankings from training data only where the benchmark matrix requires top-K lambda subsets.
4. Run the benchmark experiment matrix using the documented models:
   - nested L2 logistic regression where specified
   - Joel or autoresearch XGBoost where specified
5. Apply the HQ `exit_count > 0` override in the relevant experiments.
6. Select thresholds from pooled training OOF predictions.
7. Refit on the full training pool and score the held-out test set.
8. Write benchmark artifacts:
   - `reproduction_results.csv`
   - `reproduction_oof_predictions.csv`
   - `reproduction_test_predictions.csv`
   - `lambda_rankings.json`
   - `run_summary.md`

## Reasoning Distillation Data Flow

1. Load raw public and held-out VCBench rows.
2. Load the canonical Feature Repository splits.
3. Load the selected target family:
   - `v25_policies` for continuous regression targets
   - `taste_policies` for binary classification targets
4. Load the selected repository feature banks.
5. Build or reuse the selected intermediary banks from raw VCBench.
6. Assemble the configured feature-set comparisons from the chosen banks.
7. Align training rows to the selected target family.
8. Build stratified 3-fold CV splits from the target family values.
9. Train one model per target, per feature set, and per selected learner.
10. Write OOF predictions and public metrics.
11. If `heldout_evaluation=true`, refit on the full public set and score the held-out set.

Artifacts:

- `target_family_manifest.json`
- `feature_bank_manifests.json`
- `feature_set_manifest.json`
- `reasoning_oof_predictions.csv`
- `reasoning_metrics.csv`
- optionally `reasoning_classification_thresholds.json`
- optionally `reasoning_heldout_predictions.csv`
- optionally `reasoning_heldout_metrics.csv`
- `run_summary.md`

## Config Contract

The config in [teacher_student_distillation_v1.json](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/experiments/teacher_student_distillation_v1.json) now declares:

- raw VCBench dataset paths
- Feature Repository root and split files
- default run mode and default target family
- repository feature-bank specs
- intermediary bank specs
- distillation feature-set comparison matrix
- target-family specs
- distillation model registry
- reproduction CV, threshold, ranking, and experiment matrix settings
- distillation CV settings

## Training-Pool Behavior

The LLM-engineering feature bank has a 100-row seed exclusion in the public train split. Those rows remain in canonical founder order with NaNs at load time. When a feature set includes `llm_engineering`, assembly drops public rows with missing features so the train pool becomes the valid 4,400-row subset. Held-out rows must remain complete.

This matters because:

- reproduction mode needs the exact 4,400-vs-4,500 split behavior from the benchmark bundle
- reasoning-distillation runs using `llm_engineering` as input should inherit the same valid training subset rather than silently filling those rows
