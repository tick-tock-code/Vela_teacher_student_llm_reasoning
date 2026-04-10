# Implementation Plan

## Stage 1

Lock the active repo contract to reasoning reconstruction only:

- raw VCBench inputs in `data/VCBench_data/`
- teacher reasoning targets in `data/reasoning_feature_targets/`
- reusable intermediary feature banks in `data/intermediary_features/`
- no active downstream founder-success path

## Stage 2

Implement reusable intermediary feature generation:

- `vcbench_mirror_baseline_v1`
- `sentence_transformer_prose_v1`
- `sentence_transformer_structured_v1`
- storage manifests and cached public/private Parquet outputs

## Stage 3

Implement the reasoning-only training loop:

- explicit target list from config
- 3-fold public CV
- one regressor per selected policy target
- feature-set comparison runs
- held-out prediction and agreement scoring

## Stage 4

Add launch surfaces:

- CLI overrides for feature families, targets, models, rebuild behavior, and embedding model
- Tkinter launcher using the same override contract
- disabled future controls for founder-success and LLM-engineered features

## Expected Artifacts

Each run under `tmp/runs/<experiment_id>/...` should write:

- `resolved_config.json`
- `resolved_run_options.json`
- `reasoning_target_manifest.json`
- `intermediary_feature_manifests.json`
- `feature_set_manifest.json`
- `reasoning_oof_predictions.csv`
- `reasoning_metrics.csv`
- `reasoning_heldout_predictions.csv`
- `reasoning_heldout_metrics.csv`
- `run_summary.md`

## Important Constraints

- the active target list is the 10 policy columns shared by the held-out target bank
- train target banks may contain extra policy columns, but the default path does not train them
- intermediary feature banks must be reusable and reproducible
- sentence embeddings should be cached as committed reusable artifacts, not rebuilt ad hoc for every run
- any prompt-driven LLM-engineered path must fail explicitly until custom prompts exist
