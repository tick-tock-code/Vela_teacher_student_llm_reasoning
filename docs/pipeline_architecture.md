# Pipeline Architecture

## Active Modules

- `src/data/loading.py`: table readers and identifier helpers
- `src/data/raw_datasets.py`: public/private VCBench loading
- `src/data/targets.py`: reasoning-target bank loading and selected-target enforcement
- `src/data/splits.py`: deterministic CV split builders
- `src/intermediary_features/mirror.py`: deterministic numeric mirror features from raw VCBench rows
- `src/intermediary_features/structured_text.py`: deterministic structured-founder text rendering
- `src/intermediary_features/sentence_transformer.py`: embedding builders for rendered text banks
- `src/intermediary_features/storage.py`: cached Parquet and manifest storage for reusable banks
- `src/intermediary_features/registry.py`: auto-build, reload, and feature-set assembly
- `src/student/models.py`: reasoning regressor builders
- `src/student/reasoning_regression.py`: per-target OOF and full-train reasoning regressors
- `src/pipeline/config.py`: experiment schema
- `src/pipeline/run_options.py`: shared CLI/GUI override contract
- `src/pipeline/distillation.py`: reasoning-only orchestration
- `src/pipeline/run_distillation.py`: CLI entrypoint
- `src/gui/run_launcher.py`: Tkinter launcher

## Data Flow

1. Load public and held-out VCBench founder rows.
2. Load the reasoning target bank and restrict it to the explicitly selected targets.
3. Build or reuse the selected intermediary feature banks.
4. Assemble configured feature sets from those banks.
5. Build deterministic stratified public CV splits from quantile buckets of the row-wise mean selected target score.
6. For each feature set:
   - train one regressor per selected policy target and reasoning model
   - write OOF predictions
   - score public agreement
   - if requested, refit on the full public set
   - if requested, predict the held-out set
   - if requested, score held-out agreement

## Config Contract

The active experiment config declares:

- raw public and held-out VCBench paths
- reasoning target-bank paths
- explicit selected policy targets
- reasoning model registry
- intermediary feature registry
- feature-set comparison matrix
- CV settings

## Intermediary Feature Storage

Reusable feature banks live under `data/intermediary_features/`.

Current canonical layout:

- `data/intermediary_features/mirror/v1/`
- `data/intermediary_features/sentence_transformer/prose/all-MiniLM-L6-v2/`
- `data/intermediary_features/sentence_transformer/structured/all-MiniLM-L6-v2/`

Each bank stores:

- `public.parquet`
- `private.parquet`
- `manifest.json`

Sentence-transformer banks also store rendered text CSVs for reproducibility.

## Inactive Paths

- `src/downstream/` remains dormant future code
- `run_success_predictions` is visible in the CLI and GUI only as an inactive option
- `llm_engineered_v1` remains scaffolded until custom prompt assets exist
