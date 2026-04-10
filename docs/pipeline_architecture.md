# Pipeline Architecture

## Core Modules

- `src/utils/paths.py`: canonical repo paths and cross-repo references
- `src/utils/artifact_io.py`: JSON, CSV, Markdown, and run-directory helpers
- `src/utils/dependencies.py`: optional dependency guards for `xgboost`, `openai`, and `think_reason_learn`
- `src/utils/placeholders.py`: explicit failure helpers for unfinished custom methods

## Data Layer

- `src/data/loading.py`: table readers, identifier normalization, and numeric feature helpers
- `src/data/input_features.py`: input-feature builders for raw-founder baselines and future external banks
- `src/data/raw_datasets.py`: public/private VCBench loading
- `src/data/feature_bank.py`: reusable external feature-bank loader for future table-based inputs
- `src/data/targets.py`: reasoning-target bank loading, scaling validation, and held-out overlap tracking
- `src/data/splits.py`: deterministic 3-fold public CV splits

## Modeling Layer

- `src/student/models.py`: Ridge, LR, XGB-style model builders
- `src/student/reasoning_regression.py`: per-target student training, OOF predictions, and full-train refits
- `src/downstream/routes.py`: public CV comparison routes and promoted private prediction routes
- `src/evaluation/metrics.py`: regression agreement, founder-success metrics, and threshold selection

## Teacher And LLM Engineering

- `src/llm_engineering/cache.py`: old-style cache, rule JSON, and family/set archive compatibility
- `src/llm_engineering/adapter.py`: guarded TRL-backed engineered-rule generation adapter
- `src/llm_engineering/custom_prompts.py`: placeholder hooks for custom prompt bundles and post-processing

## Orchestration

- `src/pipeline/config.py`: experiment config parsing and validation
- `src/pipeline/distillation.py`: end-to-end run orchestration
- `src/pipeline/run_distillation.py`: CLI entrypoint

## Repo Boundary Rule

- `evaluation/` scores things
- `downstream/` fits founder-success models
- `student/` fits reasoning-prediction models

That boundary is intentional and should be preserved as the repo grows.

## Data Flow

1. Load raw public and private VCBench rows.
2. Build the configured input feature bank from those rows.
3. Load the public and held-out reasoning target bank.
4. Resolve the explicitly configured reasoning target list.
5. Restrict both public and held-out target banks to those selected target columns.
6. Join public raw rows, input features, and reasoning targets on `founder_uuid`.
7. Build 3-fold public CV splits using the public `success` label.
8. Train one student regressor per reasoning target and model spec.
9. Assemble OOF predicted reasoning features.
10. Evaluate downstream founder-success routes on the public set using the selected target list.
11. Stop unless the config's manual promotion flag is approved.
12. If approved, refit reasoning students on the full public set and predict the selected target columns on the private set.
13. Score held-out agreement and fit downstream founder-success models using that same selected target list.

## Config Contract

The experiment config must declare:

- public and private raw dataset paths
- one input feature builder spec
- one reasoning target bank spec
- an explicit per-target selection list
- reasoning student models
- downstream founder-success models
- CV settings
- a manual promotion gate

## Placeholder Hooks

The following hooks intentionally fail until you replace them with real project-specific logic:

- `load_custom_rule_prompt_bundle`
- `render_custom_rule_prompt`
- `postprocess_generated_rules`
- `generate_custom_engineered_rule_family`

These are placeholders by design because the project will later use custom prompt assets that do not exist yet.
