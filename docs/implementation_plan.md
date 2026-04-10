# Implementation Plan

## Stage 1

Build the reusable core:

- repo-relative path helpers
- artifact IO
- dependency guards
- tabular loading and alignment
- input-feature builders
- reasoning target-bank loading

## Stage 2

Build the public-set modeling workflow:

- deterministic 3-fold public CV
- per-target reasoning regression
- OOF prediction assembly
- reasoning agreement metrics
- downstream founder-success comparisons for:
  - baseline input features only
  - baseline + true reasoning
  - baseline + student-predicted reasoning

## Stage 3

Build the gated prediction workflow:

- manual promotion gate in config
- full-train refit for reasoning regressors
- private-set reasoning prediction
- optional private reasoning agreement if private labels are supplied later
- full-train downstream founder-success prediction for the private set

## Stage 4

Import the LLM-engineering layer:

- cache and archive readers
- old rule JSON compatibility
- guarded TRL-based rule generation adapter
- explicit placeholders for future custom prompts

## Expected Artifacts

Each run should write into `tmp/runs/<experiment_id>/...`:

- `resolved_config.json`
- `input_feature_manifest.json`
- `reasoning_target_manifest.json`
- `reasoning_oof_predictions.csv`
- `reasoning_metrics.csv`
- `downstream_public_summary.csv`
- `downstream_public_fold_metrics.csv`
- `promotion_status.json`

Promoted runs should additionally write:

- `reasoning_private_predictions.csv`
- `reasoning_private_metrics.csv` when private reasoning labels are available
- `downstream_private_predictions.csv`
- `run_summary.md`

## Important Constraints

- public and private raw datasets are keyed on `founder_uuid`
- policy target-bank files use `uuid` and must be renamed internally to `founder_uuid`
- the experiment config should explicitly name the policy targets to model
- the default run should restrict that list to the 10 policy columns exposed by `policy_features_test.csv`
- reasoning targets are assumed to be scaled `0-1` numeric values
- any missing future custom prompt work must fail explicitly rather than degrade to defaults
