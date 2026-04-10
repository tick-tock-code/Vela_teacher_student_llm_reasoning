# Current Workflows

This document tracks the active work sequence for the repo.

## Workflow 1: Reusable Intermediary Feature Banks

Goal:

- turn raw VCBench founder rows into reusable model inputs for reasoning reconstruction

Active feature families:

- `vcbench_mirror_baseline_v1`
- `sentence_transformer_prose_v1`
- `sentence_transformer_structured_v1`

Rules:

- store reusable outputs in `data/intermediary_features/`
- keep public and held-out columns aligned
- write manifests so every bank is traceable

Immediate work focus:

- improve the mirror bank carefully where deterministic raw-field features help
- refine structured-text rendering before chasing larger model changes
- compare prose-only, structured-only, and combined embedding banks

## Workflow 2: LLM-Engineered Feature Family

Goal:

- use custom prompt assets later to generate hard-coded engineered features aimed at reconstructing the policy targets

Current state:

- cache and archive compatibility are implemented
- custom prompts, rendering, and post-processing are still explicit placeholders

Rule:

- do not activate this feature family until custom prompts exist

## Workflow 3: Reasoning Reconstruction Model Sweep

Goal:

- compare lightweight model families once the feature banks are stable enough

Current active models:

- `ridge`
- `xgb1_regressor`

Evaluation:

- Pearson
- Spearman
- MAE
- RMSE
- R2

Priority:

- spend more time on feature building than on model proliferation
- do not add MLPs until the feature banks are stable and informative

## Dormant Future Work

- founder-success prediction is intentionally inactive
- Features A-F are intentionally irrelevant to this repo
- any future downstream study should only start after the reasoning-reconstruction stage is solid
