# Current Workflows

This document tracks the active work sequence for the repo after the Feature Repository integration.

## Workflow 1: Benchmark Reproduction

Goal:

- reproduce the current success-prediction benchmark matrix from the bundled repository files

Mode:

- `reproduction_mode`

Default inputs:

- `hq_baseline`
- `llm_engineering`
- `lambda_policies`
- `policy_v25`

Evaluation contract:

- 5-fold stratified outer CV
- inner 3-fold tuning for nested logistic regression
- held-out test evaluation always on
- HQ override preserved for experiments that use HQ features

Why this matters:

- it proves the local environment matches the current benchmark set
- it gives a fixed ceiling/reference table before any new student work

## Workflow 2: Reasoning Distillation

Goal:

- train student models that reconstruct the 16 continuous `v25_*` teacher targets

Mode:

- `reasoning_distillation_mode`

Default target family:

- `v25_policies`

Current input banks:

- repository banks:
  - `hq_baseline`
  - `llm_engineering`
  - `lambda_policies`
- intermediary banks:
  - `sentence_prose`
  - `sentence_structured`

Current learners:

- `ridge`
- `xgb1_regressor`

Evaluation contract:

- stratified 3-fold outer CV
- one model per policy target
- held-out evaluation only when explicitly requested

Research priority:

- focus on feature/input design more than model proliferation
- use `hq_baseline` as the first smoke-test student input
- then compare text-embedding banks and mixed bank combinations

## Workflow 3: Optional Taste Modeling

Goal:

- model the binary Taste targets as a secondary policy family when needed

Mode:

- `reasoning_distillation_mode`

Target family:

- `taste_policies`

Model constraint:

- classification only
- current default learner is `logreg_classifier`

Rule:

- keep Taste off unless the run is explicitly about the binary policy family

## Workflow 4: New Intermediary Feature Banks

Goal:

- generate reusable inputs from raw VCBench data that can be joined with repository banks

Current banks:

- `sentence_prose`
- `sentence_structured`

Storage rule:

- generated banks live in [data/intermediary_features](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/intermediary_features)
- raw VCBench stays in [data/VCBench_data](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/VCBench_data)

Near-term work:

- improve deterministic structured-text rendering
- compare prose-only, structured-only, and combined sentence banks
- later add custom LLM-engineered intermediary banks when prompt assets exist

## Inactive Or Guarded Paths

- the old `policy_features.csv` reasoning-target workflow is retired
- `policy_v25` cannot be used as a distillation input bank when `v25_policies` is the target family
- scaffolded custom LLM-engineered intermediary features remain disabled until prompt assets exist
