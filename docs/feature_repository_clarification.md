# Feature Repository Clarification

This document states the current project aim after the introduction of [Feature Repository](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository) and the clarifications in [project_clarification/FOR_JOEL_DISTILLATION_PROJECT.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/project_clarification/FOR_JOEL_DISTILLATION_PROJECT.md).

## Primary Aim

The main research task is now:

- take the feature universe available in `Feature Repository/` plus any new intermediary banks generated from raw VCBench data
- train lightweight student models that reconstruct the LLM-derived `v25_*` policy targets
- measure how much of the teacher signal can be recovered without calling the teacher model at inference time

The active teacher is not the older 10-policy or 48-policy contract. The active teacher is the 16-policy `v25` ensemble in [Feature Repository/policies/predictions_train.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/policies/predictions_train.csv) and [Feature Repository/policies/predictions_test.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/policies/predictions_test.csv).

## Secondary Active Requirement

The repo must also keep a faithful reproduction path for the current success-prediction benchmark set. That is why `reproduction_mode` is the default run mode.

This reproduction path is not the main research contribution, but it is required because:

- it validates the environment and data alignment
- it gives the benchmark ceiling and baseline gaps the student must be judged against
- it preserves the current reported results while the distillation work evolves

## Why The v25 Teacher Matters

The `v25_*` targets are continuous scores in `[0, 1]` and represent the strongest current policy-induction teacher in the repository bundle.

Relevant reference files:

- [Feature Repository/FOR_JOEL_DISTILLATION_PROJECT.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/FOR_JOEL_DISTILLATION_PROJECT.md)
- [Feature Repository/REPRODUCING_HEADLINE_SCORES.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/REPRODUCING_HEADLINE_SCORES.md)

The project clarification is:

- the old 10-policy set is an earlier experiment and is not the teacher to distil
- the 16 `v25_*` policies are the active teacher targets
- Taste policies are optional, binary, and should be modeled separately with logistic regression

## Current Practical Framing

Two framing options exist:

- HQ-only student inputs as a smoke test
- HQ plus richer views such as LLM-engineering features and text embeddings as the stronger research setting

The more important research question is whether a student can recover the policy-induction gain that remains complementary to the LLM-engineering feature bank. That is the harder and more meaningful setting.

## Data Layout Contract

`Feature Repository/` now owns:

- benchmark feature banks
- policy teacher targets
- split files
- success labels

`data/` now owns:

- raw VCBench public and held-out founder data
- generated intermediary banks such as sentence-transformer embeddings

The old `data/reasoning_feature_targets/` layout is no longer the active contract for current experiments.

## Modeling Contract

`reproduction_mode`:

- reproduces the benchmark success-prediction matrix
- uses the repository’s train/test split contract
- remains the default

`reasoning_distillation_mode`:

- predicts the selected target family
- defaults to `v25_policies`
- uses stratified 3-fold outer CV
- keeps held-out evaluation opt-in

The current intended distillation progression is:

1. `hq_baseline` as the first smoke-test student input
2. sentence-transformer intermediary banks from VCBench text
3. combinations of repository banks and intermediary banks
4. later, custom LLM-engineered intermediary features once prompt assets exist

## Interpretation Rule

The main question is not merely whether the student correlates with the teacher. The question is how much of the teacher’s useful signal can be reproduced from cheaper inputs and whether that recovered signal is strongest from structured metadata, text embeddings, or both.
