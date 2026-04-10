# Current Workflows

This document tracks the active workstreams for the project in the order they should converge.

## Working Rule

The project has two parallel feature-engineering tracks first.

Do not start broad model-architecture sweeps until at least one usable feature bank is stable enough to support repeatable 3-fold CV.

## Workflow 1: LLM-Engineered Features For Reasoning Prediction

Goal:

- use the reasoning prompt as the source document
- ask an LLM-engineering workflow to generate a bank of hard-coded features that try to reconstruct the teacher reasoning targets

Inputs:

- the custom reasoning prompt or prompt bundle
- founder records
- any prompt-specific instructions for rule construction

Outputs:

- a named engineered rule family
- frozen rule definitions
- train/test feature tables produced by those rules
- a manifest describing which rule family generated which feature bank

Current implementation state:

- cache and archive loading are implemented
- adapter plumbing for TRL-style rule generation is implemented
- custom prompt loading, rendering, and rule post-processing are still explicit placeholders

Files to extend:

- `src/llm_engineering/adapter.py`
- `src/llm_engineering/cache.py`
- `src/llm_engineering/custom_prompts.py`

Success condition:

- you can regenerate the same engineered feature bank from a named prompt bundle and archived rule family without manual reconstruction

## Workflow 2: Founder-To-ML Feature Engineering For Reasoning Prediction

Goal:

- transform founder data into ML-compatible inputs that can predict the reasoning targets

Candidate feature sources:

- human-curated tabular features
- LLM-engineered hard-coded features
- sentence-transformer embeddings of founder text
- other structured transforms that remain reproducible

Important constraint:

- reasoning prediction should remain a supervised distillation problem
- do not let founder-success labels leak into reasoning-feature generation

About the recursion concern:

- using features to predict reasoning targets, then using predicted reasoning plus features to predict success, is acceptable
- that is a staged teacher-student pipeline, not a logical error
- the only requirement is that the success model sits downstream from the reasoning-prediction stage and is evaluated honestly

Current implementation state:

- the repo now treats `data/reasoning_feature_targets/policy_features.csv` as the public target bank
- the repo now treats `data/reasoning_feature_targets/policy_features_test.csv` as the held-out target bank
- the default experiment config explicitly selects the 10 held-out policy columns and trains one model per selected target
- the default input side is a deterministic founder-baseline feature builder derived from raw VCBench rows
- richer feature generation paths still need to be built

Likely extension areas:

- add sentence-transformer feature builders under a future representation module
- add merge logic for combining tabular, embedding, and engineered-rule banks into explicit experiment configs
- write manifests so each bank is traceable and frozen

Success condition:

- you can produce one or more named train/test feature banks for reasoning prediction and compare them reproducibly

## Convergence Gate

The two feature-engineering workflows converge when you have at least one frozen reasoning-prediction feature bank with:

- explicit provenance
- matching public/private columns
- deterministic train/test processing
- acceptable 3-fold CV agreement on one or more reasoning targets

At that point, freeze the bank and move to model work.

## Workflow 3: Model Architecture Sweep For Reasoning Prediction

Goal:

- compare how different model families predict the teacher reasoning targets once feature engineering is good enough

Principle:

- this stage can be as black-box as needed
- the interpretability sits in the teacher reasoning targets, so the student model does not have to be inherently interpretable

Good first model families:

- ridge or elastic net as strong linear baselines
- gradient-boosted trees for non-linear tabular learning
- light MLPs only after tabular baselines are stable

Evaluation:

- Pearson
- Spearman
- MAE
- RMSE
- R2

Promotion rule:

- do not promote to private prediction until public 3-fold CV is good enough for the specific reasoning targets you care about

## Workflow 4: Downstream Founder-Success Comparison

Goal:

- test whether predicted reasoning is good enough to substitute for true reasoning in the founder-success model

Required public-set routes:

- baseline features only
- baseline features plus true reasoning
- baseline features plus predicted reasoning

Required outputs:

- public CV comparison table
- private predictions after manual promotion
- optional private reasoning agreement if private reasoning labels exist

Interpretation question:

- the key result is not only whether predicted reasoning tracks true reasoning
- it is whether downstream success performance remains close when true reasoning is replaced with predicted reasoning

## Immediate Next Steps

1. Provide the custom prompt assets for LLM-engineered rule generation.
2. Decide the first non-rule feature family to build for reasoning prediction.
3. Create frozen manifests for each feature bank rather than relying on ad hoc file names.
4. Add the next input-feature families once you decide which representation path to prioritize first.
