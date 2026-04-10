# Repo Architecture

This is the canonical folder architecture for the repo going forward.

The goal is simple:

- keep reusable inputs under `data/`
- keep generated artifacts under `tmp/`
- keep human-readable decisions and results under `docs/`
- keep executable logic under `src/`

This document sets the target structure for new work.

It does not force an immediate bulk rename of the current raw-data folders.
The existing `data/VCBench_data/` and `data/reasoning_feature_targets/` locations are the current canonical layout.

## Top-Level Layout

```text
.
|-- data/
|-- docs/
|-- experiments/
|-- src/
|-- tests/
`-- tmp/
```

## Folder Responsibilities

### `data/`

Committed, reusable inputs only.

Use this for:

- raw VCBench tables
- teacher reasoning target tables
- reusable reference artifacts that should survive reruns

Do not use this for:

- ad hoc model outputs
- temporary embeddings
- scratch CSVs from one-off debugging

Current subfolders:

- `VCBench_data/`: raw founder tables
- `reasoning_feature_targets/`: policy reasoning target banks

### `docs/`

Human-readable project memory.

Use this for:

- scoping and planning docs
- workflow trackers
- architecture decisions
- experiment readouts and interpreted findings

Recommended doc types:

- `*_summary.md` for short syntheses
- `*_architecture.md` for structure decisions
- `*_workflows.md` for active workstreams
- `reports/` later, once repeated evaluation summaries start accumulating

### `experiments/`

Committed experiment definitions, not generated artifacts.

Use this for:

- config files
- config templates
- comparison notes between named experiment variants

Do not store:

- fold outputs
- private predictions
- caches
- run directories

Those belong in `tmp/`.

### `src/`

Executable code, split by research responsibility rather than by script history.

Canonical layout:

```text
src/
|-- data/            # loading, alignment, splits, contracts around tabular inputs
|-- downstream/      # founder-success route training and prediction
|-- evaluation/      # metrics, thresholds, comparison summaries
|-- llm_engineering/ # prompt-driven rule generation adapters and cache readers
|-- pipeline/        # config parsing and end-to-end orchestration
|-- student/         # reasoning prediction models and OOF/full-train logic
|-- teacher/         # teacher target contracts and future teacher generation logic
`-- utils/           # paths, IO, dependency guards, placeholders
```

Design rule:

- `evaluation/` should score outputs
- `downstream/` should fit downstream success models
- `student/` should fit reasoning-prediction models

That separation keeps route-building logic out of the metrics layer.

### `tests/`

Tests should mirror the code structure loosely:

- unit tests for small helpers and contracts
- smoke/integration tests for end-to-end pipeline behavior

If the suite grows, split into `tests/unit/` and `tests/integration/`.

### `tmp/`

Disposable outputs only.

Use this for:

- run artifacts
- temporary embeddings
- LLM-engineering caches
- intermediate feature builds
- scratch analysis

If deleting `tmp/` would be painful, the file probably belongs in `data/` or `docs/` instead.

## Practical Rules

1. New reusable datasets go in `data/`.
2. New generated run outputs go in `tmp/`.
3. New research notes go in `docs/`.
4. New executable logic goes in `src/`.
5. Do not add standalone top-level scripts unless they are truly one-off bootstrap tools.
6. Prefer adding a module under `src/` and calling it through `python -m ...`.

## Current Refactor Decision

The repo now treats downstream founder-success modeling as its own package under `src/downstream/`.

That is the right boundary because downstream route construction is not just evaluation; it fits models and produces predictions.
