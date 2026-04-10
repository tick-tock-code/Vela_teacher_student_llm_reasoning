# Repo Architecture

This is the canonical folder layout for the reasoning-reconstruction project.

```text
.
|-- data/
|   |-- VCBench_data/
|   |-- reasoning_feature_targets/
|   `-- intermediary_features/
|-- docs/
|-- experiments/
|-- src/
|   |-- data/
|   |-- evaluation/
|   |-- gui/
|   |-- intermediary_features/
|   |-- llm_engineering/
|   |-- pipeline/
|   |-- student/
|   |-- teacher/
|   |-- downstream/   # dormant future path only
|   `-- utils/
|-- tests/
`-- tmp/
```

## Folder Rules

- `data/` stores reusable committed inputs and reusable generated feature banks
- `tmp/` stores disposable run artifacts
- `docs/` stores project memory and architecture decisions
- `experiments/` stores committed experiment definitions
- `src/intermediary_features/` owns feature-bank generation and caching logic
- `src/pipeline/` owns config parsing, overrides, and orchestration
- `src/gui/` owns the launcher surface

## Important Boundary

There is one active modeling path:

- VCBench raw data -> intermediary feature banks -> per-target reasoning regressors -> optional held-out reasoning predictions

Anything outside that path is dormant, including founder-success prediction.
