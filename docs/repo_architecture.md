# Repo Architecture

This is the canonical folder layout after the Feature Repository integration.

```text
.
|-- Feature Repository/
|   |-- hq_baseline/
|   |-- lambda_policies/
|   |-- llm_engineering/
|   |-- policies/
|   `-- splits/
|-- data/
|   |-- VCBench_data/
|   `-- intermediary_features/
|-- docs/
|-- experiments/
|-- src/
|   |-- data/
|   |-- downstream/
|   |-- evaluation/
|   |-- gui/
|   |-- intermediary_features/
|   |-- llm_engineering/
|   |-- pipeline/
|   |-- student/
|   |-- teacher/
|   `-- utils/
|-- tests/
`-- tmp/
```

## Ownership Rules

- `Feature Repository/` stores the authoritative benchmark feature banks, policy targets, labels, and canonical split order.
- `data/VCBench_data/` stores raw public and held-out VCBench founder inputs.
- `data/intermediary_features/` stores generated reusable banks such as sentence-transformer embeddings.
- `tmp/` stores disposable run artifacts.
- `experiments/` stores committed experiment definitions only.
- `docs/` stores the project contract, workflow notes, and architecture decisions.

## Code Boundaries

- `src/data/` owns loading and alignment logic.
- `src/intermediary_features/` owns creation and caching of generated feature banks.
- `src/student/` owns target-model training loops.
- `src/pipeline/` owns configuration, run-mode selection, and orchestration.
- `src/gui/` owns the Tkinter launcher.

## Active Modeling Paths

`reproduction_mode`:

- `Feature Repository/` banks plus split files
- benchmark success-prediction reproduction

`reasoning_distillation_mode`:

- repository banks plus generated intermediary banks
- teacher target family from `Feature Repository/policies/`
- one student model per target

## Guarded Or Dormant Areas

- `src/downstream/` is no longer the active contract for default runs
- custom LLM-engineered intermediary generation remains scaffolded until prompt assets exist
- the older `data/reasoning_feature_targets/` workflow is no longer the active source of truth
