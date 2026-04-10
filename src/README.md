# Source

Core code lives here and is split by responsibility:

- `teacher/` for teacher policy induction and scoring
- `student/` for distilled model training and inference
- `downstream/` for founder-success route training and prediction
- `data/` for raw-data loaders, input-feature builders, and target-bank loading
- `evaluation/` for metrics and threshold logic
- `llm_engineering/` for prompt-driven rule generation adapters and caches
- `pipeline/` for config parsing and end-to-end orchestration
- `utils/` for shared utilities
