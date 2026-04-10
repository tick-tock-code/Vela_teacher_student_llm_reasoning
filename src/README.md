# Source

Active code layout:

- `data/` for raw-data loading and target-bank contracts
- `intermediary_features/` for reusable feature-bank generation and storage
- `student/` for reasoning model builders and training loops
- `pipeline/` for config parsing, run overrides, and orchestration
- `gui/` for the Tkinter launcher
- `llm_engineering/` for the scaffolded future prompt-driven feature family
- `utils/` for paths, IO, dependency guards, and placeholders

Dormant code:

- `downstream/` is retained only as future work and is not part of the active pipeline
