# Vela Teacher-Student LLM Reasoning

Research repository for distilling a lightweight student model from a heavier reasoning-focused teacher on the VCBench founder dataset.

## Working Principles

This repo follows the operating constraints in [agents.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/agents.md):

- no silent assumptions about data, model architecture, or evaluation
- reproducible experiments with explicit inputs and outputs
- simple baselines before more complex modeling
- small, composable modules with regular refactoring

## Project Layout

```text
.
|-- data/          # canonical datasets and reusable data artifacts
|-- docs/          # reports, summaries, and interpreted results
|-- experiments/   # reproducible experiment configs and run metadata
|-- src/
|   |-- data/      # loading and preprocessing
|   |-- evaluation/# metrics and comparison logic
|   |-- student/   # distilled model code
|   |-- teacher/   # teacher policy / scoring pipeline
|   `-- utils/     # shared helpers
|-- tests/         # unit and integration tests
`-- tmp/           # disposable intermediate artifacts
```

## Next Steps

Add concrete pipeline code only after the following are explicit:

- dataset schema and storage format
- teacher output format
- student training objective
- evaluation metrics and comparison protocol
