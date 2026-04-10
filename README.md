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
|   |-- downstream/# founder-success route training and prediction
|   |-- evaluation/# metrics and threshold logic
|   |-- llm_engineering/ # imported rule-generation adapter + placeholders
|   |-- pipeline/  # config + orchestration entrypoints
|   |-- student/   # distilled model code
|   |-- teacher/   # teacher policy / scoring pipeline
|   `-- utils/     # shared helpers
|-- tests/         # unit and integration tests
`-- tmp/           # disposable intermediate artifacts
```

## Current Entry Points

- Default experiment config:
  [experiments/teacher_student_distillation_v1.json](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/experiments/teacher_student_distillation_v1.json)
- CLI entrypoint:
  `python -m src.pipeline.run_distillation --config experiments/teacher_student_distillation_v1.json`
- Scoping docs:
  [docs/scoping_summary.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/scoping_summary.md)
  [docs/implementation_plan.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/implementation_plan.md)
  [docs/pipeline_architecture.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/pipeline_architecture.md)
  [docs/repo_architecture.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/repo_architecture.md)
  [docs/current_workflows.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/current_workflows.md)

## Remaining Inputs

The default config now treats the on-disk policy files as the reasoning target bank:

- public train targets:
  [data/reasoning_feature_targets/policy_features.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/reasoning_feature_targets/policy_features.csv)
- held-out comparable targets:
  [data/reasoning_feature_targets/policy_features_test.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/reasoning_feature_targets/policy_features_test.csv)

The default config explicitly selects the 10 policy targets exposed by the held-out file and trains one model per selected target.
The default input side is a deterministic founder-baseline feature builder derived from the raw VCBench files.
Future feature families can be added through the `input_features` config block.
