# Vela Teacher-Student LLM Reasoning

This repo now has two explicit tracks built around the checked-in [Feature Repository](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository):

- `reproduction_mode` is the default. It reproduces the current VCBench success-prediction benchmark matrix as closely as possible from the bundled feature banks and targets.
- `reasoning_distillation_mode` is the active research path. It trains student models to reconstruct the LLM-derived policy targets from repository feature banks and new intermediary features built from raw VCBench data.

The primary teacher targets are the 16 continuous `v25_*` policy-induction scores in [Feature Repository/policies/predictions_train.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/policies/predictions_train.csv) and [Feature Repository/policies/predictions_test.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/policies/predictions_test.csv). Taste targets are supported as an optional binary family and are modeled with logistic regression only.

## Source Of Truth

- [Feature Repository](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository) is authoritative for the current success-prediction feature banks and policy-derived targets.
- [data](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data) is reserved for raw VCBench inputs and newly generated intermediary banks such as sentence-transformer embeddings.
- The older `policy_features.csv` contract is retired from the active pipeline.

## Active Inputs

- Raw VCBench:
  [data/VCBench_data/vcbench_final_public.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/VCBench_data/vcbench_final_public.csv)
  [data/VCBench_data/vcbench_final_private (success column removed) - vcbench_final_private.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/VCBench_data/vcbench_final_private%20(success%20column%20removed)%20-%20vcbench_final_private.csv)
- Repository feature banks:
  [Feature Repository/hq_baseline](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/hq_baseline)
  [Feature Repository/llm_engineering](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/llm_engineering)
  [Feature Repository/lambda_policies](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/lambda_policies)
  [Feature Repository/policies](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/policies)
- Split contract:
  [Feature Repository/splits/train_uuids.txt](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/splits/train_uuids.txt)
  [Feature Repository/splits/test_uuids.txt](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/splits/test_uuids.txt)
  [Feature Repository/splits/labels.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/Feature%20Repository/splits/labels.csv)
- Experiment config:
  [experiments/teacher_student_distillation_v1.json](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/experiments/teacher_student_distillation_v1.json)

## Runtime Modes

`reproduction_mode`:

- default mode
- reproduces the 9 benchmark success-prediction experiments
- uses 5-fold stratified outer CV and inner 3-fold tuning where required
- preserves the HQ `exit_count > 0` override

`reasoning_distillation_mode`:

- predicts LLM-derived policy targets from selected feature banks
- defaults to the `v25_policies` target family
- uses stratified 3-fold outer CV
- trains one model per target
- supports opt-in held-out evaluation

## Feature Families For Distillation

Repository-backed inputs:

- `hq_baseline`
- `llm_engineering`
- `lambda_policies`

Generated intermediary inputs:

- `sentence_prose`
- `sentence_structured`

`policy_v25` is allowed in reproduction mode but blocked as an input bank when `v25_policies` is the distillation target family, because that would leak the teacher targets into the student inputs.

## Entry Points

- CLI:
  `C:\Users\joelb\.conda\envs\vela_TRL\python.exe -m src.pipeline.run_distillation --config experiments/teacher_student_distillation_v1.json`
- GUI:
  `C:\Users\joelb\.conda\envs\vela_TRL\python.exe -m src.gui.run_launcher`

Representative reasoning-distillation run:

```powershell
C:\Users\joelb\.conda\envs\vela_TRL\python.exe -m src.pipeline.run_distillation `
  --config experiments\teacher_student_distillation_v1.json `
  --run-mode reasoning_distillation_mode `
  --target-family v25_policies `
  --heldout-evaluation `
  --active-feature-banks hq_baseline sentence_prose sentence_structured `
  --reasoning-models ridge
```

## Key Docs

- [docs/feature_repository_clarification.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/feature_repository_clarification.md)
- [docs/current_workflows.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/current_workflows.md)
- [docs/pipeline_architecture.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/pipeline_architecture.md)
- [docs/repo_architecture.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/repo_architecture.md)
