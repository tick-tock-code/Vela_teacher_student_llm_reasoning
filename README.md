# Vela Teacher-Student LLM Reasoning

This repo now has one active objective: reconstruct selected LLM-derived reasoning targets from raw VCBench founder data.

The active pipeline:

- builds reusable intermediary feature banks under `data/intermediary_features/`
- trains one regressor per selected policy target
- predicts the 10 held-out policy targets exposed by `policy_features_test.csv`
- compares reasoning-reconstruction quality across feature families and model choices

Features A-F are out of scope here. Founder-success prediction remains dormant future work and is not part of the default run.

## Active Inputs

- raw founder inputs:
  [data/VCBench_data/vcbench_final_public.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/VCBench_data/vcbench_final_public.csv)
  [data/VCBench_data/vcbench_final_private (success column removed) - vcbench_final_private.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/VCBench_data/vcbench_final_private%20(success%20column%20removed)%20-%20vcbench_final_private.csv)
- reasoning targets:
  [data/reasoning_feature_targets/policy_features.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/reasoning_feature_targets/policy_features.csv)
  [data/reasoning_feature_targets/policy_features_test.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/reasoning_feature_targets/policy_features_test.csv)
- experiment config:
  [experiments/teacher_student_distillation_v1.json](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/experiments/teacher_student_distillation_v1.json)

## Active Feature Families

- `vcbench_mirror_baseline_v1`: deterministic hand-built numeric features from raw VCBench fields
- `sentence_transformer_prose_v1`: embeddings of `anonymised_prose`
- `sentence_transformer_structured_v1`: embeddings of deterministic structured founder text rendered from JSON fields

LLM-engineered feature generation is scaffolded but inactive until custom prompt assets exist.

## Entry Points

- CLI:
  `C:\Users\joelb\.conda\envs\vela_TRL\python.exe -m src.pipeline.run_distillation --config experiments/teacher_student_distillation_v1.json`
- GUI:
  `C:\Users\joelb\.conda\envs\vela_TRL\python.exe -m src.gui.run_launcher`

## Key Docs

- [docs/scoping_summary.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/scoping_summary.md)
- [docs/implementation_plan.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/implementation_plan.md)
- [docs/pipeline_architecture.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/pipeline_architecture.md)
- [docs/repo_architecture.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/repo_architecture.md)
- [docs/current_workflows.md](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/current_workflows.md)
