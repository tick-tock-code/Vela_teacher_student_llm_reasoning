# Scoping Summary

The PDF thesis in [Proposed project direction.pdf](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/Proposed%20project%20direction.pdf) is still the right high-level frame:

- expensive, interpretable LLM reasoning acts as the teacher
- a cheaper student should recover that signal with much lower inference cost

The active repo implementation is narrower than the PDF’s original framing.

Here, the reasoning features are the targets:

- public teacher targets:
  [data/reasoning_feature_targets/policy_features.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/reasoning_feature_targets/policy_features.csv)
- held-out comparable targets:
  [data/reasoning_feature_targets/policy_features_test.csv](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/data/reasoning_feature_targets/policy_features_test.csv)

The model inputs are built separately from raw VCBench founder data. The active repo question is:

Can reusable intermediary feature banks extracted from VCBench reconstruct the selected 10 policy targets well enough to approximate the teacher?

That means the current project is not about Features A-F and is not currently about founder-success prediction. The active work is:

- feature extraction from raw founder data
- reusable intermediary bank storage
- per-target reasoning regression
- held-out agreement on the 10 policies shared with `policy_features_test.csv`

The three active feature families are:

- `vcbench_mirror_baseline_v1`
- `sentence_transformer_prose_v1`
- `sentence_transformer_structured_v1`

LLM-engineered features remain an intended future family, but the prompt-specific logic is still a strict placeholder and must fail loudly until custom prompts are provided.
