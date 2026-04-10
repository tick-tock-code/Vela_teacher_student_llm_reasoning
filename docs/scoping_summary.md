# Scoping Summary

This repo is the first implementation of a teacher-student reasoning distillation workflow for founder prediction.

The project direction from [Proposed project direction.pdf](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/docs/Proposed%20project%20direction.pdf) is:

- use expensive, interpretable LLM reasoning as the teacher
- train a cheaper student to recover most of that signal
- compare agreement, deployment cost, and downstream founder-success performance

This repo's v1 implementation is narrower and more operational than the PDF:

- the student predicts numeric policy reasoning targets with regression models
- the public set is the supervised training domain and the private set is the held-out prediction domain
- downstream founder-success comparisons use true reasoning targets versus student-predicted reasoning targets
- promotion to private-set prediction is manually gated

Current data reality matters:

- the raw VCBench public and private CSVs live under `data/VCBench_data/`
- the reasoning target bank lives under `data/reasoning_feature_targets/`
- `policy_features.csv` is the public teacher target table with 48 policy targets
- `policy_features_test.csv` is the held-out target table with 10 comparable policy targets
- the model input side is now separate from the target side and defaults to a deterministic founder-baseline feature builder derived from raw VCBench rows

The old `LLM_Reasoning_Main` repo is used as a source of reusable infrastructure, not as something to copy wholesale:

- keep path helpers, artifact IO, loading/alignment utilities, metrics, and model helpers
- import the LLM-engineering adapter and cache layout
- do not port instability-control study logic

The implementation also leaves strict placeholders where the project is not yet decision-complete:

- custom LLM-engineering prompt loading
- custom prompt rendering
- custom rule post-processing
- any future transforms that depend on not-yet-provided methods

Those placeholders are designed to fail loudly with `NotImplementedError` so the repo cannot silently do the wrong thing.
