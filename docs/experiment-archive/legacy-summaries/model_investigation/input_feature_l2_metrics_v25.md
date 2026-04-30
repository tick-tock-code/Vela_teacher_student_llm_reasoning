# Run Setup Summary

- Mode: `reasoning_distillation_mode`
- Source run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-15_131920_500083_reasoning_distillation`

## Run Summary

# Run Summary

- Run mode: `reasoning_distillation_mode`
- Target family: `v25_policies`
- Target task kind: `regression`
- Target count: 16
- Repository feature banks loaded: 3
- Intermediary banks prepared: 2
- Feature-set comparisons run: 15
- Distillation models run per target: 1
- Max parallel workers: 7
- Public CV strategy: stratified on quantile buckets of the row-wise mean selected target score.
- CV seed repeats: 4 (enabled=True)
- Nested hyperparameter sweep: False
- Distillation outer folds used: 3
- Distillation inner folds used for sweep: 0
- Save reasoning prediction CSVs: True
- OOF prediction columns written: 240
- Held-out prediction columns written: 0
- Held-out evaluation was skipped because it was not requested.


## Reasoning Metrics

# Reasoning Metrics Summary

- Target family: `v25_policies`
- Task kind: `regression`

## Public OOF (Averaged Across Targets)

| feature_set_id | model_id | mean_r2 | mean_rmse | mean_mae | mean_pearson | mean_spearman |
|---|---:|---:|---:|---:|---:|---:|
| lambda_policies_plus_sentence_bundle | ridge | 0.4211 | 0.2651 | 0.1775 | 0.6499 | 0.6152 |
| hq_plus_sentence_bundle | ridge | 0.4187 | 0.2659 | 0.1815 | 0.6503 | 0.6266 |
| lambda_policies_plus_sentence_prose | ridge | 0.4084 | 0.2683 | 0.1807 | 0.6400 | 0.6124 |
| hq_plus_sentence_prose | ridge | 0.4059 | 0.2691 | 0.1849 | 0.6409 | 0.6248 |
| lambda_policies_plus_sentence_structured | ridge | 0.4014 | 0.2698 | 0.1817 | 0.6338 | 0.6057 |
| hq_plus_sentence_structured | ridge | 0.3954 | 0.2714 | 0.1865 | 0.6315 | 0.6170 |
| llm_engineering_plus_sentence_bundle | ridge | 0.3894 | 0.2737 | 0.1892 | 0.6272 | 0.6072 |
| llm_engineering_plus_sentence_prose | ridge | 0.3712 | 0.2782 | 0.1942 | 0.6131 | 0.6024 |
| sentence_bundle | ridge | 0.3648 | 0.2782 | 0.1945 | 0.6079 | 0.5927 |
| llm_engineering_plus_sentence_structured | ridge | 0.3499 | 0.2831 | 0.1983 | 0.5936 | 0.5828 |
| sentence_prose | ridge | 0.3406 | 0.2838 | 0.2013 | 0.5903 | 0.5869 |
| sentence_structured | ridge | 0.3121 | 0.2899 | 0.2064 | 0.5614 | 0.5588 |
| lambda_policies | ridge | 0.2739 | 0.2990 | 0.2077 | 0.5145 | 0.4839 |
| hq_baseline | ridge | 0.2509 | 0.3048 | 0.2180 | 0.4936 | 0.4795 |
| llm_engineering | ridge | 0.1382 | 0.3303 | 0.2487 | 0.3632 | 0.2948 |
