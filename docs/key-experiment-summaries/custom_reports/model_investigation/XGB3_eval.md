# XGB depth=3 Evaluation (Partial)

- Source: interrupted depth-3 model-testing batch started on 2026-04-18.
- Scope: HQ/Lambda prose+bundle, XGB only, single-target, repeats intended=4.
- Completion status: regression completed 4/4 repeats; classification completed 3/4 repeats.
- Metric aggregation here uses `oof_overall` per target, averaged within each feature set per child run, then summarized across completed runs.

| family | feature_set_id | metric | completed_runs | mean | std_across_runs | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| taste_policies | hq_plus_sentence_bundle | F0.5 | 3 | 0.7917 | 0.0042 | 0.7890 | 0.7965 |
| taste_policies | hq_plus_sentence_prose | F0.5 | 3 | 0.7852 | 0.0039 | 0.7814 | 0.7892 |
| taste_policies | lambda_policies_plus_sentence_bundle | F0.5 | 3 | 0.7840 | 0.0029 | 0.7816 | 0.7873 |
| taste_policies | lambda_policies_plus_sentence_prose | F0.5 | 3 | 0.7762 | 0.0006 | 0.7755 | 0.7767 |
| v25_policies | lambda_policies_plus_sentence_bundle | R2 | 4 | 0.4187 | 0.0014 | 0.4170 | 0.4202 |
| v25_policies | hq_plus_sentence_bundle | R2 | 4 | 0.4138 | 0.0007 | 0.4129 | 0.4143 |
| v25_policies | lambda_policies_plus_sentence_prose | R2 | 4 | 0.4137 | 0.0018 | 0.4123 | 0.4162 |
| v25_policies | hq_plus_sentence_prose | R2 | 4 | 0.4112 | 0.0003 | 0.4109 | 0.4115 |
