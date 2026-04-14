# Feature-Set Screening Report

- Repeats: 1
- Stage A models: `linear_l2`
- Held-out features/targets: not used
- Recommendation rule: top score + any within `best - 0.005` (max 3).

## v25_policies | single_target

| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---:|---|---:|---:|---:|---:|
| 1 | lambda_policies_plus_sentence_bundle | 0.4229 | 0.0000 | 0.4229 | True |
| 2 | hq_plus_sentence_bundle | 0.4198 | 0.0000 | 0.4198 | True |
| 3 | llm_engineering_plus_sentence_bundle | 0.3886 | 0.0000 | 0.3886 | False |

## v25_policies | multi_output

| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---:|---|---:|---:|---:|---:|
| 1 | lambda_policies_plus_sentence_bundle | 0.4228 | 0.0000 | 0.4228 | True |
| 2 | hq_plus_sentence_bundle | 0.4197 | 0.0000 | 0.4197 | True |
| 3 | llm_engineering_plus_sentence_bundle | 0.3887 | 0.0000 | 0.3887 | False |

## taste_policies | single_target

| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---:|---|---:|---:|---:|---:|
| 1 | lambda_policies_plus_sentence_bundle | 0.7734 | 0.0000 | 0.7734 | True |
| 2 | hq_plus_sentence_bundle | 0.7718 | 0.0000 | 0.7718 | True |
| 3 | llm_engineering_plus_sentence_bundle | 0.7247 | 0.0000 | 0.7247 | False |

## taste_policies | multi_output

| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---:|---|---:|---:|---:|---:|
| 1 | lambda_policies_plus_sentence_bundle | 0.7736 | 0.0000 | 0.7736 | True |
| 2 | hq_plus_sentence_bundle | 0.7716 | 0.0000 | 0.7716 | True |
| 3 | llm_engineering_plus_sentence_bundle | 0.7244 | 0.0000 | 0.7244 | False |
