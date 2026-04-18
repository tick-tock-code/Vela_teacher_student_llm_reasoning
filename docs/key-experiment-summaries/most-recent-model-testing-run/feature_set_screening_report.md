# Feature-Set Screening Report

- Repeats: 1
- Stage A models: `xgb1`
- Held-out features/targets: not used
- Recommendation rule: top score + any within `best - 0.005` (max 3).

## v25_policies | single_target

| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---:|---|---:|---:|---:|---:|
| 1 | lambda_policies_plus_sentence_prose | 0.3415 | 0.0000 | 0.3415 | True |

## taste_policies | single_target

| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---:|---|---:|---:|---:|---:|
| 1 | lambda_policies_plus_sentence_prose | 0.7534 | 0.0000 | 0.7534 | True |
