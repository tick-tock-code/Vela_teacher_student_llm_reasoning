# Feature-Set Screening Report

- Repeats: 1
- Stage A models: `linear_l2`
- Organization: grouped by model architecture, then target family/output mode.
- Held-out features/targets: not used
- Recommendation rule: top score + any within `best - 0.005` (max 3).

## Architecture Coverage

| architecture | model_ids | target_families | output_modes |
|---|---|---|---|
| Linear L2 | ridge | v25_policies | single_target |

## Stage A Screening By Architecture

### Linear L2

- Model IDs: `ridge`

| target_family | output_mode | rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---|---|---:|---|---:|---:|---:|---:|
| v25_policies | single_target | 1 | sentence_prose | 0.3411 | 0.0000 | 0.3411 | True |

## Cross-Architecture Take-Forward Sets

| target_family | output_mode | recommended_feature_sets |
|---|---|---|
| v25_policies | single_target | sentence_prose |

Take-forward sets above are selected using blended screening scores across enabled Stage A models for each target/output combination.
