# Feature-Set Screening Report

- Repeats: 1
- Stage A models: `mlp`
- Organization: grouped by model architecture, then target family/output mode.
- Held-out features/targets: not used
- Recommendation rule: top score + any within `best - 0.005` (max 3).

## Architecture Coverage

| architecture | model_ids | target_families | output_modes |
|---|---|---|---|
| MLP | mlp_regressor | v25_policies | multi_output |

## Stage A Screening By Architecture

### MLP

- Model IDs: `mlp_regressor`

| target_family | output_mode | rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |
|---|---|---:|---|---:|---:|---:|---:|
| v25_policies | multi_output | 1 | lambda_policies_plus_sentence_bundle | 0.4169 | 0.0000 | 0.4169 | True |
| v25_policies | multi_output | 2 | hq_plus_sentence_bundle | 0.3502 | 0.0000 | 0.3502 | False |

## Cross-Architecture Take-Forward Sets

| target_family | output_mode | recommended_feature_sets |
|---|---|---|
| v25_policies | multi_output | lambda_policies_plus_sentence_bundle |

Take-forward sets above are selected using blended screening scores across enabled Stage A models for each target/output combination.
## Per-Target Detailed Metrics (v25)

### MLP | mlp_regressor | hq_plus_sentence_bundle | multi_output

| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.3064 | 0.3512 | 0.2696 |
| v25_p11 | 0.2524 | 0.2601 | 0.1606 |
| v25_p112 | 0.4527 | 0.3534 | 0.2828 |
| v25_p116 | 0.4388 | 0.3331 | 0.2464 |
| v25_p121 | 0.2386 | 0.1848 | 0.1004 |
| v25_p135 | 0.3598 | 0.3169 | 0.2289 |
| v25_p143 | 0.2582 | 0.1467 | 0.0677 |
| v25_p150 | 0.4128 | 0.3536 | 0.2845 |
| v25_p157 | 0.3671 | 0.2475 | 0.1481 |
| v25_p161 | 0.2090 | 0.2163 | 0.1080 |
| v25_p38 | 0.3381 | 0.1871 | 0.0897 |
| v25_p52 | 0.4354 | 0.3612 | 0.2814 |
| v25_p55 | 0.4280 | 0.3161 | 0.2256 |
| v25_p58 | 0.4251 | 0.2905 | 0.1911 |
| v25_p72 | 0.2573 | 0.2701 | 0.1716 |
| v25_p80 | 0.4291 | 0.2964 | 0.2055 |

### MLP | mlp_regressor | lambda_policies_plus_sentence_bundle | multi_output

| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.3770 | 0.3329 | 0.2477 |
| v25_p11 | 0.3262 | 0.2470 | 0.1398 |
| v25_p112 | 0.4766 | 0.3456 | 0.2665 |
| v25_p116 | 0.5023 | 0.3137 | 0.2145 |
| v25_p121 | 0.3633 | 0.1690 | 0.0743 |
| v25_p135 | 0.3983 | 0.3072 | 0.2133 |
| v25_p143 | 0.3336 | 0.1391 | 0.0568 |
| v25_p150 | 0.4539 | 0.3410 | 0.2655 |
| v25_p157 | 0.4339 | 0.2341 | 0.1396 |
| v25_p161 | 0.2810 | 0.2062 | 0.0974 |
| v25_p38 | 0.4103 | 0.1766 | 0.0814 |
| v25_p52 | 0.4859 | 0.3447 | 0.2583 |
| v25_p55 | 0.4850 | 0.2999 | 0.2013 |
| v25_p58 | 0.5250 | 0.2641 | 0.1521 |
| v25_p72 | 0.3282 | 0.2568 | 0.1507 |
| v25_p80 | 0.4945 | 0.2789 | 0.1765 |
