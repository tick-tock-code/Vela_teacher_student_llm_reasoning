# Feature-Set Screening Report

- Repeats: 16
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
| v25_policies | multi_output | 1 | llm_engineering | -1.5363 | 0.5971 | -1.8348 | True |
| v25_policies | multi_output | 2 | hq_baseline | -5.7807 | 2.3557 | -6.9586 | False |

## Cross-Architecture Take-Forward Sets

| target_family | output_mode | recommended_feature_sets |
|---|---|---|
| v25_policies | multi_output | llm_engineering |

Take-forward sets above are selected using blended screening scores across enabled Stage A models for each target/output combination.
## Per-Target Detailed Metrics (v25)

### MLP | mlp_regressor | hq_baseline | multi_output

| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | -2.9096 | 0.3077 | 0.2623 |
| v25_p1 | -7.6910 | 0.4588 | 0.3894 |
| v25_p1 | -4.4851 | 0.3645 | 0.3090 |
| v25_p1 | -3.3327 | 0.3240 | 0.2444 |
| v25_p1 | -4.8458 | 0.3763 | 0.3202 |
| v25_p1 | -0.9565 | 0.2177 | 0.1813 |
| v25_p1 | -1.2837 | 0.2352 | 0.1938 |
| v25_p1 | -6.3942 | 0.4232 | 0.3401 |
| v25_p1 | -4.1696 | 0.3539 | 0.2956 |
| v25_p1 | -3.3909 | 0.3261 | 0.2765 |
| v25_p1 | -5.4972 | 0.3967 | 0.3104 |
| v25_p1 | -0.1064 | 0.1637 | 0.1281 |
| v25_p1 | -3.2736 | 0.3218 | 0.2820 |
| v25_p1 | -1.9931 | 0.2693 | 0.2145 |
| v25_p1 | -1.6377 | 0.2528 | 0.1992 |
| v25_p1 | -10.8532 | 0.5359 | 0.4854 |
| v25_p11 | -4.5463 | 0.2444 | 0.1855 |
| v25_p11 | -8.8864 | 0.3263 | 0.2825 |
| v25_p11 | -1.9982 | 0.1797 | 0.1479 |
| v25_p11 | -4.1195 | 0.2348 | 0.2141 |
| v25_p11 | -6.8638 | 0.2910 | 0.2579 |
| v25_p11 | -3.8679 | 0.2289 | 0.1824 |
| v25_p11 | -9.3795 | 0.3343 | 0.2966 |
| v25_p11 | -11.5223 | 0.3672 | 0.3203 |
| v25_p11 | -8.6114 | 0.3217 | 0.2703 |
| v25_p11 | -8.5142 | 0.3201 | 0.2785 |
| v25_p11 | -6.4521 | 0.2833 | 0.2519 |
| v25_p11 | -8.4291 | 0.3186 | 0.2757 |
| v25_p11 | -10.8383 | 0.3570 | 0.3123 |
| v25_p11 | -7.5739 | 0.3038 | 0.2289 |
| v25_p11 | -7.7600 | 0.3071 | 0.2513 |
| v25_p11 | -10.9680 | 0.3590 | 0.2964 |

### MLP | mlp_regressor | llm_engineering | multi_output

| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | -1.1856 | 0.2044 | 0.1807 |
| v25_p1 | -0.8950 | 0.1904 | 0.1556 |
| v25_p1 | -0.6447 | 0.1774 | 0.1563 |
| v25_p1 | -0.4535 | 0.1667 | 0.1407 |
| v25_p1 | -2.5432 | 0.2603 | 0.2260 |
| v25_p1 | -0.9457 | 0.1929 | 0.1688 |
| v25_p1 | -1.4352 | 0.2158 | 0.1841 |
| v25_p1 | -1.1608 | 0.2033 | 0.1778 |
| v25_p1 | -1.0749 | 0.1992 | 0.1744 |
| v25_p1 | -1.0521 | 0.1981 | 0.1639 |
| v25_p1 | -1.5072 | 0.2190 | 0.1930 |
| v25_p1 | -0.5043 | 0.1696 | 0.1473 |
| v25_p1 | -1.4577 | 0.2168 | 0.1974 |
| v25_p1 | -1.8248 | 0.2324 | 0.2092 |
| v25_p1 | -1.4234 | 0.2153 | 0.1878 |
| v25_p1 | -0.7652 | 0.1837 | 0.1546 |
| v25_p11 | -1.1997 | 0.1367 | 0.1158 |
| v25_p11 | -1.8826 | 0.1565 | 0.1330 |
| v25_p11 | -1.0569 | 0.1322 | 0.1167 |
| v25_p11 | -0.9119 | 0.1275 | 0.1094 |
| v25_p11 | -3.1228 | 0.1872 | 0.1631 |
| v25_p11 | -1.6088 | 0.1489 | 0.1266 |
| v25_p11 | -2.5393 | 0.1734 | 0.1500 |
| v25_p11 | -2.2400 | 0.1660 | 0.1500 |
| v25_p11 | -1.4389 | 0.1440 | 0.1232 |
| v25_p11 | -2.1697 | 0.1641 | 0.1449 |
| v25_p11 | -0.6419 | 0.1181 | 0.1025 |
| v25_p11 | -1.4381 | 0.1440 | 0.1162 |
| v25_p11 | -2.5171 | 0.1729 | 0.1489 |
| v25_p11 | -1.9879 | 0.1594 | 0.1456 |
| v25_p11 | -2.6603 | 0.1764 | 0.1607 |
| v25_p11 | -1.3006 | 0.1398 | 0.1160 |
