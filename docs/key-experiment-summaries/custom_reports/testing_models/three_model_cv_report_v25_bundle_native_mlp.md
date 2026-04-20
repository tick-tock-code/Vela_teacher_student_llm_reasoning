# Three-Model CV Report (v25, Bundle Feature Sets, Native MLP)

## Scope

- Target family: `v25_policies` (regression, 16 targets)
- Feature sets:
  - `hq_plus_sentence_bundle`
  - `lambda_policies_plus_sentence_bundle`
- CV protocol: stratified 3-fold, single pass (no repeat seeds), nested CV off, held-out off

## Source Runs

- Ridge + XGB3 source:
  - `tmp/runs/teacher_student_distillation_v1/2026-04-19_195413_226932_model_testing`
- MLP source (corrected native multi-output implementation):
  - `tmp/runs/teacher_student_distillation_v1/2026-04-20_010135_130404_model_testing`

## Results (Public CV)

| model | feature_set_id | mean R² | mean RMSE | mean MAE | source run |
|---|---|---:|---:|---:|---|
| ridge | `lambda_policies_plus_sentence_bundle` | 0.4229 | 0.2647 | 0.1774 | `2026-04-19_195413_226932_model_testing` |
| xgb3_regressor | `lambda_policies_plus_sentence_bundle` | 0.4202 | 0.2656 | 0.1734 | `2026-04-19_195413_226932_model_testing` |
| mlp_regressor (native multi-output) | `lambda_policies_plus_sentence_bundle` | 0.4169 | 0.2660 | 0.1710 | `2026-04-20_010135_130404_model_testing` |
| ridge | `hq_plus_sentence_bundle` | 0.4198 | 0.2657 | 0.1812 | `2026-04-19_195413_226932_model_testing` |
| xgb3_regressor | `hq_plus_sentence_bundle` | 0.4147 | 0.2670 | 0.1758 | `2026-04-19_195413_226932_model_testing` |
| mlp_regressor (native multi-output) | `hq_plus_sentence_bundle` | 0.3502 | 0.2803 | 0.1914 | `2026-04-20_010135_130404_model_testing` |

## Notes

- This report intentionally uses the corrected MLP run where multi-output uses native MLP fitting.
- The updated MLP `lambda_plus_bundle` R² (`0.4169`) matches the prior MLP size investigation expectation (`~0.4169`) within rounding tolerance.
- On these settings, `lambda_policies_plus_sentence_bundle` remains stronger than `hq_plus_sentence_bundle` across all three models.

## Cross-Model Pattern Synthesis

- Mean performance across targets (`lambda+bundle`) is very close for all three models:
  - `ridge`: R² `0.4229`
  - `xgb3_regressor`: R² `0.4202`
  - `mlp_regressor` (native multi-output): R² `0.4172`
- On `hq+bundle`, the spread is larger because MLP drops more:
  - `ridge`: R² `0.4198`
  - `xgb3_regressor`: R² `0.4147`
  - `mlp_regressor`: R² `0.3505`
- Feature-type effect (`lambda+bundle` minus `hq+bundle`, mean R² delta):
  - `ridge`: `+0.0031`
  - `xgb3_regressor`: `+0.0055`
  - `mlp_regressor`: `+0.0666`
- This means lambda contributes modestly for linear/tree models, but materially for MLP under this setup.
- Target-level winners on `lambda+bundle` are mixed:
  - `ridge` best on 6/16 targets
  - `xgb3_regressor` best on 5/16
  - `mlp_regressor` best on 5/16
- Largest winner margins are concentrated on a few targets (e.g., `v25_p157`, `v25_p112`, `v25_p143`), while several targets are near-ties (`v25_p121`, `v25_p161`, `v25_p80`, `v25_p150`), indicating meaningful heterogeneity by target.

## Per-Target Metrics (x16 by prediction target)

- `ridge` and `xgb3_regressor` rows come from run `2026-04-19_195413_226932_model_testing`.
- `mlp_regressor` rows come from corrected native multi-output run `2026-04-20_010135_130404_model_testing`.

### Lambda+bundle per-target comparison (all three models)
| target_id | ridge_r2 | xgb3_r2 | mlp_r2 | ridge_rmse | xgb3_rmse | mlp_rmse | ridge_mae | xgb3_mae | mlp_mae | best_model_by_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v25_p1 | 0.3710 | 0.3509 | 0.3770 | 0.3345 | 0.3398 | 0.3329 | 0.2550 | 0.2591 | 0.2477 | mlp |
| v25_p11 | 0.3140 | 0.3088 | 0.3262 | 0.2492 | 0.2501 | 0.2470 | 0.1518 | 0.1480 | 0.1398 | mlp |
| v25_p112 | 0.5022 | 0.4695 | 0.4766 | 0.3370 | 0.3479 | 0.3456 | 0.2621 | 0.2730 | 0.2665 | ridge |
| v25_p116 | 0.5006 | 0.5125 | 0.5023 | 0.3142 | 0.3105 | 0.3137 | 0.2272 | 0.2176 | 0.2145 | xgb3 |
| v25_p121 | 0.3530 | 0.3626 | 0.3633 | 0.1704 | 0.1691 | 0.1690 | 0.0788 | 0.0738 | 0.0743 | mlp |
| v25_p135 | 0.3946 | 0.3937 | 0.3983 | 0.3081 | 0.3084 | 0.3072 | 0.2288 | 0.2251 | 0.2133 | mlp |
| v25_p143 | 0.3652 | 0.3901 | 0.3336 | 0.1357 | 0.1331 | 0.1391 | 0.0533 | 0.0429 | 0.0568 | xgb3 |
| v25_p150 | 0.4709 | 0.4677 | 0.4539 | 0.3356 | 0.3367 | 0.3410 | 0.2659 | 0.2668 | 0.2655 | ridge |
| v25_p157 | 0.4885 | 0.4299 | 0.4339 | 0.2225 | 0.2349 | 0.2341 | 0.1259 | 0.1263 | 0.1396 | ridge |
| v25_p161 | 0.2670 | 0.2838 | 0.2810 | 0.2082 | 0.2058 | 0.2062 | 0.1074 | 0.0968 | 0.0974 | xgb3 |
| v25_p38 | 0.4366 | 0.4236 | 0.4103 | 0.1726 | 0.1746 | 0.1766 | 0.0790 | 0.0738 | 0.0814 | ridge |
| v25_p52 | 0.4947 | 0.4819 | 0.4859 | 0.3417 | 0.3460 | 0.3447 | 0.2628 | 0.2667 | 0.2583 | ridge |
| v25_p55 | 0.4844 | 0.5072 | 0.4850 | 0.3001 | 0.2934 | 0.2999 | 0.2181 | 0.2068 | 0.2013 | xgb3 |
| v25_p58 | 0.5080 | 0.5374 | 0.5250 | 0.2687 | 0.2606 | 0.2641 | 0.1701 | 0.1584 | 0.1521 | xgb3 |
| v25_p72 | 0.3108 | 0.3025 | 0.3282 | 0.2601 | 0.2617 | 0.2568 | 0.1661 | 0.1614 | 0.1507 | mlp |
| v25_p80 | 0.5044 | 0.5015 | 0.4945 | 0.2761 | 0.2769 | 0.2789 | 0.1861 | 0.1778 | 0.1765 | ridge |

### mlp_regressor | hq_plus_sentence_bundle
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

### mlp_regressor | lambda_policies_plus_sentence_bundle
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

### ridge | hq_plus_sentence_bundle
| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.3733 | 0.3339 | 0.2561 |
| v25_p11 | 0.2906 | 0.2534 | 0.1557 |
| v25_p112 | 0.5120 | 0.3337 | 0.2608 |
| v25_p116 | 0.4785 | 0.3211 | 0.2416 |
| v25_p121 | 0.3617 | 0.1692 | 0.0805 |
| v25_p135 | 0.4036 | 0.3058 | 0.2303 |
| v25_p143 | 0.3787 | 0.1343 | 0.0524 |
| v25_p150 | 0.4868 | 0.3306 | 0.2624 |
| v25_p157 | 0.5010 | 0.2198 | 0.1241 |
| v25_p161 | 0.2748 | 0.2071 | 0.1064 |
| v25_p38 | 0.4354 | 0.1728 | 0.0790 |
| v25_p52 | 0.4829 | 0.3456 | 0.2730 |
| v25_p55 | 0.4779 | 0.3020 | 0.2243 |
| v25_p58 | 0.4492 | 0.2843 | 0.1955 |
| v25_p72 | 0.3025 | 0.2617 | 0.1688 |
| v25_p80 | 0.5079 | 0.2752 | 0.1887 |

### ridge | lambda_policies_plus_sentence_bundle
| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.3710 | 0.3345 | 0.2550 |
| v25_p11 | 0.3140 | 0.2492 | 0.1518 |
| v25_p112 | 0.5022 | 0.3370 | 0.2621 |
| v25_p116 | 0.5006 | 0.3142 | 0.2272 |
| v25_p121 | 0.3530 | 0.1704 | 0.0788 |
| v25_p135 | 0.3946 | 0.3081 | 0.2288 |
| v25_p143 | 0.3652 | 0.1357 | 0.0533 |
| v25_p150 | 0.4709 | 0.3356 | 0.2659 |
| v25_p157 | 0.4885 | 0.2225 | 0.1259 |
| v25_p161 | 0.2670 | 0.2082 | 0.1074 |
| v25_p38 | 0.4366 | 0.1726 | 0.0790 |
| v25_p52 | 0.4947 | 0.3417 | 0.2628 |
| v25_p55 | 0.4844 | 0.3001 | 0.2181 |
| v25_p58 | 0.5080 | 0.2687 | 0.1701 |
| v25_p72 | 0.3108 | 0.2601 | 0.1661 |
| v25_p80 | 0.5044 | 0.2761 | 0.1861 |

### xgb3_regressor | hq_plus_sentence_bundle
| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.3508 | 0.3398 | 0.2588 |
| v25_p11 | 0.2808 | 0.2551 | 0.1504 |
| v25_p112 | 0.4746 | 0.3462 | 0.2694 |
| v25_p116 | 0.4914 | 0.3171 | 0.2307 |
| v25_p121 | 0.3623 | 0.1691 | 0.0738 |
| v25_p135 | 0.3930 | 0.3086 | 0.2269 |
| v25_p143 | 0.3988 | 0.1321 | 0.0426 |
| v25_p150 | 0.4860 | 0.3308 | 0.2616 |
| v25_p157 | 0.4396 | 0.2329 | 0.1239 |
| v25_p161 | 0.2712 | 0.2076 | 0.0978 |
| v25_p38 | 0.4299 | 0.1736 | 0.0716 |
| v25_p52 | 0.4737 | 0.3487 | 0.2729 |
| v25_p55 | 0.5003 | 0.2954 | 0.2118 |
| v25_p58 | 0.4853 | 0.2749 | 0.1794 |
| v25_p72 | 0.3000 | 0.2622 | 0.1613 |
| v25_p80 | 0.4980 | 0.2779 | 0.1795 |

### xgb3_regressor | lambda_policies_plus_sentence_bundle
| target_id | r2 | rmse | mae |
|---|---:|---:|---:|
| v25_p1 | 0.3509 | 0.3398 | 0.2591 |
| v25_p11 | 0.3088 | 0.2501 | 0.1480 |
| v25_p112 | 0.4695 | 0.3479 | 0.2730 |
| v25_p116 | 0.5125 | 0.3105 | 0.2176 |
| v25_p121 | 0.3626 | 0.1691 | 0.0738 |
| v25_p135 | 0.3937 | 0.3084 | 0.2251 |
| v25_p143 | 0.3901 | 0.1331 | 0.0429 |
| v25_p150 | 0.4677 | 0.3367 | 0.2668 |
| v25_p157 | 0.4299 | 0.2349 | 0.1263 |
| v25_p161 | 0.2838 | 0.2058 | 0.0968 |
| v25_p38 | 0.4236 | 0.1746 | 0.0738 |
| v25_p52 | 0.4819 | 0.3460 | 0.2667 |
| v25_p55 | 0.5072 | 0.2934 | 0.2068 |
| v25_p58 | 0.5374 | 0.2606 | 0.1584 |
| v25_p72 | 0.3025 | 0.2617 | 0.1614 |
| v25_p80 | 0.5015 | 0.2769 | 0.1778 |
