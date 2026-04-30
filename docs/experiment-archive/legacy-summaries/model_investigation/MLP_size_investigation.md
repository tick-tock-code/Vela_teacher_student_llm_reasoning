# MLP Size Investigation

## Scope

- Date: 2026-04-18
- Feature set: `lambda_policies_plus_sentence_bundle`
- Script entrypoint: `scripts/sweep_mlp_hidden_size_holdout.py`
- Script mode used for final decisions: stratified CV (`--cv-splits 3`)
- Fixed optimizer settings across runs: `alpha=0.1`, `learning_rate_init=1e-3`, `max_iter=1000`, `tol=1e-3`, `n_iter_no_change=20`, `early_stopping=False`

## Executive Conclusion

Set MLP hidden layer size default to `(128,)` for the v25 reconstruction path.

Evidence from the 3-fold CV runs:
- Best mean R2 is `(128,)`: `0.416870 +/- 0.001237`
- `(64,)` is second: `0.408639 +/- 0.002175`
- Wider single layers do not improve quality:
  - `(256,)`: `0.407769 +/- 0.003954`
  - `(512,)`: `0.399645 +/- 0.005399`
- Deeper multi-layer variants tested also did not beat `(128,)`

## Run 1: Initial Holdout Sweep (v25 + taste, 9 sizes)

Artifact:
- `tmp/benchmarks/mlp_hidden_size_holdout_20260418_203212.csv`

Purpose:
- Quick screening across both families before converting the script to CV.

Key observations:
- v25 best holdout R2: `(128,)` with `0.398925`
- taste best holdout F0.5: `(16,)` with `0.804227`
- Classification was much slower than v25 regression on the same feature set:
  - `(32,)`: taste `50.153s` vs v25 `1.254s`
  - `(128,)`: taste `96.382s` vs v25 `2.691s`

## Run 2: 3-Fold CV Sweep (v25, 12 sizes including deeper models)

Artifact:
- `tmp/benchmarks/mlp_hidden_size_cv_sweep_20260418_204139.csv`

Sizes tested:
- `(2,)`, `(4,)`, `(8,)`, `(16,)`, `(32,)`, `(64,)`, `(128,)`, `(32,4)`, `(64,16)`, `(128,32)`, `(128,64,32)`, `(128,64,32,16)`

Ranking by R2 mean:

| rank | hidden_layer_sizes | r2_mean | r2_std | elapsed_seconds |
|---:|---|---:|---:|---:|
| 1 | (128,) | 0.416870 | 0.001237 | 6.258 |
| 2 | (64,) | 0.408639 | 0.002175 | 4.083 |
| 3 | (128, 64, 32) | 0.406049 | 0.006789 | 7.190 |
| 4 | (128, 32) | 0.400285 | 0.003715 | 6.639 |
| 5 | (32,) | 0.392200 | 0.002347 | 2.990 |
| 6 | (128, 64, 32, 16) | 0.363503 | 0.008087 | 7.628 |
| 7 | (64, 16) | 0.354083 | 0.004152 | 4.010 |
| 8 | (16,) | 0.317861 | 0.054711 | 2.761 |
| 9 | (8,) | 0.298272 | 0.026874 | 2.335 |
| 10 | (4,) | 0.205884 | 0.057040 | 3.122 |
| 11 | (32, 4) | 0.181988 | 0.004898 | 5.259 |
| 12 | (2,) | 0.097065 | 0.097417 | 4.459 |

Finding:
- Additional depth did not improve v25 performance over `(128,)`.

## Run 3: 3-Fold CV Sweep (v25, single-layer only + 256/512)

Artifact:
- `tmp/benchmarks/mlp_hidden_size_cv_sweep_20260418_204733.csv`

Sizes tested:
- `(2,)`, `(4,)`, `(8,)`, `(16,)`, `(32,)`, `(64,)`, `(128,)`, `(256,)`, `(512,)`

Ranking by R2 mean:

| rank | hidden_layer_sizes | r2_mean | r2_std | elapsed_seconds |
|---:|---|---:|---:|---:|
| 1 | (128,) | 0.416870 | 0.001237 | 6.818 |
| 2 | (64,) | 0.408639 | 0.002175 | 4.679 |
| 3 | (256,) | 0.407769 | 0.003954 | 33.988 |
| 4 | (512,) | 0.399645 | 0.005399 | 87.259 |
| 5 | (32,) | 0.392200 | 0.002347 | 3.981 |
| 6 | (16,) | 0.317861 | 0.054711 | 3.592 |
| 7 | (8,) | 0.298272 | 0.026874 | 2.862 |
| 8 | (4,) | 0.205884 | 0.057040 | 2.670 |
| 9 | (2,) | 0.097065 | 0.097417 | 3.354 |

Finding:
- Width beyond 128 did not help quality, and cost increased sharply.

## Recommendation for Main Pipeline

For v25 MLP defaults, use:
- `hidden_layer_sizes=(128,)`

Reasoning:
- Best observed v25 CV quality.
- Stable across repeated sweeps.
- Better quality/time tradeoff than `(256,)` and `(512,)`.
