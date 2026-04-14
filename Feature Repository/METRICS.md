# Feature Repository Metrics

## Per-source summary

| Source | # Features | LLM | Train | Test | Univariate F0.5 (range) | Best ensemble result |
|--------|-----------:|-----|:-----:|:----:|------------------------:|---------------------:|
| Policy Induction v25 | 16 | gemini-2.5-flash | yes | yes | 0.151 – 0.261 | **CV F0.5 = 0.294** (16-policy LR ensemble, OOF) |
| Taste Policy Decomposition | 20 | gemini-2.0-flash | yes | yes | 0.030 – 0.205 | CV F0.5 = 0.308 (20-policy LR ensemble) |
| RRF Curated | 20 | gpt-4o-mini | yes | NaN | 0.107 – 0.191 | (per-question only; not aggregated in this repo) |
| Joel HQ Features | 28 | (human-engineered) | yes | yes | — | **Test F0.5 = 0.274** (XGB d=1 ensemble + rule override) |

The HQ features are included as a strong baseline for comparison. They are not
LLM-derived but they are the established benchmark on VCBench.

## Top picks

### Flagship: Policy Induction v25 (16 policies)

This is the most thoroughly validated LLM-derived feature set in the
repository. The 16 policies were selected from a larger pool of 200+
candidates as the best-performing ensemble. Each policy was scored against
each founder using next-token logprobs over True/False, producing a
probability score in [0, 1].

Ensemble F0.5 (5-fold CV with nested L2 regularisation, OOF threshold tuning):
**0.294** ± 0.019.

| Feature ID | Coef | Univariate F0.5 | Short name |
|------------|-----:|----------------:|------------|
| v25_p150 | +0.232 | 0.180 | Macro-market tailwinds |
| v25_p72 | +0.180 | 0.261 | Small biz founding history (neg) |
| v25_p112 | +0.125 | 0.149 | Sales-only at large cos (neg) |
| v25_p157 | +0.122 | 0.194 | VC/PE investor perspective |
| v25_p80 | +0.116 | 0.202 | Gen. mgmt without tech (neg) |
| v25_p11 | +0.108 | 0.252 | Long careers small firms (neg) |
| v25_p135 | +0.088 | 0.189 | Clinical healthcare + academic net |
| v25_p55 | +0.082 | 0.197 | Adjacent service leadership (neg) |
| v25_p143 | +0.057 | 0.232 | Advanced STEM + R&D leadership |
| v25_p1 | +0.040 | 0.161 | Cross-industry tech transfer |
| v25_p58 | +0.038 | 0.183 | Deep exec + prior exit |
| v25_p161 | +0.037 | 0.211 | 2yr+ tech in growth markets |
| v25_p52 | +0.019 | 0.151 | Lack non-founding experience (neg) |
| v25_p38 | -0.007 | 0.179 | Prior exit/IPO predictor |
| v25_p116 | -0.017 | 0.157 | Domain depth vs scattered career |
| v25_p121 | -0.035 | 0.244 | Strong network (2000+ LinkedIn) |

Notable patterns:
- **Negatively-framed policies dominate** the top coefficients (P72, P11, P80,
  P112, P55). "Red flag" detectors give sharper score distributions than
  positive-framing policies.
- **P150 (macro-market tailwinds)** has the highest coefficient despite being
  a "partial proxy" — it references data not in the founder profiles, but
  Gemini successfully infers market quality from the founder's industry,
  timing, and competitive positioning.
- **P38 (prior exit predictor)** has near-zero coefficient in the ensemble
  despite being correlated with success. This is because `exit_count` is
  already captured by other features and the rule override.

### Taste Policy Decomposition (20 policies)

20 hand-crafted policies covering accept signals (A1-A8), reject signals
(R1-R7), and meta signals (M1-M5). Scored binary YES/NO via Gemini 2.0 Flash.

Ensemble F0.5 (10-fold CV with L2 LogReg): **0.308** ± 0.044.

Top 10 features by univariate F0.5:

| Feature ID | F0.5 | Fire rate | Lift |
|------------|-----:|----------:|-----:|
| taste_M1_elite_university | 0.205 | 12.1% | 2.16 |
| taste_A1_prior_acquisition | 0.182 | 2.9% | 2.86 |
| taste_A7_vc_investor_experience | 0.176 | 7.2% | 2.06 |
| taste_A3_top_phd | 0.165 | 4.3% | 2.22 |
| taste_A6_clevel_mid_large_co | 0.163 | 18.6% | 1.62 |
| taste_A5_deep_domain_expertise | 0.142 | 25.1% | 1.38 |
| taste_M4_tech_industry_startup | 0.138 | 50.4% | 1.28 |
| taste_M5_very_large_company_exp | 0.137 | 47.4% | 1.27 |
| taste_A2_senior_role_large_co | 0.129 | 49.5% | 1.20 |
| taste_M3_fast_career_velocity | 0.125 | 30.9% | 1.19 |

Notable: **taste_A8_prior_ipo** has lift 4.76 (the highest in the set) but
fire rate of only 0.5%, so it has low recall and low standalone F0.5. It
contributes meaningful unique signal in an ensemble.

### RRF Curated (20 questions)

20 binary YES/NO questions selected from 40 hand-curated candidates after
semantic dedup and screening. Scored via GPT-4o-mini. Metrics computed on a
~500-founder screening sample.

Top 10 questions by F-beta (0.5):

| Feature ID | F-beta | Precision | Recall | Question |
|------------|-------:|----------:|-------:|----------|
| rrf_Q000 | 0.191 | 0.178 | 0.269 | Has the founder graduated from an institution ranked in the QS top 10? |
| rrf_Q024 | 0.177 | 0.164 | 0.254 | Did the founder earn an advanced degree (MD, MBA, PhD) from an institution ranked in the top 50 by QS? |
| rrf_Q008 | 0.173 | 0.173 | 0.170 | Does the founder have experience in venture capital or private equity? |
| rrf_Q034 | 0.170 | 0.176 | 0.151 | Does the founder have experience in the Biotechnology sector? |
| rrf_Q025 | 0.166 | 0.170 | 0.151 | Has the founder worked in the VC or PE industry? |
| rrf_Q036 | 0.152 | 0.131 | 0.449 | Is the founder's undergraduate degree related to their current industry? |
| rrf_Q029 | 0.142 | 0.119 | 0.575 | Does the founder have a degree in a STEM field? |
| rrf_Q023 | 0.139 | 0.118 | 0.509 | Has the founder experienced rapid career progression within their roles? |
| rrf_Q030 | 0.135 | 0.116 | 0.398 | Does the founder possess a technical degree related to Engineering or Computer Science? |
| rrf_Q031 | 0.133 | 0.122 | 0.215 | Did the founder hold a technical position for over 6 years? |

## Reproducing the headline numbers

The `build.py --step verify` command runs sanity checks that reproduce two
key numbers from the source experiments:

1. **v25 CV F0.5 ≈ 0.294**: load `policies/predictions_train.csv`, fit an L2
   LogReg on the 16 v25 columns with C=0.005, evaluate via 5-fold OOF with
   threshold sweep for F0.5.

2. **HQ test F0.5 ≈ 0.274**: load `hq_baseline/features_*.csv`, fit Joel's
   exact XGBoost config (d=1, n=227), apply the `exit_count > 0` rule
   override, evaluate on 3 test folds of 1500.

If either of these is off by more than 0.030, the build fails.
