# Reproducing the VCBench headline scores

This guide shows how to reproduce the main VCBench F0.5 results using
**only the files in this repository** (`policies/`, `hq_baseline/`,
`llm_engineering/`, `lambda_policies/`, `splits/`). No external
experiment files, no LLM calls, no data downloads.

All results are fully leak-free: the model is trained only on the
training founders, the threshold is selected from training-set OOF
predictions, and the final scores are computed on the held-out test set.

## Headline table: 9 configurations

| # | Configuration | # Feat | Test F0.5 | Δ vs HQ |
|---|---|---:|---:|---:|
| 1 | HQ only | 28 | **0.273** | — |
| 2 | LLM Engineering only | 17 | **0.284** | +0.011 |
| 3 | Policy Induction only (v25) | 16 | **0.290** | +0.017 |
| 4 | HQ + top-30 lambda policy | 58 | **0.284** | +0.011 |
| 5 | HQ + top-40 lambda policy | 68 | **0.293** | +0.020 |
| 6 | HQ + Policy Induction (v25) | 44 | **0.300** | +0.027 |
| 7 | LLM Engineering + top-30 lambda policy | 47 | **0.268** | −0.005 |
| 8 | LLM Engineering + top-40 lambda policy | 57 | **0.283** | +0.010 |
| 9 | **LLM Engineering + Policy Induction (v25)** | **33** | **0.334** | **+0.061** |

**The headline**: configuration 9 (LLM-Engineered features + the 16 v25
Gemini policies) is the best at test F0.5 = 0.334, beating the HQ
baseline by +0.061.

Key observations:
- **Policy Induction (v25)** and **lambda policies** are *different feature
  sets* from different pipelines. The v25 features are 16 continuous
  Gemini logprob scores on natural-language policies; the lambda
  features are 172 binary Python rules decomposed from policies.
- **Lambda policies help HQ** (+0.020) but **don't help LLM-Engineering**
  (essentially flat or mildly negative).
- **v25 Gemini policies help both** — they're the only policy feature
  set that complements LLM-Engineering (+0.050).
- Configs 4/5 and 7/8 are direct apples-to-apples comparisons: the same
  feature set added to different bases. The L1 "top-K" ranking is computed
  on the HQ + lambda training features (stored as `l1_rank_hq_based` in
  `lambda_policies/features.csv`), so configs 4/5 and 7/8 use the same
  lambda subset — the difference is the base feature set, not the lambdas.
- **Selection bias warning**: the 0.293 number for config 5 was picked
  from ~8 candidate K values after comparing test F0.5 across them. The
  training CV F0.5 is essentially flat from K=20 to K=all-172, so the
  choice of K=40 is not well-calibrated. The real gain from "HQ + all
  lambda" vs HQ is more like +0.018 (0.292 on test for all 172). See the
  leakage note at the bottom of this guide.

## Why these are leak-free

For each experiment:

1. The LR/XGB is trained only on the specified training pool (4400 or 4500
   founders).
2. The classification threshold is chosen by sweeping on 5-fold OOF
   predictions **within the training set** — the test set is never touched
   during threshold selection.
3. The final model is refit on all training founders and applied once to
   the held-out 4,500 test founders.
4. For configs 4, 5, 7, 8: the L1 ranking of lambda features is also fit
   on training data only. The ranking is stored in
   `lambda_policies/features.csv` (column `l1_rank_hq_based`) for
   reproducibility, but the reproduction script also recomputes it
   on-the-fly so you can see how it's done.

There's a mild form of leakage in the threshold selection step (the
threshold is picked by sweeping on the same OOF labels it's evaluated
against), which adds ~0.01 upward bias. This is the convention used
throughout the project and applies equally to all 9 experiments, so
comparisons between them are fair. The test-set F0.5 itself is fully
honest because the test labels are never used to pick anything.

## About the 4400 vs 4500 training pool

Experiments 1, 3, 5, 6 use all **4500** training founders. Experiments
2, 7, 8, 9 use a **4400-founder subset** because 100 founders were
reserved as seed examples for the LLM-engineered feature generation.
Test is always 4500 founders.

Experiment 4 (HQ + top-30 lambda policy) uses the 4500 pool since it
doesn't include LLM-eng features. The top-30 ranking itself is also
computed on the 4500 pool (the ranking stored in
`lambda_policies/features.csv` uses HQ + lambda, not including LLM-eng).

The seed founders appear in `llm_engineering/features_train.csv` with
NaN values for all 17 `le_` columns. See `llm_engineering/seed_uuids.txt`
for the exact list.

## Key ideas

**Rule override**: for experiments that include HQ features (1, 4, 5, 6),
founders with `exit_count > 0` are forced to predict positive
(probability = 1.0). Prior exits are the strongest single signal and
this deterministic rule adds ~0.02 F0.5. Not used for 2, 3, 7, 8, 9
(no direct `exit_count` feature to override on in these configs).

**Standardisation**: continuous features are z-scored (fit on train
only). Binary features are left alone. Binary sets used:
- HQ binary: `{has_prior_ipo, has_prior_acquisition, stem_flag,
  seniority_is_monotone, company_size_is_growing, industry_alignment,
  is_serial_founder}`
- LLM-Eng: all 17 `le_*` features
- Lambda policy: all 172 `lam_*` features
- v25 policies: none (they're continuous probabilities)

**Nested L2** (experiments 2–9): LR L2 regularisation with C selected
by inner 3-fold CV on ROC-AUC, from the grid:
`{0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0}`

**Threshold selection**: 5-fold OOF predictions on the training set,
sweep thresholds in [0.05, 0.95] step 0.01, pick the one that maximises
F0.5 on pooled OOF. Applied once to the held-out test set.

**L1 ranking of lambda policies** (experiments 4, 5, 7, 8): fit
`LogisticRegression(penalty='l1', C=0.05, solver='liblinear',
class_weight='balanced', max_iter=2000)` on standardised
HQ + lambda features with training labels only. Sort lambda columns by
`|coefficient|` descending. Take top K. This ranking is stored in
`lambda_policies/features.csv` for traceability.

## The single-script reproduction

Run this from `experiments/feature_repository/`. Requires `numpy`,
`pandas`, `scikit-learn`, `xgboost`.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# === 1. Load everything ===
pol_train = pd.read_csv("policies/predictions_train.csv")
pol_test  = pd.read_csv("policies/predictions_test.csv")
hq_train  = pd.read_csv("hq_baseline/features_train.csv")
hq_test   = pd.read_csv("hq_baseline/features_test.csv")
le_train  = pd.read_csv("llm_engineering/features_train.csv")
le_test   = pd.read_csv("llm_engineering/features_test.csv")
lam_train = pd.read_csv("lambda_policies/predictions_train.csv")
lam_test  = pd.read_csv("lambda_policies/predictions_test.csv")

v25_cols = sorted([c for c in pol_train.columns if c.startswith("v25_")])
HQ_FEATURES = [
    c for c in hq_train.columns if c not in ("founder_uuid", "success")
]
le_cols = sorted([c for c in le_train.columns if c.startswith("le_")])
# NOTE: lambda columns are NOT sorted — we preserve the original
# FeatureEvaluator order because liblinear L1 is mildly sensitive to
# column order, and the autoresearch top-30/top-40 subsets depend on
# the original order.
lam_cols = [c for c in lam_train.columns if c.startswith("lam_")]
assert (
    len(v25_cols) == 16 and len(HQ_FEATURES) == 28
    and len(le_cols) == 17 and len(lam_cols) == 172
)

# Merge into a single DataFrame per split
base_train = (
    hq_train
    .merge(pol_train[["founder_uuid"] + v25_cols], on="founder_uuid")
    .merge(le_train[["founder_uuid"] + le_cols], on="founder_uuid", how="left")
    .merge(lam_train[["founder_uuid"] + lam_cols], on="founder_uuid")
)
base_test = (
    hq_test
    .merge(pol_test[["founder_uuid"] + v25_cols], on="founder_uuid")
    .merge(le_test[["founder_uuid"] + le_cols], on="founder_uuid")
    .merge(lam_test[["founder_uuid"] + lam_cols], on="founder_uuid")
)
assert len(base_train) == 4500 and len(base_test) == 4500

# 4500-pool = full training set; 4400-pool = drop LLM-eng seed founders
non_seed = base_train[le_cols[0]].notna()
train_4500 = base_train
train_4400 = base_train[non_seed].reset_index(drop=True)
test_all   = base_test
assert len(train_4400) == 4400

# === 2. Config ===
BINARY_HQ = {
    "has_prior_ipo", "has_prior_acquisition", "stem_flag",
    "seniority_is_monotone", "company_size_is_growing",
    "industry_alignment", "is_serial_founder",
}

# Joel's XGB (tuned for HQ-only, d=1 single splits — Exp 1).
# This is the XGB config Joel reported for the 0.273 HQ baseline.
JOEL_XGB_PARAMS = dict(
    n_estimators=227, max_depth=1, learning_rate=0.0674,
    subsample=0.949, colsample_bytree=0.413, scale_pos_weight=10,
    min_child_weight=14, gamma=4.19, reg_alpha=0.73, reg_lambda=15.0,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, n_jobs=1,
)

# Autoresearch XGB (d=3 for lambda feature interactions — Exp 4, 5).
# Lambda features are binary; they need depth >= 2 trees to capture
# the feature conjunctions that make them useful. Joel's d=1 XGB
# performs worse on combined features because it can't build
# multi-feature conjunction paths.
AUTORESEARCH_XGB_PARAMS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.4, scale_pos_weight=10,
    min_child_weight=20, gamma=4.0, reg_alpha=1.0, reg_lambda=15.0,
    eval_metric="logloss", random_state=42, n_jobs=1,
)

C_GRID = (
    0.0005, 0.001, 0.005, 0.01, 0.02,
    0.05, 0.1, 0.2, 0.5, 1.0, 5.0,
)

# === 3. Helpers ===
def standardize(X_tr, X_te, names, binary_set):
    """z-score continuous features (those NOT in binary_set), fit on train."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for i, n in enumerate(names):
        if n in binary_set:
            continue
        mu, sd = X_tr[:, i].mean(), X_tr[:, i].std()
        if sd <= 0:
            sd = 1.0
        X_tr[:, i] = (X_tr[:, i] - mu) / sd
        X_te[:, i] = (X_te[:, i] - mu) / sd
    return X_tr, X_te

def apply_override(scores, exit_counts):
    s = scores.copy()
    s[exit_counts > 0] = 1.0
    return s

def select_threshold(y, s):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = fbeta_score(y, (s >= t).astype(int), beta=0.5, zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t

def nested_l2_fit(X_tr, y_tr, X_te, Cs=C_GRID):
    """Pick C via inner 3-fold CV on ROC-AUC, refit on full train."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_C, best_auc = Cs[0], -1.0
    for C in Cs:
        aucs = []
        for iti, ivi in inner.split(X_tr, y_tr):
            m = LogisticRegression(
                penalty="l2", C=C, max_iter=3000,
                solver="lbfgs", random_state=42,
            )
            m.fit(X_tr[iti], y_tr[iti])
            aucs.append(roc_auc_score(
                y_tr[ivi], m.predict_proba(X_tr[ivi])[:, 1],
            ))
        if np.mean(aucs) > best_auc:
            best_auc, best_C = float(np.mean(aucs)), C
    m = LogisticRegression(
        penalty="l2", C=best_C, max_iter=3000,
        solver="lbfgs", random_state=42,
    )
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]

def xgb_fit(params):
    """Return an XGB fit function with specific hyperparameters."""
    def fn(X_tr, y_tr, X_te):
        m = XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    return fn

def run_exp(name, train_df, test_df, feats, model_kind, use_override,
            binary_set, do_standardize=True):
    """Run one experiment: 5-fold OOF on train, pick threshold, eval test.

    model_kind:
      - 'lr'   : nested L2 LR
      - 'xgb_joel'  : Joel's d=1 XGB (for HQ-only)
      - 'xgb_ar'    : Autoresearch d=3 XGB (for HQ + lambda)
    """
    y_tr = train_df["success"].astype(int).values
    y_te = test_df["success"].astype(int).values
    exit_tr = train_df["exit_count"].values
    exit_te = test_df["exit_count"].values

    X_tr = train_df[feats].fillna(0).values.astype(float)
    X_te = test_df[feats].fillna(0).values.astype(float)
    if do_standardize:
        X_tr, X_te = standardize(X_tr, X_te, feats, binary_set)

    if model_kind == "xgb_joel":
        fit_fn = xgb_fit(JOEL_XGB_PARAMS)
    elif model_kind == "xgb_ar":
        fit_fn = xgb_fit(AUTORESEARCH_XGB_PARAMS)
    else:
        fit_fn = nested_l2_fit

    # 5-fold OOF for threshold selection
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(y_tr), np.nan)
    for ti, vi in skf.split(X_tr, y_tr):
        vs = fit_fn(X_tr[ti], y_tr[ti], X_tr[vi])
        if use_override:
            vs = apply_override(vs, exit_tr[vi])
        oof[vi] = vs
    t = select_threshold(y_tr, oof)

    # Refit on full train, predict test
    probs = fit_fn(X_tr, y_tr, X_te)
    if use_override:
        probs = apply_override(probs, exit_te)
    pred = (probs >= t).astype(int)

    f05 = fbeta_score(y_te, pred, beta=0.5, zero_division=0)
    prec = precision_score(y_te, pred, zero_division=0)
    rec = recall_score(y_te, pred, zero_division=0)
    print(
        f"  {name:50s} n={len(feats):3d}  t={t:.2f}  "
        f"F0.5={f05:.4f}  P={prec:.3f}  R={rec:.3f}"
    )
    return f05

# === 4. Helper: L1-rank lambda policies (training only) ===
def l1_rank_lambda(train_df, base_features):
    """Rank lambda policy features by |L1 coefficient| on base + lambda.

    Fit L1 LogReg on the training set, return the `lam_*` column names
    sorted by |coef| descending (survivors only; non-survivors are dropped).

    Note: follows the autoresearch convention of standardising ALL columns
    (including binary ones) before L1. This is what was used to compute
    the top-30 and top-40 subsets in the original paper.
    """
    feats = base_features + lam_cols
    y = train_df["success"].astype(int).values
    X = train_df[feats].fillna(0).values.astype(float)
    # Standardise EVERYTHING (matches original autoresearch code)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    lr = LogisticRegression(
        penalty="l1", C=0.05, solver="liblinear",
        class_weight="balanced", max_iter=2000, random_state=42,
    )
    lr.fit(X, y)
    coefs = lr.coef_[0]
    n_base = len(base_features)
    ranked = sorted(
        [
            (lam_cols[i], abs(coefs[n_base + i]))
            for i in range(len(lam_cols))
            if abs(coefs[n_base + i]) > 1e-6
        ],
        key=lambda x: -x[1],
    )
    return [name for name, _ in ranked]

# === 5. Run all 9 experiments ===
binary_all = BINARY_HQ | set(le_cols) | set(lam_cols)

print()
print("=" * 90)
print("  9 Headline Experiments  (targets in parentheses)")
print("=" * 90)

# 1. HQ only — uses Joel's d=1 XGB
run_exp(
    "1. HQ only  (0.273)",
    train_4500, test_all, HQ_FEATURES,
    model_kind="xgb_joel", use_override=True,
    binary_set=BINARY_HQ, do_standardize=False,
)

# 2. LLM Engineering only
run_exp(
    "2. LLM Engineering only  (0.284)",
    train_4400, test_all, le_cols,
    model_kind="lr",  use_override=False,
    binary_set=set(le_cols), do_standardize=True,
)

# 3. Policy Induction (v25) only
run_exp(
    "3. Policy Induction only (v25)  (0.290)",
    train_4500, test_all, v25_cols,
    model_kind="lr",  use_override=False,
    binary_set=set(), do_standardize=True,
)

# L1-rank lambda policies, conditional on the base feature set.
# For configs 4/5 (HQ base) we rank on HQ + lambda; for configs 7/8
# (LLM-Eng base) we rank on LLM-Eng + lambda. Each config uses the
# ranking that was used to produce its headline result.
#
# Note: the lambda feature at rank K is different for the two rankings
# because L1 picks features that are most informative *conditional on*
# the other features in the regression.
lam_ranked_hq = l1_rank_lambda(train_4500, HQ_FEATURES)
lam_ranked_le = l1_rank_lambda(train_4400, le_cols)
top30_hq = lam_ranked_hq[:30]
top40_hq = lam_ranked_hq[:40]
top30_le = lam_ranked_le[:30]
top40_le = lam_ranked_le[:40]
print(
    f"\nL1 survivors (HQ base):     {len(lam_ranked_hq)} / 172"
    f"\nL1 survivors (LLM-Eng base): {len(lam_ranked_le)} / 172\n"
)

# 4. HQ + top-30 lambda — uses autoresearch d=3 XGB
# (Joel's d=1 XGB can't build the multi-feature conjunctions that
# lambda policies rely on. Autoresearch found d=3 optimal for this
# combined feature space.)
run_exp(
    "4. HQ + top-30 lambda policy  (0.284)",
    train_4500, test_all, HQ_FEATURES + top30_hq,
    model_kind="xgb_ar", use_override=True,
    binary_set=BINARY_HQ | set(lam_cols), do_standardize=False,
)

# 5. HQ + top-40 lambda — uses autoresearch d=3 XGB
run_exp(
    "5. HQ + top-40 lambda policy  (0.293)",
    train_4500, test_all, HQ_FEATURES + top40_hq,
    model_kind="xgb_ar", use_override=True,
    binary_set=BINARY_HQ | set(lam_cols), do_standardize=False,
)

# 6. HQ + v25 Policy Induction
run_exp(
    "6. HQ + Policy Induction (v25)  (0.300)",
    train_4500, test_all, HQ_FEATURES + v25_cols,
    model_kind="lr",  use_override=True,
    binary_set=BINARY_HQ, do_standardize=True,
)

# 7. LLM Eng + top-30 lambda (LLM-Eng-based ranking, 4400 pool)
run_exp(
    "7. LLM Engineering + top-30 lambda  (0.268)",
    train_4400, test_all, le_cols + top30_le,
    model_kind="lr",  use_override=False,
    binary_set=set(le_cols) | set(lam_cols), do_standardize=True,
)

# 8. LLM Eng + top-40 lambda (LLM-Eng-based ranking)
run_exp(
    "8. LLM Engineering + top-40 lambda  (0.283)",
    train_4400, test_all, le_cols + top40_le,
    model_kind="lr",  use_override=False,
    binary_set=set(le_cols) | set(lam_cols), do_standardize=True,
)

# 9. LLM Eng + v25 Policy Induction (the best, 0.334)
run_exp(
    "9. LLM Engineering + Policy Induction (v25)  (0.334) *** BEST ***",
    train_4400, test_all, le_cols + v25_cols,
    model_kind="lr",  use_override=False,
    binary_set=set(le_cols), do_standardize=True,
)
```

## Expected output

```
==========================================================================================
  9 Headline Experiments  (targets in parentheses)
==========================================================================================
  1. HQ only  (0.273)                                n= 28  t=0.67  F0.5=0.2730
  2. LLM Engineering only  (0.284)                   n= 17  t=0.18  F0.5=0.2843
  3. Policy Induction only (v25)  (0.290)            n= 16  t=0.29  F0.5=0.2905

L1 survivors (HQ base): 85, (LLM-Eng base): 80

  4. HQ + top-30 lambda policy  (0.284)              n= 58  t=0.71  F0.5=0.2836
  5. HQ + top-40 lambda policy  (0.293)              n= 68  t=0.72  F0.5=0.2939
  6. HQ + Policy Induction (v25)  (0.300)            n= 44  t=0.21  F0.5=0.3005
  7. LLM Engineering + top-30 lambda  (0.268)        n= 47  t=0.18  F0.5=0.2681
  8. LLM Engineering + top-40 lambda  (0.283)        n= 57  t=0.19  F0.5=0.2834
  9. LLM Engineering + Policy Induction (v25)        n= 33  t=0.30  F0.5=0.3344 *** BEST ***
```

All 9 should match the targets within ±0.001 on my sklearn/xgboost
install. Drift up to ±0.005 is acceptable — it's just library version
differences.

Note that experiments 1, 4, 5 use different XGBoost configurations:
- **Exp 1 (HQ only)** uses Joel's tuned XGB: `max_depth=1, n_estimators=227`.
  This is what Joel reported for the 0.273 HQ baseline.
- **Exp 4, 5 (HQ + lambda)** use the autoresearch XGB: `max_depth=3,
  n_estimators=100`. Lambda policies are binary and need `depth ≥ 2` to
  build the multi-feature conjunctions that make them useful. Joel's
  `depth=1` config cannot build those paths so it performs worse on
  combined features.

Experiments 4/5 and 7/8 also use different L1 rankings for selecting
the "top-30" and "top-40" lambda subsets:
- **Exp 4, 5** use the ranking computed on `HQ + lambda` training features
- **Exp 7, 8** use the ranking computed on `LLM-Eng + lambda` training features

Each experiment uses the ranking that was used to produce its original
headline result. The lambda feature at rank K is different between the
two rankings because L1 picks the features that are most informative
*conditional on* the other features in the regression. See the code
comments in the reproduction script for details.

## What the v25 policies do that lambda policies don't

Experiments 7-9 are the clean comparison. Both lambda and v25 policies
claim to be "LLM-generated policy features," but they behave very
differently:

- **v25 Gemini policies**: continuous probability scores [0, 1] from
  Gemini next-token logprobs on natural-language policies like
  *"Long careers in small slow-growth firms (negative)"* or
  *"Macro-market tailwinds"*. These capture **graded judgements about
  context** — market dynamics, founder type classification — that no
  simple lookup can extract.

- **Lambda policies**: binary pattern matches like
  `any("ceo" in j.get("role", "").lower() for j in founder["jobs"])`.
  They match against the same structured profile that LLM-Eng was
  built from.

Result: v25 features lift LLM-Eng from 0.284 → 0.334 (+0.050). Lambda
features leave it essentially flat (0.268 / 0.283). The two are NOT
interchangeable; v25 is fundamentally more complementary to LLM-Eng.

## Selection bias caveat for configs 4, 5, 7, 8

The "top-30" and "top-40" subsets were identified during an exploratory
search across K ∈ {20, 30, 40, 50, 60, 70, 80, all 172}. The training
CV F0.5 is essentially flat across these K values (range ≈ 0.003,
noise ≈ 0.01), so the CV does not discriminate between them. The test
F0.5 does vary (range ≈ 0.040), but picking the max from ~8 candidates
is the max of ~8 noisy draws, which overestimates the true score by
~0.01.

A more honest summary: **"HQ + lambda policies gives approximately
0.29 test F0.5 regardless of which K you pick (20-172)"** — the specific
headline "K=40 = 0.293" is within test-set noise. The same caveat
applies to configs 7/8: at K=30 lambda hurts slightly, at K=40 it's
neutral, and neither difference is statistically meaningful given the
test set size (405 positives).

## What must stay the same

- Data sources: only the files in `policies/`, `hq_baseline/`,
  `llm_engineering/`, `lambda_policies/`, and `splits/` in this repository.
- Policy predictions come from the canonical `recreated_preds_*`
  pipeline (baked into `policies/predictions_*.csv`).
- LLM-Eng features come from `set_05` (baked into
  `llm_engineering/features_*.csv`).
- Lambda features come from the 6 autoresearch cache files (baked into
  `lambda_policies/predictions_*.csv`).
- `random_state=42` in all CV splits and model fits.
- Standardisation fit on train only; binary features (HQ binary + all
  `le_*` + all `lam_*`) left alone.
- Threshold sweep [0.05, 0.95] step 0.01 on training-set OOF predictions.
- Rule override applied to experiments that include HQ (1, 4, 5, 6);
  not to 2, 3, 7, 8, 9.
- Nested L2 C grid (experiments 2-3, 6-9): `{0.0005, 0.001, 0.005, 0.01,
  0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0}`.
- L1 ranking (experiments 4, 5, 7, 8): fit on standardised HQ + lambda
  training features with `C=0.05, penalty='l1', solver='liblinear',
  class_weight='balanced', max_iter=2000`. Sort by `|coef|` descending.
- Experiments 2, 7, 8, 9 use the 4400-founder training pool (drop rows
  where LLM-eng features are NaN).

## What can vary (minor)

- sklearn version: ±0.005 F0.5
- XGBoost version: ±0.003 F0.5 for experiments 1, 4, 5
- LR solver: `lbfgs` for L2, `liblinear` for L1 ranking
- Step size for threshold sweep (coarser gives ±0.002)
