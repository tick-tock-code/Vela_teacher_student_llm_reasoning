# HQ Baseline Features (28 human-engineered)

Joel Beazley's hand-crafted feature set. Included here as a strong baseline
for comparison: it represents what a careful human engineer can build from
the structured founder profiles.

These are **NOT LLM-derived**. They come from `experiments/joel_hq_features/
hq_features.py` and are computed deterministically from the raw VCBench CSVs.

## What's in here

| File | Shape | Description |
|------|-------|-------------|
| `features_train.csv` | (4500, 30) | `founder_uuid`, `success`, + 28 features |
| `features_test.csv` | (4500, 30) | Same shape, for the test set |

## The 28 features by tier

The features are organised into four tiers plus a v2 interaction layer.

**Tier 1 — Direct exit signals (3):**
- `has_prior_ipo`, `has_prior_acquisition`, `exit_count`

**Tier 2 — Sacrifice signal (5):** signals that the founder gave up a stable
high-status role to start something:
- `max_company_size_before_founding`
- `prestige_sacrifice_score` (max company size × max seniority before founding)
- `years_in_large_company`
- `comfort_index` (weighted experience in stable comfort industries)
- `founding_timing` (total pre-founding experience)

**Tier 3 — Education × QS interaction (6):**
- `edu_prestige_tier` (4-tier QS ranking encoding)
- `field_relevance_score` (1-5 score for field × startup industry match)
- `prestige_x_relevance` (interaction)
- `degree_level` (PhD/MD = 4, MBA/JD = 3, MS = 2, BS = 1)
- `stem_flag`
- `best_degree_prestige`

**Tier 4 — Career trajectory (9):**
- `max_seniority_reached`
- `seniority_is_monotone` (always climbing)
- `company_size_is_growing`
- `restlessness_score` (count of jobs < 2 years)
- `founding_role_count`
- `longest_founding_tenure`
- `industry_pivot_count` (number of distinct industries)
- `industry_alignment` (whether prior job matches startup industry)
- `total_inferred_experience`

**v2 — Interaction features (5):**
- `is_serial_founder` (founding_role_count >= 2)
- `exit_x_serial`
- `sacrifice_x_serial`
- `industry_prestige_penalty` (high education × biotech/VC/PE startup)
- `persistence_score` (longest founding tenure / total experience)

## Headline result

With Joel's tuned XGBoost (depth=1, n_estimators=227) and the `exit_count > 0`
rule override, this 28-feature set achieves:

- **CV F0.5**: 0.233 (5-fold stratified, OOF threshold tuning)
- **Test F0.5**: **0.274** (mean over 3 test folds of 1500)

This is the human-engineered benchmark that LLM-derived feature combinations
need to beat. See [`../reference_docs/POLICY_DECOMPOSITION_REPORT.pdf`](../reference_docs/POLICY_DECOMPOSITION_REPORT.pdf)
for analysis of when LLM features improve on it.

## Rule override

The HQ pipeline uses a deterministic rule: founders with `exit_count > 0`
are forced to predict positive (probability = 1.0). This reflects strong
domain knowledge that prior exits are the single strongest success signal.
The rule override is responsible for ~0.02 of the F0.5.

If you build a model that uses the HQ features (or any features that don't
already capture exit history strongly), consider applying the same override.

## Reproducing the test F0.5

```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("features_train.csv")
test = pd.read_csv("features_test.csv")

HQ = [c for c in train.columns if c not in ("founder_uuid", "success")]
X_tr = train[HQ].fillna(0).values
y_tr = train["success"].values
X_te = test[HQ].fillna(0).values
y_te = test["success"].values
ex_tr = train["exit_count"].values
ex_te = test["exit_count"].values

# Joel's exact XGB config
def xgb():
    return XGBClassifier(
        n_estimators=227, max_depth=1, learning_rate=0.0674,
        subsample=0.949, colsample_bytree=0.413, scale_pos_weight=10,
        min_child_weight=14, gamma=4.19, reg_alpha=0.73, reg_lambda=15.0,
        eval_metric="logloss", random_state=42, n_jobs=1,
    )

# OOF threshold tuning on train
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.full(len(y_tr), np.nan)
for ti, vi in skf.split(X_tr, y_tr):
    m = xgb(); m.fit(X_tr[ti], y_tr[ti])
    r = m.predict_proba(X_tr[vi])[:, 1]
    r[ex_tr[vi] > 0] = 1.0  # rule override
    oof[vi] = r

best_t, best_f = 0.5, 0.0
for t in np.arange(0.05, 0.95, 0.01):
    f = fbeta_score(y_tr, (oof >= t).astype(int), beta=0.5, zero_division=0)
    if f > best_f:
        best_f, best_t = f, float(t)

# Train final, predict test, evaluate per fold
m = xgb(); m.fit(X_tr, y_tr)
probs = m.predict_proba(X_te)[:, 1]
probs[ex_te > 0] = 1.0
fold_f = []
for i in range(3):
    s, e = i * 1500, (i + 1) * 1500
    fold_f.append(
        fbeta_score(y_te[s:e], (probs[s:e] >= best_t).astype(int), beta=0.5, zero_division=0)
    )
print(f"HQ test F0.5 = {np.mean(fold_f):.4f}")  # ~0.274
```
