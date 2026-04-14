# Lambda Policy Features

172 binary features derived from LLM-generated investment policies via
**policy decomposition into Python lambda expressions**. Unlike the
Policy Induction v25 features (which are continuous Gemini logprob
scores), these are deterministic Python rules like:

```python
lambda founder: any("ceo" in j.get("role", "").lower()
                    for j in founder.get("jobs", []))
```

Each lambda was authored by an LLM (GPT-4.1, Gemini 2.5 Pro/Flash,
o3-mini) when asked to "decompose this high-level policy into concrete
testable conditions." The resulting rules are evaluated deterministically
at inference time — no LLM calls needed once the rules are defined.

## What's in here

| File | Shape | Description |
|------|-------|-------------|
| `features.csv` | (172, 11) | Metadata: feature_id, fire rate, univariate F0.5, L1 rank |
| `predictions_train.csv` | (4500, 173) | `founder_uuid` + 172 binary columns, canonical train order |
| `predictions_test.csv` | (4500, 173) | Same, for the test set |

## The 172 features

Generated from 6 cache files in
`experiments/autoresearch_vcbench/cache/`, which together contain 218
raw lambda rules. After `FeatureEvaluator` dedupes by column name, 172
unique features remain. All features are prefixed with `lam_` in this
repository to distinguish them from the `v25_` Gemini policies and the
`le_` LLM-engineered features.

The 6 source files:

| Cache file | # rules | LLM | Strategy |
|------------|---------|-----|----------|
| `policy_decomposed_v1.json` | 24 | GPT-4.1 | First decomposition pass |
| `policy_decomposed_gpt-4_1.json` | 24 | GPT-4.1 | Second decomposition with refined prompt |
| `policy_decomposed_gemini-2_5-flash.json` | 26 | Gemini 2.5 Flash | Alternative model |
| `targeted_o3-mini.json` | 46 | o3-mini | Targeted conjunctions (exit × education, etc.) |
| `targeted_gemini-2_5-pro.json` | 49 | Gemini 2.5 Pro | Targeted conjunctions |
| `targeted_gpt-4_1.json` | 49 | GPT-4.1 | Targeted conjunctions |

See the
[Policy Decomposition report](../reference_docs/POLICY_DECOMPOSITION_REPORT.pdf)
for the full pipeline: cognitive mode prompts, L1 auto-selection, and
the subsequent refinement loop. (If you want the autoresearch branch
commit history too, that's in the main think-reason-learn repo under
`experiments/autoresearch_vcbench/`.)

## L1 rank column

`features.csv` includes an `l1_rank_hq_based` column which gives each
feature a rank (1, 2, 3, ...) indicating its strength when L1
LogisticRegression is fit on the 28 HQ features + 172 lambda features
jointly on the training set (C=0.05, standardised, class-balanced).

- 85 features have a valid rank (L1 survivors)
- 87 features have `l1_rank_hq_based = NaN` (L1 zeroed them out)

This matches the ranking used in the autoresearch report's
"HQ + top-K lambda policy" experiments. The same ranking is used for
the LLM-Eng variants in the reproduction guide, so the "top-30" and
"top-40" subsets are consistent across contexts.

**Note**: the L1 ranking depends on what you condition on. If you
condition on a different base (LLM-Eng only, or LLM-Eng + HQ), you
would get a slightly different ranking. The HQ-based ranking stored
here is the one used in the autoresearch paper.

## Performance

As an ensemble (on their own) the 172 lambda features achieve roughly:

- **Test F0.5 ≈ 0.273** (XGB d=3 + rule override, no HQ, no LLM-Eng)

When combined with HQ (top-40 L1 subset, reported in autoresearch paper):

- **Test F0.5 ≈ 0.293** (+0.019 over HQ alone)

When combined with LLM-Engineering features (top-30 or top-40):

- **Test F0.5 ≈ 0.283** — **no improvement** over LLM-Eng alone (0.284).
  The lambda features are largely redundant with LLM-Engineering because
  both capture observable profile patterns. See the main reproduction
  guide for analysis.

See `../REPRODUCING_HEADLINE_SCORES.md` for all 9 headline experiments
and their reproduction.

## Important caveats

- **Binary features**: all 172 are 0/1. If you standardise continuous
  features before a linear model, leave these alone.
- **NOT in the semantic similarity index** yet. If you want to search
  for a lambda rule by text, grep `features.csv` directly for the
  `text` column.
- **L1 rank is conditional on HQ**. If you combine with a different
  base feature set (LLM-Eng, etc.) and want "the best top-K for that
  base", re-run L1 on your base + lambda training features. See the
  reproduction guide for the exact procedure.
- **Selection bias warning**: the autoresearch "HQ + top-40 lambda"
  headline of 0.293 was picked from ~8 K values after seeing test set
  performance. The difference between K=30, 40, 50, 80, and "all 172"
  on test set is within statistical noise (see leakage audit in the
  reproduction guide).
