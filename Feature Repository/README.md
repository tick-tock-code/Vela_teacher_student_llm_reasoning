# VCBench Feature Repository

A consolidated, ready-to-use catalogue of LLM-derived features for predicting
founder success on VCBench (4,500 train + 4,500 test founders).

## Why this exists

We have built up many LLM-derived feature sets across different experiments
(Policy Induction, Taste Policy Decomposition, RRF binary questions, etc.).
Each lives in a different `experiments/*/results/` directory in a different
format. If someone has a new feature idea, they typically have to re-run
expensive LLM scoring even if a near-identical feature already exists somewhere
in the repo.

This repository solves that by:

1. **Consolidating** the most valuable LLM-evaluated features into clean,
   standardised CSVs.
2. **Aligning** all features to the same canonical 4,500 train + 4,500 test
   founder UUIDs so they can be combined freely.
3. **Documenting** provenance (LLM, prompt, generation date, performance).
4. **Providing** a semantic similarity lookup tool: paste a description of
   your new feature, get the top-K cached features by similarity. If something
   above your threshold already exists, just use the cached predictions.

## What's inside

| Source | # Features | LLM | Train | Test | Best F0.5 |
|--------|-----------:|-----|:-----:|:----:|----------:|
| Policy Induction v25 | 16 | gemini-2.5-flash | yes | yes | 0.290 (test, LR) |
| Taste Policy Decomposition | 20 | gemini-2.0-flash | yes | yes | 0.308 (CV, ensemble) |
| RRF Curated | 20 | gpt-4o-mini | yes | **NaN** | 0.19 (best single Q) |
| LLM-Engineered (set_05) | 17 | gpt-4.1-nano | 4400* | yes | 0.284 (test, LR) |
| Joel HQ Features (baseline) | 28 | (human-engineered) | yes | yes | 0.274 (test, XGB) |

*LLM-Engineered features reserve 100 training founders as LLM seed examples;
those 100 rows are NaN in `llm_engineering/features_train.csv`.

**Total LLM-derived features**: 73 (36 policies + 20 RRF questions + 17 LLM-engineered).
**Total founders**: 9,000 (4,500 train + 4,500 test, both fully labelled).

**Headline combined result: test F0.5 = 0.334** (LLM-Engineered + Policy
Induction, 33 features, nested L2 LR). Beats the HQ baseline (0.273) by
+0.061 and Policy Induction alone (0.290) by +0.044. See
[REPRODUCING_HEADLINE_SCORES.md](REPRODUCING_HEADLINE_SCORES.md) for the
full 7-experiment reproduction.

The HQ features are included as a strong baseline for comparison; they are
NOT LLM-derived.

## Directory layout

```
feature_repository/
├── README.md               (this file)
├── METRICS.md              per-source performance summary, top picks called out
├── build.py                idempotent script that rebuilds everything from source
├── splits/
│   ├── train_uuids.txt     canonical 4500 train UUIDs (line-aligned with all CSVs)
│   ├── test_uuids.txt      canonical 4500 test UUIDs
│   └── labels.csv          founder_uuid, split, success
├── policies/
│   ├── README.md           explains the two policy sources, their differences
│   ├── policies.csv        36 policies: id, source, text, llm, metrics, coefficient
│   ├── predictions_train.csv  (4500, 37) — founder_uuid + 36 policy columns
│   ├── predictions_test.csv   (4500, 37)
│   └── prompts/            prompt templates used to score each policy
├── rrf_questions/
│   ├── README.md           explains the RRF approach, screening pipeline
│   ├── questions.csv       20 questions: id, text, llm, precision/recall/f-beta
│   ├── predictions_train.csv  (4500, 21) — binary YES/NO
│   ├── predictions_test.csv   (4500, 21) — all NaN (test not cached)
│   └── prompts/
├── hq_baseline/
│   ├── README.md           defines each of the 28 HQ features by tier
│   ├── features_train.csv  (4500, 30) — uuid + success + 28 features
│   └── features_test.csv   (4500, 30)
├── llm_engineering/
│   ├── README.md           17 LLM-engineered features, seed holdout, provenance
│   ├── features.csv        id, name, llm, fire rate, univariate F0.5
│   ├── features_train.csv  (4500, 18) — uuid + 17 features (100 seed rows NaN)
│   ├── features_test.csv   (4500, 18)
│   └── seed_uuids.txt      100 founders reserved as LLM seed examples
├── REPRODUCING_HEADLINE_SCORES.md   how to recreate the 0.273 / 0.290 / 0.334 etc.
└── similarity_lookup/
    ├── README.md           how to use the lookup tool
    ├── lookup.py           CLI + Python API
    ├── embeddings.npy      cached embeddings (73, D)
    ├── embedding_backend.txt   which backend was used
    ├── feature_index.csv   id, source, type, text, row_idx (aligned with embeddings)
    └── tfidf_vocab.json    cached TF-IDF vocabulary (only present if TF-IDF backend)
```

## Is this self-contained?

**Yes**, for the common use cases. If you received this as a zip file, you
can unzip it standalone (no parent repository required) and do the following
without any external setup:

- **Reproduce all 9 headline F0.5 results** from
  [`REPRODUCING_HEADLINE_SCORES.md`](REPRODUCING_HEADLINE_SCORES.md) — the
  single-script reproduction reads only from files inside this bundle.
- **Load any feature file** from `policies/`, `hq_baseline/`,
  `llm_engineering/`, `lambda_policies/`, or `rrf_questions/` into pandas.
- **Run semantic similarity lookup** with `python similarity_lookup/lookup.py
  "your query text"`.
- **Read the policy decomposition report** at
  [`reference_docs/POLICY_DECOMPOSITION_REPORT.pdf`](reference_docs/POLICY_DECOMPOSITION_REPORT.pdf).

Required Python packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`.
Optional: `sentence-transformers` for higher-quality semantic similarity
(the bundle ships with a TF-IDF fallback that works without it).

**What is NOT self-contained**:

- [`build.py`](build.py) — reads raw source data from `.private/` and
  other experiment folders in the full think-reason-learn repo. It's
  included as reference only, you do NOT need to run it to use the
  bundle. The data files it would generate are already present.
- References to `experiments/policy_induction_vcbench/`,
  `experiments/autoresearch_vcbench/`, `experiments/joel_hq_features/`
  etc. in various READMEs are prose references to the source
  experiments that generated the data — they're documentation of
  provenance, not runnable pointers.

## Getting started

### 1. Look up whether your feature idea already exists

```bash
cd experiments/feature_repository

python similarity_lookup/lookup.py "founder has previously been a CEO at a large company"
```

If you find a match above your similarity threshold, you can skip generating
this feature yourself and just use the cached predictions. Read the matching
feature's text to confirm it really captures what you want.

### 2. Load predictions for modelling

```python
import pandas as pd

# Load the policy features
pol_train = pd.read_csv("policies/predictions_train.csv")
pol_test  = pd.read_csv("policies/predictions_test.csv")

# Load the HQ baseline features
hq_train = pd.read_csv("hq_baseline/features_train.csv")
hq_test  = pd.read_csv("hq_baseline/features_test.csv")

# Combine
import functools
train = functools.reduce(
    lambda l, r: l.merge(r, on="founder_uuid"),
    [hq_train, pol_train],
)
# train.shape == (4500, 28 + 36 + 2)  # 28 HQ + 36 policy + uuid + success
```

All CSVs use `founder_uuid` as the key and are line-aligned to
`splits/{train,test}_uuids.txt`.

### 3. Combining with the rule override

The HQ baseline uses a deterministic rule: founders with `exit_count > 0` are
forced to predict positive. This is a strong domain prior that improves test
F0.5 by ~0.02. If you build a model that uses HQ features, consider applying
the same override on inference.

## Recommended workflow when adding a new feature

1. Write a clear, single-sentence description of the feature.
2. Run `python similarity_lookup/lookup.py "<your description>"`.
3. Inspect the top-5 matches. If any are semantically the same as your idea
   AND have similarity above your threshold (0.85 with sentence-transformers,
   0.3 with TF-IDF), use those cached predictions.
4. If nothing matches, generate your feature yourself. Then **contribute back**
   by adding it to a new sub-directory under `feature_repository/` and
   re-running `build.py` to refresh the embeddings.

## Known limitations

- **RRF test predictions are NaN.** The original RRF run only scored the
  training set. Re-running on the test set would require ~90,000 LLM calls
  (4,500 founders × 20 questions). The questions and metrics are still
  documented for reference.
- **v25 policy scores are probabilities, not binary.** They come from
  next-token logprobs over True/False. Apply your own threshold if you need
  binary predictions.
- **Taste policy scores are binary.** Convert to 0/1 (already done in
  `predictions_*.csv`).
- **TF-IDF vs sentence-transformers.** The embeddings were built with TF-IDF
  by default (sentence-transformers is not in the project dependencies).
  TF-IDF gives weaker semantic matching: the recommended threshold is ~0.3
  rather than 0.85. To use sentence-transformers instead, install it (`pip
  install sentence-transformers`) and re-run `python build.py --step
  similarity` — the script will detect the install and use the better backend.

## Reproducibility

The entire repository is built deterministically from source files in the
parent repo. To rebuild from scratch:

```bash
poetry run python experiments/feature_repository/build.py
```

This will:
1. Extract canonical UUIDs from `.private/vcbench_final_*.csv`
2. Concatenate v25 fold predictions and parse policy text from
   `experiments/micro-internship-project/POLICY_ABLATION_REPORT.md`
3. Read Taste predictions and parse policy definitions from
   `experiments/taste_policy_decomposition/run_taste_policies.py`
4. Read RRF questions and predictions from
   `experiments/rrf_vcbench/results/vcbench_rrf_curated/`
5. Re-extract HQ features by calling
   `experiments/joel_hq_features/hq_features.extract_features()` on both CSVs
6. Compute embeddings for all 56 LLM-evaluated feature texts
7. Run validation: shape checks, UUID alignment, and reproduce the v25 CV F0.5
   ≈ 0.294 and HQ test F0.5 ≈ 0.274 numbers as a sanity check.

## Source experiments

| Subdirectory | Source experiment |
|--------------|-------------------|
| `policies/` (v25) | `experiments/policy_induction_vcbench/results/v25_*` + `experiments/micro-internship-project/` |
| `policies/` (Taste) | `experiments/taste_policy_decomposition/` |
| `rrf_questions/` | `experiments/rrf_vcbench/results/vcbench_rrf_curated/` |
| `hq_baseline/` | `experiments/joel_hq_features/` |

See each sub-directory README for details on the source files and methodology.
