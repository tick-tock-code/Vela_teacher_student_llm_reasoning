# Similarity Lookup

A simple tool for checking whether a feature you want to compute already
exists in the repository. Paste a free-text description of your feature; get
back the top-K most similar features by semantic similarity.

If a match is above your threshold, you can use the cached predictions
instead of paying to re-run LLM scoring.

## Usage

### CLI

```bash
python lookup.py "founder has previously been a CEO at a large company"
python lookup.py "founder went to MIT or Stanford" --top-k 10
python lookup.py "deep technical expertise" --threshold 0.3
```

### Python

```python
from lookup import lookup

matches = lookup(
    "founder has a PhD from a top university",
    top_k=5,
    threshold=0.85,  # use 0.3 for TF-IDF backend
)
print(matches)
# DataFrame with: feature_id, source, type, text, similarity, recommendation
```

## Backend

The repository supports two embedding backends:

| Backend | Quality | Install | Recommended threshold |
|---------|---------|---------|----------------------|
| `sentence-transformers/all-mpnet-base-v2` | High (semantic) | `pip install sentence-transformers` | **0.85** |
| `tfidf` (default) | Moderate (lexical) | none — uses sklearn | **0.30** |

The build script (`../build.py --step similarity`) prefers
sentence-transformers if available, falls back to TF-IDF otherwise. The
choice is recorded in `embedding_backend.txt` and the lookup script uses
the same backend automatically.

**TF-IDF limitations**: relies on shared words. "Founder went to MIT" won't
match "elite university" because there's no word overlap. For better semantic
matching, install sentence-transformers and re-run the build:

```bash
pip install sentence-transformers
python ../build.py --step similarity
```

## What's in here

| File | Description |
|------|-------------|
| `lookup.py` | The lookup script (CLI + Python API) |
| `embeddings.npy` | Cached embeddings, shape (56, D), L2-normalised |
| `embedding_backend.txt` | Which backend was used |
| `feature_index.csv` | id, source, type, text, row_idx (aligned with embeddings) |
| `tfidf_vocab.json` | Cached TF-IDF vocabulary (only present with TF-IDF backend) |

## How it works

1. The build script collates the text of all 56 LLM-evaluated features (36
   policies + 20 RRF questions).
2. It encodes them into a (56, D) embedding matrix and writes
   `embeddings.npy` and `feature_index.csv`.
3. The lookup script encodes your query string with the same backend and
   computes cosine similarity (a single matrix-vector product since
   embeddings are L2-normalised) against the cached matrix.
4. Returns top-K matches sorted by similarity, with a `recommendation`
   column flagging matches above your threshold.

## Workflow when adding a new feature

1. Write a clear, single-sentence description of the feature.
2. Run `python lookup.py "<your description>"`.
3. Inspect the top-5 matches.
4. If any match is **semantically the same as your idea** AND above your
   threshold, use the cached predictions from `../policies/predictions_*.csv`
   or `../rrf_questions/predictions_*.csv`.
5. Otherwise, generate your feature yourself, then **contribute it back** by
   adding it to a sub-directory under `feature_repository/` and re-running
   `../build.py` to refresh the embeddings.

## Caveats

- **Always read the matched feature's text** to confirm it really captures
  what you want. Semantic similarity is a heuristic; two features can have
  high similarity but mean subtly different things.
- **Threshold is a heuristic too.** A high similarity match isn't a guarantee
  the predictions are interchangeable. Consider running both your new feature
  and the cached one on a small subset and comparing.
- **Coverage**: only LLM-evaluated features are in the lookup index (56
  total). The HQ baseline features are NOT in the index because they are
  computed deterministically from the raw data (no LLM calls to skip).
