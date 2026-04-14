"""Semantic similarity lookup for the feature repository.

Given a free-text description of a new feature you want to compute, this
script finds the most similar features in the repository. If a match has
similarity above the threshold (default 0.85), you can use the cached
predictions instead of paying to re-run LLM scoring.

Usage:
    # CLI
    python lookup.py "founder has prior IPO experience"
    python lookup.py "founder went to MIT" --top-k 10
    python lookup.py "deep technical expertise" --threshold 0.7

    # Python
    from lookup import lookup
    matches = lookup("founder has a PhD from a top university", top_k=5)
    print(matches)

Backend: prefers sentence-transformers (all-mpnet-base-v2) if installed,
otherwise falls back to the cached TF-IDF vocabulary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EMBEDDINGS_NPY = HERE / "embeddings.npy"
FEATURE_INDEX_CSV = HERE / "feature_index.csv"
BACKEND_FILE = HERE / "embedding_backend.txt"
TFIDF_VOCAB_JSON = HERE / "tfidf_vocab.json"


def _load_repo_embeddings() -> tuple[str, np.ndarray, pd.DataFrame]:
    """Load the cached embeddings + feature index."""
    backend = BACKEND_FILE.read_text().strip()
    embeddings = np.load(EMBEDDINGS_NPY)
    feature_index = pd.read_csv(FEATURE_INDEX_CSV)
    return backend, embeddings, feature_index


def _embed_query(text: str, backend: str) -> np.ndarray:
    """Embed a single query string using the same backend that built the cache."""
    if backend.startswith("sentence-transformers"):
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(backend)
        emb = model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return emb[0]
    elif backend == "tfidf":
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import normalize

        vocab_data = json.loads(TFIDF_VOCAB_JSON.read_text())
        # Use CountVectorizer with the cached vocabulary, then apply cached IDF
        vec = CountVectorizer(
            ngram_range=tuple(vocab_data["ngram_range"]),
            stop_words="english",
            vocabulary=vocab_data["vocabulary"],
        )
        counts = vec.transform([text]).toarray().astype(np.float32)
        idf = np.asarray(vocab_data["idf"], dtype=np.float32)
        tfidf = counts * idf
        tfidf = normalize(tfidf, norm="l2", axis=1)
        return tfidf[0]
    else:
        raise ValueError(f"Unknown backend: {backend}")


def lookup(
    text: str,
    top_k: int = 5,
    threshold: float = 0.85,
) -> pd.DataFrame:
    """Find the top-k most similar features in the repository.

    Args:
        text: Description of the new feature you want to compute.
        top_k: Number of matches to return.
        threshold: Similarity threshold for the "use cached" recommendation.

    Returns:
        DataFrame with columns: feature_id, source, type, text, similarity,
        recommendation. Sorted by similarity descending.
    """
    backend, embeddings, feature_index = _load_repo_embeddings()
    query = _embed_query(text, backend)
    # Cosine similarity (embeddings are L2-normalised)
    sims = embeddings @ query
    top_idx = np.argsort(-sims)[:top_k]

    rows = []
    for idx in top_idx:
        sim = float(sims[idx])
        rec = "USE CACHED" if sim >= threshold else "GENERATE"
        rows.append(
            {
                "feature_id": feature_index.loc[idx, "feature_id"],
                "source": feature_index.loc[idx, "source"],
                "type": feature_index.loc[idx, "type"],
                "text": feature_index.loc[idx, "text"],
                "similarity": round(sim, 4),
                "recommendation": rec,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Semantic similarity lookup for the feature repository."
    )
    parser.add_argument("query", help="Free-text description of the new feature.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    matches = lookup(args.query, top_k=args.top_k, threshold=args.threshold)

    print()
    print(f"Query: {args.query}")
    print(f"Backend: {BACKEND_FILE.read_text().strip()}")
    print(f"Threshold: {args.threshold}")
    print()
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)
    print(matches.to_string(index=False))
    print()

    above = (matches["similarity"] >= args.threshold).sum()
    if above > 0:
        print(
            f"Found {above} match(es) above threshold {args.threshold} - "
            "you can use the cached predictions for these features."
        )
    else:
        print(
            f"No matches above threshold {args.threshold} - "
            "consider generating this feature yourself."
        )


if __name__ == "__main__":
    main()
