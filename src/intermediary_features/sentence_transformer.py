from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.intermediary_features.mirror import clean_text
from src.utils.dependencies import require_dependency


TextEncoder = Callable[[list[str]], np.ndarray]


def build_sentence_transformer_encoder(model_name: str) -> TextEncoder:
    require_dependency("sentence_transformers", "build sentence-transformer intermediary features")
    from sentence_transformers import SentenceTransformer  # type: ignore

    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sentence-transformer model '{model_name}'. "
                "Cache it locally or allow a one-time download."
            ) from exc

    def encode_texts(texts: list[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ),
            dtype=float,
        )

    return encode_texts


def _encode_frame(
    *,
    frame: pd.DataFrame,
    text_column: str,
    feature_prefix: str,
    encode_texts: TextEncoder,
) -> tuple[pd.DataFrame, list[str]]:
    texts = [clean_text(value) for value in frame[text_column].tolist()]
    vectors = np.asarray(encode_texts(texts), dtype=float)
    if vectors.ndim != 2:
        raise RuntimeError(
            f"Sentence-transformer encoder for '{feature_prefix}' returned shape {vectors.shape}, "
            "but a 2D array was required."
        )
    feature_columns = [
        f"{feature_prefix}__dim_{index:03d}"
        for index in range(vectors.shape[1])
    ]
    encoded = pd.DataFrame(vectors, columns=feature_columns)
    encoded.insert(0, "founder_uuid", frame["founder_uuid"].astype(str).tolist())
    return encoded, feature_columns


def build_sentence_transformer_frames(
    *,
    public_text_frame: pd.DataFrame,
    private_text_frame: pd.DataFrame,
    model_name: str,
    feature_prefix: str,
    encode_texts: TextEncoder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, object]]:
    encoder = encode_texts or build_sentence_transformer_encoder(model_name)
    public_frame, feature_columns = _encode_frame(
        frame=public_text_frame,
        text_column="rendered_text",
        feature_prefix=feature_prefix,
        encode_texts=encoder,
    )
    private_frame, private_columns = _encode_frame(
        frame=private_text_frame,
        text_column="rendered_text",
        feature_prefix=feature_prefix,
        encode_texts=encoder,
    )
    if feature_columns != private_columns:
        raise RuntimeError(
            f"Sentence-transformer feature columns diverged for '{feature_prefix}'."
        )
    manifest = {
        "embedding_model_name": model_name,
        "embedding_dimension": len(feature_columns),
        "text_column": "rendered_text",
        "feature_prefix": feature_prefix,
    }
    return public_frame, private_frame, feature_columns, manifest
