from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


NON_FEATURE_COLUMNS = {"founder_uuid", "uuid", "success", "row_index", "set_id"}


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to read parquet file {path}. Install pyarrow or fastparquet."
            ) from exc
    raise ValueError(f"Unsupported table format: {path}")


def ensure_row_index(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "row_index" not in prepared.columns:
        prepared = prepared.reset_index(drop=True)
        prepared["row_index"] = prepared.index
    return prepared


def rename_identifier_column(
    frame: pd.DataFrame,
    *,
    source_id_column: str,
    target_id_column: str = "founder_uuid",
) -> pd.DataFrame:
    if source_id_column not in frame.columns:
        raise KeyError(f"Missing identifier column '{source_id_column}'.")
    prepared = frame.copy()
    if source_id_column != target_id_column:
        prepared = prepared.rename(columns={source_id_column: target_id_column})
    prepared[target_id_column] = prepared[target_id_column].astype(str)
    return prepared


def validate_unique_ids(frame: pd.DataFrame, *, id_column: str) -> None:
    if frame[id_column].isna().any():
        raise RuntimeError(f"Identifier column '{id_column}' contains missing values.")
    if not frame[id_column].is_unique:
        raise RuntimeError(f"Identifier column '{id_column}' contains duplicate values.")


def select_numeric_feature_columns(
    frame: pd.DataFrame,
    *,
    include_columns: Iterable[str] | None = None,
    exclude_columns: Iterable[str] | None = None,
) -> list[str]:
    excluded = set(NON_FEATURE_COLUMNS)
    excluded.update(exclude_columns or [])
    if include_columns is not None:
        return [
            column
            for column in include_columns
            if column in frame.columns
            and column not in excluded
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
