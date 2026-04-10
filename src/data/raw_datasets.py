from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.loading import ensure_row_index, read_table, rename_identifier_column, validate_unique_ids


@dataclass(frozen=True)
class LoadedRawDatasets:
    public_frame: pd.DataFrame
    private_frame: pd.DataFrame


def load_public_dataset(
    path: Path,
    *,
    id_column: str = "founder_uuid",
    label_column: str = "success",
) -> pd.DataFrame:
    frame = read_table(path)
    frame = rename_identifier_column(frame, source_id_column=id_column, target_id_column="founder_uuid")
    frame = ensure_row_index(frame)
    if label_column not in frame.columns:
        raise RuntimeError(f"Public dataset is missing label column '{label_column}': {path}")
    frame[label_column] = frame[label_column].astype(int)
    validate_unique_ids(frame, id_column="founder_uuid")
    return frame


def load_private_dataset(
    path: Path,
    *,
    id_column: str = "founder_uuid",
) -> pd.DataFrame:
    frame = read_table(path)
    frame = rename_identifier_column(frame, source_id_column=id_column, target_id_column="founder_uuid")
    frame = ensure_row_index(frame)
    validate_unique_ids(frame, id_column="founder_uuid")
    return frame


def load_raw_datasets(
    public_path: Path,
    private_path: Path,
    *,
    public_id_column: str = "founder_uuid",
    private_id_column: str = "founder_uuid",
) -> LoadedRawDatasets:
    return LoadedRawDatasets(
        public_frame=load_public_dataset(public_path, id_column=public_id_column),
        private_frame=load_private_dataset(private_path, id_column=private_id_column),
    )
