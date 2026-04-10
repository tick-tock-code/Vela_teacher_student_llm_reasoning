from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from src.data.loading import read_table, rename_identifier_column, validate_unique_ids
from src.pipeline.config import ReasoningTargetBankSpec


@dataclass(frozen=True)
class LoadedReasoningTargetBank:
    train_frame: pd.DataFrame
    train_target_columns: list[str]
    test_frame: pd.DataFrame | None
    test_target_columns: list[str]
    available_train_target_columns: list[str]
    available_test_target_columns: list[str]
    scale_min: float
    scale_max: float


def _resolve_target_columns(
    frame: pd.DataFrame,
    *,
    target_id_column: str,
    target_regex: str,
) -> list[str]:
    pattern = re.compile(target_regex)
    return [
        column
        for column in frame.columns
        if column != target_id_column
        and pattern.fullmatch(column)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _prepare_target_frame(
    *,
    path: str | Path,
    source_id_column: str,
    target_id_column: str,
    target_regex: str,
    scale_min: float,
    scale_max: float,
) -> tuple[pd.DataFrame, list[str]]:
    path_use = Path(path)
    frame = read_table(path_use)
    frame = rename_identifier_column(
        frame,
        source_id_column=source_id_column,
        target_id_column=target_id_column,
    )
    validate_unique_ids(frame, id_column=target_id_column)

    target_columns = _resolve_target_columns(
        frame,
        target_id_column=target_id_column,
        target_regex=target_regex,
    )
    if not target_columns:
        raise RuntimeError(f"No numeric target columns matched '{target_regex}' in {path_use}.")

    prepared = frame[[target_id_column] + target_columns].copy()
    for column in target_columns:
        values = pd.to_numeric(prepared[column], errors="coerce")
        if values.isna().any():
            raise RuntimeError(f"Reasoning target column '{column}' contains non-numeric values in {path_use}.")
        if ((values < scale_min) | (values > scale_max)).any():
            raise RuntimeError(
                f"Reasoning target column '{column}' contains values outside "
                f"[{scale_min}, {scale_max}] in {path_use}."
            )
        prepared[column] = values.astype(float)

    return prepared, target_columns


def load_reasoning_target_bank(
    spec: ReasoningTargetBankSpec,
    *,
    selected_targets: list[str],
) -> LoadedReasoningTargetBank:
    raw_train_frame, available_train_target_columns = _prepare_target_frame(
        path=spec.train_path,
        source_id_column=spec.source_id_column,
        target_id_column=spec.target_id_column,
        target_regex=spec.target_regex,
        scale_min=spec.scale_min,
        scale_max=spec.scale_max,
    )
    missing_train_targets = [
        column for column in selected_targets if column not in set(available_train_target_columns)
    ]
    if missing_train_targets:
        raise RuntimeError(
            f"Selected reasoning targets are missing from the public target bank: {missing_train_targets}"
        )
    train_frame = raw_train_frame[[spec.target_id_column] + selected_targets].copy()
    train_target_columns = selected_targets.copy()

    test_frame = None
    test_target_columns: list[str] = selected_targets.copy()
    available_test_target_columns: list[str] = []
    if spec.test_path is not None:
        raw_test_frame, available_test_target_columns = _prepare_target_frame(
            path=spec.test_path,
            source_id_column=spec.source_id_column,
            target_id_column=spec.target_id_column,
            target_regex=spec.target_regex,
            scale_min=spec.scale_min,
            scale_max=spec.scale_max,
        )
        missing_test_targets = [
            column for column in selected_targets if column not in set(available_test_target_columns)
        ]
        if missing_test_targets:
            raise RuntimeError(
                f"Selected reasoning targets are missing from the held-out target bank: {missing_test_targets}"
            )
        test_frame = raw_test_frame[[spec.target_id_column] + selected_targets].copy()

    return LoadedReasoningTargetBank(
        train_frame=train_frame,
        train_target_columns=train_target_columns,
        test_frame=test_frame,
        test_target_columns=test_target_columns,
        available_train_target_columns=available_train_target_columns,
        available_test_target_columns=available_test_target_columns,
        scale_min=spec.scale_min,
        scale_max=spec.scale_max,
    )


def target_manifest_payload(target_bank: LoadedReasoningTargetBank) -> dict[str, object]:
    return {
        "train_target_columns": target_bank.train_target_columns,
        "train_target_count": len(target_bank.train_target_columns),
        "test_target_columns": target_bank.test_target_columns,
        "test_target_count": len(target_bank.test_target_columns),
        "available_train_target_columns": target_bank.available_train_target_columns,
        "available_train_target_count": len(target_bank.available_train_target_columns),
        "available_test_target_columns": target_bank.available_test_target_columns,
        "available_test_target_count": len(target_bank.available_test_target_columns),
        "scale_min": target_bank.scale_min,
        "scale_max": target_bank.scale_max,
    }
