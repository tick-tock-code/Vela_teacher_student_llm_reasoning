from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import pandas as pd

from src.data.loading import read_table, rename_identifier_column, validate_unique_ids


@dataclass(frozen=True)
class ResolvedFeatureBank:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    feature_columns: list[str]
    source_id_column: str
    target_id_column: str
    feature_regex: str
    expected_feature_count: int


def _resolve_test_feature_columns(
    test_frame: pd.DataFrame,
    *,
    target_id_column: str,
    feature_regex: str,
    expected_feature_count: int,
) -> list[str]:
    pattern = re.compile(feature_regex)
    matches = [
        column
        for column in test_frame.columns
        if column != target_id_column
        and pattern.fullmatch(column)
        and pd.api.types.is_numeric_dtype(test_frame[column])
    ]
    if len(matches) != expected_feature_count:
        raise RuntimeError(
            f"Expected exactly {expected_feature_count} numeric feature columns in the test bank "
            f"matching '{feature_regex}', but found {len(matches)}: {matches}"
        )
    return matches


def load_feature_bank(
    train_path: Path,
    test_path: Path,
    *,
    source_id_column: str,
    target_id_column: str,
    feature_regex: str,
    expected_feature_count: int,
) -> ResolvedFeatureBank:
    train_frame = read_table(train_path)
    test_frame = read_table(test_path)

    train_frame = rename_identifier_column(
        train_frame,
        source_id_column=source_id_column,
        target_id_column=target_id_column,
    )
    test_frame = rename_identifier_column(
        test_frame,
        source_id_column=source_id_column,
        target_id_column=target_id_column,
    )

    validate_unique_ids(train_frame, id_column=target_id_column)
    validate_unique_ids(test_frame, id_column=target_id_column)

    feature_columns = _resolve_test_feature_columns(
        test_frame,
        target_id_column=target_id_column,
        feature_regex=feature_regex,
        expected_feature_count=expected_feature_count,
    )
    missing_columns = [column for column in feature_columns if column not in train_frame.columns]
    if missing_columns:
        raise RuntimeError(
            f"Train feature bank is missing required test-bank columns: {missing_columns}"
        )

    non_numeric_train = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(train_frame[column])
    ]
    if non_numeric_train:
        raise RuntimeError(f"Train feature bank has non-numeric required columns: {non_numeric_train}")

    train_use = train_frame[[target_id_column] + feature_columns].copy()
    test_use = test_frame[[target_id_column] + feature_columns].copy()

    return ResolvedFeatureBank(
        train_frame=train_use,
        test_frame=test_use,
        feature_columns=feature_columns,
        source_id_column=source_id_column,
        target_id_column=target_id_column,
        feature_regex=feature_regex,
        expected_feature_count=expected_feature_count,
    )


def feature_manifest_payload(feature_bank: ResolvedFeatureBank) -> dict[str, object]:
    return {
        "source_id_column": feature_bank.source_id_column,
        "target_id_column": feature_bank.target_id_column,
        "feature_regex": feature_bank.feature_regex,
        "expected_feature_count": feature_bank.expected_feature_count,
        "feature_columns": feature_bank.feature_columns,
    }
