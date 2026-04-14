from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.loading import read_table, rename_identifier_column, validate_unique_ids
from src.pipeline.config import TargetFamilySpec


@dataclass(frozen=True)
class LoadedTargetFamily:
    family_id: str
    task_kind: str
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame | None
    target_columns: list[str]
    available_train_target_columns: list[str]
    available_test_target_columns: list[str]
    scale_min: float | None
    scale_max: float | None


def _select_target_columns(
    frame: pd.DataFrame,
    *,
    target_id_column: str,
    prefixes: list[str],
) -> list[str]:
    return [
        column
        for column in frame.columns
        if column != target_id_column
        and any(column.startswith(prefix) for prefix in prefixes)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _prepare_target_frame(
    *,
    path: str | Path,
    source_id_column: str,
    target_id_column: str,
    prefixes: list[str],
    task_kind: str,
    scale_min: float | None,
    scale_max: float | None,
) -> tuple[pd.DataFrame, list[str]]:
    path_use = Path(path)
    frame = read_table(path_use)
    frame = rename_identifier_column(
        frame,
        source_id_column=source_id_column,
        target_id_column=target_id_column,
    )
    validate_unique_ids(frame, id_column=target_id_column)
    target_columns = _select_target_columns(
        frame,
        target_id_column=target_id_column,
        prefixes=prefixes,
    )
    if not target_columns:
        raise RuntimeError(f"No target columns matched prefixes {prefixes} in {path_use}.")

    prepared = frame[[target_id_column] + target_columns].copy()
    for column in target_columns:
        values = pd.to_numeric(prepared[column], errors="coerce")
        if values.isna().any():
            raise RuntimeError(f"Target column '{column}' contains non-numeric values in {path_use}.")
        if task_kind == "regression":
            if scale_min is None or scale_max is None:
                raise RuntimeError("Regression target families require scale_min and scale_max.")
            tolerance = 1e-5
            if ((values < scale_min - tolerance) | (values > scale_max + tolerance)).any():
                raise RuntimeError(
                    f"Regression target column '{column}' contains values outside "
                    f"[{scale_min}, {scale_max}] in {path_use}."
                )
            prepared[column] = values.clip(lower=scale_min, upper=scale_max).astype(float)
        else:
            unique_values = set(values.astype(int).tolist())
            if unique_values - {0, 1}:
                raise RuntimeError(
                    f"Classification target column '{column}' must be binary 0/1 in {path_use}."
                )
            prepared[column] = values.astype(int)
    return prepared, target_columns


def load_target_family(spec: TargetFamilySpec) -> LoadedTargetFamily:
    train_frame, available_train_target_columns = _prepare_target_frame(
        path=spec.train_path,
        source_id_column=spec.source_id_column,
        target_id_column=spec.target_id_column,
        prefixes=spec.target_prefixes,
        task_kind=spec.task_kind,
        scale_min=spec.scale_min,
        scale_max=spec.scale_max,
    )
    test_frame = None
    available_test_target_columns: list[str] = []
    if spec.test_path is not None:
        test_frame, available_test_target_columns = _prepare_target_frame(
            path=spec.test_path,
            source_id_column=spec.source_id_column,
            target_id_column=spec.target_id_column,
            prefixes=spec.target_prefixes,
            task_kind=spec.task_kind,
            scale_min=spec.scale_min,
            scale_max=spec.scale_max,
        )

    return LoadedTargetFamily(
        family_id=spec.family_id,
        task_kind=spec.task_kind,
        train_frame=train_frame,
        test_frame=test_frame,
        target_columns=available_train_target_columns,
        available_train_target_columns=available_train_target_columns,
        available_test_target_columns=available_test_target_columns,
        scale_min=spec.scale_min,
        scale_max=spec.scale_max,
    )


def target_manifest_payload(target_family: LoadedTargetFamily) -> dict[str, object]:
    return {
        "family_id": target_family.family_id,
        "task_kind": target_family.task_kind,
        "target_columns": target_family.target_columns,
        "target_count": len(target_family.target_columns),
        "available_train_target_columns": target_family.available_train_target_columns,
        "available_train_target_count": len(target_family.available_train_target_columns),
        "available_test_target_columns": target_family.available_test_target_columns,
        "available_test_target_count": len(target_family.available_test_target_columns),
        "scale_min": target_family.scale_min,
        "scale_max": target_family.scale_max,
    }
