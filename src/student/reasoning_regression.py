from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.pipeline.config import DistillationModelSpec
from src.student.models import build_reasoning_regressor
from src.utils.parallel import bounded_worker_count


@dataclass(frozen=True)
class ReasoningCVOutputs:
    oof_predictions: pd.DataFrame
    metrics: pd.DataFrame


def _clip_predictions(values: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), lower, upper)


def train_reasoning_regressors_oof(
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    founder_ids: pd.Series,
    target_columns: list[str],
    model_specs: list[DistillationModelSpec],
    splits: list,
    random_state: int,
    scale_min: float,
    scale_max: float,
    model_param_overrides_by_model_id: dict[str, dict[str, object]] | None = None,
    max_parallel_workers: int = 1,
) -> ReasoningCVOutputs:
    predictions = pd.DataFrame({"founder_uuid": founder_ids.astype(str).tolist()})
    X = feature_frame.to_numpy(dtype=float)
    tasks: list[tuple[str, np.ndarray, int, DistillationModelSpec]] = []
    for target_column in target_columns:
        y = target_frame[target_column].to_numpy(dtype=float)
        for model_offset, model_spec in enumerate(model_specs):
            tasks.append((target_column, y, model_offset, model_spec))

    def _run_task(task: tuple[str, np.ndarray, int, DistillationModelSpec]) -> tuple[str, np.ndarray, list[dict[str, object]]]:
        target_column, y, model_offset, model_spec = task
        column_name = f"{target_column}__{model_spec.model_id}"
        oof = np.full(len(feature_frame), np.nan, dtype=float)
        metric_rows_local: list[dict[str, object]] = []
        for fold_offset, split in enumerate(splits):
            model = build_reasoning_regressor(
                model_spec.kind,
                random_state=random_state + model_offset + fold_offset,
                param_overrides=(model_param_overrides_by_model_id or {}).get(model_spec.model_id),
            )
            X_train = X[split.train_idx]
            X_test = X[split.test_idx]
            y_train = y[split.train_idx]
            y_test = y[split.test_idx]
            model.fit(X_train, y_train)
            preds = _clip_predictions(
                model.predict(X_test),
                lower=scale_min,
                upper=scale_max,
            )
            oof[split.test_idx] = preds
            fold_metrics = regression_metrics(y_test, preds)
            metric_rows_local.append(
                {
                    "target_id": target_column,
                    "model_id": model_spec.model_id,
                    "split_id": split.split_id,
                    **fold_metrics,
                }
            )
        if np.isnan(oof).any():
            raise RuntimeError(
                f"Reasoning OOF predictions contain NaNs for target '{target_column}' "
                f"and model '{model_spec.model_id}'."
            )
        metric_rows_local.append(
            {
                "target_id": target_column,
                "model_id": model_spec.model_id,
                "split_id": "oof_overall",
                **regression_metrics(y, oof),
            }
        )
        return column_name, oof, metric_rows_local

    workers = bounded_worker_count(max_parallel_workers=max_parallel_workers, task_count=len(tasks))
    if workers == 1:
        task_outputs = [_run_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            task_outputs = list(executor.map(_run_task, tasks))

    metric_rows: list[dict[str, object]] = []
    for column_name, oof, task_metric_rows in task_outputs:
        predictions[column_name] = oof
        metric_rows.extend(task_metric_rows)

    return ReasoningCVOutputs(
        oof_predictions=predictions,
        metrics=pd.DataFrame(metric_rows),
    )


def fit_reasoning_regressors_full(
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    target_columns: list[str],
    model_specs: list[DistillationModelSpec],
    random_state: int,
    model_param_overrides_by_model_id: dict[str, dict[str, object]] | None = None,
    max_parallel_workers: int = 1,
) -> dict[tuple[str, str], object]:
    X = feature_frame.to_numpy(dtype=float)
    tasks: list[tuple[str, int, DistillationModelSpec]] = []
    for model_offset, model_spec in enumerate(model_specs):
        for target_column in target_columns:
            tasks.append((target_column, model_offset, model_spec))

    def _fit_task(task: tuple[str, int, DistillationModelSpec]) -> tuple[tuple[str, str], object]:
        target_column, model_offset, model_spec = task
        model = build_reasoning_regressor(
            model_spec.kind,
            random_state=random_state + model_offset,
            param_overrides=(model_param_overrides_by_model_id or {}).get(model_spec.model_id),
        )
        model.fit(X, target_frame[target_column].to_numpy(dtype=float))
        return (target_column, model_spec.model_id), model

    workers = bounded_worker_count(max_parallel_workers=max_parallel_workers, task_count=len(tasks))
    if workers == 1:
        fitted_pairs = [_fit_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fitted_pairs = list(executor.map(_fit_task, tasks))
    fitted: dict[tuple[str, str], object] = dict(fitted_pairs)
    return fitted


def predict_reasoning_regressors_full(
    feature_frame: pd.DataFrame,
    *,
    founder_ids: pd.Series,
    target_columns: list[str],
    model_specs: list[DistillationModelSpec],
    fitted_models: dict[tuple[str, str], object],
    scale_min: float,
    scale_max: float,
) -> pd.DataFrame:
    predictions = pd.DataFrame({"founder_uuid": founder_ids.astype(str).tolist()})
    X = feature_frame.to_numpy(dtype=float)
    for target_column in target_columns:
        for model_spec in model_specs:
            model = fitted_models[(target_column, model_spec.model_id)]
            preds = _clip_predictions(
                model.predict(X),
                lower=scale_min,
                upper=scale_max,
            )
            predictions[f"{target_column}__{model_spec.model_id}"] = preds
    return predictions
