from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_classification_metrics, select_f05_threshold
from src.pipeline.config import DistillationModelSpec
from src.student.models import build_reasoning_classifier
from src.utils.parallel import bounded_worker_count


@dataclass(frozen=True)
class ReasoningClassificationCVOutputs:
    oof_predictions: pd.DataFrame
    metrics: pd.DataFrame
    selected_thresholds: dict[tuple[str, str], float]


def train_reasoning_classifiers_oof(
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    founder_ids: pd.Series,
    target_columns: list[str],
    model_specs: list[DistillationModelSpec],
    splits: list,
    random_state: int,
    model_param_overrides_by_model_id: dict[str, dict[str, float | int]] | None = None,
    max_parallel_workers: int = 1,
) -> ReasoningClassificationCVOutputs:
    predictions = pd.DataFrame({"founder_uuid": founder_ids.astype(str).tolist()})
    X = feature_frame.to_numpy(dtype=float)
    tasks: list[tuple[str, np.ndarray, int, DistillationModelSpec]] = []
    for target_column in target_columns:
        y = target_frame[target_column].to_numpy(dtype=int)
        for model_offset, model_spec in enumerate(model_specs):
            tasks.append((target_column, y, model_offset, model_spec))

    def _run_task(
        task: tuple[str, np.ndarray, int, DistillationModelSpec],
    ) -> tuple[str, np.ndarray, tuple[str, str], float, list[dict[str, object]]]:
        target_column, y, model_offset, model_spec = task
        column_name = f"{target_column}__{model_spec.model_id}"
        oof = np.full(len(feature_frame), np.nan, dtype=float)
        fold_predictions: dict[str, np.ndarray] = {}
        metric_rows_local: list[dict[str, object]] = []

        for fold_offset, split in enumerate(splits):
            model = build_reasoning_classifier(
                model_spec.kind,
                random_state=random_state + model_offset + fold_offset,
                param_overrides=(model_param_overrides_by_model_id or {}).get(model_spec.model_id),
            )
            model.fit(X[split.train_idx], y[split.train_idx])
            preds = model.predict_proba(X[split.test_idx])[:, 1]
            oof[split.test_idx] = preds
            fold_predictions[split.split_id] = preds

        if np.isnan(oof).any():
            raise RuntimeError(
                f"Reasoning OOF classification predictions contain NaNs for target "
                f"'{target_column}' and model '{model_spec.model_id}'."
            )

        threshold = select_f05_threshold(y, oof)
        for split in splits:
            fold_metrics = binary_classification_metrics(
                y[split.test_idx],
                fold_predictions[split.split_id],
                threshold=threshold,
            )
            metric_rows_local.append(
                {
                    "target_id": target_column,
                    "model_id": model_spec.model_id,
                    "split_id": split.split_id,
                    **fold_metrics,
                }
            )
        metric_rows_local.append(
            {
                "target_id": target_column,
                "model_id": model_spec.model_id,
                "split_id": "oof_overall",
                **binary_classification_metrics(y, oof, threshold=threshold),
            }
        )
        return column_name, oof, (target_column, model_spec.model_id), threshold, metric_rows_local

    workers = bounded_worker_count(max_parallel_workers=max_parallel_workers, task_count=len(tasks))
    if workers == 1:
        task_outputs = [_run_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            task_outputs = list(executor.map(_run_task, tasks))

    metric_rows: list[dict[str, object]] = []
    thresholds: dict[tuple[str, str], float] = {}
    for column_name, oof, threshold_key, threshold_value, task_metric_rows in task_outputs:
        predictions[column_name] = oof
        thresholds[threshold_key] = threshold_value
        metric_rows.extend(task_metric_rows)

    return ReasoningClassificationCVOutputs(
        oof_predictions=predictions,
        metrics=pd.DataFrame(metric_rows),
        selected_thresholds=thresholds,
    )


def fit_reasoning_classifiers_full(
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    target_columns: list[str],
    model_specs: list[DistillationModelSpec],
    random_state: int,
    model_param_overrides_by_model_id: dict[str, dict[str, float | int]] | None = None,
    max_parallel_workers: int = 1,
) -> dict[tuple[str, str], object]:
    X = feature_frame.to_numpy(dtype=float)
    tasks: list[tuple[str, int, DistillationModelSpec]] = []
    for model_offset, model_spec in enumerate(model_specs):
        for target_column in target_columns:
            tasks.append((target_column, model_offset, model_spec))

    def _fit_task(task: tuple[str, int, DistillationModelSpec]) -> tuple[tuple[str, str], object]:
        target_column, model_offset, model_spec = task
        model = build_reasoning_classifier(
            model_spec.kind,
            random_state=random_state + model_offset,
            param_overrides=(model_param_overrides_by_model_id or {}).get(model_spec.model_id),
        )
        model.fit(X, target_frame[target_column].to_numpy(dtype=int))
        return (target_column, model_spec.model_id), model

    workers = bounded_worker_count(max_parallel_workers=max_parallel_workers, task_count=len(tasks))
    if workers == 1:
        fitted_pairs = [_fit_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fitted_pairs = list(executor.map(_fit_task, tasks))
    fitted: dict[tuple[str, str], object] = dict(fitted_pairs)
    return fitted


def predict_reasoning_classifiers_full(
    feature_frame: pd.DataFrame,
    *,
    founder_ids: pd.Series,
    target_columns: list[str],
    model_specs: list[DistillationModelSpec],
    fitted_models: dict[tuple[str, str], object],
) -> pd.DataFrame:
    predictions = pd.DataFrame({"founder_uuid": founder_ids.astype(str).tolist()})
    X = feature_frame.to_numpy(dtype=float)
    for target_column in target_columns:
        for model_spec in model_specs:
            model = fitted_models[(target_column, model_spec.model_id)]
            predictions[f"{target_column}__{model_spec.model_id}"] = model.predict_proba(X)[:, 1]
    return predictions
