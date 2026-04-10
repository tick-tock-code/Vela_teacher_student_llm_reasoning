from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.pipeline.config import ModelSpec
from src.student.models import build_reasoning_regressor


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
    model_specs: list[ModelSpec],
    splits: list,
    random_state: int,
    scale_min: float,
    scale_max: float,
) -> ReasoningCVOutputs:
    predictions = pd.DataFrame({"founder_uuid": founder_ids.astype(str).tolist()})
    metric_rows: list[dict[str, object]] = []

    X = feature_frame.to_numpy(dtype=float)
    for target_column in target_columns:
        y = target_frame[target_column].to_numpy(dtype=float)
        for model_offset, model_spec in enumerate(model_specs):
            column_name = f"{target_column}__{model_spec.model_id}"
            oof = np.full(len(feature_frame), np.nan, dtype=float)
            for fold_offset, split in enumerate(splits):
                model = build_reasoning_regressor(
                    model_spec.kind,
                    random_state=random_state + model_offset + fold_offset,
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
                metric_rows.append(
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

            predictions[column_name] = oof
            overall_metrics = regression_metrics(y, oof)
            metric_rows.append(
                {
                    "target_id": target_column,
                    "model_id": model_spec.model_id,
                    "split_id": "oof_overall",
                    **overall_metrics,
                }
            )

    return ReasoningCVOutputs(
        oof_predictions=predictions,
        metrics=pd.DataFrame(metric_rows),
    )


def fit_reasoning_regressors_full(
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    target_columns: list[str],
    model_specs: list[ModelSpec],
    random_state: int,
) -> dict[tuple[str, str], object]:
    fitted: dict[tuple[str, str], object] = {}
    X = feature_frame.to_numpy(dtype=float)
    for model_offset, model_spec in enumerate(model_specs):
        for target_column in target_columns:
            model = build_reasoning_regressor(
                model_spec.kind,
                random_state=random_state + model_offset,
            )
            model.fit(X, target_frame[target_column].to_numpy(dtype=float))
            fitted[(target_column, model_spec.model_id)] = model
    return fitted


def predict_reasoning_regressors_full(
    feature_frame: pd.DataFrame,
    *,
    founder_ids: pd.Series,
    target_columns: list[str],
    model_specs: list[ModelSpec],
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
