from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_classification_metrics, select_f05_threshold
from src.pipeline.config import DistillationModelSpec
from src.student.models import build_reasoning_classifier


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
) -> ReasoningClassificationCVOutputs:
    predictions = pd.DataFrame({"founder_uuid": founder_ids.astype(str).tolist()})
    metric_rows: list[dict[str, object]] = []
    thresholds: dict[tuple[str, str], float] = {}

    X = feature_frame.to_numpy(dtype=float)
    for target_column in target_columns:
        y = target_frame[target_column].to_numpy(dtype=int)
        for model_offset, model_spec in enumerate(model_specs):
            column_name = f"{target_column}__{model_spec.model_id}"
            oof = np.full(len(feature_frame), np.nan, dtype=float)
            fold_predictions: dict[str, np.ndarray] = {}

            for fold_offset, split in enumerate(splits):
                model = build_reasoning_classifier(
                    model_spec.kind,
                    random_state=random_state + model_offset + fold_offset,
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
            thresholds[(target_column, model_spec.model_id)] = threshold
            predictions[column_name] = oof

            for split in splits:
                fold_metrics = binary_classification_metrics(
                    y[split.test_idx],
                    fold_predictions[split.split_id],
                    threshold=threshold,
                )
                metric_rows.append(
                    {
                        "target_id": target_column,
                        "model_id": model_spec.model_id,
                        "split_id": split.split_id,
                        **fold_metrics,
                    }
                )

            overall_metrics = binary_classification_metrics(
                y,
                oof,
                threshold=threshold,
            )
            metric_rows.append(
                {
                    "target_id": target_column,
                    "model_id": model_spec.model_id,
                    "split_id": "oof_overall",
                    **overall_metrics,
                }
            )

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
) -> dict[tuple[str, str], object]:
    fitted: dict[tuple[str, str], object] = {}
    X = feature_frame.to_numpy(dtype=float)
    for model_offset, model_spec in enumerate(model_specs):
        for target_column in target_columns:
            model = build_reasoning_classifier(
                model_spec.kind,
                random_state=random_state + model_offset,
            )
            model.fit(X, target_frame[target_column].to_numpy(dtype=int))
            fitted[(target_column, model_spec.model_id)] = model
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
