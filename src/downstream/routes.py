from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_classification_metrics, select_f05_threshold
from src.pipeline.config import ModelSpec
from src.student.models import build_downstream_classifier


@dataclass(frozen=True)
class DownstreamPublicOutputs:
    summary: pd.DataFrame
    fold_metrics: pd.DataFrame


def _fill_missing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    model_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_kind == "xgb1_classifier":
        return train_df.fillna(0.0), test_df.fillna(0.0)
    fill_values = train_df.mean(numeric_only=True)
    return (
        train_df.fillna(fill_values).fillna(0.0),
        test_df.fillna(fill_values).fillna(0.0),
    )


def _standardize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = train_df.mean(numeric_only=True)
    stds = train_df.std(numeric_only=True).replace(0.0, 1.0)
    return (
        ((train_df - means) / stds).fillna(0.0),
        ((test_df - means) / stds).fillna(0.0),
    )


def _prepare_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    model_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_use, test_use = _fill_missing(train_df.copy(), test_df.copy(), model_kind=model_kind)
    if model_kind == "lr_classifier":
        train_use, test_use = _standardize(train_use, test_use)
    return train_use.to_numpy(dtype=float), test_use.to_numpy(dtype=float)


def _build_public_routes(
    base_features: pd.DataFrame,
    true_reasoning: pd.DataFrame,
    predicted_reasoning_by_model: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    routes = {
        "baseline_only": base_features.copy(),
        "true_reasoning": pd.concat([base_features, true_reasoning], axis=1),
    }
    for reasoning_model_id, predicted_frame in predicted_reasoning_by_model.items():
        routes[f"predicted_reasoning__{reasoning_model_id}"] = pd.concat(
            [base_features, predicted_frame],
            axis=1,
        )
    return routes


def _summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    group_columns = ["route_id", "downstream_model_id"]
    metric_columns = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f0_5",
        "brier",
        "precision_at_01",
        "precision_at_05",
        "precision_at_10",
        "threshold",
    ]
    grouped = fold_metrics.groupby(group_columns, as_index=False)
    for (route_id, downstream_model_id), subset in grouped:
        row: dict[str, object] = {
            "route_id": route_id,
            "downstream_model_id": downstream_model_id,
            "fold_count": int(len(subset)),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(subset[column].mean())
            row[f"{column}_std"] = float(subset[column].std(ddof=0))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def evaluate_public_downstream_routes(
    *,
    base_features: pd.DataFrame,
    labels: pd.Series,
    true_reasoning: pd.DataFrame,
    predicted_reasoning_by_model: dict[str, pd.DataFrame],
    model_specs: list[ModelSpec],
    splits: list,
    random_state: int,
) -> DownstreamPublicOutputs:
    y = labels.to_numpy(dtype=int)
    routes = _build_public_routes(base_features, true_reasoning, predicted_reasoning_by_model)
    fold_rows: list[dict[str, object]] = []

    for route_id, route_frame in routes.items():
        for model_offset, model_spec in enumerate(model_specs):
            for fold_offset, split in enumerate(splits):
                X_train, X_test = _prepare_arrays(
                    route_frame.iloc[split.train_idx].copy(),
                    route_frame.iloc[split.test_idx].copy(),
                    model_kind=model_spec.kind,
                )
                y_train = y[split.train_idx]
                y_test = y[split.test_idx]
                model = build_downstream_classifier(
                    model_spec.kind,
                    random_state=random_state + model_offset + fold_offset,
                )
                model.fit(X_train, y_train)
                train_scores = model.predict_proba(X_train)[:, 1]
                test_scores = model.predict_proba(X_test)[:, 1]
                threshold = select_f05_threshold(y_train, train_scores)
                metrics = binary_classification_metrics(y_test, test_scores, threshold=threshold)
                fold_rows.append(
                    {
                        "route_id": route_id,
                        "downstream_model_id": model_spec.model_id,
                        "split_id": split.split_id,
                        **metrics,
                    }
                )

    fold_metrics = pd.DataFrame(fold_rows)
    return DownstreamPublicOutputs(
        summary=_summarize_fold_metrics(fold_metrics),
        fold_metrics=fold_metrics,
    )


def extract_predicted_reasoning_features(
    prediction_frame: pd.DataFrame,
    *,
    model_id: str,
) -> pd.DataFrame:
    columns = [
        column
        for column in prediction_frame.columns
        if column != "founder_uuid" and column.endswith(f"__{model_id}")
    ]
    rename_map = {
        column: column.rsplit("__", 1)[0]
        for column in columns
    }
    return prediction_frame[columns].rename(columns=rename_map).copy()


def predict_private_downstream_routes(
    *,
    public_base_features: pd.DataFrame,
    public_labels: pd.Series,
    private_founder_ids: pd.Series,
    private_base_features: pd.DataFrame,
    true_reasoning_public: pd.DataFrame,
    true_reasoning_private: pd.DataFrame | None,
    predicted_reasoning_public_by_model: dict[str, pd.DataFrame],
    predicted_reasoning_private_by_model: dict[str, pd.DataFrame],
    model_specs: list[ModelSpec],
    random_state: int,
) -> pd.DataFrame:
    public_routes = _build_public_routes(
        public_base_features,
        true_reasoning_public,
        predicted_reasoning_public_by_model,
    )
    private_routes = {
        "baseline_only": private_base_features.copy(),
    }
    if true_reasoning_private is not None:
        private_routes["true_reasoning"] = pd.concat(
            [private_base_features, true_reasoning_private],
            axis=1,
        )
    for model_id, predicted_frame in predicted_reasoning_private_by_model.items():
        private_routes[f"predicted_reasoning__{model_id}"] = pd.concat(
            [private_base_features, predicted_frame],
            axis=1,
        )

    rows: list[dict[str, object]] = []
    y_train = public_labels.to_numpy(dtype=int)
    for route_id, private_route_frame in private_routes.items():
        public_route_frame = public_routes[route_id]
        for model_offset, model_spec in enumerate(model_specs):
            X_train, X_test = _prepare_arrays(
                public_route_frame.copy(),
                private_route_frame.copy(),
                model_kind=model_spec.kind,
            )
            model = build_downstream_classifier(
                model_spec.kind,
                random_state=random_state + model_offset,
            )
            model.fit(X_train, y_train)
            train_scores = model.predict_proba(X_train)[:, 1]
            private_scores = model.predict_proba(X_test)[:, 1]
            threshold = select_f05_threshold(y_train, train_scores)
            for founder_uuid, score in zip(private_founder_ids.astype(str), private_scores):
                rows.append(
                    {
                        "founder_uuid": founder_uuid,
                        "route_id": route_id,
                        "downstream_model_id": model_spec.model_id,
                        "score": float(score),
                        "predicted_label": int(score >= threshold),
                        "threshold": float(threshold),
                    }
                )
    return pd.DataFrame(rows)

