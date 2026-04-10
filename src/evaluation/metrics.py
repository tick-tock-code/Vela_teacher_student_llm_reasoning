from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _safe_correlation(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    left = pd.Series(np.asarray(y_true, dtype=float))
    right = pd.Series(np.asarray(y_pred, dtype=float))
    if left.nunique(dropna=False) <= 1 or right.nunique(dropna=False) <= 1:
        return 0.0
    value = left.corr(right, method=method)
    if value is None or math.isnan(value):
        return 0.0
    return float(value)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_use = np.asarray(y_true, dtype=float)
    y_pred_use = np.asarray(y_pred, dtype=float)
    return {
        "pearson": _safe_correlation(y_true_use, y_pred_use, method="pearson"),
        "spearman": _safe_correlation(y_true_use, y_pred_use, method="spearman"),
        "mae": float(mean_absolute_error(y_true_use, y_pred_use)),
        "rmse": float(mean_squared_error(y_true_use, y_pred_use) ** 0.5),
        "r2": float(r2_score(y_true_use, y_pred_use)),
    }


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, pct: float) -> float:
    if len(y_true) == 0:
        return 0.0
    k = max(1, int(math.ceil(len(y_true) * pct)))
    order = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[order]))


def select_f05_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_threshold = 0.5
    best_value = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        value = fbeta_score(y_true, (scores >= threshold).astype(int), beta=0.5, zero_division=0.0)
        if value > best_value:
            best_value = float(value)
            best_threshold = float(threshold)
    return best_threshold


def binary_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    y_true_use = np.asarray(y_true, dtype=int)
    score_use = np.asarray(scores, dtype=float)
    preds = (score_use >= threshold).astype(int)
    pr_auc = average_precision_score(y_true_use, score_use) if np.any(y_true_use == 1) else 0.0
    roc_auc = roc_auc_score(y_true_use, score_use) if len(np.unique(y_true_use)) > 1 else 0.5
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision_score(y_true_use, preds, zero_division=0.0)),
        "recall": float(recall_score(y_true_use, preds, zero_division=0.0)),
        "f0_5": float(fbeta_score(y_true_use, preds, beta=0.5, zero_division=0.0)),
        "brier": float(brier_score_loss(y_true_use, score_use)),
        "precision_at_01": precision_at_k(y_true_use, score_use, 0.01),
        "precision_at_05": precision_at_k(y_true_use, score_use, 0.05),
        "precision_at_10": precision_at_k(y_true_use, score_use, 0.10),
        "threshold": float(threshold),
    }
