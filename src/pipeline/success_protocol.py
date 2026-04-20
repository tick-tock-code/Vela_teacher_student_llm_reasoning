from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.splits import build_public_cv_splits
from src.evaluation.metrics import binary_classification_metrics


def continuous_indices(
    feature_columns: list[str],
    binary_feature_columns: list[str],
) -> list[int]:
    binary = set(binary_feature_columns)
    return [index for index, column in enumerate(feature_columns) if column not in binary]


def standardize_arrays(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    continuous_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not continuous_indices:
        return X_train.copy(), X_eval.copy()
    X_train_use = X_train.copy()
    X_eval_use = X_eval.copy()
    train_continuous = X_train_use[:, continuous_indices]
    eval_continuous = X_eval_use[:, continuous_indices]
    scaler = StandardScaler()
    X_train_use[:, continuous_indices] = scaler.fit_transform(train_continuous)
    X_eval_use[:, continuous_indices] = scaler.transform(eval_continuous)
    return X_train_use, X_eval_use


def select_threshold_from_grid(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    start: float,
    stop: float,
    step: float,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_value = -1.0
    threshold = start
    while threshold <= stop + 1e-9:
        metrics = binary_classification_metrics(y_true, scores, threshold=threshold)
        if metrics["f0_5"] > best_value:
            best_threshold = round(float(threshold), 6)
            best_value = float(metrics["f0_5"])
        threshold += step
    return best_threshold, best_value


def default_l2_c(c_grid: list[float]) -> float:
    if 5.0 in c_grid:
        return 5.0
    if 1.0 in c_grid:
        return 1.0
    if not c_grid:
        raise RuntimeError("logistic C grid must contain at least one value.")
    return float(c_grid[len(c_grid) // 2])


def apply_exit_override(scores: np.ndarray, exit_counts: np.ndarray) -> np.ndarray:
    updated = scores.copy()
    updated[np.asarray(exit_counts, dtype=float) > 0] = 1.0
    return updated


def nested_l2_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    continuous_indices: list[int],
    inner_n_splits: int,
    inner_shuffle: bool,
    inner_random_state: int,
    c_grid: list[float],
    random_state: int,
) -> tuple[np.ndarray, float]:
    best_c = c_grid[0]
    best_auc = -1.0
    for c_value in c_grid:
        aucs: list[float] = []
        inner_splits = build_public_cv_splits(
            y_train,
            n_splits=inner_n_splits,
            shuffle=inner_shuffle,
            random_state=inner_random_state,
        )
        for split in inner_splits:
            X_inner_train = X_train[split.train_idx]
            X_inner_eval = X_train[split.test_idx]
            y_inner_train = y_train[split.train_idx]
            y_inner_eval = y_train[split.test_idx]
            X_inner_train, X_inner_eval = standardize_arrays(
                X_inner_train,
                X_inner_eval,
                continuous_indices=continuous_indices,
            )
            model = LogisticRegression(
                C=c_value,
                max_iter=3000,
                solver="lbfgs",
                random_state=random_state,
            )
            model.fit(X_inner_train, y_inner_train)
            probs = model.predict_proba(X_inner_eval)[:, 1]
            aucs.append(float(roc_auc_score(y_inner_eval, probs)))
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_c = c_value

    X_train_final, X_eval_final = standardize_arrays(
        X_train,
        X_eval,
        continuous_indices=continuous_indices,
    )
    final_model = LogisticRegression(
        C=best_c,
        max_iter=3000,
        solver="lbfgs",
        random_state=random_state,
    )
    final_model.fit(X_train_final, y_train)
    return final_model.predict_proba(X_eval_final)[:, 1], float(best_c)


def fixed_l2_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    continuous_indices: list[int],
    c_value: float,
    random_state: int,
) -> np.ndarray:
    X_train_final, X_eval_final = standardize_arrays(
        X_train,
        X_eval,
        continuous_indices=continuous_indices,
    )
    final_model = LogisticRegression(
        C=c_value,
        max_iter=3000,
        solver="lbfgs",
        random_state=random_state,
    )
    final_model.fit(X_train_final, y_train)
    return final_model.predict_proba(X_eval_final)[:, 1]


def run_nested_l2_success_protocol(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    continuous_indices: list[int],
    outer_n_splits: int,
    outer_shuffle: bool,
    outer_random_state: int,
    inner_n_splits: int,
    inner_shuffle: bool,
    inner_random_state: int,
    c_grid: list[float],
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
    use_nested: bool,
    fixed_c_value: float | None = None,
    use_exit_override: bool = False,
    train_exit_counts: np.ndarray | None = None,
    test_exit_counts: np.ndarray | None = None,
    random_state_offset: int = 0,
) -> dict[str, Any]:
    outer_splits = build_public_cv_splits(
        y_train,
        n_splits=outer_n_splits,
        shuffle=outer_shuffle,
        random_state=outer_random_state,
    )

    oof = np.full(len(y_train), np.nan, dtype=float)
    selected_cs: list[float] = []
    for fold_offset, split in enumerate(outer_splits):
        X_outer_train = X_train[split.train_idx]
        X_outer_eval = X_train[split.test_idx]
        y_outer_train = y_train[split.train_idx]
        chosen_c_local: float | None = None
        if use_nested:
            preds_local, chosen_c_local = nested_l2_predict_proba(
                X_outer_train,
                y_outer_train,
                X_outer_eval,
                continuous_indices=continuous_indices,
                inner_n_splits=inner_n_splits,
                inner_shuffle=inner_shuffle,
                inner_random_state=inner_random_state,
                c_grid=c_grid,
                random_state=outer_random_state + fold_offset,
            )
        else:
            c_value = fixed_c_value if fixed_c_value is not None else default_l2_c(c_grid)
            chosen_c_local = float(c_value)
            preds_local = fixed_l2_predict_proba(
                X_outer_train,
                y_outer_train,
                X_outer_eval,
                continuous_indices=continuous_indices,
                c_value=c_value,
                random_state=outer_random_state + fold_offset,
            )
        if use_exit_override and train_exit_counts is not None:
            preds_local = apply_exit_override(preds_local, train_exit_counts[split.test_idx])
        oof[split.test_idx] = preds_local
        if chosen_c_local is not None:
            selected_cs.append(float(chosen_c_local))

    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions contain NaN values in nested L2 success protocol.")

    threshold, _ = select_threshold_from_grid(
        y_train,
        oof,
        start=threshold_start,
        stop=threshold_stop,
        step=threshold_step,
    )
    cv_metrics = binary_classification_metrics(y_train, oof, threshold=threshold)

    if use_nested:
        test_probs, final_c = nested_l2_predict_proba(
            X_train,
            y_train,
            X_test,
            continuous_indices=continuous_indices,
            inner_n_splits=inner_n_splits,
            inner_shuffle=inner_shuffle,
            inner_random_state=inner_random_state,
            c_grid=c_grid,
            random_state=outer_random_state + random_state_offset,
        )
    else:
        final_c = float(fixed_c_value if fixed_c_value is not None else default_l2_c(c_grid))
        test_probs = fixed_l2_predict_proba(
            X_train,
            y_train,
            X_test,
            continuous_indices=continuous_indices,
            c_value=final_c,
            random_state=outer_random_state + random_state_offset,
        )
    if use_exit_override and test_exit_counts is not None:
        test_probs = apply_exit_override(test_probs, test_exit_counts)
    test_metrics = binary_classification_metrics(y_test, test_probs, threshold=threshold)

    return {
        "oof_scores": oof,
        "threshold": float(threshold),
        "cv_metrics": cv_metrics,
        "test_scores": test_probs,
        "test_metrics": test_metrics,
        "selected_c_oof_mean": float(np.mean(selected_cs)) if selected_cs else None,
        "selected_c_final": float(final_c),
    }
