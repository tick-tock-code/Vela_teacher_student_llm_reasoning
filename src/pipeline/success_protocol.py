from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.splits import build_public_cv_splits
from src.evaluation.metrics import binary_classification_metrics


# Legacy democratic threshold defaults kept for backward compatibility wrappers/scripts.
DEMOCRATIC_VOTE_THRESHOLD_START = 0.30
DEMOCRATIC_VOTE_THRESHOLD_STOP = 0.70
DEMOCRATIC_VOTE_THRESHOLD_STEP = 0.05


@dataclass(frozen=True)
class _L2FoldVoter:
    model: LogisticRegression
    scaler: StandardScaler | None
    quality: float


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


def sweep_threshold_grid(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    start: float,
    stop: float,
    step: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    threshold = start
    while threshold <= stop + 1e-9:
        threshold_use = round(float(threshold), 6)
        metrics = binary_classification_metrics(y_true, scores, threshold=threshold_use)
        rows.append(
            {
                "threshold": threshold_use,
                "f0_5": float(metrics["f0_5"]),
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
            }
        )
        threshold += step
    return rows


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
    if 1.0 in c_grid:
        return 1.0
    if 5.0 in c_grid:
        return 5.0
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


def _fit_l2_fold_voter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    continuous_indices: list[int],
    c_value: float,
    random_state: int,
) -> _L2FoldVoter:
    X_train_use = X_train.copy()
    scaler: StandardScaler | None = None
    if continuous_indices:
        scaler = StandardScaler()
        X_train_use[:, continuous_indices] = scaler.fit_transform(X_train_use[:, continuous_indices])
    model = LogisticRegression(
        C=c_value,
        max_iter=3000,
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(X_train_use, y_train)
    return _L2FoldVoter(model=model, scaler=scaler, quality=0.0)


def _predict_with_fold_voter(
    voter: _L2FoldVoter,
    X_eval: np.ndarray,
    *,
    continuous_indices: list[int],
) -> np.ndarray:
    X_eval_use = X_eval.copy()
    if voter.scaler is not None and continuous_indices:
        X_eval_use[:, continuous_indices] = voter.scaler.transform(X_eval_use[:, continuous_indices])
    return np.asarray(voter.model.predict_proba(X_eval_use)[:, 1], dtype=float)


def _choose_fold_c_value(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    continuous_indices: list[int],
    inner_n_splits: int,
    inner_shuffle: bool,
    inner_random_state: int,
    c_grid: list[float],
    random_state: int,
    use_nested: bool,
    fixed_c_value: float | None,
) -> float:
    if use_nested:
        _, chosen_c = nested_l2_predict_proba(
            X_train,
            y_train,
            X_train[:1],
            continuous_indices=continuous_indices,
            inner_n_splits=inner_n_splits,
            inner_shuffle=inner_shuffle,
            inner_random_state=inner_random_state,
            c_grid=c_grid,
            random_state=random_state,
        )
        return float(chosen_c)
    return float(fixed_c_value if fixed_c_value is not None else default_l2_c(c_grid))


def _normalized_weights_from_quality(
    qualities: np.ndarray,
) -> np.ndarray:
    if qualities.size == 0:
        raise RuntimeError("Cannot normalize empty voter-quality weights.")
    safe = np.where(np.isfinite(qualities), qualities, 0.0).astype(float)
    safe = np.maximum(safe, 1e-6)
    total = float(np.sum(safe))
    if not np.isfinite(total) or total <= 0:
        return np.full(len(safe), 1.0 / float(len(safe)), dtype=float)
    return safe / total


def _soft_score_from_voters(
    voters: list[_L2FoldVoter],
    X_eval: np.ndarray,
    *,
    continuous_indices: list[int],
    use_exit_override: bool,
    exit_counts: np.ndarray | None,
    weighted: bool,
) -> np.ndarray:
    if not voters:
        raise RuntimeError("Soft ensemble success protocol requires at least one trained voter model.")
    probs_rows: list[np.ndarray] = []
    for voter in voters:
        probs = _predict_with_fold_voter(voter, X_eval, continuous_indices=continuous_indices)
        if use_exit_override and exit_counts is not None:
            probs = apply_exit_override(probs, exit_counts)
        probs_rows.append(np.asarray(probs, dtype=float))
    probs_matrix = np.vstack(probs_rows)
    if weighted:
        quality = np.asarray([float(voter.quality) for voter in voters], dtype=float)
        weights = _normalized_weights_from_quality(quality)
        scores = np.average(probs_matrix, axis=0, weights=weights)
        return np.clip(scores, 0.0, 1.0)
    scores = np.mean(probs_matrix, axis=0)
    return np.clip(scores, 0.0, 1.0)


def _build_soft_variant_result(
    *,
    y_train: np.ndarray,
    y_test: np.ndarray | None,
    train_scores: np.ndarray,
    test_scores: np.ndarray | None,
    repeat_train_scores: list[np.ndarray],
    repeat_test_scores: list[np.ndarray],
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
) -> dict[str, Any]:
    threshold_sweep = sweep_threshold_grid(
        y_train,
        train_scores,
        start=threshold_start,
        stop=threshold_stop,
        step=threshold_step,
    )
    if not threshold_sweep:
        raise RuntimeError("Soft ensemble threshold sweep produced no rows.")
    best_row = max(threshold_sweep, key=lambda row: row["f0_5"])
    selected_threshold = float(best_row["threshold"])
    cv_metrics = binary_classification_metrics(y_train, train_scores, threshold=selected_threshold)
    repeat_train_metrics = [
        binary_classification_metrics(y_train, scores, threshold=selected_threshold)
        for scores in repeat_train_scores
    ]

    test_metrics = None
    repeat_test_metrics: list[dict[str, float]] = []
    if test_scores is not None and y_test is not None:
        test_metrics = binary_classification_metrics(y_test, test_scores, threshold=selected_threshold)
        repeat_test_metrics = [
            binary_classification_metrics(y_test, scores, threshold=selected_threshold)
            for scores in repeat_test_scores
        ]

    return {
        "threshold": selected_threshold,
        "threshold_sweep": threshold_sweep,
        "selected_train_f0_5": float(best_row["f0_5"]),
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "repeat_train_metrics": repeat_train_metrics,
        "repeat_test_metrics": repeat_test_metrics,
    }


def run_nested_l2_soft_ensemble_success_protocol(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    continuous_indices: list[int],
    outer_n_splits: int,
    outer_shuffle: bool,
    outer_random_state: int,
    inner_n_splits: int,
    inner_shuffle: bool,
    inner_random_state: int,
    c_grid: list[float],
    use_nested: bool,
    fixed_c_value: float | None = None,
    use_exit_override: bool = False,
    train_exit_counts: np.ndarray | None = None,
    test_exit_counts: np.ndarray | None = None,
    repeat_count: int = 1,
    threshold_start: float = 0.05,
    threshold_stop: float = 0.95,
    threshold_step: float = 0.01,
) -> dict[str, Any]:
    repeat_count_use = max(1, int(repeat_count))
    all_voters: list[_L2FoldVoter] = []
    selected_cs: list[float] = []
    repeat_train_soft_scores: list[np.ndarray] = []
    repeat_train_weighted_soft_scores: list[np.ndarray] = []
    repeat_test_soft_scores: list[np.ndarray] = []
    repeat_test_weighted_soft_scores: list[np.ndarray] = []

    for repeat_index in range(repeat_count_use):
        seed_offset = repeat_index * 10_000
        repeat_outer_random_state = int(outer_random_state) + seed_offset
        repeat_inner_random_state = int(inner_random_state) + seed_offset
        splits = build_public_cv_splits(
            y_train,
            n_splits=outer_n_splits,
            shuffle=outer_shuffle,
            random_state=repeat_outer_random_state,
        )
        repeat_voters: list[_L2FoldVoter] = []
        for fold_offset, split in enumerate(splits):
            X_outer_train = X_train[split.train_idx]
            y_outer_train = y_train[split.train_idx]
            chosen_c = _choose_fold_c_value(
                X_outer_train,
                y_outer_train,
                continuous_indices=continuous_indices,
                inner_n_splits=inner_n_splits,
                inner_shuffle=inner_shuffle,
                inner_random_state=repeat_inner_random_state,
                c_grid=c_grid,
                random_state=repeat_outer_random_state + fold_offset,
                use_nested=use_nested,
                fixed_c_value=fixed_c_value,
            )
            voter = _fit_l2_fold_voter(
                X_outer_train,
                y_outer_train,
                continuous_indices=continuous_indices,
                c_value=chosen_c,
                random_state=repeat_outer_random_state + fold_offset,
            )
            X_outer_eval = X_train[split.test_idx]
            y_outer_eval = y_train[split.test_idx]
            fold_scores = _predict_with_fold_voter(voter, X_outer_eval, continuous_indices=continuous_indices)
            if use_exit_override and train_exit_counts is not None:
                fold_scores = apply_exit_override(fold_scores, train_exit_counts[split.test_idx])
            fold_sweep = sweep_threshold_grid(
                y_outer_eval,
                fold_scores,
                start=threshold_start,
                stop=threshold_stop,
                step=threshold_step,
            )
            fold_best_f0_5 = float(max(fold_sweep, key=lambda row: row["f0_5"])["f0_5"]) if fold_sweep else 0.0
            voter = _L2FoldVoter(model=voter.model, scaler=voter.scaler, quality=fold_best_f0_5)
            repeat_voters.append(voter)
            selected_cs.append(float(chosen_c))
        all_voters.extend(repeat_voters)
        repeat_train_soft_scores.append(
            _soft_score_from_voters(
                repeat_voters,
                X_train,
                continuous_indices=continuous_indices,
                use_exit_override=use_exit_override,
                exit_counts=train_exit_counts,
                weighted=False,
            )
        )
        repeat_train_weighted_soft_scores.append(
            _soft_score_from_voters(
                repeat_voters,
                X_train,
                continuous_indices=continuous_indices,
                use_exit_override=use_exit_override,
                exit_counts=train_exit_counts,
                weighted=True,
            )
        )
        if X_test is not None:
            repeat_test_soft_scores.append(
                _soft_score_from_voters(
                    repeat_voters,
                    X_test,
                    continuous_indices=continuous_indices,
                    use_exit_override=use_exit_override,
                    exit_counts=test_exit_counts,
                    weighted=False,
                )
            )
            repeat_test_weighted_soft_scores.append(
                _soft_score_from_voters(
                    repeat_voters,
                    X_test,
                    continuous_indices=continuous_indices,
                    use_exit_override=use_exit_override,
                    exit_counts=test_exit_counts,
                    weighted=True,
                )
            )

    train_soft_scores = _soft_score_from_voters(
        all_voters,
        X_train,
        continuous_indices=continuous_indices,
        use_exit_override=use_exit_override,
        exit_counts=train_exit_counts,
        weighted=False,
    )
    train_weighted_soft_scores = _soft_score_from_voters(
        all_voters,
        X_train,
        continuous_indices=continuous_indices,
        use_exit_override=use_exit_override,
        exit_counts=train_exit_counts,
        weighted=True,
    )
    test_soft_scores = None
    test_weighted_soft_scores = None
    if X_test is not None:
        test_soft_scores = _soft_score_from_voters(
            all_voters,
            X_test,
            continuous_indices=continuous_indices,
            use_exit_override=use_exit_override,
            exit_counts=test_exit_counts,
            weighted=False,
        )
        test_weighted_soft_scores = _soft_score_from_voters(
            all_voters,
            X_test,
            continuous_indices=continuous_indices,
            use_exit_override=use_exit_override,
            exit_counts=test_exit_counts,
            weighted=True,
        )

    soft_avg_result = _build_soft_variant_result(
        y_train=y_train,
        y_test=y_test,
        train_scores=train_soft_scores,
        test_scores=test_soft_scores,
        repeat_train_scores=repeat_train_soft_scores,
        repeat_test_scores=repeat_test_soft_scores,
        threshold_start=threshold_start,
        threshold_stop=threshold_stop,
        threshold_step=threshold_step,
    )
    soft_avg_weighted_result = _build_soft_variant_result(
        y_train=y_train,
        y_test=y_test,
        train_scores=train_weighted_soft_scores,
        test_scores=test_weighted_soft_scores,
        repeat_train_scores=repeat_train_weighted_soft_scores,
        repeat_test_scores=repeat_test_weighted_soft_scores,
        threshold_start=threshold_start,
        threshold_stop=threshold_stop,
        threshold_step=threshold_step,
    )

    return {
        "variants": {
            "soft_avg_model": soft_avg_result,
            "soft_avg_weighted_model": soft_avg_weighted_result,
        },
        "selected_c_oof_mean": float(np.mean(selected_cs)) if selected_cs else None,
        "selected_c_final": float(np.mean(selected_cs)) if selected_cs else None,
        "voter_count": int(len(all_voters)),
        "repeat_count": int(repeat_count_use),
    }


def run_nested_l2_democratic_success_protocol(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    continuous_indices: list[int],
    outer_n_splits: int,
    outer_shuffle: bool,
    outer_random_state: int,
    inner_n_splits: int,
    inner_shuffle: bool,
    inner_random_state: int,
    c_grid: list[float],
    use_nested: bool,
    fixed_c_value: float | None = None,
    use_exit_override: bool = False,
    train_exit_counts: np.ndarray | None = None,
    test_exit_counts: np.ndarray | None = None,
    repeat_count: int = 1,
    vote_threshold_start: float = DEMOCRATIC_VOTE_THRESHOLD_START,
    vote_threshold_stop: float = DEMOCRATIC_VOTE_THRESHOLD_STOP,
    vote_threshold_step: float = DEMOCRATIC_VOTE_THRESHOLD_STEP,
) -> dict[str, Any]:
    # Backward-compatible alias for older scripts: now returns the unweighted soft ensemble path.
    ensemble = run_nested_l2_soft_ensemble_success_protocol(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        continuous_indices=continuous_indices,
        outer_n_splits=outer_n_splits,
        outer_shuffle=outer_shuffle,
        outer_random_state=outer_random_state,
        inner_n_splits=inner_n_splits,
        inner_shuffle=inner_shuffle,
        inner_random_state=inner_random_state,
        c_grid=c_grid,
        use_nested=use_nested,
        fixed_c_value=fixed_c_value,
        use_exit_override=use_exit_override,
        train_exit_counts=train_exit_counts,
        test_exit_counts=test_exit_counts,
        repeat_count=repeat_count,
        threshold_start=vote_threshold_start,
        threshold_stop=vote_threshold_stop,
        threshold_step=vote_threshold_step,
    )
    soft_avg = dict(ensemble["variants"]["soft_avg_model"])
    return {
        "threshold": float(soft_avg["threshold"]),
        "vote_threshold_sweep": list(soft_avg["threshold_sweep"]),
        "cv_metrics": dict(soft_avg["cv_metrics"]),
        "test_metrics": soft_avg["test_metrics"],
        "train_vote_share": np.asarray(soft_avg["train_scores"], dtype=float),
        "test_vote_share": (
            np.asarray(soft_avg["test_scores"], dtype=float) if soft_avg["test_scores"] is not None else None
        ),
        "repeat_train_metrics": list(soft_avg["repeat_train_metrics"]),
        "repeat_test_metrics": list(soft_avg["repeat_test_metrics"]),
        "selected_c_oof_mean": ensemble["selected_c_oof_mean"],
        "selected_c_final": ensemble["selected_c_final"],
        "voter_count": ensemble["voter_count"],
        "repeat_count": ensemble["repeat_count"],
    }


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


def run_nested_l2_success_cv_only(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
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
        raise RuntimeError("OOF predictions contain NaN values in nested L2 success CV protocol.")

    threshold, _ = select_threshold_from_grid(
        y_train,
        oof,
        start=threshold_start,
        stop=threshold_stop,
        step=threshold_step,
    )
    cv_metrics = binary_classification_metrics(y_train, oof, threshold=threshold)
    selected_c_final = (
        float(np.mean(selected_cs))
        if selected_cs
        else float(fixed_c_value if fixed_c_value is not None else default_l2_c(c_grid))
    )

    return {
        "oof_scores": oof,
        "threshold": float(threshold),
        "cv_metrics": cv_metrics,
        "selected_c_oof_mean": float(np.mean(selected_cs)) if selected_cs else None,
        "selected_c_final": float(selected_c_final),
    }
