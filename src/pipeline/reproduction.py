from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.data.feature_repository import LoadedFeatureRepositorySplits, load_feature_repository_splits, load_repository_feature_banks
from src.data.splits import build_public_cv_splits
from src.evaluation.metrics import binary_classification_metrics
from src.pipeline.config import ExperimentConfig, ReproductionExperimentSpec
from src.student.models import build_reproduction_classifier
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.paths import RUNS_DIR


Logger = Callable[[str], None]


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _load_test_success_labels_for_final_evaluation(
    repository_splits: LoadedFeatureRepositorySplits,
) -> np.ndarray:
    labels_test = repository_splits.test_labels.set_index("founder_uuid").reindex(repository_splits.test_ids)
    return labels_test["success"].to_numpy(dtype=int)


def _merge_feature_bank(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    bank_id: str,
    split_label: str,
) -> pd.DataFrame:
    merged = left_frame.merge(
        right_frame[["founder_uuid"] + feature_columns],
        on="founder_uuid",
        how="left",
        validate="one_to_one",
    )
    missing_ids = sorted(set(left_frame["founder_uuid"].astype(str)) - set(right_frame["founder_uuid"].astype(str)))
    if missing_ids:
        raise RuntimeError(
            f"Feature bank '{bank_id}' is missing {split_label} rows for {len(missing_ids)} founders. "
            f"Examples: {missing_ids[:5]}"
        )
    return merged


def _continuous_indices(feature_columns: list[str], binary_feature_columns: list[str]) -> list[int]:
    binary = set(binary_feature_columns)
    return [index for index, column in enumerate(feature_columns) if column not in binary]


def _standardize_arrays(
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


def _select_threshold_from_grid(
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


def _nested_l2_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    continuous_indices: list[int],
    inner_splits: list,
    c_grid: list[float],
    random_state: int,
) -> tuple[np.ndarray, float]:
    best_c = c_grid[0]
    best_auc = -1.0
    for c_value in c_grid:
        aucs: list[float] = []
        for split in inner_splits:
            X_inner_train = X_train[split.train_idx]
            X_inner_eval = X_train[split.test_idx]
            y_inner_train = y_train[split.train_idx]
            y_inner_eval = y_train[split.test_idx]
            X_inner_train, X_inner_eval = _standardize_arrays(
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

    X_train_final, X_eval_final = _standardize_arrays(
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


def _fixed_l2_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    continuous_indices: list[int],
    c_value: float,
    random_state: int,
) -> np.ndarray:
    X_train_final, X_eval_final = _standardize_arrays(
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


def _default_l2_c(c_grid: list[float]) -> float:
    if 1.0 in c_grid:
        return 1.0
    if not c_grid:
        raise RuntimeError("reproduction.logistic_c_grid must contain at least one value.")
    return float(c_grid[len(c_grid) // 2])


def _apply_exit_override(scores: np.ndarray, exit_counts: np.ndarray) -> np.ndarray:
    updated = scores.copy()
    updated[np.asarray(exit_counts, dtype=float) > 0] = 1.0
    return updated


def _rank_lambda_features(
    train_frame: pd.DataFrame,
    *,
    base_feature_columns: list[str],
    lambda_feature_columns: list[str],
    y_train: np.ndarray,
    config: ExperimentConfig,
) -> list[str]:
    features = base_feature_columns + lambda_feature_columns
    X = train_frame[features].fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ranking_spec = config.reproduction.lambda_ranking
    model = LogisticRegression(
        penalty="l1",
        C=ranking_spec.c,
        solver=ranking_spec.solver,
        class_weight=ranking_spec.class_weight,
        max_iter=ranking_spec.max_iter,
        random_state=ranking_spec.random_state,
    )
    model.fit(X_scaled, y_train)
    coefficients = model.coef_[0]
    n_base = len(base_feature_columns)
    ranked = sorted(
        [
            (lambda_feature_columns[index], abs(coefficients[n_base + index]))
            for index in range(len(lambda_feature_columns))
            if abs(coefficients[n_base + index]) > 1e-6
        ],
        key=lambda item: -item[1],
    )
    return [name for name, _ in ranked]


def _assemble_experiment_frames(
    experiment: ReproductionExperimentSpec,
    *,
    repository_splits: LoadedFeatureRepositorySplits,
    banks_by_id: dict,
    ranked_lambda_columns: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_frame = pd.DataFrame({"founder_uuid": repository_splits.train_ids})
    test_frame = pd.DataFrame({"founder_uuid": repository_splits.test_ids})
    feature_columns: list[str] = []
    binary_feature_columns: list[str] = []

    for bank_id in experiment.feature_bank_ids:
        bank = banks_by_id[bank_id]
        selected_columns = bank.feature_columns
        if bank_id == "lambda_policies" and experiment.lambda_top_k is not None:
            ranking = ranked_lambda_columns[experiment.lambda_rank_base_bank_id or ""]
            selected_columns = ranking[: experiment.lambda_top_k]
        train_frame = _merge_feature_bank(
            train_frame,
            bank.public_frame,
            feature_columns=selected_columns,
            bank_id=bank_id,
            split_label="train",
        )
        test_frame = _merge_feature_bank(
            test_frame,
            bank.private_frame,
            feature_columns=selected_columns,
            bank_id=bank_id,
            split_label="test",
        )
        feature_columns.extend(selected_columns)
        binary_feature_columns.extend(
            [column for column in selected_columns if column in set(bank.binary_feature_columns)]
        )

    if test_frame[feature_columns].isna().any().any():
        raise RuntimeError(f"Reproduction experiment '{experiment.experiment_id}' has missing test features.")

    if experiment.training_pool == "full":
        train_use = train_frame.copy()
    else:
        missing_mask = train_frame[feature_columns].isna().any(axis=1)
        train_use = train_frame.loc[~missing_mask].reset_index(drop=True)
    return train_use, test_frame, feature_columns, sorted(set(binary_feature_columns))


def run_reproduction_mode(
    config: ExperimentConfig,
    *,
    use_nested_hyperparameter_cv: bool = True,
    logger: Logger | None = None,
) -> Path:
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "success_reproduction")
    write_json(run_dir / "resolved_config.json", asdict(config))

    _log(logger, "Loading Feature Repository splits and baseline banks.")
    repository_splits = load_feature_repository_splits(config.feature_repository)
    banks_by_id = load_repository_feature_banks(
        repository_splits=repository_splits,
        specs=config.repository_feature_banks,
    )
    write_json(
        run_dir / "repository_feature_banks.json",
        {feature_id: bank.manifest for feature_id, bank in banks_by_id.items()},
    )

    labels_train = repository_splits.train_labels.set_index("founder_uuid").reindex(repository_splits.train_ids)
    y_test: np.ndarray | None = None

    ranked_lambda_columns: dict[str, list[str]] = {}
    lambda_bank = banks_by_id["lambda_policies"]
    for base_bank_id in sorted(
        {
            experiment.lambda_rank_base_bank_id
            for experiment in config.reproduction.experiments
            if experiment.lambda_rank_base_bank_id
        }
    ):
        base_bank = banks_by_id[base_bank_id]
        base_train = pd.DataFrame({"founder_uuid": repository_splits.train_ids})
        base_train = _merge_feature_bank(
            base_train,
            base_bank.public_frame,
            feature_columns=base_bank.feature_columns,
            bank_id=base_bank_id,
            split_label="train",
        )
        base_train = _merge_feature_bank(
            base_train,
            lambda_bank.public_frame,
            feature_columns=lambda_bank.feature_columns,
            bank_id="lambda_policies",
            split_label="train",
        )
        valid_mask = ~base_train[base_bank.feature_columns + lambda_bank.feature_columns].isna().any(axis=1)
        base_train = base_train.loc[valid_mask].reset_index(drop=True)
        y_rank = labels_train.reindex(base_train["founder_uuid"])["success"].to_numpy(dtype=int)
        ranked_lambda_columns[base_bank_id] = _rank_lambda_features(
            base_train,
            base_feature_columns=base_bank.feature_columns,
            lambda_feature_columns=lambda_bank.feature_columns,
            y_train=y_rank,
            config=config,
        )

    result_rows: list[dict[str, object]] = []
    oof_predictions = pd.DataFrame({"founder_uuid": repository_splits.train_ids})
    test_predictions = pd.DataFrame({"founder_uuid": repository_splits.test_ids})

    for experiment_index, experiment in enumerate(config.reproduction.experiments):
        _log(logger, f"Running reproduction experiment '{experiment.experiment_id}'.")
        train_frame, test_frame, feature_columns, binary_feature_columns = _assemble_experiment_frames(
            experiment,
            repository_splits=repository_splits,
            banks_by_id=banks_by_id,
            ranked_lambda_columns=ranked_lambda_columns,
        )
        y_train = labels_train.reindex(train_frame["founder_uuid"])["success"].to_numpy(dtype=int)
        X_train = train_frame[feature_columns].fillna(0.0).to_numpy(dtype=float)
        X_test = test_frame[feature_columns].fillna(0.0).to_numpy(dtype=float)
        exit_train = train_frame["exit_count"].to_numpy(dtype=float) if "exit_count" in train_frame.columns else np.zeros(len(train_frame))
        exit_test = test_frame["exit_count"].to_numpy(dtype=float) if "exit_count" in test_frame.columns else np.zeros(len(test_frame))

        outer_splits = build_public_cv_splits(
            y_train,
            n_splits=config.reproduction.outer_cv.n_splits,
            shuffle=config.reproduction.outer_cv.shuffle,
            random_state=config.reproduction.outer_cv.random_state,
        )
        continuous_indices = _continuous_indices(feature_columns, binary_feature_columns)
        oof = np.full(len(train_frame), np.nan, dtype=float)
        selected_cs: list[float] = []

        for fold_offset, split in enumerate(outer_splits):
            X_outer_train = X_train[split.train_idx]
            X_outer_eval = X_train[split.test_idx]
            y_outer_train = y_train[split.train_idx]
            if experiment.model_kind == "nested_l2_logreg" and use_nested_hyperparameter_cv:
                inner_splits = build_public_cv_splits(
                    y_outer_train,
                    n_splits=config.reproduction.inner_cv.n_splits,
                    shuffle=config.reproduction.inner_cv.shuffle,
                    random_state=config.reproduction.inner_cv.random_state,
                )
                preds, chosen_c = _nested_l2_predict_proba(
                    X_outer_train,
                    y_outer_train,
                    X_outer_eval,
                    continuous_indices=continuous_indices if experiment.standardize else [],
                    inner_splits=inner_splits,
                    c_grid=config.reproduction.logistic_c_grid,
                    random_state=config.reproduction.outer_cv.random_state + fold_offset,
                )
                selected_cs.append(chosen_c)
            elif experiment.model_kind == "nested_l2_logreg":
                fixed_c = _default_l2_c(config.reproduction.logistic_c_grid)
                preds = _fixed_l2_predict_proba(
                    X_outer_train,
                    y_outer_train,
                    X_outer_eval,
                    continuous_indices=continuous_indices if experiment.standardize else [],
                    c_value=fixed_c,
                    random_state=config.reproduction.outer_cv.random_state + fold_offset,
                )
                selected_cs.append(fixed_c)
            else:
                if experiment.standardize:
                    X_outer_train_use, X_outer_eval_use = _standardize_arrays(
                        X_outer_train,
                        X_outer_eval,
                        continuous_indices=continuous_indices,
                    )
                else:
                    X_outer_train_use, X_outer_eval_use = X_outer_train, X_outer_eval
                model = build_reproduction_classifier(
                    experiment.model_kind,
                    random_state=config.reproduction.outer_cv.random_state + fold_offset,
                )
                model.fit(X_outer_train_use, y_outer_train)
                preds = model.predict_proba(X_outer_eval_use)[:, 1]

            if experiment.use_exit_override:
                preds = _apply_exit_override(preds, exit_train[split.test_idx])
            oof[split.test_idx] = preds

        threshold, cv_f05 = _select_threshold_from_grid(
            y_train,
            oof,
            start=config.reproduction.threshold_grid.start,
            stop=config.reproduction.threshold_grid.stop,
            step=config.reproduction.threshold_grid.step,
        )
        cv_metrics = binary_classification_metrics(y_train, oof, threshold=threshold)

        if experiment.model_kind == "nested_l2_logreg" and use_nested_hyperparameter_cv:
            inner_splits = build_public_cv_splits(
                y_train,
                n_splits=config.reproduction.inner_cv.n_splits,
                shuffle=config.reproduction.inner_cv.shuffle,
                random_state=config.reproduction.inner_cv.random_state,
            )
            test_probs, final_c = _nested_l2_predict_proba(
                X_train,
                y_train,
                X_test,
                continuous_indices=continuous_indices if experiment.standardize else [],
                inner_splits=inner_splits,
                c_grid=config.reproduction.logistic_c_grid,
                random_state=config.reproduction.outer_cv.random_state + experiment_index,
            )
        elif experiment.model_kind == "nested_l2_logreg":
            final_c = _default_l2_c(config.reproduction.logistic_c_grid)
            test_probs = _fixed_l2_predict_proba(
                X_train,
                y_train,
                X_test,
                continuous_indices=continuous_indices if experiment.standardize else [],
                c_value=final_c,
                random_state=config.reproduction.outer_cv.random_state + experiment_index,
            )
        else:
            if experiment.standardize:
                X_train_final, X_test_final = _standardize_arrays(
                    X_train,
                    X_test,
                    continuous_indices=continuous_indices,
                )
            else:
                X_train_final, X_test_final = X_train, X_test
            model = build_reproduction_classifier(
                experiment.model_kind,
                random_state=config.reproduction.outer_cv.random_state + experiment_index,
            )
            model.fit(X_train_final, y_train)
            test_probs = model.predict_proba(X_test_final)[:, 1]
            final_c = None

        if experiment.use_exit_override:
            test_probs = _apply_exit_override(test_probs, exit_test)
        if y_test is None:
            # Test labels are resolved lazily and used only for final held-out evaluation.
            y_test = _load_test_success_labels_for_final_evaluation(repository_splits)
        test_metrics = binary_classification_metrics(y_test, test_probs, threshold=threshold)

        result_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "title": experiment.title,
                "feature_bank_ids": ",".join(experiment.feature_bank_ids),
                "training_pool": experiment.training_pool,
                "model_kind": experiment.model_kind,
                "n_features": len(feature_columns),
                "train_row_count": len(train_frame),
                "test_row_count": len(test_frame),
                "threshold": threshold,
                "selected_c_final": final_c,
                "selected_c_oof_mean": float(np.mean(selected_cs)) if selected_cs else None,
                "cv_f0_5": cv_f05,
                "cv_precision": cv_metrics["precision"],
                "cv_recall": cv_metrics["recall"],
                "cv_roc_auc": cv_metrics["roc_auc"],
                "test_f0_5": test_metrics["f0_5"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_brier": test_metrics["brier"],
            }
        )

        oof_column = f"{experiment.experiment_id}__oof_score"
        oof_series = pd.Series(np.nan, index=pd.Index(repository_splits.train_ids, name="founder_uuid"), dtype=float)
        oof_series.loc[train_frame["founder_uuid"].astype(str)] = oof
        oof_predictions[oof_column] = oof_series.reindex(repository_splits.train_ids).to_numpy(dtype=float)
        test_predictions[f"{experiment.experiment_id}__test_score"] = test_probs

    results_frame = pd.DataFrame(result_rows).sort_values("experiment_id").reset_index(drop=True)
    write_csv(run_dir / "reproduction_results.csv", results_frame)
    write_csv(run_dir / "reproduction_oof_predictions.csv", oof_predictions)
    write_csv(run_dir / "reproduction_test_predictions.csv", test_predictions)
    write_json(run_dir / "lambda_rankings.json", ranked_lambda_columns)

    best_row = results_frame.sort_values("test_f0_5", ascending=False).iloc[0]
    summary_lines = [
        "# Reproduction Summary",
        "",
        f"- Experiments run: {len(results_frame)}",
        f"- Best experiment: `{best_row['experiment_id']}`",
        f"- Best held-out F0.5: {best_row['test_f0_5']:.4f}",
        f"- Default run mode: `{config.defaults.run_mode}`",
        f"- Nested hyperparameter CV enabled: {use_nested_hyperparameter_cv}",
        "- This mode reproduces the success-prediction headline matrix from Feature Repository.",
    ]
    write_markdown(run_dir / "run_summary.md", "\n".join(summary_lines))
    _log(logger, f"Reproduction run complete. Artifacts written to {run_dir}.")
    return run_dir
