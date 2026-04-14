from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from src.data.feature_repository import load_feature_repository_splits, load_repository_feature_banks
from src.data.raw_datasets import load_raw_datasets
from src.data.splits import build_stratified_reasoning_cv_splits
from src.data.targets import load_target_family
from src.evaluation.metrics import binary_classification_metrics, regression_metrics, select_f05_threshold
from src.intermediary_features.registry import assemble_feature_sets, prepare_intermediary_banks
from src.pipeline.config import ExperimentConfig
from src.pipeline.run_options import MODEL_FAMILY_TO_MODEL_ID, RunOverrides, resolve_run_options
from src.student.models import build_reasoning_classifier, build_reasoning_regressor
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.paths import RUNS_DIR


Logger = Callable[[str], None]


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _require_full_overlap(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    *,
    on: str,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    merged = left_frame.merge(right_frame, on=on, how="inner", validate="one_to_one")
    if len(merged) != len(left_frame):
        missing = sorted(set(left_frame[on].astype(str)) - set(right_frame[on].astype(str)))
        raise RuntimeError(
            f"{right_name} is missing {len(missing)} ids required by {left_name}. Examples: {missing[:5]}"
        )
    return merged


def _resolve_stage_a_model_families(requested_families: list[str]) -> list[str]:
    allowed = {"linear_l2", "xgb1"}
    selected = [family for family in requested_families if family in allowed]
    if not selected:
        raise RuntimeError(
            "Model-testing Stage A requires at least one screening family enabled: "
            "`linear_l2` and/or `xgb1`."
        )
    return selected


def _nested_param_grid(model_kind: str, task_kind: str) -> list[dict[str, float | int]]:
    if task_kind == "regression":
        if model_kind == "ridge":
            return [{"alpha": value} for value in (0.1, 1.0, 10.0, 50.0)]
        if model_kind == "xgb1_regressor":
            grid: list[dict[str, float | int]] = []
            for n_estimators in (120, 227):
                for learning_rate in (0.04, 0.0674):
                    for max_depth in (1, 2):
                        grid.append(
                            {
                                "n_estimators": n_estimators,
                                "learning_rate": learning_rate,
                                "max_depth": max_depth,
                            }
                        )
            return grid
        if model_kind == "elasticnet_regressor":
            return [
                {"alpha": alpha, "l1_ratio": l1_ratio}
                for alpha in (0.001, 0.01, 0.05)
                for l1_ratio in (0.2, 0.5, 0.8)
            ]
    if task_kind == "classification":
        if model_kind == "logreg_classifier":
            return [{"C": value} for value in (0.05, 0.1, 0.5, 1.0, 5.0)]
        if model_kind == "xgb1_classifier":
            grid: list[dict[str, float | int]] = []
            for n_estimators in (120, 227):
                for learning_rate in (0.04, 0.0674):
                    for max_depth in (1, 2):
                        grid.append(
                            {
                                "n_estimators": n_estimators,
                                "learning_rate": learning_rate,
                                "max_depth": max_depth,
                            }
                        )
            return grid
        if model_kind == "elasticnet_logreg_classifier":
            return [
                {"C": c, "l1_ratio": l1_ratio}
                for c in (0.1, 1.0, 5.0)
                for l1_ratio in (0.2, 0.5, 0.8)
            ]
    return [{}]


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _predict_multi_output_probabilities(model: MultiOutputClassifier, X: np.ndarray, n_targets: int) -> np.ndarray:
    probs_list = model.predict_proba(X)
    if not isinstance(probs_list, list):
        probs_arr = np.asarray(probs_list, dtype=float)
        if probs_arr.ndim == 2 and probs_arr.shape[1] == n_targets:
            return probs_arr
        raise RuntimeError("Unexpected predict_proba shape for multi-output classification model.")
    columns: list[np.ndarray] = []
    for target_idx, probs in enumerate(probs_list):
        if target_idx >= n_targets:
            break
        arr = np.asarray(probs, dtype=float)
        if arr.ndim == 1:
            columns.append(arr)
        elif arr.shape[1] == 1:
            columns.append(arr[:, 0])
        else:
            columns.append(arr[:, 1])
    if len(columns) != n_targets:
        raise RuntimeError("Multi-output probability matrix column count does not match target count.")
    return np.column_stack(columns)


def _select_best_params_multi_output_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_kind: str,
    random_state: int,
    inner_n_splits: int,
    inner_shuffle: bool,
) -> dict[str, float | int]:
    candidates = _nested_param_grid(model_kind, "regression")
    inner_splits = build_stratified_reasoning_cv_splits(
        pd.DataFrame(y_train, columns=[f"target_{idx}" for idx in range(y_train.shape[1])]),
        n_splits=inner_n_splits,
        shuffle=inner_shuffle,
        random_state=random_state,
    )
    best_candidate = candidates[0]
    best_score = -np.inf
    for candidate in candidates:
        fold_scores: list[float] = []
        for split in inner_splits:
            base = build_reasoning_regressor(
                model_kind,
                random_state=random_state,
                param_overrides=candidate,
            )
            model = MultiOutputRegressor(base)
            model.fit(X_train[split.train_idx], y_train[split.train_idx])
            preds = model.predict(X_train[split.test_idx])
            per_target_scores = [
                float(r2_score(y_train[split.test_idx, target_idx], preds[:, target_idx]))
                for target_idx in range(y_train.shape[1])
            ]
            fold_scores.append(float(np.mean(per_target_scores)))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_candidate = candidate
    return best_candidate


def _select_best_params_multi_output_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_kind: str,
    random_state: int,
    inner_n_splits: int,
    inner_shuffle: bool,
) -> dict[str, float | int]:
    candidates = _nested_param_grid(model_kind, "classification")
    inner_splits = build_stratified_reasoning_cv_splits(
        pd.DataFrame(y_train, columns=[f"target_{idx}" for idx in range(y_train.shape[1])]),
        n_splits=inner_n_splits,
        shuffle=inner_shuffle,
        random_state=random_state,
    )
    best_candidate = candidates[0]
    best_score = -np.inf
    for candidate in candidates:
        fold_scores: list[float] = []
        for split in inner_splits:
            base = build_reasoning_classifier(
                model_kind,
                random_state=random_state,
                param_overrides=candidate,
            )
            model = MultiOutputClassifier(base)
            model.fit(X_train[split.train_idx], y_train[split.train_idx])
            probs = _predict_multi_output_probabilities(model, X_train[split.test_idx], y_train.shape[1])
            per_target_scores = [
                _safe_roc_auc(y_train[split.test_idx, target_idx], probs[:, target_idx])
                for target_idx in range(y_train.shape[1])
            ]
            fold_scores.append(float(np.mean(per_target_scores)))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_candidate = candidate
    return best_candidate


def _load_reasoning_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "reasoning_metrics.csv"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing reasoning metrics artifact: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    return frame[frame["split_id"] == "oof_overall"].copy()


def _repeat_seeds(config: ExperimentConfig, repeat_count: int) -> list[int]:
    base = config.distillation_cv.random_state
    return [base + (index * 10_000) for index in range(repeat_count)]


def _resolve_model_ids(
    config: ExperimentConfig,
    *,
    task_kind: str,
    model_families: list[str],
) -> list[str]:
    available_by_id = {spec.model_id: spec for spec in config.distillation_models}
    mapped_ids = [MODEL_FAMILY_TO_MODEL_ID[task_kind][family] for family in model_families]
    missing = [model_id for model_id in mapped_ids if model_id not in available_by_id]
    if missing:
        raise RuntimeError(
            f"model_testing_mode requested model ids not present in distillation_models: {missing}"
        )
    return mapped_ids


def _group_repeat_metrics(
    metrics_frame: pd.DataFrame,
    *,
    repeat_index: int,
    repeat_seed: int,
    target_family: str,
    stage: str,
    output_mode: str,
) -> pd.DataFrame:
    grouped = (
        metrics_frame.groupby(["feature_set_id", "model_id", "output_mode"], as_index=False)
        .mean(numeric_only=True)
    )
    grouped["repeat_index"] = repeat_index
    grouped["repeat_seed"] = repeat_seed
    grouped["target_family"] = target_family
    grouped["stage"] = stage
    grouped["output_mode"] = output_mode
    return grouped


def _aggregate_screening_metrics(
    repeat_model_metrics: pd.DataFrame,
    *,
    task_kind: str,
    score_delta: float,
    max_recommended: int,
) -> pd.DataFrame:
    if repeat_model_metrics.empty:
        return pd.DataFrame()
    for column in ("r2", "rmse", "mae", "f0_5", "roc_auc", "pr_auc"):
        if column not in repeat_model_metrics.columns:
            repeat_model_metrics[column] = float("nan")

    primary_column = "r2" if task_kind == "regression" else "f0_5"

    per_repeat_feature = (
        repeat_model_metrics.groupby(
            ["target_family", "output_mode", "feature_set_id", "repeat_index"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )
    screening = (
        per_repeat_feature.groupby(["target_family", "output_mode", "feature_set_id"], as_index=False)
        .agg(
            primary_mean=(primary_column, "mean"),
            primary_std=(primary_column, "std"),
            r2_mean=("r2", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            f0_5_mean=("f0_5", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
        )
    )
    screening["primary_std"] = screening["primary_std"].fillna(0.0)
    screening["screen_score"] = screening["primary_mean"] - (0.5 * screening["primary_std"])
    screening["primary_metric"] = primary_column
    screening["recommended_take_forward"] = False
    screening["rank"] = 0

    output_rows: list[pd.DataFrame] = []
    for _, family_frame in screening.groupby(["target_family", "output_mode"], sort=False):
        ranked = family_frame.sort_values("screen_score", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        best_score = float(ranked.iloc[0]["screen_score"])
        threshold = best_score - float(score_delta)
        mask = ranked["screen_score"] >= threshold
        recommended_indices = ranked.index[mask].tolist()
        if not recommended_indices:
            recommended_indices = [0]
        recommended_indices = recommended_indices[:max_recommended]
        ranked.loc[recommended_indices, "recommended_take_forward"] = True
        output_rows.append(ranked)

    output = pd.concat(output_rows, ignore_index=True)
    columns = [
        "target_family",
        "output_mode",
        "feature_set_id",
        "rank",
        "primary_metric",
        "primary_mean",
        "primary_std",
        "screen_score",
        "recommended_take_forward",
        "r2_mean",
        "rmse_mean",
        "mae_mean",
        "f0_5_mean",
        "roc_auc_mean",
        "pr_auc_mean",
    ]
    return output[columns].sort_values(["target_family", "output_mode", "rank"]).reset_index(drop=True)


def _aggregate_model_results(
    repeat_model_metrics: pd.DataFrame,
    *,
    task_kind: str,
) -> pd.DataFrame:
    if repeat_model_metrics.empty:
        return pd.DataFrame()
    for column in ("r2", "rmse", "mae", "f0_5", "roc_auc", "pr_auc"):
        if column not in repeat_model_metrics.columns:
            repeat_model_metrics[column] = float("nan")
    primary_column = "r2" if task_kind == "regression" else "f0_5"
    grouped = (
        repeat_model_metrics.groupby(["target_family", "output_mode", "feature_set_id", "model_id"], as_index=False)
        .agg(
            primary_mean=(primary_column, "mean"),
            primary_std=(primary_column, "std"),
            r2_mean=("r2", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            f0_5_mean=("f0_5", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
        )
    )
    grouped["primary_std"] = grouped["primary_std"].fillna(0.0)
    grouped["primary_metric"] = primary_column
    columns = [
        "target_family",
        "output_mode",
        "feature_set_id",
        "model_id",
        "primary_metric",
        "primary_mean",
        "primary_std",
        "r2_mean",
        "rmse_mean",
        "mae_mean",
        "f0_5_mean",
        "roc_auc_mean",
        "pr_auc_mean",
    ]
    return grouped[columns].sort_values(
        ["target_family", "output_mode", "feature_set_id", "primary_mean"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def _render_screening_markdown(
    screening: pd.DataFrame,
    *,
    repeat_count: int,
    stage_a_model_families: list[str],
    score_delta: float,
    max_recommended: int,
) -> str:
    lines = [
        "# Feature-Set Screening Report",
        "",
        f"- Repeats: {repeat_count}",
        f"- Stage A models: {', '.join(f'`{family}`' for family in stage_a_model_families)}",
        "- Held-out features/targets: not used",
        f"- Recommendation rule: top score + any within `best - {score_delta}` (max {max_recommended}).",
        "",
    ]
    if screening.empty:
        lines.append("No screening results were produced.")
        return "\n".join(lines)

    for (target_family, output_mode), frame in screening.groupby(["target_family", "output_mode"], sort=False):
        lines.extend(
            [
                f"## {target_family} | {output_mode}",
                "",
                "| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in frame.sort_values("rank").itertuples(index=False):
            lines.append(
                f"| {row.rank} | {row.feature_set_id} | {row.primary_mean:.4f} | "
                f"{row.primary_std:.4f} | {row.screen_score:.4f} | {bool(row.recommended_take_forward)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _render_model_testing_markdown(
    results: pd.DataFrame,
    *,
    repeat_count: int,
) -> str:
    lines = [
        "# Model Testing Report",
        "",
        f"- Repeats: {repeat_count}",
        "- This report compares shortlisted feature sets by model family.",
        "",
    ]
    if results.empty:
        lines.append("Advanced model stage was skipped or produced no rows.")
        return "\n".join(lines)

    for (target_family, output_mode, feature_set_id), frame in results.groupby(
        ["target_family", "output_mode", "feature_set_id"], sort=False
    ):
        lines.extend(
            [
                f"## {target_family} | {output_mode} | {feature_set_id}",
                "",
                "| model_id | primary_mean | primary_std |",
                "|---|---:|---:|",
            ]
        )
        for row in frame.sort_values("primary_mean", ascending=False).itertuples(index=False):
            lines.append(
                f"| {row.model_id} | {row.primary_mean:.4f} | {row.primary_std:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _resolve_family_sequence(
    *,
    overrides: RunOverrides,
    resolved_target_family: str,
) -> list[str]:
    if overrides.target_family == "v25_and_taste":
        return ["v25_policies", "taste_policies"]
    return [resolved_target_family]


def _run_stage(
    config: ExperimentConfig,
    *,
    base_overrides: RunOverrides,
    family_id: str,
    feature_set_ids: list[str],
    model_ids: list[str],
    repeat_count: int,
    force_rebuild_intermediary_features: bool,
    nested_sweep: bool,
    output_mode: str,
    logger: Logger | None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if output_mode == "single_target":
        from src.pipeline.distillation import run_reasoning_distillation_mode

        seeds = _repeat_seeds(config, repeat_count)
        stage_rows: list[pd.DataFrame] = []
        child_runs: list[dict[str, object]] = []
        for repeat_index, seed in enumerate(seeds):
            if repeat_index > 0:
                _log(logger, f"{family_id}: repeat {repeat_index + 1}/{repeat_count} (seed={seed})")
            config_seeded = replace(
                config,
                distillation_cv=replace(config.distillation_cv, random_state=seed),
                reproduction=replace(
                    config.reproduction,
                    outer_cv=replace(config.reproduction.outer_cv, random_state=seed),
                    inner_cv=replace(config.reproduction.inner_cv, random_state=seed),
                ),
            )
            run_overrides = replace(
                base_overrides,
                run_mode="reasoning_distillation_mode",
                target_family=family_id,
                output_modes=[output_mode],
                heldout_evaluation=False,
                repeat_cv_with_new_seeds=False,
                cv_seed_repeat_count=1,
                distillation_nested_sweep=nested_sweep,
                save_reasoning_predictions=False,
                force_rebuild_intermediary_features=(
                    force_rebuild_intermediary_features if repeat_index == 0 else False
                ),
                reasoning_models=model_ids,
                candidate_feature_sets=feature_set_ids,
            )
            run_dir = run_reasoning_distillation_mode(config_seeded, run_overrides, logger=logger)
            child_runs.append(
                {
                    "target_family": family_id,
                    "repeat_index": repeat_index,
                    "repeat_seed": seed,
                    "run_dir": str(run_dir),
                    "output_mode": output_mode,
                }
            )
            metrics = _load_reasoning_metrics(run_dir)
            metrics["output_mode"] = output_mode
            stage_rows.append(
                _group_repeat_metrics(
                    metrics,
                    repeat_index=repeat_index,
                    repeat_seed=seed,
                    target_family=family_id,
                    stage="screening",
                    output_mode=output_mode,
                )
            )
        return pd.concat(stage_rows, ignore_index=True), child_runs

    if output_mode != "multi_output":
        raise RuntimeError(f"Unsupported output_mode '{output_mode}'.")

    seeds = _repeat_seeds(config, repeat_count)
    stage_rows = []
    child_runs = []
    for repeat_index, seed in enumerate(seeds):
        if repeat_index > 0:
            _log(
                logger,
                f"{family_id} ({output_mode}): repeat {repeat_index + 1}/{repeat_count} (seed={seed})",
            )
        config_seeded = replace(
            config,
            distillation_cv=replace(config.distillation_cv, random_state=seed),
            reproduction=replace(
                config.reproduction,
                outer_cv=replace(config.reproduction.outer_cv, random_state=seed),
                inner_cv=replace(config.reproduction.inner_cv, random_state=seed),
            ),
        )
        run_overrides = replace(
            base_overrides,
            run_mode="reasoning_distillation_mode",
            target_family=family_id,
            output_modes=[output_mode],
            heldout_evaluation=False,
            repeat_cv_with_new_seeds=False,
            cv_seed_repeat_count=1,
            distillation_nested_sweep=nested_sweep,
            save_reasoning_predictions=False,
            force_rebuild_intermediary_features=(
                force_rebuild_intermediary_features if repeat_index == 0 else False
            ),
            reasoning_models=model_ids,
            candidate_feature_sets=feature_set_ids,
        )
        resolved_run = resolve_run_options(config_seeded, run_overrides)
        raw_datasets = load_raw_datasets(
            Path(config_seeded.datasets.public_train_csv),
            Path(config_seeded.datasets.private_test_csv),
        )
        repository_splits = load_feature_repository_splits(config_seeded.feature_repository)
        target_family = load_target_family(resolved_run.target_family)
        repository_banks = load_repository_feature_banks(
            repository_splits=repository_splits,
            specs=resolved_run.repository_feature_banks,
        )
        intermediary_banks = prepare_intermediary_banks(
            public_raw=raw_datasets.public_frame,
            private_raw=raw_datasets.private_frame,
            feature_specs=resolved_run.intermediary_features,
            force_rebuild=resolved_run.force_rebuild_intermediary_features,
            logger=logger,
        )
        feature_sets = assemble_feature_sets(
            public_founder_ids=raw_datasets.public_frame["founder_uuid"],
            private_founder_ids=raw_datasets.private_frame["founder_uuid"],
            banks_by_id={**repository_banks, **intermediary_banks},
            feature_sets=resolved_run.distillation_feature_sets,
        )
        outer_n_splits = (
            config_seeded.reproduction.outer_cv.n_splits
            if nested_sweep
            else config_seeded.distillation_cv.n_splits
        )

        repeat_metric_rows: list[dict[str, object]] = []
        for feature_set in feature_sets:
            public_target_rows = _require_full_overlap(
                feature_set.public_frame[["founder_uuid"]],
                target_family.train_frame,
                on="founder_uuid",
                left_name=f"feature set '{feature_set.feature_set_id}' public rows",
                right_name=f"target family '{target_family.family_id}' public targets",
            )
            X_public = feature_set.public_frame[feature_set.feature_columns].to_numpy(dtype=float)
            target_columns = target_family.target_columns
            if target_family.task_kind == "regression":
                y_public = public_target_rows[target_columns].to_numpy(dtype=float)
            else:
                y_public = public_target_rows[target_columns].to_numpy(dtype=int)
            splits = build_stratified_reasoning_cv_splits(
                pd.DataFrame(y_public, columns=target_columns),
                n_splits=outer_n_splits,
                shuffle=config_seeded.distillation_cv.shuffle,
                random_state=seed,
            )
            for model_offset, model_spec in enumerate(resolved_run.distillation_models):
                oof = np.full((len(X_public), len(target_columns)), np.nan, dtype=float)
                fold_predictions: dict[str, np.ndarray] = {}
                for fold_offset, split in enumerate(splits):
                    X_train = X_public[split.train_idx]
                    X_test = X_public[split.test_idx]
                    y_train = y_public[split.train_idx]
                    if target_family.task_kind == "regression":
                        best_params = (
                            _select_best_params_multi_output_regression(
                                X_train,
                                y_train,
                                model_kind=model_spec.kind,
                                random_state=seed + model_offset + fold_offset,
                                inner_n_splits=config_seeded.reproduction.inner_cv.n_splits,
                                inner_shuffle=config_seeded.reproduction.inner_cv.shuffle,
                            )
                            if nested_sweep
                            else {}
                        )
                        base = build_reasoning_regressor(
                            model_spec.kind,
                            random_state=seed + model_offset + fold_offset,
                            param_overrides=best_params,
                        )
                        model = MultiOutputRegressor(base)
                        model.fit(X_train, y_train)
                        preds = np.clip(
                            model.predict(X_test),
                            target_family.scale_min or 0.0,
                            target_family.scale_max or 1.0,
                        )
                    else:
                        best_params = (
                            _select_best_params_multi_output_classification(
                                X_train,
                                y_train,
                                model_kind=model_spec.kind,
                                random_state=seed + model_offset + fold_offset,
                                inner_n_splits=config_seeded.reproduction.inner_cv.n_splits,
                                inner_shuffle=config_seeded.reproduction.inner_cv.shuffle,
                            )
                            if nested_sweep
                            else {}
                        )
                        base = build_reasoning_classifier(
                            model_spec.kind,
                            random_state=seed + model_offset + fold_offset,
                            param_overrides=best_params,
                        )
                        model = MultiOutputClassifier(base)
                        model.fit(X_train, y_train)
                        preds = _predict_multi_output_probabilities(model, X_test, len(target_columns))
                    oof[split.test_idx] = preds
                    fold_predictions[split.split_id] = preds

                if np.isnan(oof).any():
                    raise RuntimeError(
                        f"Multi-output OOF contains NaNs for model '{model_spec.model_id}'."
                    )

                if target_family.task_kind == "regression":
                    for target_idx, target_column in enumerate(target_columns):
                        for split in splits:
                            preds = fold_predictions[split.split_id][:, target_idx]
                            y_true = y_public[split.test_idx, target_idx]
                            repeat_metric_rows.append(
                                {
                                    "feature_set_id": feature_set.feature_set_id,
                                    "target_id": target_column,
                                    "model_id": model_spec.model_id,
                                    "split_id": split.split_id,
                                    **regression_metrics(y_true, preds),
                                    "output_mode": output_mode,
                                }
                            )
                        repeat_metric_rows.append(
                            {
                                "feature_set_id": feature_set.feature_set_id,
                                "target_id": target_column,
                                "model_id": model_spec.model_id,
                                "split_id": "oof_overall",
                                **regression_metrics(y_public[:, target_idx], oof[:, target_idx]),
                                "output_mode": output_mode,
                            }
                        )
                else:
                    for target_idx, target_column in enumerate(target_columns):
                        threshold = select_f05_threshold(y_public[:, target_idx], oof[:, target_idx])
                        for split in splits:
                            preds = fold_predictions[split.split_id][:, target_idx]
                            y_true = y_public[split.test_idx, target_idx]
                            repeat_metric_rows.append(
                                {
                                    "feature_set_id": feature_set.feature_set_id,
                                    "target_id": target_column,
                                    "model_id": model_spec.model_id,
                                    "split_id": split.split_id,
                                    **binary_classification_metrics(y_true, preds, threshold=threshold),
                                    "output_mode": output_mode,
                                }
                            )
                        repeat_metric_rows.append(
                            {
                                "feature_set_id": feature_set.feature_set_id,
                                "target_id": target_column,
                                "model_id": model_spec.model_id,
                                "split_id": "oof_overall",
                                **binary_classification_metrics(
                                    y_public[:, target_idx],
                                    oof[:, target_idx],
                                    threshold=threshold,
                                ),
                                "output_mode": output_mode,
                            }
                        )

        repeat_metrics_frame = pd.DataFrame(repeat_metric_rows)
        stage_rows.append(
            _group_repeat_metrics(
                repeat_metrics_frame,
                repeat_index=repeat_index,
                repeat_seed=seed,
                target_family=family_id,
                stage="screening",
                output_mode=output_mode,
            )
        )
        child_runs.append(
            {
                "target_family": family_id,
                "repeat_index": repeat_index,
                "repeat_seed": seed,
                "run_dir": "in_memory_multi_output",
                "output_mode": output_mode,
            }
        )
    return pd.concat(stage_rows, ignore_index=True), child_runs


def run_model_testing_mode(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    overrides_use = overrides or RunOverrides()
    if overrides_use.heldout_evaluation is True:
        raise RuntimeError("model_testing_mode is training-only. heldout_evaluation must be false.")

    resolved = resolve_run_options(config, replace(overrides_use, run_mode="model_testing_mode"))
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "model_testing")
    write_json(run_dir / "resolved_config.json", asdict(config))
    write_json(run_dir / "resolved_run_options.json", asdict(resolved))

    feature_set_ids = [spec.feature_set_id for spec in resolved.distillation_feature_sets]
    if not feature_set_ids:
        raise RuntimeError("No candidate feature sets are available for model_testing_mode.")

    repeat_count = resolved.cv_seed_repeat_count if resolved.repeat_cv_with_new_seeds else 1
    family_sequence = _resolve_family_sequence(
        overrides=overrides_use,
        resolved_target_family=resolved.target_family.family_id,
    )
    family_map = {spec.family_id: spec for spec in config.target_families}
    output_modes = resolved.output_modes
    stage_a_model_families = _resolve_stage_a_model_families(resolved.model_families)

    stage_a_rows: list[pd.DataFrame] = []
    stage_a_child_runs: list[dict[str, object]] = []
    for family_id in family_sequence:
        for output_mode in output_modes:
            if family_id not in family_map:
                raise RuntimeError(f"Unknown target family '{family_id}'.")
            task_kind = family_map[family_id].task_kind
            stage_a_model_ids = _resolve_model_ids(
                config,
                task_kind=task_kind,
                model_families=stage_a_model_families,
            )
            _log(
                logger,
                f"Stage A screening for '{family_id}' ({output_mode}) with feature sets: {feature_set_ids}.",
            )
            stage_metrics, child_runs = _run_stage(
                config,
                base_overrides=overrides_use,
                family_id=family_id,
                feature_set_ids=feature_set_ids,
                model_ids=stage_a_model_ids,
                repeat_count=repeat_count,
                force_rebuild_intermediary_features=resolved.force_rebuild_intermediary_features,
                nested_sweep=resolved.distillation_nested_sweep,
                output_mode=output_mode,
                logger=logger,
            )
            stage_a_rows.append(stage_metrics)
            stage_a_child_runs.extend(child_runs)

    stage_a_repeat_metrics = pd.concat(stage_a_rows, ignore_index=True) if stage_a_rows else pd.DataFrame()
    write_csv(run_dir / "feature_set_screening_repeat_metrics.csv", stage_a_repeat_metrics)
    write_json(run_dir / "feature_set_screening_child_runs.json", stage_a_child_runs)

    screening_rows: list[pd.DataFrame] = []
    recommended_sets: dict[tuple[str, str], list[str]] = {}
    for family_id in family_sequence:
        for output_mode in output_modes:
            task_kind = family_map[family_id].task_kind
            family_metrics = stage_a_repeat_metrics[
                (stage_a_repeat_metrics["target_family"] == family_id)
                & (stage_a_repeat_metrics["output_mode"] == output_mode)
            ].copy()
            screening_family = _aggregate_screening_metrics(
                family_metrics,
                task_kind=task_kind,
                score_delta=config.model_testing.screening_score_delta,
                max_recommended=config.model_testing.max_recommended_feature_sets,
            )
            screening_rows.append(screening_family)
            if screening_family.empty:
                recommended_sets[(family_id, output_mode)] = []
                continue
            recommended_sets[(family_id, output_mode)] = (
                screening_family.loc[
                    screening_family["recommended_take_forward"],
                    "feature_set_id",
                ]
                .drop_duplicates()
                .astype(str)
                .tolist()
            )

    screening_frame = pd.concat(screening_rows, ignore_index=True) if screening_rows else pd.DataFrame()
    write_csv(run_dir / "feature_set_screening.csv", screening_frame)
    write_markdown(
        run_dir / "feature_set_screening_report.md",
        _render_screening_markdown(
            screening_frame,
            repeat_count=repeat_count,
            stage_a_model_families=stage_a_model_families,
            score_delta=config.model_testing.screening_score_delta,
            max_recommended=config.model_testing.max_recommended_feature_sets,
        ),
    )

    model_testing_results = pd.DataFrame()
    model_testing_runs: list[dict[str, object]] = []
    if resolved.run_advanced_models:
        stage_b_rows: list[pd.DataFrame] = []
        for family_id in family_sequence:
            for output_mode in output_modes:
                shortlisted = recommended_sets.get((family_id, output_mode), [])
                if not shortlisted:
                    continue
                task_kind = family_map[family_id].task_kind
                model_ids = _resolve_model_ids(
                    config,
                    task_kind=task_kind,
                    model_families=resolved.model_families,
                )
                _log(
                    logger,
                    f"Stage B model testing for '{family_id}' ({output_mode}) on shortlisted sets: {shortlisted}.",
                )
                stage_metrics, child_runs = _run_stage(
                    config,
                    base_overrides=overrides_use,
                    family_id=family_id,
                    feature_set_ids=shortlisted,
                    model_ids=model_ids,
                    repeat_count=repeat_count,
                    force_rebuild_intermediary_features=False,
                    nested_sweep=resolved.distillation_nested_sweep,
                    output_mode=output_mode,
                    logger=logger,
                )
                stage_metrics["stage"] = "advanced"
                stage_b_rows.append(stage_metrics)
                model_testing_runs.extend(child_runs)

        if stage_b_rows:
            stage_b_repeat_metrics = pd.concat(stage_b_rows, ignore_index=True)
            write_csv(run_dir / "model_testing_repeat_metrics.csv", stage_b_repeat_metrics)
            write_json(run_dir / "model_testing_child_runs.json", model_testing_runs)

            result_rows: list[pd.DataFrame] = []
            for family_id in family_sequence:
                for output_mode in output_modes:
                    family_task_kind = family_map[family_id].task_kind
                    family_frame = stage_b_repeat_metrics[
                        (stage_b_repeat_metrics["target_family"] == family_id)
                        & (stage_b_repeat_metrics["output_mode"] == output_mode)
                    ].copy()
                    result_rows.append(
                        _aggregate_model_results(family_frame, task_kind=family_task_kind)
                    )
            model_testing_results = pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()
    write_csv(run_dir / "model_testing_results.csv", model_testing_results)
    write_markdown(
        run_dir / "model_testing_report.md",
        _render_model_testing_markdown(
            model_testing_results,
            repeat_count=repeat_count,
        ),
    )

    summary_lines = [
        "# Model Testing Summary",
        "",
        f"- Candidate feature sets: {len(feature_set_ids)}",
        f"- Repeats: {repeat_count}",
        f"- Stage A model families: {', '.join(stage_a_model_families)}",
        f"- Output modes: {', '.join(output_modes)}",
        f"- Stage B enabled: {resolved.run_advanced_models}",
        "- Held-out/test features or targets are not used in this mode.",
    ]
    write_markdown(run_dir / "run_summary.md", "\n".join(summary_lines))
    _log(logger, f"Model testing run complete. Artifacts written to {run_dir}.")
    return run_dir
