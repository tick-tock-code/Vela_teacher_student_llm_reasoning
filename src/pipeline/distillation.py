from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

from src.data.feature_repository import load_feature_repository_splits, load_repository_feature_banks
from src.data.raw_datasets import load_raw_datasets
from src.data.splits import build_public_cv_splits, build_stratified_reasoning_cv_splits
from src.data.targets import load_target_family, target_manifest_payload
from src.evaluation.metrics import (
    binary_classification_metrics,
    regression_metrics,
    select_f05_threshold,
)
from src.intermediary_features.registry import assemble_feature_sets, prepare_intermediary_banks
from src.pipeline.config import ExperimentConfig
from src.pipeline.reproduction import run_reproduction_mode
from src.pipeline.run_options import RunOverrides, resolve_run_options
from src.student.models import build_reasoning_classifier, build_reasoning_regressor
from src.student.reasoning_classification import (
    fit_reasoning_classifiers_full,
    predict_reasoning_classifiers_full,
    train_reasoning_classifiers_oof,
)
from src.student.reasoning_regression import (
    fit_reasoning_regressors_full,
    predict_reasoning_regressors_full,
    train_reasoning_regressors_oof,
)
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.parallel import bounded_worker_count
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


def _prefix_prediction_columns(frame: pd.DataFrame, *, feature_set_id: str) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [
        column if column == "founder_uuid" else f"{feature_set_id}__{column}"
        for column in renamed.columns
    ]
    return renamed


def _merge_prediction_tables(current: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    if current is None:
        return incoming
    return current.merge(incoming, on="founder_uuid", how="outer", validate="one_to_one")


def _average_prediction_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise RuntimeError("Expected at least one prediction table to average.")
    if len(frames) == 1:
        return frames[0]
    indexed_frames = [frame.set_index("founder_uuid").sort_index() for frame in frames]
    template_columns = indexed_frames[0].columns.tolist()
    for frame in indexed_frames[1:]:
        if frame.columns.tolist() != template_columns:
            raise RuntimeError("Prediction tables do not share identical columns for averaging.")
        if not frame.index.equals(indexed_frames[0].index):
            raise RuntimeError("Prediction tables do not share identical founder_uuid indices for averaging.")
    stacked = np.stack([frame.to_numpy(dtype=float) for frame in indexed_frames], axis=0)
    averaged = np.mean(stacked, axis=0)
    averaged_frame = pd.DataFrame(
        averaged,
        index=indexed_frames[0].index,
        columns=template_columns,
    ).reset_index()
    return averaged_frame


def _average_metric_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise RuntimeError("Expected at least one metric table to average.")
    if len(frames) == 1:
        return frames[0]
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["target_id", "model_id", "split_id"], as_index=False)
        .mean(numeric_only=True)
    )
    return grouped


def _average_threshold_maps(
    threshold_maps: list[dict[tuple[str, str], float]],
) -> dict[tuple[str, str], float]:
    if not threshold_maps:
        return {}
    keys = set().union(*(threshold_map.keys() for threshold_map in threshold_maps))
    averaged: dict[tuple[str, str], float] = {}
    for key in keys:
        values = [threshold_map[key] for threshold_map in threshold_maps if key in threshold_map]
        averaged[key] = float(np.mean(values))
    return averaged


def _model_param_overrides_for(
    resolved_run,
    model_id: str,
) -> dict[str, float | int]:
    return dict((resolved_run.xgb_model_param_overrides_by_model_id or {}).get(model_id, {}))


def _render_reasoning_metrics_summary(
    *,
    target_family_id: str,
    task_kind: str,
    metrics_frame: pd.DataFrame,
    heldout_metrics_frame: pd.DataFrame | None,
) -> str:
    lines: list[str] = [
        "# Reasoning Metrics Summary",
        "",
        f"- Target family: `{target_family_id}`",
        f"- Task kind: `{task_kind}`",
        "",
    ]
    if metrics_frame.empty:
        lines.append("No public metrics were produced.")
        return "\n".join(lines)

    oof = metrics_frame[metrics_frame["split_id"] == "oof_overall"].copy()
    if oof.empty:
        lines.append("No `oof_overall` rows were found in public metrics.")
        return "\n".join(lines)

    if task_kind == "regression":
        summary = (
            oof.groupby(["feature_set_id", "model_id"], as_index=False)
            .agg(
                mean_r2=("r2", "mean"),
                mean_rmse=("rmse", "mean"),
                mean_mae=("mae", "mean"),
                mean_pearson=("pearson", "mean"),
                mean_spearman=("spearman", "mean"),
            )
            .sort_values("mean_r2", ascending=False)
        )
        lines.extend([
            "## Public OOF (Averaged Across Targets)",
            "",
            "| feature_set_id | model_id | mean_r2 | mean_rmse | mean_mae | mean_pearson | mean_spearman |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.feature_set_id} | {row.model_id} | {row.mean_r2:.4f} | {row.mean_rmse:.4f} | "
                f"{row.mean_mae:.4f} | {row.mean_pearson:.4f} | {row.mean_spearman:.4f} |"
            )
    else:
        summary = (
            oof.groupby(["feature_set_id", "model_id"], as_index=False)
            .agg(
                mean_f0_5=("f0_5", "mean"),
                mean_roc_auc=("roc_auc", "mean"),
                mean_pr_auc=("pr_auc", "mean"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
            )
            .sort_values("mean_f0_5", ascending=False)
        )
        lines.extend([
            "## Public OOF (Averaged Across Targets)",
            "",
            "| feature_set_id | model_id | mean_f0_5 | mean_roc_auc | mean_pr_auc | mean_precision | mean_recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.feature_set_id} | {row.model_id} | {row.mean_f0_5:.4f} | {row.mean_roc_auc:.4f} | "
                f"{row.mean_pr_auc:.4f} | {row.mean_precision:.4f} | {row.mean_recall:.4f} |"
            )

    if heldout_metrics_frame is not None and not heldout_metrics_frame.empty:
        lines.extend(["", "## Held-Out (Averaged Across Targets)", ""])
        if task_kind == "regression":
            heldout_summary = (
                heldout_metrics_frame.groupby(["feature_set_id", "model_id"], as_index=False)
                .agg(
                    mean_r2=("r2", "mean"),
                    mean_rmse=("rmse", "mean"),
                    mean_mae=("mae", "mean"),
                    mean_pearson=("pearson", "mean"),
                    mean_spearman=("spearman", "mean"),
                )
                .sort_values("mean_r2", ascending=False)
            )
            lines.extend([
                "| feature_set_id | model_id | mean_r2 | mean_rmse | mean_mae | mean_pearson | mean_spearman |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for row in heldout_summary.itertuples(index=False):
                lines.append(
                    f"| {row.feature_set_id} | {row.model_id} | {row.mean_r2:.4f} | {row.mean_rmse:.4f} | "
                    f"{row.mean_mae:.4f} | {row.mean_pearson:.4f} | {row.mean_spearman:.4f} |"
                )
        else:
            heldout_summary = (
                heldout_metrics_frame.groupby(["feature_set_id", "model_id"], as_index=False)
                .agg(
                    mean_f0_5=("f0_5", "mean"),
                    mean_roc_auc=("roc_auc", "mean"),
                    mean_pr_auc=("pr_auc", "mean"),
                    mean_precision=("precision", "mean"),
                    mean_recall=("recall", "mean"),
                )
                .sort_values("mean_f0_5", ascending=False)
            )
            lines.extend([
                "| feature_set_id | model_id | mean_f0_5 | mean_roc_auc | mean_pr_auc | mean_precision | mean_recall |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for row in heldout_summary.itertuples(index=False):
                lines.append(
                    f"| {row.feature_set_id} | {row.model_id} | {row.mean_f0_5:.4f} | {row.mean_roc_auc:.4f} | "
                    f"{row.mean_pr_auc:.4f} | {row.mean_precision:.4f} | {row.mean_recall:.4f} |"
                )

    return "\n".join(lines)


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
    return [{}]


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _select_best_params_regression(
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
        pd.DataFrame({"target": y_train}),
        n_splits=inner_n_splits,
        shuffle=inner_shuffle,
        random_state=random_state,
    )
    best_candidate = candidates[0]
    best_score = -np.inf
    for candidate in candidates:
        scores: list[float] = []
        for split in inner_splits:
            model = build_reasoning_regressor(
                model_kind,
                random_state=random_state,
                param_overrides=candidate,
            )
            model.fit(X_train[split.train_idx], y_train[split.train_idx])
            preds = model.predict(X_train[split.test_idx])
            scores.append(float(r2_score(y_train[split.test_idx], preds)))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_candidate = candidate
    return best_candidate


def _select_best_params_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_kind: str,
    random_state: int,
    inner_n_splits: int,
    inner_shuffle: bool,
) -> dict[str, float | int]:
    candidates = _nested_param_grid(model_kind, "classification")
    inner_splits = build_public_cv_splits(
        y_train,
        n_splits=inner_n_splits,
        shuffle=inner_shuffle,
        random_state=random_state,
    )
    best_candidate = candidates[0]
    best_score = -np.inf
    for candidate in candidates:
        scores: list[float] = []
        for split in inner_splits:
            model = build_reasoning_classifier(
                model_kind,
                random_state=random_state,
                param_overrides=candidate,
            )
            model.fit(X_train[split.train_idx], y_train[split.train_idx])
            probs = model.predict_proba(X_train[split.test_idx])[:, 1]
            scores.append(_safe_roc_auc(y_train[split.test_idx], probs))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_candidate = candidate
    return best_candidate


def _train_nested_single_target_regression_oof(
    *,
    X_public: np.ndarray,
    y: np.ndarray,
    target_column: str,
    model_spec,
    model_offset: int,
    splits: list,
    repeat_seed: int,
    inner_n_splits: int,
    inner_shuffle: bool,
    scale_min: float,
    scale_max: float,
) -> tuple[str, np.ndarray, list[dict[str, object]]]:
    column_name = f"{target_column}__{model_spec.model_id}"
    oof = np.full(len(X_public), np.nan, dtype=float)
    metric_rows: list[dict[str, object]] = []
    for fold_offset, split in enumerate(splits):
        X_train = X_public[split.train_idx]
        X_test = X_public[split.test_idx]
        y_train = y[split.train_idx]
        y_test = y[split.test_idx]
        best_params = _select_best_params_regression(
            X_train,
            y_train,
            model_kind=model_spec.kind,
            random_state=repeat_seed + model_offset + fold_offset,
            inner_n_splits=inner_n_splits,
            inner_shuffle=inner_shuffle,
        )
        model = build_reasoning_regressor(
            model_spec.kind,
            random_state=repeat_seed + model_offset + fold_offset,
            param_overrides=best_params,
        )
        model.fit(X_train, y_train)
        preds = np.clip(
            model.predict(X_test),
            scale_min,
            scale_max,
        )
        oof[split.test_idx] = preds
        metric_rows.append(
            {
                "target_id": target_column,
                "model_id": model_spec.model_id,
                "split_id": split.split_id,
                **regression_metrics(y_test, preds),
            }
        )
    if np.isnan(oof).any():
        raise RuntimeError(
            f"Nested distillation OOF contains NaNs for target '{target_column}' and model '{model_spec.model_id}'."
        )
    metric_rows.append(
        {
            "target_id": target_column,
            "model_id": model_spec.model_id,
            "split_id": "oof_overall",
            **regression_metrics(y, oof),
        }
    )
    return column_name, oof, metric_rows


def _train_nested_single_target_classification_oof(
    *,
    X_public: np.ndarray,
    y: np.ndarray,
    target_column: str,
    model_spec,
    model_offset: int,
    splits: list,
    repeat_seed: int,
    inner_n_splits: int,
    inner_shuffle: bool,
) -> tuple[str, np.ndarray, tuple[str, str], float, list[dict[str, object]]]:
    column_name = f"{target_column}__{model_spec.model_id}"
    oof = np.full(len(X_public), np.nan, dtype=float)
    fold_probs: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, object]] = []
    for fold_offset, split in enumerate(splits):
        X_train = X_public[split.train_idx]
        X_test = X_public[split.test_idx]
        y_train = y[split.train_idx]
        best_params = _select_best_params_classification(
            X_train,
            y_train,
            model_kind=model_spec.kind,
            random_state=repeat_seed + model_offset + fold_offset,
            inner_n_splits=inner_n_splits,
            inner_shuffle=inner_shuffle,
        )
        model = build_reasoning_classifier(
            model_spec.kind,
            random_state=repeat_seed + model_offset + fold_offset,
            param_overrides=best_params,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        oof[split.test_idx] = probs
        fold_probs[split.split_id] = probs
    if np.isnan(oof).any():
        raise RuntimeError(
            f"Nested distillation OOF contains NaNs for target '{target_column}' and model '{model_spec.model_id}'."
        )
    threshold = select_f05_threshold(y, oof)
    for split in splits:
        metric_rows.append(
            {
                "target_id": target_column,
                "model_id": model_spec.model_id,
                "split_id": split.split_id,
                **binary_classification_metrics(
                    y[split.test_idx],
                    fold_probs[split.split_id],
                    threshold=threshold,
                ),
            }
        )
    metric_rows.append(
        {
            "target_id": target_column,
            "model_id": model_spec.model_id,
            "split_id": "oof_overall",
            **binary_classification_metrics(y, oof, threshold=threshold),
        }
    )
    return column_name, oof, (target_column, model_spec.model_id), threshold, metric_rows


def _fit_nested_single_target_regression_full(
    *,
    X_public: np.ndarray,
    y_train_full: np.ndarray,
    target_column: str,
    model_spec,
    model_offset: int,
    target_offset: int,
    inner_n_splits: int,
    inner_shuffle: bool,
    random_state_base: int,
) -> tuple[tuple[str, str], object]:
    random_state = random_state_base + model_offset + target_offset
    best_params = _select_best_params_regression(
        X_public,
        y_train_full,
        model_kind=model_spec.kind,
        random_state=random_state,
        inner_n_splits=inner_n_splits,
        inner_shuffle=inner_shuffle,
    )
    model = build_reasoning_regressor(
        model_spec.kind,
        random_state=random_state,
        param_overrides=best_params,
    )
    model.fit(X_public, y_train_full)
    return (target_column, model_spec.model_id), model


def _fit_nested_single_target_classification_full(
    *,
    X_public: np.ndarray,
    y_train_full: np.ndarray,
    target_column: str,
    model_spec,
    model_offset: int,
    target_offset: int,
    inner_n_splits: int,
    inner_shuffle: bool,
    random_state_base: int,
) -> tuple[tuple[str, str], object]:
    random_state = random_state_base + model_offset + target_offset
    best_params = _select_best_params_classification(
        X_public,
        y_train_full,
        model_kind=model_spec.kind,
        random_state=random_state,
        inner_n_splits=inner_n_splits,
        inner_shuffle=inner_shuffle,
    )
    model = build_reasoning_classifier(
        model_spec.kind,
        random_state=random_state,
        param_overrides=best_params,
    )
    model.fit(X_public, y_train_full)
    return (target_column, model_spec.model_id), model


def _evaluate_heldout_regression_predictions(
    *,
    feature_set_id: str,
    heldout_targets: pd.DataFrame,
    heldout_predictions: pd.DataFrame,
    target_columns: list[str],
    model_specs: list,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_column in target_columns:
        for model_spec in model_specs:
            pred_column = f"{target_column}__{model_spec.model_id}"
            metrics = regression_metrics(
                heldout_targets[target_column].to_numpy(dtype=float),
                heldout_predictions[pred_column].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "feature_set_id": feature_set_id,
                    "target_id": target_column,
                    "model_id": model_spec.model_id,
                    "split_id": "heldout_overall",
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _evaluate_heldout_classification_predictions(
    *,
    feature_set_id: str,
    heldout_targets: pd.DataFrame,
    heldout_predictions: pd.DataFrame,
    target_columns: list[str],
    model_specs: list,
    thresholds: dict[tuple[str, str], float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_column in target_columns:
        for model_spec in model_specs:
            pred_column = f"{target_column}__{model_spec.model_id}"
            metrics = binary_classification_metrics(
                heldout_targets[target_column].to_numpy(dtype=int),
                heldout_predictions[pred_column].to_numpy(dtype=float),
                threshold=thresholds[(target_column, model_spec.model_id)],
            )
            rows.append(
                {
                    "feature_set_id": feature_set_id,
                    "target_id": target_column,
                    "model_id": model_spec.model_id,
                    "split_id": "heldout_overall",
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_reasoning_distillation_mode(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    resolved_run = resolve_run_options(config, overrides)
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "reasoning_distillation")

    write_json(run_dir / "resolved_config.json", asdict(config))
    write_json(run_dir / "resolved_run_options.json", asdict(resolved_run))

    _log(logger, "Loading raw VCBench datasets and Feature Repository splits.")
    raw_datasets = load_raw_datasets(
        Path(config.datasets.public_train_csv),
        Path(config.datasets.private_test_csv),
    )
    repository_splits = load_feature_repository_splits(config.feature_repository)

    _log(logger, f"Loading target family '{resolved_run.target_family.family_id}'.")
    target_family = load_target_family(resolved_run.target_family)
    write_json(run_dir / "target_family_manifest.json", target_manifest_payload(target_family))

    _log(logger, "Loading repository-backed feature banks.")
    repository_banks = load_repository_feature_banks(
        repository_splits=repository_splits,
        specs=resolved_run.repository_feature_banks,
    )

    _log(logger, "Preparing intermediary feature banks.")
    intermediary_banks = prepare_intermediary_banks(
        public_raw=raw_datasets.public_frame,
        private_raw=raw_datasets.private_frame,
        feature_specs=resolved_run.intermediary_features,
        force_rebuild=resolved_run.force_rebuild_intermediary_features,
        logger=logger,
    )
    banks_by_id = {**repository_banks, **intermediary_banks}
    write_json(
        run_dir / "feature_bank_manifests.json",
        {feature_id: bank.manifest for feature_id, bank in banks_by_id.items()},
    )

    feature_sets = assemble_feature_sets(
        public_founder_ids=raw_datasets.public_frame["founder_uuid"],
        private_founder_ids=raw_datasets.private_frame["founder_uuid"],
        banks_by_id=banks_by_id,
        feature_sets=resolved_run.distillation_feature_sets,
    )
    write_json(
        run_dir / "feature_set_manifest.json",
        {feature_set.feature_set_id: feature_set.manifest for feature_set in feature_sets},
    )

    target_columns = target_family.target_columns
    combined_oof_predictions: pd.DataFrame | None = None
    combined_heldout_predictions: pd.DataFrame | None = None
    public_metric_frames: list[pd.DataFrame] = []
    heldout_metric_frames: list[pd.DataFrame] = []
    classification_thresholds_payload: dict[str, dict[str, float]] = {}

    for feature_set in feature_sets:
        _log(logger, f"Training target models for feature set '{feature_set.feature_set_id}'.")
        public_target_rows = _require_full_overlap(
            feature_set.public_frame[["founder_uuid"]],
            target_family.train_frame,
            on="founder_uuid",
            left_name=f"feature set '{feature_set.feature_set_id}' public rows",
            right_name=f"target family '{target_family.family_id}' public targets",
        )

        repeat_count = resolved_run.cv_seed_repeat_count
        output_prediction_tables: list[pd.DataFrame] = []
        output_metric_tables: list[pd.DataFrame] = []
        output_threshold_maps: list[dict[tuple[str, str], float]] = []
        X_public = feature_set.public_frame[feature_set.feature_columns].to_numpy(dtype=float)
        founder_ids_public = feature_set.public_frame["founder_uuid"].astype(str).tolist()

        for repeat_index in range(repeat_count):
            repeat_seed = config.distillation_cv.random_state + (repeat_index * 10_000)
            if resolved_run.repeat_cv_with_new_seeds:
                _log(
                    logger,
                    f"Feature set '{feature_set.feature_set_id}': CV repeat "
                    f"{repeat_index + 1}/{repeat_count} with random_seed={repeat_seed}.",
                )
            outer_n_splits = (
                config.reproduction.outer_cv.n_splits
                if resolved_run.distillation_nested_sweep
                else config.distillation_cv.n_splits
            )
            splits = build_stratified_reasoning_cv_splits(
                public_target_rows[target_columns],
                n_splits=outer_n_splits,
                shuffle=config.distillation_cv.shuffle,
                random_state=repeat_seed,
            )

            if target_family.task_kind == "regression" and not resolved_run.distillation_nested_sweep:
                outputs = train_reasoning_regressors_oof(
                    feature_set.public_frame[feature_set.feature_columns],
                    public_target_rows[target_columns],
                    founder_ids=feature_set.public_frame["founder_uuid"],
                    target_columns=target_columns,
                    model_specs=resolved_run.distillation_models,
                    splits=splits,
                    random_state=repeat_seed,
                    scale_min=target_family.scale_min or 0.0,
                    scale_max=target_family.scale_max or 1.0,
                    model_param_overrides_by_model_id=resolved_run.xgb_model_param_overrides_by_model_id,
                    max_parallel_workers=resolved_run.max_parallel_workers,
                )
                output_prediction_tables.append(outputs.oof_predictions)
                output_metric_tables.append(outputs.metrics.copy())
            elif target_family.task_kind == "classification" and not resolved_run.distillation_nested_sweep:
                outputs = train_reasoning_classifiers_oof(
                    feature_set.public_frame[feature_set.feature_columns],
                    public_target_rows[target_columns],
                    founder_ids=feature_set.public_frame["founder_uuid"],
                    target_columns=target_columns,
                    model_specs=resolved_run.distillation_models,
                    splits=splits,
                    random_state=repeat_seed,
                    model_param_overrides_by_model_id=resolved_run.xgb_model_param_overrides_by_model_id,
                    max_parallel_workers=resolved_run.max_parallel_workers,
                )
                output_prediction_tables.append(outputs.oof_predictions)
                output_metric_tables.append(outputs.metrics.copy())
                output_threshold_maps.append(outputs.selected_thresholds)
            elif target_family.task_kind == "regression":
                oof_predictions = pd.DataFrame({"founder_uuid": founder_ids_public})
                tasks: list[tuple[str, np.ndarray, int, object]] = []
                for target_column in target_columns:
                    y = public_target_rows[target_column].to_numpy(dtype=float)
                    for model_offset, model_spec in enumerate(resolved_run.distillation_models):
                        tasks.append((target_column, y, model_offset, model_spec))
                workers = bounded_worker_count(
                    max_parallel_workers=resolved_run.max_parallel_workers,
                    task_count=len(tasks),
                )
                if workers == 1:
                    task_outputs = [
                        _train_nested_single_target_regression_oof(
                            X_public=X_public,
                            y=y,
                            target_column=target_column,
                            model_spec=model_spec,
                            model_offset=model_offset,
                            splits=splits,
                            repeat_seed=repeat_seed,
                            inner_n_splits=config.reproduction.inner_cv.n_splits,
                            inner_shuffle=config.reproduction.inner_cv.shuffle,
                            scale_min=target_family.scale_min or 0.0,
                            scale_max=target_family.scale_max or 1.0,
                        )
                        for target_column, y, model_offset, model_spec in tasks
                    ]
                else:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        task_outputs = list(
                            executor.map(
                                lambda task: _train_nested_single_target_regression_oof(
                                    X_public=X_public,
                                    y=task[1],
                                    target_column=task[0],
                                    model_spec=task[3],
                                    model_offset=task[2],
                                    splits=splits,
                                    repeat_seed=repeat_seed,
                                    inner_n_splits=config.reproduction.inner_cv.n_splits,
                                    inner_shuffle=config.reproduction.inner_cv.shuffle,
                                    scale_min=target_family.scale_min or 0.0,
                                    scale_max=target_family.scale_max or 1.0,
                                ),
                                tasks,
                            )
                        )
                metric_rows: list[dict[str, object]] = []
                for column_name, oof, task_metric_rows in task_outputs:
                    oof_predictions[column_name] = oof
                    metric_rows.extend(task_metric_rows)
                output_prediction_tables.append(oof_predictions)
                output_metric_tables.append(pd.DataFrame(metric_rows))
            else:
                oof_predictions = pd.DataFrame({"founder_uuid": founder_ids_public})
                thresholds_for_repeat: dict[tuple[str, str], float] = {}
                tasks: list[tuple[str, np.ndarray, int, object]] = []
                for target_column in target_columns:
                    y = public_target_rows[target_column].to_numpy(dtype=int)
                    for model_offset, model_spec in enumerate(resolved_run.distillation_models):
                        tasks.append((target_column, y, model_offset, model_spec))
                workers = bounded_worker_count(
                    max_parallel_workers=resolved_run.max_parallel_workers,
                    task_count=len(tasks),
                )
                if workers == 1:
                    task_outputs = [
                        _train_nested_single_target_classification_oof(
                            X_public=X_public,
                            y=y,
                            target_column=target_column,
                            model_spec=model_spec,
                            model_offset=model_offset,
                            splits=splits,
                            repeat_seed=repeat_seed,
                            inner_n_splits=config.reproduction.inner_cv.n_splits,
                            inner_shuffle=config.reproduction.inner_cv.shuffle,
                        )
                        for target_column, y, model_offset, model_spec in tasks
                    ]
                else:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        task_outputs = list(
                            executor.map(
                                lambda task: _train_nested_single_target_classification_oof(
                                    X_public=X_public,
                                    y=task[1],
                                    target_column=task[0],
                                    model_spec=task[3],
                                    model_offset=task[2],
                                    splits=splits,
                                    repeat_seed=repeat_seed,
                                    inner_n_splits=config.reproduction.inner_cv.n_splits,
                                    inner_shuffle=config.reproduction.inner_cv.shuffle,
                                ),
                                tasks,
                            )
                        )
                metric_rows: list[dict[str, object]] = []
                for column_name, oof, threshold_key, threshold, task_metric_rows in task_outputs:
                    oof_predictions[column_name] = oof
                    thresholds_for_repeat[threshold_key] = threshold
                    metric_rows.extend(task_metric_rows)
                output_prediction_tables.append(oof_predictions)
                output_metric_tables.append(pd.DataFrame(metric_rows))
                output_threshold_maps.append(thresholds_for_repeat)

        averaged_outputs = _average_prediction_tables(output_prediction_tables)
        metrics_frame = _average_metric_tables(output_metric_tables)
        threshold_map = _average_threshold_maps(output_threshold_maps)
        if threshold_map:
            classification_thresholds_payload[feature_set.feature_set_id] = {
                f"{target_id}__{model_id}": threshold
                for (target_id, model_id), threshold in threshold_map.items()
            }

        metrics_frame.insert(0, "feature_set_id", feature_set.feature_set_id)
        metrics_frame["cv_seed_repeat_count"] = repeat_count
        public_metric_frames.append(metrics_frame)
        combined_oof_predictions = _merge_prediction_tables(
            combined_oof_predictions,
            _prefix_prediction_columns(averaged_outputs, feature_set_id=feature_set.feature_set_id),
        )

        if resolved_run.heldout_evaluation:
            heldout_feature_rows = feature_set.private_frame
            heldout_target_rows = None
            if target_family.test_frame is not None:
                heldout_target_rows = _require_full_overlap(
                    heldout_feature_rows[["founder_uuid"]],
                    target_family.test_frame,
                    on="founder_uuid",
                    left_name=f"held-out rows for feature set '{feature_set.feature_set_id}'",
                    right_name=f"held-out target family '{target_family.family_id}'",
                )

            if target_family.task_kind == "regression":
                if resolved_run.distillation_nested_sweep:
                    fit_tasks: list[tuple[str, np.ndarray, int, int, object]] = []
                    for model_offset, model_spec in enumerate(resolved_run.distillation_models):
                        for target_offset, target_column in enumerate(target_columns):
                            fit_tasks.append(
                                (
                                    target_column,
                                    public_target_rows[target_column].to_numpy(dtype=float),
                                    model_offset,
                                    target_offset,
                                    model_spec,
                                )
                            )
                    workers = bounded_worker_count(
                        max_parallel_workers=resolved_run.max_parallel_workers,
                        task_count=len(fit_tasks),
                    )
                    if workers == 1:
                        fitted_pairs = [
                            _fit_nested_single_target_regression_full(
                                X_public=X_public,
                                y_train_full=y_train_full,
                                target_column=target_column,
                                model_spec=model_spec,
                                model_offset=model_offset,
                                target_offset=target_offset,
                                inner_n_splits=config.reproduction.inner_cv.n_splits,
                                inner_shuffle=config.reproduction.inner_cv.shuffle,
                                random_state_base=config.distillation_cv.random_state,
                            )
                            for target_column, y_train_full, model_offset, target_offset, model_spec in fit_tasks
                        ]
                    else:
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            fitted_pairs = list(
                                executor.map(
                                    lambda task: _fit_nested_single_target_regression_full(
                                        X_public=X_public,
                                        y_train_full=task[1],
                                        target_column=task[0],
                                        model_spec=task[4],
                                        model_offset=task[2],
                                        target_offset=task[3],
                                        inner_n_splits=config.reproduction.inner_cv.n_splits,
                                        inner_shuffle=config.reproduction.inner_cv.shuffle,
                                        random_state_base=config.distillation_cv.random_state,
                                    ),
                                    fit_tasks,
                                )
                            )
                    fitted_models = dict(fitted_pairs)
                else:
                    fitted_models = fit_reasoning_regressors_full(
                        feature_set.public_frame[feature_set.feature_columns],
                        public_target_rows[target_columns],
                        target_columns=target_columns,
                        model_specs=resolved_run.distillation_models,
                        random_state=config.distillation_cv.random_state,
                        model_param_overrides_by_model_id=resolved_run.xgb_model_param_overrides_by_model_id,
                        max_parallel_workers=resolved_run.max_parallel_workers,
                    )
                heldout_predictions = predict_reasoning_regressors_full(
                    heldout_feature_rows[feature_set.feature_columns],
                    founder_ids=heldout_feature_rows["founder_uuid"],
                    target_columns=target_columns,
                    model_specs=resolved_run.distillation_models,
                    fitted_models=fitted_models,
                    scale_min=target_family.scale_min or 0.0,
                    scale_max=target_family.scale_max or 1.0,
                )
                if heldout_target_rows is not None:
                    heldout_metric_frames.append(
                        _evaluate_heldout_regression_predictions(
                            feature_set_id=feature_set.feature_set_id,
                            heldout_targets=heldout_target_rows,
                            heldout_predictions=heldout_predictions,
                            target_columns=target_columns,
                            model_specs=resolved_run.distillation_models,
                        )
                    )
            else:
                if resolved_run.distillation_nested_sweep:
                    fit_tasks: list[tuple[str, np.ndarray, int, int, object]] = []
                    for model_offset, model_spec in enumerate(resolved_run.distillation_models):
                        for target_offset, target_column in enumerate(target_columns):
                            fit_tasks.append(
                                (
                                    target_column,
                                    public_target_rows[target_column].to_numpy(dtype=int),
                                    model_offset,
                                    target_offset,
                                    model_spec,
                                )
                            )
                    workers = bounded_worker_count(
                        max_parallel_workers=resolved_run.max_parallel_workers,
                        task_count=len(fit_tasks),
                    )
                    if workers == 1:
                        fitted_pairs = [
                            _fit_nested_single_target_classification_full(
                                X_public=X_public,
                                y_train_full=y_train_full,
                                target_column=target_column,
                                model_spec=model_spec,
                                model_offset=model_offset,
                                target_offset=target_offset,
                                inner_n_splits=config.reproduction.inner_cv.n_splits,
                                inner_shuffle=config.reproduction.inner_cv.shuffle,
                                random_state_base=config.distillation_cv.random_state,
                            )
                            for target_column, y_train_full, model_offset, target_offset, model_spec in fit_tasks
                        ]
                    else:
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            fitted_pairs = list(
                                executor.map(
                                    lambda task: _fit_nested_single_target_classification_full(
                                        X_public=X_public,
                                        y_train_full=task[1],
                                        target_column=task[0],
                                        model_spec=task[4],
                                        model_offset=task[2],
                                        target_offset=task[3],
                                        inner_n_splits=config.reproduction.inner_cv.n_splits,
                                        inner_shuffle=config.reproduction.inner_cv.shuffle,
                                        random_state_base=config.distillation_cv.random_state,
                                    ),
                                    fit_tasks,
                                )
                            )
                    fitted_models = dict(fitted_pairs)
                else:
                    fitted_models = fit_reasoning_classifiers_full(
                        feature_set.public_frame[feature_set.feature_columns],
                        public_target_rows[target_columns],
                        target_columns=target_columns,
                        model_specs=resolved_run.distillation_models,
                        random_state=config.distillation_cv.random_state,
                        model_param_overrides_by_model_id=resolved_run.xgb_model_param_overrides_by_model_id,
                        max_parallel_workers=resolved_run.max_parallel_workers,
                    )
                heldout_predictions = predict_reasoning_classifiers_full(
                    heldout_feature_rows[feature_set.feature_columns],
                    founder_ids=heldout_feature_rows["founder_uuid"],
                    target_columns=target_columns,
                    model_specs=resolved_run.distillation_models,
                    fitted_models=fitted_models,
                )
                if heldout_target_rows is not None:
                    heldout_metric_frames.append(
                        _evaluate_heldout_classification_predictions(
                            feature_set_id=feature_set.feature_set_id,
                            heldout_targets=heldout_target_rows,
                            heldout_predictions=heldout_predictions,
                            target_columns=target_columns,
                            model_specs=resolved_run.distillation_models,
                            thresholds=threshold_map,
                        )
                    )

            combined_heldout_predictions = _merge_prediction_tables(
                combined_heldout_predictions,
                _prefix_prediction_columns(
                    heldout_predictions,
                    feature_set_id=feature_set.feature_set_id,
                ),
            )

    if resolved_run.save_reasoning_predictions:
        write_csv(
            run_dir / "reasoning_oof_predictions.csv",
            combined_oof_predictions if combined_oof_predictions is not None else pd.DataFrame({"founder_uuid": []}),
        )
    public_metrics_frame = pd.concat(public_metric_frames, ignore_index=True) if public_metric_frames else pd.DataFrame()
    write_csv(
        run_dir / "reasoning_metrics.csv",
        public_metrics_frame,
    )

    if classification_thresholds_payload:
        write_json(run_dir / "reasoning_classification_thresholds.json", classification_thresholds_payload)

    if resolved_run.heldout_evaluation and resolved_run.save_reasoning_predictions:
        write_csv(
            run_dir / "reasoning_heldout_predictions.csv",
            combined_heldout_predictions if combined_heldout_predictions is not None else pd.DataFrame({"founder_uuid": []}),
        )
    if resolved_run.heldout_evaluation:
        if heldout_metric_frames:
            heldout_metrics_frame = pd.concat(heldout_metric_frames, ignore_index=True)
            write_csv(
                run_dir / "reasoning_heldout_metrics.csv",
                heldout_metrics_frame,
            )
        else:
            heldout_metrics_frame = pd.DataFrame()
    else:
        heldout_metrics_frame = None

    write_markdown(
        run_dir / "reasoning_metrics_summary.md",
        _render_reasoning_metrics_summary(
            target_family_id=target_family.family_id,
            task_kind=target_family.task_kind,
            metrics_frame=public_metrics_frame,
            heldout_metrics_frame=heldout_metrics_frame,
        ),
    )

    summary_lines = [
        "# Run Summary",
        "",
        f"- Run mode: `{resolved_run.run_mode}`",
        f"- Target family: `{target_family.family_id}`",
        f"- Target task kind: `{target_family.task_kind}`",
        f"- Target count: {len(target_columns)}",
        f"- Repository feature banks loaded: {len(repository_banks)}",
        f"- Intermediary banks prepared: {len(intermediary_banks)}",
        f"- Feature-set comparisons run: {len(feature_sets)}",
        f"- Distillation models run per target: {len(resolved_run.distillation_models)}",
        f"- Max parallel workers: {resolved_run.max_parallel_workers}",
        "- Public CV strategy: stratified on quantile buckets of the row-wise mean selected target score.",
        f"- CV seed repeats: {resolved_run.cv_seed_repeat_count} (enabled={resolved_run.repeat_cv_with_new_seeds})",
        f"- Nested hyperparameter sweep: {resolved_run.distillation_nested_sweep}",
        f"- Distillation outer folds used: {config.reproduction.outer_cv.n_splits if resolved_run.distillation_nested_sweep else config.distillation_cv.n_splits}",
        f"- Distillation inner folds used for sweep: {config.reproduction.inner_cv.n_splits if resolved_run.distillation_nested_sweep else 0}",
        f"- Save reasoning prediction CSVs: {resolved_run.save_reasoning_predictions}",
        f"- OOF prediction columns written: {0 if combined_oof_predictions is None else len(combined_oof_predictions.columns) - 1}",
        f"- Held-out prediction columns written: {0 if combined_heldout_predictions is None else len(combined_heldout_predictions.columns) - 1}",
    ]
    if not resolved_run.heldout_evaluation:
        summary_lines.append("- Held-out evaluation was skipped because it was not requested.")
    write_markdown(run_dir / "run_summary.md", "\n".join(summary_lines))
    _log(logger, f"Reasoning-distillation run complete. Artifacts written to {run_dir}.")
    return run_dir


def run_pipeline(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    overrides_use = overrides or RunOverrides()
    if overrides_use.run_mode == "model_testing_mode":
        from src.pipeline.model_testing import run_model_testing_mode

        return run_model_testing_mode(config, overrides_use, logger=logger)
    if overrides_use.run_mode == "xgb_calibration_mode":
        from src.pipeline.xgb_calibration import run_xgb_calibration_mode

        return run_xgb_calibration_mode(config, overrides_use, logger=logger)

    if (
        (overrides_use.run_mode == "reasoning_distillation_mode")
        and (overrides_use.target_family == "v25_and_taste")
    ):
        run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "reasoning_distillation_multi")
        child_runs: list[dict[str, str]] = []
        for family_id in ("v25_policies", "taste_policies"):
            child_overrides = replace(overrides_use, target_family=family_id)
            child_run_dir = run_reasoning_distillation_mode(config, child_overrides, logger=logger)
            child_runs.append({"target_family": family_id, "run_dir": str(child_run_dir)})
        write_json(run_dir / "child_runs.json", child_runs)
        summary_lines = [
            "# Multi-Family Distillation Summary",
            "",
            "- Composite selection: `v25_and_taste`",
            "- Child runs:",
        ]
        summary_lines.extend([f"  - `{item['target_family']}` -> `{item['run_dir']}`" for item in child_runs])
        write_markdown(run_dir / "run_summary.md", "\n".join(summary_lines))
        _log(logger, f"Multi-family distillation run complete. Artifacts written to {run_dir}.")
        return run_dir

    resolved = resolve_run_options(config, overrides)
    if resolved.run_mode == "reproduction_mode":
        return run_reproduction_mode(
            config,
            use_nested_hyperparameter_cv=resolved.distillation_nested_sweep,
            max_parallel_workers=resolved.max_parallel_workers,
            logger=logger,
        )
    return run_reasoning_distillation_mode(config, overrides, logger=logger)


def run_reasoning_reconstruction(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    return run_pipeline(config, overrides, logger=logger)
