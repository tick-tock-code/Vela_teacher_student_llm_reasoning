from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import re
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
from src.pipeline.mlp_calibration import load_latest_mlp_calibration
from src.pipeline.saved_model_configs import bundle_dir_from_run_id, save_pickle, write_bundle_manifest
from src.pipeline.run_options import MODEL_FAMILY_TO_MODEL_ID, RunOverrides, resolve_run_options
from src.pipeline.rf_calibration import load_latest_rf_calibration
from src.pipeline.xgb_calibration import load_latest_xgb_calibration
from src.student.models import build_reasoning_classifier, build_reasoning_regressor
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.model_ids import (
    LEGACY_XGB_CLASSIFIER_MODEL_KIND,
    LEGACY_XGB_REGRESSOR_MODEL_KIND,
    XGB_CLASSIFIER_MODEL_KIND,
    XGB_FAMILY_ID,
    XGB_REGRESSOR_MODEL_KIND,
    normalize_xgb_family_id,
    normalize_xgb_model_kind,
)
from src.utils.parallel import apply_global_thread_env, bounded_worker_count
from src.utils.paths import RUNS_DIR


Logger = Callable[[str], None]
HEARTBEAT_SECONDS = 30.0
DEFAULT_MODEL_TESTING_MLP_TOL = 1e-3

MODEL_ID_TO_ARCHITECTURE: dict[str, str] = {
    "ridge": "linear_l2",
    "logreg_classifier": "linear_l2",
    "linear_svr_regressor": "linear_svm",
    "linear_svm_classifier": "linear_svm",
    LEGACY_XGB_REGRESSOR_MODEL_KIND: XGB_FAMILY_ID,
    LEGACY_XGB_CLASSIFIER_MODEL_KIND: XGB_FAMILY_ID,
    XGB_REGRESSOR_MODEL_KIND: XGB_FAMILY_ID,
    XGB_CLASSIFIER_MODEL_KIND: XGB_FAMILY_ID,
    "mlp_regressor": "mlp",
    "mlp_classifier": "mlp",
    "randomforest_regressor": "randomforest",
    "randomforest_classifier": "randomforest",
    "elasticnet_regressor": "elasticnet",
    "elasticnet_logreg_classifier": "elasticnet",
}
ARCHITECTURE_LABELS: dict[str, str] = {
    "linear_l2": "Linear L2",
    "linear_svm": "Linear SVM",
    XGB_FAMILY_ID: "XGBoost",
    "mlp": "MLP",
    "randomforest": "Random Forest",
    "elasticnet": "Elastic Net",
}
ARCHITECTURE_ORDER = ["linear_l2", "linear_svm", XGB_FAMILY_ID, "mlp", "randomforest", "elasticnet"]
ARCHITECTURE_SORT_INDEX = {name: index for index, name in enumerate(ARCHITECTURE_ORDER)}


def _model_architecture(model_id: str) -> str:
    return MODEL_ID_TO_ARCHITECTURE.get(model_id, model_id)


def _architecture_label(model_architecture: str) -> str:
    return ARCHITECTURE_LABELS.get(model_architecture, model_architecture)


def _architecture_sort_key(model_architecture: str) -> tuple[int, str]:
    return (ARCHITECTURE_SORT_INDEX.get(model_architecture, 999), model_architecture)


def _sorted_architectures(values: list[str]) -> list[str]:
    unique = sorted({str(value) for value in values if str(value).strip()})
    return sorted(unique, key=_architecture_sort_key)


def _split_csv_tokens(values: pd.Series) -> list[str]:
    tokens: set[str] = set()
    for value in values.dropna().astype(str):
        for token in value.split(","):
            cleaned = token.strip()
            if cleaned:
                tokens.add(cleaned)
    return sorted(tokens)


def _parse_mlp_hidden_layer_sizes(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        parsed = [int(v) for v in value]
        if not parsed:
            raise RuntimeError("MLP calibration hidden_layer_sizes cannot be empty.")
        return tuple(parsed)
    text = str(value).strip()
    if not text:
        raise RuntimeError("MLP calibration hidden_layer_sizes cannot be empty.")
    text = text.strip("()[] ")
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        raise RuntimeError(f"Could not parse MLP hidden_layer_sizes value: {value!r}")
    return tuple(int(token) for token in tokens)


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


@contextmanager
def _fit_heartbeat(
    logger: Logger | None,
    *,
    stage_label: str,
    interval_seconds: float = HEARTBEAT_SECONDS,
):
    if logger is None:
        yield
        return

    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.wait(interval_seconds):
            _log(logger, f"still doing {stage_label}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=0.2)


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
    allowed = {"linear_l2", "linear_svm", XGB_FAMILY_ID, "mlp"}
    selected = [family for family in [normalize_xgb_family_id(v) for v in requested_families] if family in allowed]
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise RuntimeError(
            "Model-testing Stage A requires at least one screening family enabled: "
            f"`linear_l2`, `linear_svm`, `{XGB_FAMILY_ID}`, and/or `mlp`."
        )
    return selected


def _nested_param_grid(model_kind: str, task_kind: str) -> list[dict[str, float | int]]:
    model_kind = normalize_xgb_model_kind(model_kind)
    if task_kind == "regression":
        if model_kind == "ridge":
            return [{"alpha": value} for value in (0.1, 1.0, 10.0, 50.0)]
        if model_kind == XGB_REGRESSOR_MODEL_KIND:
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
        if model_kind == XGB_CLASSIFIER_MODEL_KIND:
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


def _model_has_sweepable_grid(model_kind: str, task_kind: str) -> bool:
    return len(_nested_param_grid(model_kind, task_kind)) > 1


def _effective_nested_flag_for_models(
    *,
    requested_nested: bool,
    model_specs: list[object],
    task_kind: str,
    logger: Logger | None,
    context_label: str,
) -> bool:
    if not requested_nested:
        return False
    if any(_model_has_sweepable_grid(spec.kind, task_kind) for spec in model_specs):
        return True
    _log(
        logger,
        f"{context_label}: nested CV requested, but selected models have no sweep grid; running without nested CV.",
    )
    return False


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _safe_filename_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token.strip("._") or "target"


def _predict_binary_probabilities(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        if proba.ndim == 2:
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    raise RuntimeError("Model does not expose predict_proba or decision_function for classification.")


def _predict_multi_output_probabilities(model: object, X: np.ndarray, n_targets: int) -> np.ndarray:
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


def _prefer_native_multi_output(model_kind: str) -> bool:
    return str(model_kind) in {"mlp_regressor", "mlp_classifier"}


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
            if _prefer_native_multi_output(model_kind):
                model = build_reasoning_regressor(
                    model_kind,
                    random_state=random_state,
                    param_overrides=candidate,
                )
            else:
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
            if _prefer_native_multi_output(model_kind):
                model = build_reasoning_classifier(
                    model_kind,
                    random_state=random_state,
                    param_overrides=candidate,
                )
            else:
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
    grouped["model_architecture"] = grouped["model_id"].astype(str).map(_model_architecture)
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


def _aggregate_screening_metrics_by_architecture(
    repeat_model_metrics: pd.DataFrame,
    *,
    task_kind: str,
    score_delta: float,
    max_recommended: int,
) -> pd.DataFrame:
    if repeat_model_metrics.empty:
        return pd.DataFrame()

    metrics = repeat_model_metrics.copy()
    metrics["model_architecture"] = metrics["model_id"].astype(str).map(_model_architecture)

    for column in ("r2", "rmse", "mae", "f0_5", "roc_auc", "pr_auc"):
        if column not in metrics.columns:
            metrics[column] = float("nan")

    primary_column = "r2" if task_kind == "regression" else "f0_5"

    per_repeat_feature = (
        metrics.groupby(
            ["target_family", "output_mode", "model_architecture", "feature_set_id", "repeat_index"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )
    screening = (
        per_repeat_feature.groupby(
            ["target_family", "output_mode", "model_architecture", "feature_set_id"],
            as_index=False,
        )
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
    model_ids_by_architecture = (
        metrics.groupby(["target_family", "output_mode", "model_architecture"], as_index=False)
        .agg(model_ids=("model_id", lambda values: ", ".join(sorted({str(value) for value in values}))))
    )
    screening = screening.merge(
        model_ids_by_architecture,
        on=["target_family", "output_mode", "model_architecture"],
        how="left",
    )
    screening["primary_std"] = screening["primary_std"].fillna(0.0)
    screening["screen_score"] = screening["primary_mean"] - (0.5 * screening["primary_std"])
    screening["primary_metric"] = primary_column
    screening["recommended_take_forward"] = False
    screening["rank"] = 0

    output_rows: list[pd.DataFrame] = []
    for _, architecture_frame in screening.groupby(
        ["target_family", "output_mode", "model_architecture"],
        sort=False,
    ):
        ranked = architecture_frame.sort_values("screen_score", ascending=False).reset_index(drop=True)
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
        "model_architecture",
        "model_ids",
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

    output["architecture_sort_idx"] = (
        output["model_architecture"].astype(str).map(ARCHITECTURE_SORT_INDEX).fillna(999).astype(int)
    )
    return output[columns + ["architecture_sort_idx"]].sort_values(
        ["architecture_sort_idx", "target_family", "output_mode", "rank", "feature_set_id"],
        ascending=[True, True, True, True, True],
    ).drop(columns=["architecture_sort_idx"]).reset_index(drop=True)


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
    grouped["model_architecture"] = grouped["model_id"].astype(str).map(_model_architecture)
    columns = [
        "target_family",
        "output_mode",
        "feature_set_id",
        "model_architecture",
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
    grouped["architecture_sort_idx"] = (
        grouped["model_architecture"].astype(str).map(ARCHITECTURE_SORT_INDEX).fillna(999).astype(int)
    )
    return grouped[columns + ["architecture_sort_idx"]].sort_values(
        ["architecture_sort_idx", "target_family", "output_mode", "feature_set_id", "primary_mean"],
        ascending=[True, True, True, True, False],
    ).drop(columns=["architecture_sort_idx"]).reset_index(drop=True)


def _build_v25_per_target_screening(stage_a_repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target_family",
        "output_mode",
        "model_architecture",
        "model_id",
        "feature_set_id",
        "target_id",
        "r2",
        "rmse",
        "mae",
    ]
    if stage_a_repeat_metrics.empty:
        return pd.DataFrame(columns=columns)

    required_columns = {"target_family", "split_id", "feature_set_id", "model_id", "output_mode", "target_id"}
    if not required_columns.issubset(set(stage_a_repeat_metrics.columns)):
        return pd.DataFrame(columns=columns)

    for metric_column in ("r2", "rmse", "mae"):
        if metric_column not in stage_a_repeat_metrics.columns:
            stage_a_repeat_metrics[metric_column] = float("nan")

    filtered = stage_a_repeat_metrics[
        (stage_a_repeat_metrics["target_family"] == "v25_policies")
        & (stage_a_repeat_metrics["split_id"] == "oof_overall")
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=columns)

    filtered["model_architecture"] = filtered["model_id"].astype(str).map(_model_architecture)
    filtered["architecture_sort_idx"] = (
        filtered["model_architecture"].astype(str).map(ARCHITECTURE_SORT_INDEX).fillna(999).astype(int)
    )
    return filtered[columns + ["architecture_sort_idx"]].sort_values(
        ["architecture_sort_idx", "model_id", "feature_set_id", "output_mode", "target_id"],
        ascending=[True, True, True, True, True],
    ).drop(columns=["architecture_sort_idx"]).reset_index(drop=True)


def _append_v25_per_target_screening_markdown(
    lines: list[str],
    per_target_details: pd.DataFrame | None,
) -> None:
    lines.extend(["## Per-Target Detailed Metrics (v25)", ""])
    if per_target_details is None or per_target_details.empty:
        lines.extend(["No v25 per-target rows are available.", ""])
        return

    group_columns = ["model_architecture", "model_id", "feature_set_id", "output_mode"]
    for (model_architecture, model_id, feature_set_id, output_mode), combo_frame in per_target_details.groupby(
        group_columns,
        sort=False,
    ):
        lines.extend(
            [
                f"### {_architecture_label(str(model_architecture))} | {model_id} | {feature_set_id} | {output_mode}",
                "",
                "| target_id | r2 | rmse | mae |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in combo_frame.itertuples(index=False):
            lines.append(
                f"| {row.target_id} | {float(row.r2):.4f} | {float(row.rmse):.4f} | {float(row.mae):.4f} |"
            )
        lines.append("")


def _render_screening_markdown(
    screening_overall: pd.DataFrame,
    screening_by_architecture: pd.DataFrame,
    *,
    repeat_count: int,
    stage_a_model_families: list[str],
    score_delta: float,
    max_recommended: int,
    per_target_details: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# Feature-Set Screening Report",
        "",
        f"- Repeats: {repeat_count}",
        f"- Stage A models: {', '.join(f'`{family}`' for family in stage_a_model_families)}",
        "- Organization: grouped by model architecture, then target family/output mode.",
        "- Held-out features/targets: not used",
        f"- Recommendation rule: top score + any within `best - {score_delta}` (max {max_recommended}).",
        "",
    ]
    if screening_overall.empty and screening_by_architecture.empty:
        lines.append("No screening results were produced.")
        _append_v25_per_target_screening_markdown(lines, per_target_details)
        return "\n".join(lines)

    if not screening_by_architecture.empty:
        architecture_values = screening_by_architecture["model_architecture"].astype(str).tolist()
        lines.extend(["## Architecture Coverage", ""])
        lines.extend([
            "| architecture | model_ids | target_families | output_modes |",
            "|---|---|---|---|",
        ])
        for architecture in _sorted_architectures(architecture_values):
            architecture_frame = screening_by_architecture[
                screening_by_architecture["model_architecture"] == architecture
            ]
            model_ids = ", ".join(_split_csv_tokens(architecture_frame["model_ids"]))
            target_families = ", ".join(sorted({str(value) for value in architecture_frame["target_family"]}))
            output_modes = ", ".join(sorted({str(value) for value in architecture_frame["output_mode"]}))
            lines.append(
                f"| {_architecture_label(architecture)} | {model_ids} | {target_families} | {output_modes} |"
            )
        lines.append("")

        lines.extend(["## Stage A Screening By Architecture", ""])
        for architecture in _sorted_architectures(architecture_values):
            architecture_frame = screening_by_architecture[
                screening_by_architecture["model_architecture"] == architecture
            ].copy()
            architecture_frame = architecture_frame.sort_values(
                ["target_family", "output_mode", "rank", "feature_set_id"],
                ascending=[True, True, True, True],
            )
            model_ids = ", ".join(_split_csv_tokens(architecture_frame["model_ids"]))
            lines.extend([
                f"### {_architecture_label(architecture)}",
                "",
                f"- Model IDs: `{model_ids}`",
                "",
                "| target_family | output_mode | rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |",
                "|---|---|---:|---|---:|---:|---:|---:|",
            ])
            for row in architecture_frame.itertuples(index=False):
                lines.append(
                    f"| {row.target_family} | {row.output_mode} | {row.rank} | {row.feature_set_id} | "
                    f"{row.primary_mean:.4f} | {row.primary_std:.4f} | {row.screen_score:.4f} | "
                    f"{bool(row.recommended_take_forward)} |"
                )
            lines.append("")

    if screening_overall.empty:
        lines.append("Cross-architecture recommendation rows are not available.")
        _append_v25_per_target_screening_markdown(lines, per_target_details)
        return "\n".join(lines)

    recommended_rows = screening_overall[screening_overall["recommended_take_forward"]].copy()
    lines.extend(["## Cross-Architecture Take-Forward Sets", ""])
    if recommended_rows.empty:
        lines.append("No feature sets were marked for take-forward.")
        return "\n".join(lines)

    recommendation_summary = (
        recommended_rows.groupby(["target_family", "output_mode"], as_index=False)
        .agg(
            recommended_feature_sets=(
                "feature_set_id",
                lambda values: ", ".join(sorted({str(value) for value in values})),
            )
        )
        .sort_values(["target_family", "output_mode"]) 
    )
    lines.extend([
        "| target_family | output_mode | recommended_feature_sets |",
        "|---|---|---|",
    ])
    for row in recommendation_summary.itertuples(index=False):
        lines.append(
            f"| {row.target_family} | {row.output_mode} | {row.recommended_feature_sets} |"
        )
    lines.append("")
    lines.append(
        "Take-forward sets above are selected using blended screening scores across enabled Stage A models for each target/output combination."
    )
    _append_v25_per_target_screening_markdown(lines, per_target_details)
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
        "- Stage B advanced comparisons are inactive in the current pipeline.",
        "- Use Saved Config Eval mode to run test-set evaluations from persisted Stage A bundles.",
        "",
    ]
    if results.empty:
        lines.append("No Stage B rows were produced because Stage B execution is inactive.")
        return "\n".join(lines)

    results_with_architecture = results.copy()
    results_with_architecture["model_architecture"] = (
        results_with_architecture["model_architecture"].astype(str)
        if "model_architecture" in results_with_architecture.columns
        else results_with_architecture["model_id"].astype(str).map(_model_architecture)
    )
    architecture_values = results_with_architecture["model_architecture"].astype(str).tolist()

    lines.extend(["## Architecture Coverage", ""])
    lines.extend([
        "| architecture | model_ids | target_families | output_modes |",
        "|---|---|---|---|",
    ])
    for architecture in _sorted_architectures(architecture_values):
        architecture_frame = results_with_architecture[
            results_with_architecture["model_architecture"] == architecture
        ]
        model_ids = ", ".join(sorted({str(value) for value in architecture_frame["model_id"]}))
        target_families = ", ".join(sorted({str(value) for value in architecture_frame["target_family"]}))
        output_modes = ", ".join(sorted({str(value) for value in architecture_frame["output_mode"]}))
        lines.append(
            f"| {_architecture_label(architecture)} | {model_ids} | {target_families} | {output_modes} |"
        )
    lines.append("")

    lines.extend(["## Stage B Results By Architecture", ""])
    for architecture in _sorted_architectures(architecture_values):
        architecture_frame = results_with_architecture[
            results_with_architecture["model_architecture"] == architecture
        ].copy()
        architecture_frame = architecture_frame.sort_values(
            ["target_family", "output_mode", "primary_mean", "feature_set_id"],
            ascending=[True, True, False, True],
        )
        lines.extend([
            f"### {_architecture_label(architecture)}",
            "",
            "| target_family | output_mode | feature_set_id | model_id | primary_mean | primary_std |",
            "|---|---|---|---|---:|---:|",
        ])
        for row in architecture_frame.itertuples(index=False):
            lines.append(
                f"| {row.target_family} | {row.output_mode} | {row.feature_set_id} | {row.model_id} | "
                f"{row.primary_mean:.4f} | {row.primary_std:.4f} |"
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


def _estimate_stage_a_outer_fit_count(
    *,
    config: ExperimentConfig,
    family_sequence: list[str],
    family_map: dict[str, object],
    output_modes: list[str],
    model_family_output_modes: dict[str, list[str]],
    stage_a_model_families: list[str],
    repeat_count: int,
    feature_set_count: int,
    nested_requested: bool,
) -> int:
    total = 0
    for family_id in family_sequence:
        target_family = family_map[family_id]
        target_count = len(load_target_family(target_family).target_columns)
        for output_mode in output_modes:
            families_for_mode = [
                family_name
                for family_name in stage_a_model_families
                if output_mode in model_family_output_modes.get(family_name, [])
            ]
            if not families_for_mode:
                continue
            model_ids = _resolve_model_ids(
                config,
                task_kind=target_family.task_kind,
                model_families=families_for_mode,
            )
            model_specs = [
                spec for spec in config.distillation_models if spec.model_id in set(model_ids)
            ]
            nested_effective = _effective_nested_flag_for_models(
                requested_nested=nested_requested,
                model_specs=model_specs,
                task_kind=target_family.task_kind,
                logger=None,
                context_label=f"{family_id} ({output_mode})",
            )
            outer_n_splits = (
                config.reproduction.outer_cv.n_splits
                if nested_effective
                else config.distillation_cv.n_splits
            )
            per_fold_multiplier = 1 if output_mode == "multi_output" else target_count
            total += (
                feature_set_count
                * repeat_count
                * outer_n_splits
                * len(model_ids)
                * per_fold_multiplier
            )
    return int(total)


def _run_stage(
    config: ExperimentConfig,
    *,
    stage_name: str,
    base_overrides: RunOverrides,
    family_id: str,
    feature_set_ids: list[str],
    model_ids: list[str],
    repeat_count: int,
    force_rebuild_intermediary_features: bool,
    nested_sweep: bool,
    output_mode: str,
    model_param_overrides_by_model_id: dict[str, dict[str, object]] | None,
    logger: Logger | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    if output_mode == "single_target":
        from src.pipeline.distillation import run_reasoning_distillation_mode

        seeds = _repeat_seeds(config, repeat_count)
        stage_rows: list[pd.DataFrame] = []
        stage_detail_rows: list[pd.DataFrame] = []
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
                xgb_model_param_overrides_by_model_id=dict(model_param_overrides_by_model_id or {}),
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
            stage_label = (
                f"{stage_name} {family_id} {output_mode} repeat {repeat_index + 1}/{repeat_count} "
                "child distillation run"
            )
            _log(logger, f"{stage_label}: about to fit models.")
            with _fit_heartbeat(logger, stage_label=stage_label):
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
            detail_metrics = metrics.copy()
            detail_metrics["repeat_index"] = repeat_index
            detail_metrics["repeat_seed"] = seed
            detail_metrics["target_family"] = family_id
            detail_metrics["stage"] = "screening"
            detail_metrics["output_mode"] = output_mode
            stage_detail_rows.append(detail_metrics)
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
        return (
            pd.concat(stage_rows, ignore_index=True),
            pd.concat(stage_detail_rows, ignore_index=True),
            child_runs,
        )

    if output_mode != "multi_output":
        raise RuntimeError(f"Unsupported output_mode '{output_mode}'.")

    run_overrides = replace(
        base_overrides,
        run_mode="reasoning_distillation_mode",
        target_family=family_id,
        output_modes=[output_mode],
        xgb_model_param_overrides_by_model_id=dict(model_param_overrides_by_model_id or {}),
        heldout_evaluation=False,
        repeat_cv_with_new_seeds=False,
        cv_seed_repeat_count=1,
        distillation_nested_sweep=nested_sweep,
        save_reasoning_predictions=False,
        force_rebuild_intermediary_features=force_rebuild_intermediary_features,
        reasoning_models=model_ids,
        candidate_feature_sets=feature_set_ids,
    )
    resolved_run = resolve_run_options(config, run_overrides)
    raw_datasets = load_raw_datasets(
        Path(config.datasets.public_train_csv),
        Path(config.datasets.private_test_csv),
    )
    repository_splits = load_feature_repository_splits(config.feature_repository)
    target_family = load_target_family(resolved_run.target_family)
    effective_nested_sweep = _effective_nested_flag_for_models(
        requested_nested=nested_sweep,
        model_specs=resolved_run.distillation_models,
        task_kind=target_family.task_kind,
        logger=logger,
        context_label=f"{family_id} ({output_mode})",
    )
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
        config.reproduction.outer_cv.n_splits
        if effective_nested_sweep
        else config.distillation_cv.n_splits
    )

    prepared_feature_sets: list[dict[str, object]] = []
    for feature_set in feature_sets:
        feature_set_id = feature_set.feature_set_id
        public_target_rows = _require_full_overlap(
            feature_set.public_frame[["founder_uuid"]],
            target_family.train_frame,
            on="founder_uuid",
            left_name=f"feature set '{feature_set_id}' public rows",
            right_name=f"target family '{target_family.family_id}' public targets",
        )
        target_columns = target_family.target_columns
        if target_family.task_kind == "regression":
            y_public = public_target_rows[target_columns].to_numpy(dtype=float)
        else:
            y_public = public_target_rows[target_columns].to_numpy(dtype=int)
        prepared_feature_sets.append(
            {
                "feature_set_id": feature_set_id,
                "X_public": feature_set.public_frame[feature_set.feature_columns].to_numpy(dtype=float),
                "y_public": y_public,
                "target_columns": target_columns,
            }
        )

    seeds = _repeat_seeds(config, repeat_count)
    stage_rows = []
    stage_detail_rows = []
    child_runs = []
    for repeat_index, seed in enumerate(seeds):
        if repeat_index > 0:
            _log(
                logger,
                f"{family_id} ({output_mode}): repeat {repeat_index + 1}/{repeat_count} (seed={seed})",
            )
        feature_states: dict[str, dict[str, object]] = {}
        fold_tasks: list[dict[str, object]] = []
        for prepared in prepared_feature_sets:
            feature_set_id = str(prepared["feature_set_id"])
            X_public = np.asarray(prepared["X_public"], dtype=float)
            y_public = np.asarray(prepared["y_public"])
            target_columns = list(prepared["target_columns"])
            splits = build_stratified_reasoning_cv_splits(
                pd.DataFrame(y_public, columns=target_columns),
                n_splits=outer_n_splits,
                shuffle=config.distillation_cv.shuffle,
                random_state=seed,
            )
            feature_states[feature_set_id] = {
                "feature_set_id": feature_set_id,
                "X_public": X_public,
                "y_public": y_public,
                "target_columns": target_columns,
                "splits": splits,
                "model_outputs": {
                    model_spec.model_id: {
                        "oof": np.full((len(X_public), len(target_columns)), np.nan, dtype=float),
                        "fold_predictions": {},
                    }
                    for model_spec in resolved_run.distillation_models
                },
            }
            for model_offset, model_spec in enumerate(resolved_run.distillation_models):
                _log(
                    logger,
                    f"{family_id} ({output_mode}) repeat {repeat_index + 1}/{repeat_count}: "
                    f"{feature_set_id} -> model {model_spec.model_id} scheduled across {len(splits)} folds.",
                )
                for fold_index, split in enumerate(splits, start=1):
                    fold_tasks.append(
                        {
                            "feature_set_id": feature_set_id,
                            "model_spec": model_spec,
                            "model_offset": model_offset,
                            "fold_index": fold_index,
                            "split": split,
                            "X_public": X_public,
                            "y_public": y_public,
                        }
                    )

        def _run_fold_task(task: dict[str, object]) -> tuple[str, str, str, np.ndarray, np.ndarray]:
            feature_set_id = str(task["feature_set_id"])
            model_spec = task["model_spec"]
            model_offset = int(task["model_offset"])
            fold_index = int(task["fold_index"])
            split = task["split"]
            X_public = np.asarray(task["X_public"], dtype=float)
            y_public = np.asarray(task["y_public"])
            fold_offset = fold_index - 1
            stage_label = (
                f"{stage_name} {family_id} {output_mode} repeat {repeat_index + 1}/{repeat_count} "
                f"feature_set={feature_set_id} model={model_spec.model_id} fold {fold_index}/{outer_n_splits}"
            )
            _log(logger, f"{stage_label}: about to fit.")

            X_train = X_public[split.train_idx]
            X_test = X_public[split.test_idx]
            y_train = y_public[split.train_idx]
            if target_family.task_kind == "regression":
                run_nested_for_model = effective_nested_sweep and _model_has_sweepable_grid(
                    model_spec.kind,
                    target_family.task_kind,
                )
                best_params = (
                    _select_best_params_multi_output_regression(
                        X_train,
                        y_train,
                        model_kind=model_spec.kind,
                        random_state=seed + model_offset + fold_offset,
                        inner_n_splits=config.reproduction.inner_cv.n_splits,
                        inner_shuffle=config.reproduction.inner_cv.shuffle,
                    )
                    if run_nested_for_model
                    else {}
                )
                best_params = {
                    **dict((model_param_overrides_by_model_id or {}).get(model_spec.model_id, {})),
                    **best_params,
                }
                if _prefer_native_multi_output(model_spec.kind):
                    model = build_reasoning_regressor(
                        model_spec.kind,
                        random_state=seed + model_offset + fold_offset,
                        param_overrides=best_params,
                    )
                else:
                    base = build_reasoning_regressor(
                        model_spec.kind,
                        random_state=seed + model_offset + fold_offset,
                        param_overrides=best_params,
                    )
                    model = MultiOutputRegressor(base)
                with _fit_heartbeat(logger, stage_label=stage_label):
                    model.fit(X_train, y_train)
                preds = np.clip(
                    model.predict(X_test),
                    target_family.scale_min or 0.0,
                    target_family.scale_max or 1.0,
                )
            else:
                run_nested_for_model = effective_nested_sweep and _model_has_sweepable_grid(
                    model_spec.kind,
                    target_family.task_kind,
                )
                best_params = (
                    _select_best_params_multi_output_classification(
                        X_train,
                        y_train,
                        model_kind=model_spec.kind,
                        random_state=seed + model_offset + fold_offset,
                        inner_n_splits=config.reproduction.inner_cv.n_splits,
                        inner_shuffle=config.reproduction.inner_cv.shuffle,
                    )
                    if run_nested_for_model
                    else {}
                )
                best_params = {
                    **dict((model_param_overrides_by_model_id or {}).get(model_spec.model_id, {})),
                    **best_params,
                }
                if _prefer_native_multi_output(model_spec.kind):
                    model = build_reasoning_classifier(
                        model_spec.kind,
                        random_state=seed + model_offset + fold_offset,
                        param_overrides=best_params,
                    )
                else:
                    base = build_reasoning_classifier(
                        model_spec.kind,
                        random_state=seed + model_offset + fold_offset,
                        param_overrides=best_params,
                    )
                    model = MultiOutputClassifier(base)
                with _fit_heartbeat(logger, stage_label=stage_label):
                    model.fit(X_train, y_train)
                preds = _predict_multi_output_probabilities(model, X_test, y_public.shape[1])
            _log(
                logger,
                f"{family_id} ({output_mode}) repeat {repeat_index + 1}/{repeat_count}: "
                f"{feature_set_id} -> model {model_spec.model_id} fold {fold_index}/{outer_n_splits} completed.",
            )
            return feature_set_id, model_spec.model_id, split.split_id, split.test_idx, preds

        fit_workers = bounded_worker_count(
            max_parallel_workers=resolved_run.max_parallel_workers,
            task_count=len(fold_tasks),
        )
        _log(
            logger,
            f"{family_id} ({output_mode}) repeat {repeat_index + 1}/{repeat_count}: "
            f"parallelizing across {fit_workers} fold x feature-set fits.",
        )

        if fit_workers == 1:
            task_results = [_run_fold_task(task) for task in fold_tasks]
        else:
            with ThreadPoolExecutor(max_workers=fit_workers) as executor:
                task_results = list(executor.map(_run_fold_task, fold_tasks))

        for feature_set_id, model_id, split_id, test_idx, preds in task_results:
            model_output = feature_states[feature_set_id]["model_outputs"][model_id]
            model_output["oof"][test_idx] = preds
            model_output["fold_predictions"][split_id] = preds

        repeat_metric_rows: list[dict[str, object]] = []
        for feature_set_id, state in feature_states.items():
            y_public = np.asarray(state["y_public"])
            target_columns = list(state["target_columns"])
            splits = state["splits"]
            for model_spec in resolved_run.distillation_models:
                model_output = state["model_outputs"][model_spec.model_id]
                oof = np.asarray(model_output["oof"], dtype=float)
                fold_predictions = model_output["fold_predictions"]
                if np.isnan(oof).any():
                    raise RuntimeError(
                        f"Multi-output OOF contains NaNs for feature_set '{feature_set_id}' "
                        f"and model '{model_spec.model_id}'."
                    )
                if target_family.task_kind == "regression":
                    for target_idx, target_column in enumerate(target_columns):
                        for split in splits:
                            preds = fold_predictions[split.split_id][:, target_idx]
                            y_true = y_public[split.test_idx, target_idx]
                            repeat_metric_rows.append(
                                {
                                    "feature_set_id": feature_set_id,
                                    "target_id": target_column,
                                    "model_id": model_spec.model_id,
                                    "split_id": split.split_id,
                                    **regression_metrics(y_true, preds),
                                    "output_mode": output_mode,
                                }
                            )
                        repeat_metric_rows.append(
                            {
                                "feature_set_id": feature_set_id,
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
                                    "feature_set_id": feature_set_id,
                                    "target_id": target_column,
                                    "model_id": model_spec.model_id,
                                    "split_id": split.split_id,
                                    **binary_classification_metrics(y_true, preds, threshold=threshold),
                                    "output_mode": output_mode,
                                }
                            )
                        repeat_metric_rows.append(
                            {
                                "feature_set_id": feature_set_id,
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
        repeat_details = repeat_metrics_frame[repeat_metrics_frame["split_id"] == "oof_overall"].copy()
        repeat_details["repeat_index"] = repeat_index
        repeat_details["repeat_seed"] = seed
        repeat_details["target_family"] = family_id
        repeat_details["stage"] = "screening"
        repeat_details["output_mode"] = output_mode
        stage_detail_rows.append(repeat_details)
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
    return (
        pd.concat(stage_rows, ignore_index=True),
        pd.concat(stage_detail_rows, ignore_index=True),
        child_runs,
    )


def _save_stage_a_model_bundle(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    combo_requests: list[dict[str, object]],
    logger: Logger | None,
) -> Path:
    if not combo_requests:
        raise RuntimeError("No Stage A combos were provided for bundle persistence.")

    family_map = {spec.family_id: spec for spec in config.target_families}
    distillation_model_map = {spec.model_id: spec for spec in config.distillation_models}
    feature_set_map = {spec.feature_set_id: spec for spec in config.distillation_feature_sets}

    required_feature_set_ids = sorted(
        {
            str(feature_set_id)
            for request in combo_requests
            for feature_set_id in list(request["feature_set_ids"])
        }
    )
    required_feature_bank_ids = sorted(
        {
            feature_bank_id
            for feature_set_id in required_feature_set_ids
            for feature_bank_id in feature_set_map[feature_set_id].feature_bank_ids
        }
    )

    _log(logger, "Saving model config bundle: loading repository splits and feature banks.")
    repository_splits = load_feature_repository_splits(config.feature_repository)
    raw_datasets = load_raw_datasets(
        Path(config.datasets.public_train_csv),
        Path(config.datasets.private_test_csv),
    )
    repository_banks = load_repository_feature_banks(
        repository_splits=repository_splits,
        specs=[
            spec
            for spec in config.repository_feature_banks
            if spec.enabled and spec.feature_bank_id in set(required_feature_bank_ids)
        ],
    )
    intermediary_banks = prepare_intermediary_banks(
        public_raw=raw_datasets.public_frame,
        private_raw=raw_datasets.private_frame,
        feature_specs=[
            spec
            for spec in config.intermediary_features
            if spec.enabled and spec.feature_bank_id in set(required_feature_bank_ids)
        ],
        force_rebuild=False,
        logger=logger,
    )
    feature_sets = assemble_feature_sets(
        public_founder_ids=raw_datasets.public_frame["founder_uuid"],
        private_founder_ids=raw_datasets.private_frame["founder_uuid"],
        banks_by_id={**repository_banks, **intermediary_banks},
        feature_sets=[feature_set_map[feature_set_id] for feature_set_id in required_feature_set_ids],
    )
    resolved_feature_sets = {feature_set.feature_set_id: feature_set for feature_set in feature_sets}

    bundle_dir = bundle_dir_from_run_id(run_dir.name)
    bundle_entries: list[dict[str, object]] = []
    combo_counter = 0

    for request in combo_requests:
        family_id = str(request["target_family"])
        output_mode = str(request["output_mode"])
        task_kind = str(request["task_kind"])
        model_ids = [str(value) for value in list(request["model_ids"])]
        feature_set_ids = [str(value) for value in list(request["feature_set_ids"])]
        model_overrides = {
            str(model_id): dict(params)
            for model_id, params in dict(request["model_param_overrides_by_model_id"]).items()
        }
        target_family = load_target_family(family_map[family_id])
        target_columns = list(target_family.target_columns)

        for feature_set_id in feature_set_ids:
            if feature_set_id not in resolved_feature_sets:
                raise RuntimeError(f"Feature set '{feature_set_id}' was not assembled for bundle persistence.")
            feature_set = resolved_feature_sets[feature_set_id]
            aligned_targets = _require_full_overlap(
                feature_set.public_frame[["founder_uuid"]],
                target_family.train_frame,
                on="founder_uuid",
                left_name=f"bundle fit feature set '{feature_set_id}'",
                right_name=f"target family '{target_family.family_id}'",
            )
            if task_kind == "regression":
                y_public = aligned_targets[target_columns].to_numpy(dtype=float)
            else:
                y_public = aligned_targets[target_columns].to_numpy(dtype=int)
            X_public = feature_set.public_frame[feature_set.feature_columns].to_numpy(dtype=float)

            for model_offset, model_id in enumerate(model_ids):
                combo_counter += 1
                model_spec = distillation_model_map[model_id]
                combo_id = (
                    f"{family_id}__{feature_set_id}__{model_id}__{output_mode}__{combo_counter:04d}"
                )
                _log(
                    logger,
                    f"Saving bundle model {combo_counter}: "
                    f"{family_id} | {feature_set_id} | {model_id} | {output_mode}.",
                )
                thresholds_by_target: dict[str, float] = {}

                if output_mode == "multi_output":
                    if task_kind == "regression":
                        if _prefer_native_multi_output(model_spec.kind):
                            model = build_reasoning_regressor(
                                model_spec.kind,
                                random_state=config.distillation_cv.random_state + model_offset,
                                param_overrides=model_overrides.get(model_id, {}),
                            )
                        else:
                            base_model = build_reasoning_regressor(
                                model_spec.kind,
                                random_state=config.distillation_cv.random_state + model_offset,
                                param_overrides=model_overrides.get(model_id, {}),
                            )
                            model = MultiOutputRegressor(base_model)
                        model.fit(X_public, y_public)
                    else:
                        if _prefer_native_multi_output(model_spec.kind):
                            model = build_reasoning_classifier(
                                model_spec.kind,
                                random_state=config.distillation_cv.random_state + model_offset,
                                param_overrides=model_overrides.get(model_id, {}),
                            )
                        else:
                            base_model = build_reasoning_classifier(
                                model_spec.kind,
                                random_state=config.distillation_cv.random_state + model_offset,
                                param_overrides=model_overrides.get(model_id, {}),
                            )
                            model = MultiOutputClassifier(base_model)
                        model.fit(X_public, y_public)
                        train_probs = _predict_multi_output_probabilities(
                            model,
                            X_public,
                            len(target_columns),
                        )
                        for target_idx, target_name in enumerate(target_columns):
                            thresholds_by_target[target_name] = float(
                                select_f05_threshold(y_public[:, target_idx], train_probs[:, target_idx])
                            )
                    rel_path = Path("models") / f"combo_{combo_counter:04d}.pkl"
                    save_pickle(bundle_dir / rel_path, model)
                    entry = {
                        "combo_id": combo_id,
                        "target_family": family_id,
                        "task_kind": task_kind,
                        "output_mode": output_mode,
                        "feature_set_id": feature_set_id,
                        "model_id": model_id,
                        "feature_columns": list(feature_set.feature_columns),
                        "target_columns": target_columns,
                        "thresholds_by_target": thresholds_by_target,
                        "model_artifact_relpath": str(rel_path.as_posix()),
                    }
                    bundle_entries.append(entry)
                    continue

                if output_mode != "single_target":
                    raise RuntimeError(f"Unsupported output_mode '{output_mode}' for bundle persistence.")
                target_model_artifacts: dict[str, str] = {}
                for target_idx, target_name in enumerate(target_columns):
                    if task_kind == "regression":
                        model = build_reasoning_regressor(
                            model_spec.kind,
                            random_state=config.distillation_cv.random_state + model_offset + target_idx,
                            param_overrides=model_overrides.get(model_id, {}),
                        )
                        model.fit(X_public, y_public[:, target_idx])
                    else:
                        model = build_reasoning_classifier(
                            model_spec.kind,
                            random_state=config.distillation_cv.random_state + model_offset + target_idx,
                            param_overrides=model_overrides.get(model_id, {}),
                        )
                        model.fit(X_public, y_public[:, target_idx])
                        train_probs = _predict_binary_probabilities(model, X_public)
                        thresholds_by_target[target_name] = float(
                            select_f05_threshold(y_public[:, target_idx], train_probs)
                        )
                    safe_target_name = _safe_filename_token(target_name)
                    rel_path = Path("models") / f"combo_{combo_counter:04d}__t{target_idx:02d}__{safe_target_name}.pkl"
                    save_pickle(bundle_dir / rel_path, model)
                    target_model_artifacts[target_name] = str(rel_path.as_posix())
                bundle_entries.append(
                    {
                        "combo_id": combo_id,
                        "target_family": family_id,
                        "task_kind": task_kind,
                        "output_mode": output_mode,
                        "feature_set_id": feature_set_id,
                        "model_id": model_id,
                        "feature_columns": list(feature_set.feature_columns),
                        "target_columns": target_columns,
                        "thresholds_by_target": thresholds_by_target,
                        "target_model_artifacts": target_model_artifacts,
                    }
                )

    manifest = {
        "bundle_version": 1,
        "experiment_id": config.experiment_id,
        "source_run_id": run_dir.name,
        "source_run_dir": str(run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "saved_models_kind": "final_full_train_only",
        "combos": bundle_entries,
    }
    write_bundle_manifest(bundle_dir, manifest)
    write_json(bundle_dir / "resolved_config.json", asdict(config))
    _log(logger, f"Saved model config bundle with {len(bundle_entries)} combos at {bundle_dir}.")
    return bundle_dir


def run_model_testing_mode(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    overrides_use = overrides or RunOverrides()
    per_fit_threads = (
        int(overrides_use.model_testing_per_fit_threads)
        if overrides_use.model_testing_per_fit_threads is not None
        else 1
    )
    if per_fit_threads < 1:
        raise RuntimeError("model_testing_per_fit_threads must be >= 1.")
    thread_count = apply_global_thread_env(per_fit_threads)
    _log(
        logger,
        f"Model-testing compute settings: BLAS/OpenMP thread env vars set to {thread_count} for per-fit linear algebra.",
    )
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
    model_family_output_modes = resolved.model_family_output_modes
    estimated_stage_a_outer_fits = _estimate_stage_a_outer_fit_count(
        config=config,
        family_sequence=family_sequence,
        family_map=family_map,
        output_modes=output_modes,
        model_family_output_modes=model_family_output_modes,
        stage_a_model_families=stage_a_model_families,
        repeat_count=repeat_count,
        feature_set_count=len(feature_set_ids),
        nested_requested=resolved.distillation_nested_sweep,
    )
    _log(
        logger,
        f"Planned Stage A outer fits: {estimated_stage_a_outer_fits} "
        f"(families={family_sequence}, feature_sets={len(feature_set_ids)}, "
        f"repeats={repeat_count}, output_modes={output_modes}).",
    )
    model_param_overrides_by_model_id: dict[str, dict[str, object]] = {
        model_id: dict(params)
        for model_id, params in {
            **resolved.xgb_model_param_overrides_by_model_id,
            **resolved.rf_model_param_overrides_by_model_id,
        }.items()
    }
    model_param_overrides_by_model_id.setdefault("mlp_regressor", {}).setdefault(
        "tol",
        DEFAULT_MODEL_TESTING_MLP_TOL,
    )
    model_param_overrides_by_model_id.setdefault("mlp_regressor", {}).setdefault(
        "early_stopping",
        False,
    )
    model_param_overrides_by_model_id.setdefault("mlp_classifier", {}).setdefault(
        "tol",
        DEFAULT_MODEL_TESTING_MLP_TOL,
    )
    model_param_overrides_by_model_id.setdefault("mlp_classifier", {}).setdefault(
        "early_stopping",
        False,
    )
    latest_calibration: dict[str, object] | None = None
    latest_rf_calibration: dict[str, object] | None = None
    latest_mlp_calibration: dict[str, object] | None = None
    if resolved.use_latest_xgb_calibration:
        latest_calibration = load_latest_xgb_calibration(config.experiment_id)
        if latest_calibration is None:
            raise RuntimeError(
                "use_latest_xgb_calibration was enabled, but no xgb calibration artifact was found."
            )
    if resolved.use_latest_rf_calibration:
        latest_rf_calibration = load_latest_rf_calibration(config.experiment_id)
        if latest_rf_calibration is None:
            raise RuntimeError(
                "use_latest_rf_calibration was enabled, but no rf calibration artifact was found."
            )
    if resolved.use_latest_mlp_calibration:
        latest_mlp_calibration = load_latest_mlp_calibration(config.experiment_id)
        if latest_mlp_calibration is None:
            raise RuntimeError(
                "use_latest_mlp_calibration was enabled, but no mlp calibration artifact was found."
            )

    stage_a_rows: list[pd.DataFrame] = []
    stage_a_detail_rows: list[pd.DataFrame] = []
    stage_a_child_runs: list[dict[str, object]] = []
    stage_a_nested_effective_map: dict[str, bool] = {}
    stage_a_combo_requests: list[dict[str, object]] = []
    for family_id in family_sequence:
        for output_mode in output_modes:
            stage_a_families_for_mode = [
                family_name
                for family_name in stage_a_model_families
                if output_mode in model_family_output_modes.get(family_name, [])
            ]
            if not stage_a_families_for_mode:
                continue
            if family_id not in family_map:
                raise RuntimeError(f"Unknown target family '{family_id}'.")
            task_kind = family_map[family_id].task_kind
            stage_a_model_ids = _resolve_model_ids(
                config,
                task_kind=task_kind,
                model_families=stage_a_families_for_mode,
            )
            stage_a_model_specs = [
                spec for spec in config.distillation_models if spec.model_id in set(stage_a_model_ids)
            ]
            stage_a_nested_effective = _effective_nested_flag_for_models(
                requested_nested=resolved.distillation_nested_sweep,
                model_specs=stage_a_model_specs,
                task_kind=task_kind,
                logger=logger,
                context_label=f"{family_id} ({output_mode}) Stage A",
            )
            stage_a_nested_effective_map[f"{family_id}::{output_mode}"] = stage_a_nested_effective
            stage_model_overrides = dict(model_param_overrides_by_model_id)
            if latest_calibration is not None:
                selected_map = dict(latest_calibration.get("selected_n_estimators_by_family", {}))
                selected_n = selected_map.get(family_id)
                if selected_n is not None:
                    xgb_model_id = XGB_REGRESSOR_MODEL_KIND if task_kind == "regression" else XGB_CLASSIFIER_MODEL_KIND
                    if xgb_model_id in stage_a_model_ids:
                        stage_model_overrides[xgb_model_id] = {
                            **dict(stage_model_overrides.get(xgb_model_id, {})),
                            "n_estimators": int(selected_n),
                        }
            if latest_rf_calibration is not None:
                selected_map = dict(latest_rf_calibration.get("selected_params_by_family", {}))
                selected_params = selected_map.get(family_id)
                if isinstance(selected_params, dict):
                    rf_model_id = "randomforest_regressor" if task_kind == "regression" else "randomforest_classifier"
                    if rf_model_id in stage_a_model_ids:
                        stage_model_overrides[rf_model_id] = {
                            **dict(stage_model_overrides.get(rf_model_id, {})),
                            **dict(selected_params),
                        }
            if latest_mlp_calibration is not None:
                selected_map = dict(latest_mlp_calibration.get("selected_params_by_family", {}))
                selected_params = selected_map.get(family_id)
                if isinstance(selected_params, dict):
                    mlp_model_id = "mlp_regressor" if task_kind == "regression" else "mlp_classifier"
                    if mlp_model_id in stage_a_model_ids:
                        stage_model_overrides[mlp_model_id] = {
                                **dict(stage_model_overrides.get(mlp_model_id, {})),
                            "hidden_layer_sizes": _parse_mlp_hidden_layer_sizes(
                                selected_params.get("hidden_layer_sizes", "")
                            ),
                            "alpha": float(selected_params["alpha"]),
                            "early_stopping": False,
                        }
            _log(
                logger,
                f"Stage A screening for '{family_id}' ({output_mode}) with feature sets: {feature_set_ids}. "
                f"Nested requested={resolved.distillation_nested_sweep}, effective={stage_a_nested_effective}.",
            )
            stage_metrics, stage_metrics_detail, child_runs = _run_stage(
                config,
                stage_name="Stage A",
                base_overrides=overrides_use,
                family_id=family_id,
                feature_set_ids=feature_set_ids,
                model_ids=stage_a_model_ids,
                repeat_count=repeat_count,
                force_rebuild_intermediary_features=resolved.force_rebuild_intermediary_features,
                nested_sweep=stage_a_nested_effective,
                output_mode=output_mode,
                model_param_overrides_by_model_id=stage_model_overrides,
                logger=logger,
            )
            stage_a_rows.append(stage_metrics)
            stage_a_detail_rows.append(stage_metrics_detail)
            stage_a_child_runs.extend(child_runs)
            stage_a_combo_requests.append(
                {
                    "target_family": family_id,
                    "task_kind": task_kind,
                    "output_mode": output_mode,
                    "feature_set_ids": list(feature_set_ids),
                    "model_ids": list(stage_a_model_ids),
                    "model_param_overrides_by_model_id": dict(stage_model_overrides),
                }
            )

    stage_a_repeat_metrics = pd.concat(stage_a_rows, ignore_index=True) if stage_a_rows else pd.DataFrame()
    stage_a_repeat_metrics_detailed = (
        pd.concat(stage_a_detail_rows, ignore_index=True) if stage_a_detail_rows else pd.DataFrame()
    )
    write_csv(run_dir / "feature_set_screening_repeat_metrics.csv", stage_a_repeat_metrics_detailed)
    write_csv(run_dir / "feature_set_screening_repeat_summary.csv", stage_a_repeat_metrics)
    write_json(run_dir / "feature_set_screening_child_runs.json", stage_a_child_runs)
    v25_per_target_screening = _build_v25_per_target_screening(stage_a_repeat_metrics_detailed)
    write_csv(run_dir / "feature_set_screening_per_target.csv", v25_per_target_screening)

    screening_rows: list[pd.DataFrame] = []
    screening_by_architecture_rows: list[pd.DataFrame] = []
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
            screening_by_architecture_rows.append(
                _aggregate_screening_metrics_by_architecture(
                    family_metrics,
                    task_kind=task_kind,
                    score_delta=config.model_testing.screening_score_delta,
                    max_recommended=config.model_testing.max_recommended_feature_sets,
                )
            )
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
    screening_by_architecture_frame = (
        pd.concat(screening_by_architecture_rows, ignore_index=True)
        if screening_by_architecture_rows
        else pd.DataFrame()
    )
    write_csv(run_dir / "feature_set_screening.csv", screening_frame)
    write_csv(run_dir / "feature_set_screening_by_architecture.csv", screening_by_architecture_frame)
    write_markdown(
        run_dir / "feature_set_screening_report.md",
        _render_screening_markdown(
            screening_frame,
            screening_by_architecture_frame,
            repeat_count=repeat_count,
            stage_a_model_families=stage_a_model_families,
            score_delta=config.model_testing.screening_score_delta,
            max_recommended=config.model_testing.max_recommended_feature_sets,
            per_target_details=v25_per_target_screening,
        ),
    )

    model_testing_results = pd.DataFrame()
    write_csv(run_dir / "model_testing_results.csv", model_testing_results)
    write_markdown(
        run_dir / "model_testing_report.md",
        _render_model_testing_markdown(
            model_testing_results,
            repeat_count=repeat_count,
        ),
    )

    saved_bundle_path: Path | None = None
    if resolved.save_model_configs_after_training:
        saved_bundle_path = _save_stage_a_model_bundle(
            config=config,
            run_dir=run_dir,
            combo_requests=stage_a_combo_requests,
            logger=logger,
        )
        write_json(saved_bundle_path / "resolved_run_options.json", asdict(resolved))
        write_json(saved_bundle_path / "resolved_config.json", asdict(config))

    summary_lines = [
        "# Model Testing Summary",
        "",
        f"- Candidate feature sets: {len(feature_set_ids)}",
        f"- Repeats: {repeat_count}",
        f"- Estimated Stage A outer fits: {estimated_stage_a_outer_fits}",
        f"- Max parallel workers: {resolved.max_parallel_workers}",
        f"- Stage A model families: {', '.join(stage_a_model_families)}",
        f"- Output modes: {', '.join(output_modes)}",
        f"- Model-family output modes: {model_family_output_modes}",
        f"- Nested requested: {resolved.distillation_nested_sweep}",
        f"- Nested effective (Stage A): {stage_a_nested_effective_map}",
        "- Stage B advanced execution: inactive (removed from active model-testing flow).",
        f"- Save model configs after training: {resolved.save_model_configs_after_training}",
        f"- Saved model bundle path: {saved_bundle_path if saved_bundle_path is not None else 'not_saved'}",
        f"- Use latest xgb calibration: {resolved.use_latest_xgb_calibration}",
        f"- Use latest rf calibration: {resolved.use_latest_rf_calibration}",
        f"- Use latest mlp calibration: {resolved.use_latest_mlp_calibration}",
        "- Held-out/test features or targets are not used in this mode.",
    ]
    write_markdown(run_dir / "run_summary.md", "\n".join(summary_lines))
    _log(logger, f"Model testing run complete. Artifacts written to {run_dir}.")
    return run_dir
