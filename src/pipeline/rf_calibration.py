from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data.feature_repository import load_feature_repository_splits, load_repository_feature_banks
from src.data.raw_datasets import load_raw_datasets
from src.data.splits import build_stratified_reasoning_cv_splits
from src.data.targets import load_target_family
from src.evaluation.metrics import binary_classification_metrics, regression_metrics, select_f05_threshold
from src.intermediary_features.registry import assemble_feature_sets, prepare_intermediary_banks
from src.pipeline.config import ExperimentConfig
from src.pipeline.run_options import RunOverrides, resolve_run_options
from src.student.models import build_reasoning_classifier, build_reasoning_regressor
from src.utils.artifact_io import read_json, timestamped_run_dir, write_json, write_markdown
from src.utils.paths import DOCS_DIR, RUNS_DIR


Logger = Callable[[str], None]
RF_CALIBRATION_ARTIFACT_NAME = "rf_calibration_recommendations.json"
RF_FIXED_N_ESTIMATORS = 500
RF_FIXED_BOOTSTRAP = True


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


def _family_sequence(resolved_target_family: str, requested: str | None) -> list[str]:
    if requested == "v25_and_taste":
        return ["v25_policies", "taste_policies"]
    return [resolved_target_family]


def _rf_sweep_grid(
    *,
    min_samples_leaf_values: list[int],
    max_depth_values: list[int | None],
    max_features_values: list[str | float],
) -> list[dict[str, object]]:
    grid: list[dict[str, object]] = []
    for min_samples_leaf in min_samples_leaf_values:
        for max_depth in max_depth_values:
            for max_features in max_features_values:
                grid.append(
                    {
                        "n_estimators": RF_FIXED_N_ESTIMATORS,
                        "bootstrap": RF_FIXED_BOOTSTRAP,
                        "min_samples_leaf": int(min_samples_leaf),
                        "max_depth": (None if max_depth is None else int(max_depth)),
                        "max_features": max_features,
                    }
                )
    return grid


def _params_signature(params: dict[str, object]) -> str:
    max_depth = "none" if params["max_depth"] is None else str(params["max_depth"])
    max_features = str(params["max_features"])
    return (
        f"min_samples_leaf={params['min_samples_leaf']}|"
        f"max_depth={max_depth}|max_features={max_features}"
    )


def _select_recommended_params(metrics_frame: pd.DataFrame) -> dict[str, object]:
    ranked = (
        metrics_frame.groupby("params_signature", as_index=False)
        .agg(primary_mean=("primary_mean", "mean"), primary_std=("primary_std", "mean"))
        .fillna({"primary_std": 0.0})
        .sort_values(
            ["primary_mean", "primary_std", "params_signature"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    if ranked.empty:
        raise RuntimeError("RF calibration produced no candidate rows to select parameters.")
    return {"params_signature": str(ranked.iloc[0]["params_signature"])}


def _render_summary_markdown(
    *,
    run_dir: Path,
    rows_frame: pd.DataFrame,
    chosen_by_family: dict[str, dict[str, object]],
    top_sets_by_family: dict[str, list[str]],
    cv_outer_splits: int,
    cv_random_state: int,
    parallel_workers: int,
) -> str:
    lines: list[str] = [
        "# Random Forest Calibration Summary",
        "",
        f"- Run artifacts: `{run_dir}`",
        "- Calibration type: training CV only, no held-out/test usage.",
        f"- Fixed params: n_estimators={RF_FIXED_N_ESTIMATORS}, bootstrap={RF_FIXED_BOOTSTRAP}",
        f"- Outer CV: {cv_outer_splits}-fold stratified (random_state={cv_random_state})",
        f"- Parallel target workers: {parallel_workers}",
        "",
        "## Selected Defaults",
        "",
    ]
    for family_id, params in chosen_by_family.items():
        lines.append(
            f"- `{family_id}`: `min_samples_leaf={params['min_samples_leaf']}`, "
            f"`max_depth={params['max_depth']}`, `max_features={params['max_features']}`"
        )
    lines.extend(["", "## Top Feature Sets At Selected Params", ""])
    for family_id, feature_sets in top_sets_by_family.items():
        joined = ", ".join(f"`{feature_set}`" for feature_set in feature_sets) if feature_sets else "_none_"
        lines.append(f"- `{family_id}`: {joined}")
    lines.extend(
        [
            "",
            "## Calibration Table (feature_set x parameter combo)",
            "",
            "| target_family | feature_set_id | min_samples_leaf | max_depth | max_features | primary_metric | primary_mean | primary_std |",
            "|---|---|---:|---|---|---|---:|---:|",
        ]
    )
    table = rows_frame.sort_values(
        ["target_family", "feature_set_id", "min_samples_leaf", "max_depth_sort", "max_features_sort"],
        ascending=[True, True, True, True, True],
    )
    for row in table.itertuples(index=False):
        max_depth = "None" if row.max_depth is None else str(int(row.max_depth))
        lines.append(
            f"| {row.target_family} | {row.feature_set_id} | {int(row.min_samples_leaf)} | {max_depth} | "
            f"{row.max_features} | {row.primary_metric} | {float(row.primary_mean):.4f} | {float(row.primary_std):.4f} |"
        )
    return "\n".join(lines)


def _latest_path() -> Path:
    docs_dir = DOCS_DIR / "experiment-archive" / "generated-reports"
    return docs_dir / "rf_calibration_summary_latest.md"


def _compute_primary_metrics(
    *,
    X: np.ndarray,
    y: np.ndarray,
    target_columns: list[str],
    task_kind: str,
    params: dict[str, object],
    random_state: int,
    n_splits: int,
    shuffle: bool,
    scale_min: float | None,
    scale_max: float | None,
    parallel_workers: int,
) -> list[float]:
    splits = build_stratified_reasoning_cv_splits(
        pd.DataFrame(y, columns=target_columns),
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    def _run_target(target_index: int) -> float:
        target_column = target_columns[target_index]
        y_target = y[:, target_index]
        oof = np.full(len(y_target), np.nan, dtype=float)
        for fold_offset, split in enumerate(splits):
            seed = random_state + target_index + fold_offset
            if task_kind == "regression":
                model = build_reasoning_regressor(
                    "randomforest_regressor",
                    random_state=seed,
                    param_overrides=params,
                )
                model.fit(X[split.train_idx], y_target[split.train_idx])
                preds = model.predict(X[split.test_idx])
                preds = np.clip(
                    preds,
                    scale_min if scale_min is not None else 0.0,
                    scale_max if scale_max is not None else 1.0,
                )
                oof[split.test_idx] = preds
            else:
                model = build_reasoning_classifier(
                    "randomforest_classifier",
                    random_state=seed,
                    param_overrides=params,
                )
                model.fit(X[split.train_idx], y_target[split.train_idx].astype(int))
                probs = model.predict_proba(X[split.test_idx])[:, 1]
                oof[split.test_idx] = probs
        if np.isnan(oof).any():
            raise RuntimeError(f"RF calibration OOF contains NaNs for target '{target_column}'.")
        if task_kind == "regression":
            return float(regression_metrics(y_target, oof)["r2"])
        threshold = select_f05_threshold(y_target.astype(int), oof)
        return float(binary_classification_metrics(y_target.astype(int), oof, threshold=threshold)["f0_5"])

    if parallel_workers <= 1 or len(target_columns) <= 1:
        return [_run_target(target_index) for target_index in range(len(target_columns))]

    use_workers = min(parallel_workers, len(target_columns))
    with ThreadPoolExecutor(max_workers=use_workers) as executor:
        futures = [executor.submit(_run_target, target_index) for target_index in range(len(target_columns))]
        return [float(future.result()) for future in futures]


def load_latest_rf_calibration(experiment_id: str) -> dict[str, object] | None:
    root = RUNS_DIR / experiment_id
    if not root.exists():
        return None
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.endswith("_rf_calibration")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        artifact_path = candidate / RF_CALIBRATION_ARTIFACT_NAME
        if artifact_path.exists():
            payload = read_json(artifact_path)
            return {
                "run_dir": str(candidate),
                "artifact_path": str(artifact_path),
                **payload,
            }
    return None


def run_rf_calibration_mode(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    overrides_use = overrides or RunOverrides()
    if overrides_use.heldout_evaluation is True:
        raise RuntimeError("rf_calibration_mode is training-only. heldout_evaluation must be false.")

    resolved = resolve_run_options(config, replace(overrides_use, run_mode="rf_calibration_mode"))
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "rf_calibration")
    write_json(run_dir / "resolved_config.json", asdict(config))
    write_json(run_dir / "resolved_run_options.json", asdict(resolved))

    raw_datasets = load_raw_datasets(
        Path(config.datasets.public_train_csv),
        Path(config.datasets.private_test_csv),
    )
    repository_splits = load_feature_repository_splits(config.feature_repository)
    repository_banks = load_repository_feature_banks(
        repository_splits=repository_splits,
        specs=resolved.repository_feature_banks,
    )
    intermediary_banks = prepare_intermediary_banks(
        public_raw=raw_datasets.public_frame,
        private_raw=raw_datasets.private_frame,
        feature_specs=resolved.intermediary_features,
        force_rebuild=resolved.force_rebuild_intermediary_features,
        logger=logger,
    )
    feature_sets = assemble_feature_sets(
        public_founder_ids=raw_datasets.public_frame["founder_uuid"],
        private_founder_ids=raw_datasets.private_frame["founder_uuid"],
        banks_by_id={**repository_banks, **intermediary_banks},
        feature_sets=resolved.distillation_feature_sets,
    )
    if not feature_sets:
        raise RuntimeError("No feature sets are available for rf_calibration_mode.")

    family_sequence = _family_sequence(resolved.target_family.family_id, overrides_use.target_family)
    family_map = {spec.family_id: spec for spec in config.target_families}
    rows: list[dict[str, object]] = []
    selected_by_family: dict[str, dict[str, object]] = {}
    top_sets_by_family: dict[str, list[str]] = {}
    cpu_count = os.cpu_count() or 1
    parallel_workers = resolved.max_parallel_workers
    _log(logger, f"RF calibration parallel workers: {parallel_workers} (cpu_count={cpu_count}).")
    sweep_grid = _rf_sweep_grid(
        min_samples_leaf_values=resolved.rf_calibration_min_samples_leaf,
        max_depth_values=resolved.rf_calibration_max_depth,
        max_features_values=resolved.rf_calibration_max_features,
    )

    for family_id in family_sequence:
        target_spec = family_map[family_id]
        target_family = load_target_family(target_spec)
        for feature_set in feature_sets:
            _log(logger, f"Calibrating RF {family_id} on {feature_set.feature_set_id}.")
            target_rows = _require_full_overlap(
                feature_set.public_frame[["founder_uuid"]],
                target_family.train_frame,
                on="founder_uuid",
                left_name=f"feature set '{feature_set.feature_set_id}' public rows",
                right_name=f"target family '{family_id}' public targets",
            )
            X_public = feature_set.public_frame[feature_set.feature_columns].to_numpy(dtype=float)
            target_columns = list(target_family.target_columns)
            y_public = target_rows[target_columns].to_numpy(
                dtype=float if target_family.task_kind == "regression" else int
            )
            for params in sweep_grid:
                primary_values = _compute_primary_metrics(
                    X=X_public,
                    y=y_public,
                    target_columns=target_columns,
                    task_kind=target_family.task_kind,
                    params=params,
                    random_state=config.distillation_cv.random_state,
                    n_splits=config.reproduction.outer_cv.n_splits,
                    shuffle=config.reproduction.outer_cv.shuffle,
                    scale_min=target_family.scale_min,
                    scale_max=target_family.scale_max,
                    parallel_workers=parallel_workers,
                )
                max_depth = params["max_depth"]
                max_features = params["max_features"]
                rows.append(
                    {
                        "target_family": family_id,
                        "feature_set_id": feature_set.feature_set_id,
                        "min_samples_leaf": int(params["min_samples_leaf"]),
                        "max_depth": max_depth,
                        "max_depth_sort": (10_000 if max_depth is None else int(max_depth)),
                        "max_features": max_features,
                        "max_features_sort": str(max_features),
                        "params_signature": _params_signature(params),
                        "primary_metric": "r2" if target_family.task_kind == "regression" else "f0_5",
                        "primary_mean": float(np.mean(primary_values)),
                        "primary_std": float(np.std(primary_values)),
                    }
                )

    metrics_frame = pd.DataFrame(rows)
    if metrics_frame.empty:
        raise RuntimeError("RF calibration produced no rows.")
    for family_id in family_sequence:
        family_rows = metrics_frame[metrics_frame["target_family"] == family_id].copy()
        selected_sig_payload = _select_recommended_params(family_rows)
        selected_sig = str(selected_sig_payload["params_signature"])
        selected_rows = family_rows[family_rows["params_signature"] == selected_sig].copy()
        if selected_rows.empty:
            raise RuntimeError(f"RF calibration selected signature '{selected_sig}' had no rows.")
        first = selected_rows.iloc[0]
        selected_by_family[family_id] = {
            "n_estimators": RF_FIXED_N_ESTIMATORS,
            "bootstrap": RF_FIXED_BOOTSTRAP,
            "min_samples_leaf": int(first["min_samples_leaf"]),
            "max_depth": (None if pd.isna(first["max_depth"]) else int(first["max_depth"])),
            "max_features": first["max_features"],
        }
        top_rows = (
            selected_rows.sort_values(["primary_mean", "primary_std"], ascending=[False, True]).head(2)
        )
        top_sets_by_family[family_id] = top_rows["feature_set_id"].astype(str).tolist()

    artifact_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "fixed_params": {
            "n_estimators": RF_FIXED_N_ESTIMATORS,
            "bootstrap": RF_FIXED_BOOTSTRAP,
        },
        "sweep": {
            "min_samples_leaf": [int(value) for value in resolved.rf_calibration_min_samples_leaf],
            "max_depth": [value for value in resolved.rf_calibration_max_depth],
            "max_features": [value for value in resolved.rf_calibration_max_features],
        },
        "selected_params_by_family": selected_by_family,
        "top_feature_sets_by_family": top_sets_by_family,
        "cv": {
            "n_splits": config.reproduction.outer_cv.n_splits,
            "shuffle": config.reproduction.outer_cv.shuffle,
            "random_state": config.reproduction.outer_cv.random_state,
            "type": "stratified_reasoning_cv",
        },
        "parallel_workers": parallel_workers,
        "metrics_table": metrics_frame.drop(columns=["max_depth_sort", "max_features_sort"]).to_dict(orient="records"),
    }
    write_json(run_dir / RF_CALIBRATION_ARTIFACT_NAME, artifact_payload)

    summary_text = _render_summary_markdown(
        run_dir=run_dir,
        rows_frame=metrics_frame,
        chosen_by_family=selected_by_family,
        top_sets_by_family=top_sets_by_family,
        cv_outer_splits=config.reproduction.outer_cv.n_splits,
        cv_random_state=config.reproduction.outer_cv.random_state,
        parallel_workers=parallel_workers,
    )
    write_markdown(run_dir / "rf_calibration_summary.md", summary_text)
    write_markdown(_latest_path(), summary_text)

    _log(logger, f"RF calibration run complete. Artifacts written to {run_dir}.")
    return run_dir
