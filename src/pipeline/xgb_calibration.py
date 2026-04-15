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
CALIBRATION_ARTIFACT_NAME = "xgb_calibration_recommendations.json"


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


def _select_recommended_n_estimators(metrics_frame: pd.DataFrame, *, task_kind: str) -> int:
    primary = "primary_mean"
    ranked = (
        metrics_frame.groupby("n_estimators", as_index=False)
        .agg(primary_mean=(primary, "mean"), primary_std=(primary, "std"))
        .fillna({"primary_std": 0.0})
        .sort_values(["primary_mean", "primary_std", "n_estimators"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    if ranked.empty:
        raise RuntimeError("Calibration produced no candidate rows to select n_estimators.")
    _ = task_kind
    return int(ranked.iloc[0]["n_estimators"])


def _render_summary_markdown(
    *,
    run_dir: Path,
    estimator_sweep: list[int],
    rows_frame: pd.DataFrame,
    chosen_by_family: dict[str, int],
    top_sets_by_family: dict[str, list[str]],
    cv_outer_splits: int,
    cv_random_state: int,
    parallel_workers: int,
) -> str:
    lines: list[str] = [
        "# XGB Calibration Summary",
        "",
        f"- Run artifacts: `{run_dir}`",
        "- Calibration type: training CV only, no held-out/test usage.",
        "- Early stopping: disabled (no early stopping rounds/eval-set callbacks used).",
        f"- n_estimators sweep: {estimator_sweep}",
        f"- Outer CV: {cv_outer_splits}-fold stratified (random_state={cv_random_state})",
        f"- Parallel target workers: {parallel_workers}",
        "",
        "## Selected Defaults",
        "",
    ]
    for family_id, value in chosen_by_family.items():
        lines.append(f"- `{family_id}`: `n_estimators={value}`")
    lines.extend(["", "## Top Feature Sets At Selected n_estimators", ""])
    for family_id, feature_sets in top_sets_by_family.items():
        joined = ", ".join(f"`{feature_set}`" for feature_set in feature_sets) if feature_sets else "_none_"
        lines.append(f"- `{family_id}`: {joined}")
    lines.extend(
        [
            "",
            "## Step 2 (Default Routine Run)",
            "",
            "Run model testing on all selected feature sets with `use_latest_xgb_calibration=true` and nested CV off.",
            "",
            "## Step 3 (Confirmatory Run)",
            "",
            "Run nested CV only on top-2 feature sets per target family, then compare tuned-vs-fixed deltas and stability.",
            "",
            "## Calibration Table (feature_set x n_estimators)",
            "",
            "| target_family | feature_set_id | n_estimators | primary_metric | primary_mean | primary_std |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in rows_frame.sort_values(["target_family", "n_estimators", "feature_set_id"]).itertuples(index=False):
        lines.append(
            f"| {row.target_family} | {row.feature_set_id} | {int(row.n_estimators)} | {row.primary_metric} | "
            f"{float(row.primary_mean):.4f} | {float(row.primary_std):.4f} |"
        )
    return "\n".join(lines)


def _latest_and_archive_paths(run_dir: Path) -> tuple[Path, Path]:
    docs_dir = DOCS_DIR / "key-experiment-summaries"
    stamp = run_dir.name.replace("_xgb_calibration", "")
    latest = docs_dir / "xgb_calibration_summary_latest.md"
    archive = docs_dir / f"xgb_calibration_summary_{stamp}.md"
    return latest, archive


def _compute_primary_metrics(
    *,
    X: np.ndarray,
    y: np.ndarray,
    target_columns: list[str],
    task_kind: str,
    n_estimators: int,
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
                    "xgb1_regressor",
                    random_state=seed,
                    param_overrides={"n_estimators": n_estimators},
                )
                model.fit(X[split.train_idx], y_target[split.train_idx])
                preds = model.predict(X[split.test_idx])
                preds = np.clip(preds, scale_min if scale_min is not None else 0.0, scale_max if scale_max is not None else 1.0)
                oof[split.test_idx] = preds
            else:
                model = build_reasoning_classifier(
                    "xgb1_classifier",
                    random_state=seed,
                    param_overrides={"n_estimators": n_estimators},
                )
                model.fit(X[split.train_idx], y_target[split.train_idx].astype(int))
                probs = model.predict_proba(X[split.test_idx])[:, 1]
                oof[split.test_idx] = probs
        if np.isnan(oof).any():
            raise RuntimeError(f"Calibration OOF contains NaNs for target '{target_column}'.")
        if task_kind == "regression":
            return float(regression_metrics(y_target, oof)["r2"])
        threshold = select_f05_threshold(y_target.astype(int), oof)
        return float(
            binary_classification_metrics(
                y_target.astype(int),
                oof,
                threshold=threshold,
            )["f0_5"]
        )

    if parallel_workers <= 1 or len(target_columns) <= 1:
        return [_run_target(target_index) for target_index in range(len(target_columns))]

    use_workers = min(parallel_workers, len(target_columns))
    with ThreadPoolExecutor(max_workers=use_workers) as executor:
        futures = [executor.submit(_run_target, target_index) for target_index in range(len(target_columns))]
        return [float(future.result()) for future in futures]


def load_latest_xgb_calibration(experiment_id: str) -> dict[str, object] | None:
    root = RUNS_DIR / experiment_id
    if not root.exists():
        return None
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.endswith("_xgb_calibration")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        artifact_path = candidate / CALIBRATION_ARTIFACT_NAME
        if artifact_path.exists():
            payload = read_json(artifact_path)
            return {
                "run_dir": str(candidate),
                "artifact_path": str(artifact_path),
                **payload,
            }
    return None


def run_xgb_calibration_mode(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    overrides_use = overrides or RunOverrides()
    if overrides_use.heldout_evaluation is True:
        raise RuntimeError("xgb_calibration_mode is training-only. heldout_evaluation must be false.")

    resolved = resolve_run_options(config, replace(overrides_use, run_mode="xgb_calibration_mode"))
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "xgb_calibration")
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
        raise RuntimeError("No feature sets are available for xgb_calibration_mode.")

    family_sequence = _family_sequence(resolved.target_family.family_id, overrides_use.target_family)
    family_map = {spec.family_id: spec for spec in config.target_families}
    rows: list[dict[str, object]] = []
    selected_by_family: dict[str, int] = {}
    top_sets_by_family: dict[str, list[str]] = {}
    cpu_count = os.cpu_count() or 1
    parallel_workers = resolved.max_parallel_workers
    _log(logger, f"XGB calibration parallel workers: {parallel_workers} (cpu_count={cpu_count}).")

    for family_id in family_sequence:
        target_spec = family_map[family_id]
        target_family = load_target_family(target_spec)
        for feature_set in feature_sets:
            _log(logger, f"Calibrating {family_id} on {feature_set.feature_set_id}.")
            target_rows = _require_full_overlap(
                feature_set.public_frame[["founder_uuid"]],
                target_family.train_frame,
                on="founder_uuid",
                left_name=f"feature set '{feature_set.feature_set_id}' public rows",
                right_name=f"target family '{family_id}' public targets",
            )
            X_public = feature_set.public_frame[feature_set.feature_columns].to_numpy(dtype=float)
            target_columns = list(target_family.target_columns)
            y_public = target_rows[target_columns].to_numpy(dtype=float if target_family.task_kind == "regression" else int)
            for n_estimators in resolved.xgb_calibration_estimators:
                primary_values = _compute_primary_metrics(
                    X=X_public,
                    y=y_public,
                    target_columns=target_columns,
                    task_kind=target_family.task_kind,
                    n_estimators=int(n_estimators),
                    random_state=config.distillation_cv.random_state,
                    n_splits=config.reproduction.outer_cv.n_splits,
                    shuffle=config.reproduction.outer_cv.shuffle,
                    scale_min=target_family.scale_min,
                    scale_max=target_family.scale_max,
                    parallel_workers=parallel_workers,
                )
                rows.append(
                    {
                        "target_family": family_id,
                        "feature_set_id": feature_set.feature_set_id,
                        "n_estimators": int(n_estimators),
                        "primary_metric": "r2" if target_family.task_kind == "regression" else "f0_5",
                        "primary_mean": float(np.mean(primary_values)),
                        "primary_std": float(np.std(primary_values)),
                    }
                )

    metrics_frame = pd.DataFrame(rows)
    if metrics_frame.empty:
        raise RuntimeError("XGB calibration produced no rows.")
    for family_id in family_sequence:
        family_rows = metrics_frame[metrics_frame["target_family"] == family_id].copy()
        family_task_kind = family_map[family_id].task_kind
        selected = _select_recommended_n_estimators(family_rows, task_kind=family_task_kind)
        selected_by_family[family_id] = int(selected)
        top_rows = (
            family_rows[family_rows["n_estimators"] == selected]
            .sort_values(["primary_mean", "primary_std"], ascending=[False, True])
            .head(2)
        )
        top_sets_by_family[family_id] = top_rows["feature_set_id"].astype(str).tolist()

    artifact_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "calibration_estimator_sweep": [int(value) for value in resolved.xgb_calibration_estimators],
        "selected_n_estimators_by_family": selected_by_family,
        "top_feature_sets_by_family": top_sets_by_family,
        "cv": {
            "n_splits": config.reproduction.outer_cv.n_splits,
            "shuffle": config.reproduction.outer_cv.shuffle,
            "random_state": config.reproduction.outer_cv.random_state,
            "type": "stratified_reasoning_cv",
        },
        "parallel_workers": parallel_workers,
        "metrics_table": metrics_frame.to_dict(orient="records"),
    }
    write_json(run_dir / CALIBRATION_ARTIFACT_NAME, artifact_payload)

    summary_text = _render_summary_markdown(
        run_dir=run_dir,
        estimator_sweep=[int(value) for value in resolved.xgb_calibration_estimators],
        rows_frame=metrics_frame,
        chosen_by_family=selected_by_family,
        top_sets_by_family=top_sets_by_family,
        cv_outer_splits=config.reproduction.outer_cv.n_splits,
        cv_random_state=config.reproduction.outer_cv.random_state,
        parallel_workers=parallel_workers,
    )
    write_markdown(run_dir / "xgb_calibration_summary.md", summary_text)
    latest_path, archive_path = _latest_and_archive_paths(run_dir)
    write_markdown(latest_path, summary_text)
    write_markdown(archive_path, summary_text)

    _log(logger, f"XGB calibration run complete. Artifacts written to {run_dir}.")
    return run_dir
