from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
import os
from pathlib import Path
import time
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
from src.utils.parallel import bounded_worker_count
from src.utils.paths import DOCS_DIR, RUNS_DIR


Logger = Callable[[str], None]
MLP_CALIBRATION_ARTIFACT_NAME = "mlp_calibration_recommendations.json"


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


def _mlp_sweep_grid(*, hidden_layer_sizes: list[list[int]], alpha_values: list[float]) -> list[dict[str, object]]:
    grid: list[dict[str, object]] = []
    for layers in hidden_layer_sizes:
        for alpha in alpha_values:
            grid.append(
                {
                    "hidden_layer_sizes": tuple(int(v) for v in layers),
                    "alpha": float(alpha),
                }
            )
    return grid


def _params_signature(params: dict[str, object]) -> str:
    layers = ",".join(str(v) for v in params["hidden_layer_sizes"])
    return f"hidden_layer_sizes={layers}|alpha={params['alpha']}"


def _select_recommended_params(metrics_frame: pd.DataFrame) -> str:
    ranked = (
        metrics_frame.groupby("params_signature", as_index=False)
        .agg(primary_mean=("primary_mean", "mean"), primary_std=("primary_std", "mean"))
        .fillna({"primary_std": 0.0})
        .sort_values(["primary_mean", "primary_std", "params_signature"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    if ranked.empty:
        raise RuntimeError("MLP calibration produced no candidate rows to select parameters.")
    return str(ranked.iloc[0]["params_signature"])


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
        "# MLP Calibration Summary",
        "",
        f"- Run artifacts: `{run_dir}`",
        "- Calibration type: training CV only, no held-out/test usage.",
        "- Training form: one native multi-output MLP per fold/parameter combo.",
        f"- Outer CV: {cv_outer_splits}-fold stratified (random_state={cv_random_state})",
        f"- Parallel target workers: {parallel_workers}",
        "",
        "## Selected Defaults",
        "",
    ]
    for family_id, params in chosen_by_family.items():
        layers = tuple(int(v) for v in params["hidden_layer_sizes"])
        lines.append(
            f"- `{family_id}`: `hidden_layer_sizes={layers}`, "
            f"`alpha={params['alpha']}`"
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
            "| target_family | feature_set_id | hidden_layer_sizes | alpha | primary_metric | primary_mean | primary_std |",
            "|---|---|---|---:|---|---:|---:|",
        ]
    )
    table = rows_frame.sort_values(
        ["target_family", "feature_set_id", "hidden_layer_sizes_sort", "alpha"],
        ascending=[True, True, True, True],
    )
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.target_family} | {row.feature_set_id} | {row.hidden_layer_sizes} | {float(row.alpha):.4f} | "
            f"{row.primary_metric} | {float(row.primary_mean):.4f} | {float(row.primary_std):.4f} |"
        )
    return "\n".join(lines)


def _latest_path() -> Path:
    docs_dir = DOCS_DIR / "experiment-archive" / "generated-reports"
    return docs_dir / "mlp_calibration_summary_latest.md"


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

    # For MLP calibration, always use one native multi-output model per fold.
    # This shares representation across targets and avoids per-target model duplication.
    n_targets = len(target_columns)
    oof = np.full((len(y), n_targets), np.nan, dtype=float)

    for fold_offset, split in enumerate(splits):
        seed = random_state + fold_offset
        if task_kind == "regression":
            model = build_reasoning_regressor(
                "mlp_regressor",
                random_state=seed,
                param_overrides=params,
            )
            model.fit(X[split.train_idx], y[split.train_idx])
            preds = np.asarray(model.predict(X[split.test_idx]), dtype=float)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            preds = np.clip(
                preds,
                scale_min if scale_min is not None else 0.0,
                scale_max if scale_max is not None else 1.0,
            )
            oof[split.test_idx] = preds
        else:
            model = build_reasoning_classifier(
                "mlp_classifier",
                random_state=seed,
                param_overrides=params,
            )
            model.fit(X[split.train_idx], y[split.train_idx].astype(int))
            probs_raw = model.predict_proba(X[split.test_idx])
            if isinstance(probs_raw, list):
                cols: list[np.ndarray] = []
                for probs in probs_raw:
                    arr = np.asarray(probs, dtype=float)
                    if arr.ndim == 1:
                        cols.append(arr)
                    elif arr.shape[1] == 1:
                        cols.append(arr[:, 0])
                    else:
                        cols.append(arr[:, 1])
                probs = np.column_stack(cols)
            else:
                probs = np.asarray(probs_raw, dtype=float)
                if probs.ndim == 1:
                    probs = probs.reshape(-1, 1)
            if probs.shape[1] != n_targets:
                raise RuntimeError(
                    f"MLP calibration probability output has {probs.shape[1]} columns; expected {n_targets}."
                )
            oof[split.test_idx] = probs

    if np.isnan(oof).any():
        raise RuntimeError("MLP calibration OOF contains NaNs in multi-output predictions.")

    if task_kind == "regression":
        return [
            float(regression_metrics(y[:, target_index], oof[:, target_index])["r2"])
            for target_index in range(n_targets)
        ]

    values: list[float] = []
    for target_index in range(n_targets):
        y_target = y[:, target_index].astype(int)
        threshold = select_f05_threshold(y_target, oof[:, target_index])
        values.append(float(binary_classification_metrics(y_target, oof[:, target_index], threshold=threshold)["f0_5"]))
    return values


def load_latest_mlp_calibration(experiment_id: str) -> dict[str, object] | None:
    root = RUNS_DIR / experiment_id
    if not root.exists():
        return None
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.endswith("_mlp_calibration")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        artifact_path = candidate / MLP_CALIBRATION_ARTIFACT_NAME
        if artifact_path.exists():
            payload = read_json(artifact_path)
            return {
                "run_dir": str(candidate),
                "artifact_path": str(artifact_path),
                **payload,
            }
    return None


def run_mlp_calibration_mode(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    overrides_use = overrides or RunOverrides()
    if overrides_use.heldout_evaluation is True:
        raise RuntimeError("mlp_calibration_mode is training-only. heldout_evaluation must be false.")

    resolved = resolve_run_options(config, replace(overrides_use, run_mode="mlp_calibration_mode"))
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "mlp_calibration")
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
        raise RuntimeError("No feature sets are available for mlp_calibration_mode.")

    family_sequence = _family_sequence(resolved.target_family.family_id, overrides_use.target_family)
    family_map = {spec.family_id: spec for spec in config.target_families}
    rows: list[dict[str, object]] = []
    selected_by_family: dict[str, dict[str, object]] = {}
    top_sets_by_family: dict[str, list[str]] = {}
    cpu_count = os.cpu_count() or 1
    parallel_workers = resolved.max_parallel_workers
    _log(logger, f"MLP calibration parallel workers: {parallel_workers} (cpu_count={cpu_count}).")
    sweep_grid = _mlp_sweep_grid(
        hidden_layer_sizes=resolved.mlp_calibration_hidden_layer_sizes,
        alpha_values=resolved.mlp_calibration_alpha,
    )
    total_tasks = len(family_sequence) * len(feature_sets) * len(sweep_grid)
    task_index = 0
    t0 = time.perf_counter()
    _log(
        logger,
        f"MLP calibration plan: families={len(family_sequence)}, feature_sets={len(feature_sets)}, "
        f"param_combos={len(sweep_grid)}, total_tasks={total_tasks}.",
    )

    calibration_tasks: list[dict[str, object]] = []
    for family_id in family_sequence:
        target_spec = family_map[family_id]
        target_family = load_target_family(target_spec)
        for feature_set in feature_sets:
            _log(logger, f"Preparing MLP calibration data for {family_id} on {feature_set.feature_set_id}.")
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
                calibration_tasks.append(
                    {
                        "family_id": family_id,
                        "feature_set_id": feature_set.feature_set_id,
                        "task_kind": target_family.task_kind,
                        "target_columns": target_columns,
                        "X_public": X_public,
                        "y_public": y_public,
                        "params": params,
                        "scale_min": target_family.scale_min,
                        "scale_max": target_family.scale_max,
                    }
                )

    worker_count = bounded_worker_count(
        max_parallel_workers=parallel_workers,
        task_count=len(calibration_tasks),
    )
    _log(logger, f"MLP calibration task workers: {worker_count}.")

    def _run_calibration_task(task: dict[str, object]) -> dict[str, object]:
        family_id = str(task["family_id"])
        feature_set_id = str(task["feature_set_id"])
        params = {**dict(task["params"]), "early_stopping": False}
        task_kind = str(task["task_kind"])
        target_columns = list(task["target_columns"])
        primary_values = _compute_primary_metrics(
            X=np.asarray(task["X_public"], dtype=float),
            y=np.asarray(task["y_public"]),
            target_columns=target_columns,
            task_kind=task_kind,
            params=params,
            random_state=config.distillation_cv.random_state,
            n_splits=config.reproduction.outer_cv.n_splits,
            shuffle=config.reproduction.outer_cv.shuffle,
            scale_min=task["scale_min"] if task["scale_min"] is None else float(task["scale_min"]),
            scale_max=task["scale_max"] if task["scale_max"] is None else float(task["scale_max"]),
            parallel_workers=parallel_workers,
        )
        layers_tuple = tuple(int(v) for v in params["hidden_layer_sizes"])
        return {
            "target_family": family_id,
            "feature_set_id": feature_set_id,
            "hidden_layer_sizes": str(layers_tuple),
            "hidden_layer_sizes_sort": "-".join(f"{v:04d}" for v in layers_tuple),
            "alpha": float(params["alpha"]),
            "params_signature": _params_signature(params),
            "primary_metric": "r2" if task_kind == "regression" else "f0_5",
            "primary_mean": float(np.mean(primary_values)),
            "primary_std": float(np.std(primary_values)),
        }

    if worker_count == 1:
        for task in calibration_tasks:
            task_index += 1
            row = _run_calibration_task(task)
            rows.append(row)
            elapsed = max(time.perf_counter() - t0, 1e-9)
            rate = task_index / elapsed
            remaining = max(total_tasks - task_index, 0)
            eta_seconds = remaining / rate if rate > 0 else 0.0
            _log(
                logger,
                f"[{task_index}/{total_tasks}] family={row['target_family']} feature_set={row['feature_set_id']} "
                f"params={row['params_signature']} elapsed={elapsed/60:.1f}m eta={eta_seconds/60:.1f}m",
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_run_calibration_task, task): task
                for task in calibration_tasks
            }
            for future in as_completed(future_map):
                row = future.result()
                rows.append(row)
                task_index += 1
                elapsed = max(time.perf_counter() - t0, 1e-9)
                rate = task_index / elapsed
                remaining = max(total_tasks - task_index, 0)
                eta_seconds = remaining / rate if rate > 0 else 0.0
                _log(
                    logger,
                    f"[{task_index}/{total_tasks}] family={row['target_family']} feature_set={row['feature_set_id']} "
                    f"params={row['params_signature']} elapsed={elapsed/60:.1f}m eta={eta_seconds/60:.1f}m",
                )

    metrics_frame = pd.DataFrame(rows)
    if metrics_frame.empty:
        raise RuntimeError("MLP calibration produced no rows.")
    for family_id in family_sequence:
        family_rows = metrics_frame[metrics_frame["target_family"] == family_id].copy()
        selected_sig = _select_recommended_params(family_rows)
        selected_rows = family_rows[family_rows["params_signature"] == selected_sig].copy()
        if selected_rows.empty:
            raise RuntimeError(f"MLP calibration selected signature '{selected_sig}' had no rows.")
        first = selected_rows.iloc[0]
        selected_by_family[family_id] = {
            "hidden_layer_sizes": [int(v) for v in str(first["hidden_layer_sizes"]).strip("() ").split(",") if str(v).strip()],
            "alpha": float(first["alpha"]),
        }
        top_rows = selected_rows.sort_values(["primary_mean", "primary_std"], ascending=[False, True]).head(2)
        top_sets_by_family[family_id] = top_rows["feature_set_id"].astype(str).tolist()

    artifact_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "sweep": {
            "hidden_layer_sizes": [list(layer) for layer in resolved.mlp_calibration_hidden_layer_sizes],
            "alpha": [float(v) for v in resolved.mlp_calibration_alpha],
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
        "metrics_table": metrics_frame.to_dict(orient="records"),
    }
    write_json(run_dir / MLP_CALIBRATION_ARTIFACT_NAME, artifact_payload)

    summary_text = _render_summary_markdown(
        run_dir=run_dir,
        rows_frame=metrics_frame,
        chosen_by_family=selected_by_family,
        top_sets_by_family=top_sets_by_family,
        cv_outer_splits=config.reproduction.outer_cv.n_splits,
        cv_random_state=config.reproduction.outer_cv.random_state,
        parallel_workers=parallel_workers,
    )
    write_markdown(run_dir / "mlp_calibration_summary.md", summary_text)
    write_markdown(_latest_path(), summary_text)

    total_minutes = (time.perf_counter() - t0) / 60.0
    _log(logger, f"MLP calibration run complete in {total_minutes:.1f}m. Artifacts written to {run_dir}.")
    return run_dir
