from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.feature_repository import (
    LoadedFeatureRepositorySplits,
    load_feature_repository_splits,
    load_repository_feature_banks,
)
from src.data.raw_datasets import load_raw_datasets
from src.data.splits import build_public_cv_splits
from src.data.targets import LoadedTargetFamily, load_target_family
from src.evaluation.metrics import (
    binary_classification_metrics,
    regression_metrics,
    select_f05_threshold,
)
from src.intermediary_features.registry import assemble_feature_sets, prepare_intermediary_banks
from src.pipeline.config import ExperimentConfig
from src.pipeline.saved_model_configs import load_bundle_manifest, load_pickle
from src.pipeline.run_options import RunOverrides, resolve_run_options
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.paths import RUNS_DIR


Logger = Callable[[str], None]


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _align_by_founder_uuid(
    left_ids: pd.Series,
    right_frame: pd.DataFrame,
    *,
    id_column: str = "founder_uuid",
    frame_label: str,
) -> pd.DataFrame:
    right_idx = right_frame.set_index(id_column)
    missing = [value for value in left_ids.astype(str) if value not in right_idx.index.astype(str)]
    if missing:
        raise RuntimeError(
            f"{frame_label} is missing {len(missing)} founders required for evaluation. Examples: {missing[:5]}"
        )
    aligned = right_idx.reindex(left_ids.astype(str)).reset_index()
    return aligned


def _predict_binary_probabilities(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X), dtype=float)
        if probs.ndim == 2:
            if probs.shape[1] == 1:
                return probs[:, 0]
            return probs[:, 1]
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    raise RuntimeError("Loaded classifier does not expose predict_proba or decision_function.")


def _predict_multi_output_probabilities(model: object, X: np.ndarray, n_targets: int) -> np.ndarray:
    probs = model.predict_proba(X)
    if isinstance(probs, list):
        columns: list[np.ndarray] = []
        for target_idx, value in enumerate(probs):
            if target_idx >= n_targets:
                break
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 1:
                columns.append(arr)
            elif arr.shape[1] == 1:
                columns.append(arr[:, 0])
            else:
                columns.append(arr[:, 1])
        if len(columns) != n_targets:
            raise RuntimeError("Loaded multi-output classifier returned unexpected probability shape.")
        return np.column_stack(columns)
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == n_targets:
        return arr
    raise RuntimeError("Loaded multi-output classifier returned unexpected probability matrix.")


def _predict_combo_on_frame(
    *,
    bundle_dir: Path,
    combo: dict[str, object],
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = [str(value) for value in list(combo["feature_columns"])]
    target_columns = [str(value) for value in list(combo["target_columns"])]
    output_mode = str(combo["output_mode"])
    task_kind = str(combo["task_kind"])
    X = feature_frame[feature_columns].to_numpy(dtype=float)

    if output_mode == "multi_output":
        rel_path = str(combo["model_artifact_relpath"])
        model = load_pickle(bundle_dir / rel_path)
        if task_kind == "regression":
            preds = np.asarray(model.predict(X), dtype=float)
        else:
            preds = _predict_multi_output_probabilities(model, X, len(target_columns))
        return pd.DataFrame(preds, columns=target_columns)

    if output_mode != "single_target":
        raise RuntimeError(f"Unsupported output_mode '{output_mode}' in saved combo.")

    artifacts = {str(k): str(v) for k, v in dict(combo["target_model_artifacts"]).items()}
    columns: list[np.ndarray] = []
    for target_column in target_columns:
        if target_column not in artifacts:
            raise RuntimeError(f"Missing artifact path for target '{target_column}' in combo.")
        model = load_pickle(bundle_dir / artifacts[target_column])
        if task_kind == "regression":
            preds = np.asarray(model.predict(X), dtype=float)
        else:
            preds = _predict_binary_probabilities(model, X)
        columns.append(preds)
    return pd.DataFrame(np.column_stack(columns), columns=target_columns)


def _load_feature_sets_for_bundle(
    *,
    config: ExperimentConfig,
    combos: list[dict[str, object]],
    required_extra_feature_bank_ids: set[str] | None = None,
    logger: Logger | None,
) -> tuple[
    dict[str, object],
    LoadedFeatureRepositorySplits,
    dict[str, object],
]:
    feature_set_map = {spec.feature_set_id: spec for spec in config.distillation_feature_sets}
    required_feature_set_ids = sorted({str(combo["feature_set_id"]) for combo in combos})
    required_feature_bank_ids = sorted(
        {
            feature_bank_id
            for feature_set_id in required_feature_set_ids
            for feature_bank_id in feature_set_map[feature_set_id].feature_bank_ids
        }
    )
    if required_extra_feature_bank_ids:
        required_feature_bank_ids = sorted(
            set(required_feature_bank_ids).union(required_extra_feature_bank_ids)
        )
    _log(logger, "Loading Feature Repository splits and feature banks for saved-config evaluation.")
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
    return (
        {feature_set.feature_set_id: feature_set for feature_set in feature_sets},
        repository_splits,
        repository_banks,
    )


def _evaluate_reasoning_test_metrics(
    *,
    config: ExperimentConfig,
    bundle_dir: Path,
    combos: list[dict[str, object]],
    feature_sets_by_id: dict[str, object],
    logger: Logger | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_map = {spec.family_id: spec for spec in config.target_families}
    loaded_targets: dict[str, LoadedTargetFamily] = {}
    rows: list[dict[str, object]] = []

    for combo in combos:
        family_id = str(combo["target_family"])
        if family_id not in loaded_targets:
            loaded_targets[family_id] = load_target_family(family_map[family_id])
        target_family = loaded_targets[family_id]
        if target_family.test_frame is None:
            raise RuntimeError(f"Target family '{family_id}' has no test frame for saved-config evaluation.")
        feature_set_id = str(combo["feature_set_id"])
        feature_set = feature_sets_by_id[feature_set_id]
        founder_ids = feature_set.private_frame["founder_uuid"].astype(str)
        aligned_test = _align_by_founder_uuid(
            founder_ids,
            target_family.test_frame,
            frame_label=f"target family '{family_id}' test frame",
        )
        predictions = _predict_combo_on_frame(
            bundle_dir=bundle_dir,
            combo=combo,
            feature_frame=feature_set.private_frame,
        )
        thresholds = {str(k): float(v) for k, v in dict(combo.get("thresholds_by_target", {})).items()}
        task_kind = str(combo["task_kind"])
        model_id = str(combo["model_id"])
        combo_id = str(combo["combo_id"])
        output_mode = str(combo["output_mode"])

        for target_name in [str(value) for value in list(combo["target_columns"])]:
            y_true = aligned_test[target_name].to_numpy(dtype=float if task_kind == "regression" else int)
            y_pred = predictions[target_name].to_numpy(dtype=float)
            if task_kind == "regression":
                metrics = regression_metrics(y_true, y_pred)
            else:
                metrics = binary_classification_metrics(
                    y_true.astype(int),
                    y_pred,
                    threshold=float(thresholds.get(target_name, 0.5)),
                )
            rows.append(
                {
                    "combo_id": combo_id,
                    "target_family": family_id,
                    "feature_set_id": feature_set_id,
                    "model_id": model_id,
                    "output_mode": output_mode,
                    "task_kind": task_kind,
                    "target_id": target_name,
                    **metrics,
                }
            )
        _log(logger, f"Evaluated reasoning test metrics for combo '{combo_id}'.")

    per_target = pd.DataFrame(rows)
    if per_target.empty:
        return per_target, pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    for (combo_id, family_id, feature_set_id, model_id, output_mode, task_kind), frame in per_target.groupby(
        ["combo_id", "target_family", "feature_set_id", "model_id", "output_mode", "task_kind"],
        as_index=False,
    ):
        row: dict[str, object] = {
            "combo_id": combo_id,
            "target_family": family_id,
            "feature_set_id": feature_set_id,
            "model_id": model_id,
            "output_mode": output_mode,
            "task_kind": task_kind,
            "target_count": int(len(frame)),
        }
        if task_kind == "regression":
            row["primary_metric"] = "r2"
            row["primary_value"] = float(frame["r2"].mean())
            row["mae_mean"] = float(frame["mae"].mean())
            row["rmse_mean"] = float(frame["rmse"].mean())
        else:
            row["primary_metric"] = "f0_5"
            row["primary_value"] = float(frame["f0_5"].mean())
            row["roc_auc_mean"] = float(frame["roc_auc"].mean())
            row["pr_auc_mean"] = float(frame["pr_auc"].mean())
        summary_rows.append(row)
    per_combo = pd.DataFrame(summary_rows).sort_values(
        ["target_family", "task_kind", "primary_value"],
        ascending=[True, True, False],
    )
    return per_target, per_combo


def _prepare_success_arrays(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    continuous_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_use = train_df.copy()
    eval_use = eval_df.copy()
    all_columns = list(train_use.columns)
    continuous_set = set(continuous_columns)
    for column in all_columns:
        if column in continuous_set:
            train_values = pd.to_numeric(train_use[column], errors="coerce")
            fill_value = float(train_values.mean()) if not train_values.isna().all() else 0.0
            train_values = train_values.fillna(fill_value)
            eval_values = pd.to_numeric(eval_use[column], errors="coerce").fillna(fill_value)
            mean = float(train_values.mean())
            std = float(train_values.std(ddof=0))
            if std <= 0.0:
                std = 1.0
            train_use[column] = ((train_values - mean) / std).astype(float)
            eval_use[column] = ((eval_values - mean) / std).astype(float)
            continue
        train_use[column] = pd.to_numeric(train_use[column], errors="coerce").fillna(0.0).astype(float)
        eval_use[column] = pd.to_numeric(eval_use[column], errors="coerce").fillna(0.0).astype(float)
    return train_use.to_numpy(dtype=float), eval_use.to_numpy(dtype=float)


def _fit_success_branch(
    *,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_exit_counts: np.ndarray | None,
    test_exit_counts: np.ndarray | None,
    use_hq_exit_override: bool,
    config: ExperimentConfig,
) -> dict[str, float]:
    splits = build_public_cv_splits(
        y_train,
        n_splits=config.reproduction.outer_cv.n_splits,
        shuffle=config.reproduction.outer_cv.shuffle,
        random_state=config.reproduction.outer_cv.random_state,
    )
    continuous_columns = list(train_features.columns)
    oof = np.full(len(train_features), np.nan, dtype=float)
    for fold_index, split in enumerate(splits):
        X_train_fold, X_eval_fold = _prepare_success_arrays(
            train_features.iloc[split.train_idx],
            train_features.iloc[split.test_idx],
            continuous_columns=continuous_columns,
        )
        model = LogisticRegression(
            solver="lbfgs",
            C=5.0,
            max_iter=3000,
            random_state=config.reproduction.outer_cv.random_state + fold_index,
        )
        model.fit(X_train_fold, y_train[split.train_idx])
        fold_probs = model.predict_proba(X_eval_fold)[:, 1]
        if use_hq_exit_override and train_exit_counts is not None:
            fold_probs = np.where(train_exit_counts[split.test_idx] > 0, 1.0, fold_probs)
        oof[split.test_idx] = fold_probs
    if np.isnan(oof).any():
        raise RuntimeError("OOF success predictions contain NaNs.")
    threshold = float(select_f05_threshold(y_train, oof))

    X_train_full, X_test_full = _prepare_success_arrays(
        train_features,
        test_features,
        continuous_columns=continuous_columns,
    )
    final_model = LogisticRegression(
        solver="lbfgs",
        C=5.0,
        max_iter=3000,
        random_state=config.reproduction.outer_cv.random_state + 10_000,
    )
    final_model.fit(X_train_full, y_train)
    test_probs = final_model.predict_proba(X_test_full)[:, 1]
    if use_hq_exit_override and test_exit_counts is not None:
        test_probs = np.where(test_exit_counts > 0, 1.0, test_probs)
    return binary_classification_metrics(y_test, test_probs, threshold=threshold)


def _evaluate_success_with_pred_reasoning(
    *,
    config: ExperimentConfig,
    bundle_dir: Path,
    combos: list[dict[str, object]],
    feature_sets_by_id: dict[str, object],
    repository_splits: LoadedFeatureRepositorySplits,
    repository_banks: dict[str, object],
    hq_exit_override_mode: str,
    logger: Logger | None,
) -> pd.DataFrame:
    if "hq_baseline" not in repository_banks:
        raise RuntimeError("HQ baseline feature bank is required for success_with_pred_reasoning.")
    if "llm_engineering" not in repository_banks:
        raise RuntimeError("LLM-engineering feature bank is required for success_with_pred_reasoning.")

    train_labels = (
        repository_splits.train_labels.set_index("founder_uuid")
        .reindex(repository_splits.train_ids)["success"]
        .astype(int)
    )
    test_labels = (
        repository_splits.test_labels.set_index("founder_uuid")
        .reindex(repository_splits.test_ids)["success"]
        .astype(int)
    )

    hq_bank = repository_banks["hq_baseline"]
    llm_bank = repository_banks["llm_engineering"]
    hq_train = hq_bank.public_frame.set_index("founder_uuid").reindex(repository_splits.train_ids)
    hq_test = hq_bank.private_frame.set_index("founder_uuid").reindex(repository_splits.test_ids)
    llm_train = llm_bank.public_frame.set_index("founder_uuid").reindex(repository_splits.train_ids)
    llm_test = llm_bank.private_frame.set_index("founder_uuid").reindex(repository_splits.test_ids)

    hq_exit_train = hq_train["exit_count"].to_numpy(dtype=float) if "exit_count" in hq_train.columns else None
    hq_exit_test = hq_test["exit_count"].to_numpy(dtype=float) if "exit_count" in hq_test.columns else None
    hq_binary = set(getattr(hq_bank, "binary_feature_columns", []))
    llm_binary = set(getattr(llm_bank, "binary_feature_columns", []))
    rows: list[dict[str, object]] = []

    for combo in combos:
        feature_set_id = str(combo["feature_set_id"])
        combo_id = str(combo["combo_id"])
        model_id = str(combo["model_id"])
        feature_set = feature_sets_by_id[feature_set_id]

        pred_train = _predict_combo_on_frame(
            bundle_dir=bundle_dir,
            combo=combo,
            feature_frame=feature_set.public_frame,
        )
        pred_test = _predict_combo_on_frame(
            bundle_dir=bundle_dir,
            combo=combo,
            feature_frame=feature_set.private_frame,
        )
        pred_train.index = repository_splits.train_ids
        pred_test.index = repository_splits.test_ids

        for base_bank_id, base_train, base_test, binary_columns in [
            ("hq_baseline", hq_train, hq_test, hq_binary),
            ("llm_engineering", llm_train, llm_test, llm_binary),
        ]:
            branch_variants = [("with_override", True)] if base_bank_id == "hq_baseline" else [("without_override", False)]
            if base_bank_id == "hq_baseline" and hq_exit_override_mode == "both_with_and_without":
                branch_variants = [("with_override", True), ("without_override", False)]

            base_columns = [
                column for column in base_train.columns
                if column != "founder_uuid"
            ]
            base_train_frame = base_train[base_columns].reset_index(drop=True)
            base_test_frame = base_test[base_columns].reset_index(drop=True)
            pred_train_frame = pred_train.reset_index(drop=True)
            pred_test_frame = pred_test.reset_index(drop=True)

            train_features = pd.concat([base_train_frame, pred_train_frame], axis=1)
            test_features = pd.concat([base_test_frame, pred_test_frame], axis=1)
            continuous_columns = [column for column in train_features.columns if column not in binary_columns]

            for branch_label, use_override in branch_variants:
                metrics = _fit_success_branch(
                    train_features=train_features,
                    test_features=test_features,
                    y_train=train_labels.to_numpy(dtype=int),
                    y_test=test_labels.to_numpy(dtype=int),
                    train_exit_counts=hq_exit_train if base_bank_id == "hq_baseline" else None,
                    test_exit_counts=hq_exit_test if base_bank_id == "hq_baseline" else None,
                    use_hq_exit_override=(base_bank_id == "hq_baseline" and use_override),
                    config=config,
                )
                rows.append(
                    {
                        "combo_id": combo_id,
                        "target_family": str(combo["target_family"]),
                        "feature_set_id": feature_set_id,
                        "model_id": model_id,
                        "base_bank": base_bank_id,
                        "hq_exit_override_branch": branch_label,
                        **metrics,
                    }
                )
        _log(logger, f"Evaluated success test metrics for combo '{combo_id}'.")
    return pd.DataFrame(rows)


def run_saved_config_evaluation_mode(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    overrides_use = overrides or RunOverrides()
    resolved = resolve_run_options(config, replace(overrides_use, run_mode="saved_config_evaluation_mode"))
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "saved_config_evaluation")
    write_json(run_dir / "resolved_config.json", asdict(config))
    write_json(run_dir / "resolved_run_options.json", asdict(resolved))

    bundle_dir, bundle_manifest = load_bundle_manifest(str(resolved.saved_config_bundle_path))
    combos = [dict(item) for item in list(bundle_manifest.get("combos", []))]
    if not combos:
        raise RuntimeError("Selected saved-config bundle has no persisted combos.")
    _log(logger, f"Loaded saved model bundle from {bundle_dir}.")

    extra_bank_ids = (
        {"hq_baseline", "llm_engineering"}
        if resolved.saved_eval_mode == "success_with_pred_reasoning"
        else None
    )
    feature_sets_by_id, repository_splits, repository_banks = _load_feature_sets_for_bundle(
        config=config,
        combos=combos,
        required_extra_feature_bank_ids=extra_bank_ids,
        logger=logger,
    )

    if resolved.saved_eval_mode == "reasoning_test_metrics":
        per_target, per_combo = _evaluate_reasoning_test_metrics(
            config=config,
            bundle_dir=bundle_dir,
            combos=combos,
            feature_sets_by_id=feature_sets_by_id,
            logger=logger,
        )
        write_csv(run_dir / "reasoning_test_metrics_per_target.csv", per_target)
        write_csv(run_dir / "reasoning_test_metrics_per_combo.csv", per_combo)
        summary_lines = [
            "# Saved Config Evaluation Summary",
            "",
            "- Mode: reasoning_test_metrics",
            f"- Bundle: {bundle_dir}",
            f"- Combo count: {len(per_combo)}",
            "",
        ]
        if per_combo.empty:
            summary_lines.append("No reasoning test metrics were produced.")
        else:
            summary_lines.extend(
                [
                    "| target_family | feature_set_id | model_id | output_mode | primary_metric | primary_value |",
                    "|---|---|---|---|---|---:|",
                ]
            )
            for row in per_combo.itertuples(index=False):
                summary_lines.append(
                    f"| {row.target_family} | {row.feature_set_id} | {row.model_id} | {row.output_mode} | "
                    f"{row.primary_metric} | {row.primary_value:.4f} |"
                )
        write_markdown(run_dir / "saved_config_eval_summary.md", "\n".join(summary_lines))
        _log(logger, f"Saved-config reasoning evaluation complete. Artifacts written to {run_dir}.")
        return run_dir

    success_results = _evaluate_success_with_pred_reasoning(
        config=config,
        bundle_dir=bundle_dir,
        combos=combos,
        feature_sets_by_id=feature_sets_by_id,
        repository_splits=repository_splits,
        repository_banks=repository_banks,
        hq_exit_override_mode=resolved.hq_exit_override_mode,
        logger=logger,
    )
    write_csv(run_dir / "success_with_pred_reasoning_metrics.csv", success_results)

    summary_lines = [
        "# Saved Config Evaluation Summary",
        "",
        "- Mode: success_with_pred_reasoning",
        f"- Bundle: {bundle_dir}",
        f"- HQ override mode: {resolved.hq_exit_override_mode}",
        "",
    ]
    if success_results.empty:
        summary_lines.append("No success metrics were produced.")
    else:
        ordered = success_results.sort_values(["f0_5", "roc_auc"], ascending=[False, False])
        summary_lines.extend(
            [
                "| combo_id | base_bank | hq_exit_override_branch | f0_5 | roc_auc | pr_auc |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in ordered.itertuples(index=False):
            summary_lines.append(
                f"| {row.combo_id} | {row.base_bank} | {row.hq_exit_override_branch} | "
                f"{row.f0_5:.4f} | {row.roc_auc:.4f} | {row.pr_auc:.4f} |"
            )
    write_markdown(run_dir / "saved_config_eval_summary.md", "\n".join(summary_lines))
    _log(logger, f"Saved-config success evaluation complete. Artifacts written to {run_dir}.")
    return run_dir
