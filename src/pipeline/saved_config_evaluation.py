from __future__ import annotations

from collections import defaultdict
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
from src.pipeline.reproduction import run_reproduction_mode
from src.pipeline.saved_model_configs import load_bundle_manifest, load_pickle
from src.pipeline.success_protocol import continuous_indices, run_nested_l2_success_protocol
from src.pipeline.run_options import RunOverrides, resolve_run_options
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.paths import DOCS_DIR, RUNS_DIR


Logger = Callable[[str], None]

FULL_TRANSFER_TARGET_FAMILY = "v25_policies"
FULL_TRANSFER_FEATURE_SET_ID = "lambda_policies_plus_sentence_bundle"
FULL_TRANSFER_MODEL_ORDER = ["ridge", "xgb3_regressor", "mlp_regressor"]
FULL_TRANSFER_REPORT_MODEL_ORDER = [*FULL_TRANSFER_MODEL_ORDER, "combined_best"]
FULL_TRANSFER_TIEBREAK_PRIORITY = {
    "ridge": 0,
    "xgb3_regressor": 1,
    "mlp_regressor": 2,
}
FULL_TRANSFER_REPRO_EXPERIMENT_IDS = [
    "hq_only",
    "hq_plus_policy_induction",
    "llm_engineering_only",
    "llm_engineering_plus_policy_induction",
]
FULL_TRANSFER_REPRO_HEADLINE_F05 = {
    "hq_only": 0.2730,
    "hq_plus_policy_induction": 0.3000,
    "llm_engineering_only": 0.2840,
    "llm_engineering_plus_policy_induction": 0.3340,
}
FULL_TRANSFER_REPRO_TOLERANCE = 0.005


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


def _filter_combos_by_requested_ids(
    combos: list[dict[str, object]],
    selected_combo_ids: list[str] | None,
) -> list[dict[str, object]]:
    if selected_combo_ids is None:
        return combos
    if not selected_combo_ids:
        raise RuntimeError("saved_eval_combo_ids is empty. Select at least one combo_id.")

    combo_map = {str(combo["combo_id"]): combo for combo in combos}
    unknown = [combo_id for combo_id in selected_combo_ids if combo_id not in combo_map]
    if unknown:
        raise RuntimeError(
            f"Unknown saved_eval_combo_ids: {unknown}. "
            "Select combo ids that exist in the selected bundle manifest."
        )
    return [combo_map[combo_id] for combo_id in selected_combo_ids]


def _parse_combo_ref(value: str) -> tuple[str, str]:
    raw = str(value).strip()
    if "::" not in raw:
        raise RuntimeError(
            f"Invalid combo ref '{raw}'. Expected format: <bundle_path_or_id>::<combo_id>."
        )
    bundle_part, combo_part = raw.split("::", 1)
    bundle_token = bundle_part.strip()
    combo_id = combo_part.strip()
    if not bundle_token or not combo_id:
        raise RuntimeError(
            f"Invalid combo ref '{raw}'. Expected format: <bundle_path_or_id>::<combo_id>."
        )
    return bundle_token, combo_id


def _tag_combo_with_bundle_metadata(
    combo: dict[str, object],
    *,
    bundle_dir: Path,
    source_run_dir: str | None,
) -> dict[str, object]:
    tagged = dict(combo)
    tagged["bundle_dir"] = str(bundle_dir)
    if source_run_dir:
        tagged["source_run_dir"] = str(source_run_dir)
    return tagged


def _load_requested_combos(
    *,
    bundle_dir_or_id: str | Path | None,
    selected_combo_ids: list[str] | None,
    selected_combo_refs: list[str] | None,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    combos: list[dict[str, object]] = []
    bundle_dirs_used: list[str] = []
    selected_ref_lines: list[str] = []

    if selected_combo_refs:
        refs_by_bundle: dict[str, list[str]] = defaultdict(list)
        for value in selected_combo_refs:
            bundle_token, combo_id = _parse_combo_ref(value)
            refs_by_bundle[bundle_token].append(combo_id)
        for bundle_token, combo_ids in refs_by_bundle.items():
            bundle_dir, manifest = load_bundle_manifest(bundle_token)
            bundle_combo_list = [dict(item) for item in list(manifest.get("combos", []))]
            bundle_combo_map = {str(item.get("combo_id", "")).strip(): item for item in bundle_combo_list}
            unknown = [combo_id for combo_id in combo_ids if combo_id not in bundle_combo_map]
            if unknown:
                raise RuntimeError(
                    f"Unknown combo ids for bundle '{bundle_dir}': {unknown}."
                )
            source_run_dir = (
                str(manifest.get("source_run_dir")).strip()
                if str(manifest.get("source_run_dir", "")).strip()
                else None
            )
            for combo_id in combo_ids:
                selected_ref_lines.append(f"{bundle_dir}::{combo_id}")
                combos.append(
                    _tag_combo_with_bundle_metadata(
                        bundle_combo_map[combo_id],
                        bundle_dir=bundle_dir,
                        source_run_dir=source_run_dir,
                    )
                )
            bundle_dirs_used.append(str(bundle_dir))
        return combos, sorted(set(bundle_dirs_used)), selected_ref_lines

    if not bundle_dir_or_id:
        raise RuntimeError("No saved bundle path provided.")
    bundle_dir, manifest = load_bundle_manifest(str(bundle_dir_or_id))
    combo_list = [dict(item) for item in list(manifest.get("combos", []))]
    combo_list = _filter_combos_by_requested_ids(combo_list, selected_combo_ids)
    source_run_dir = (
        str(manifest.get("source_run_dir")).strip()
        if str(manifest.get("source_run_dir", "")).strip()
        else None
    )
    tagged_list = [
        _tag_combo_with_bundle_metadata(item, bundle_dir=bundle_dir, source_run_dir=source_run_dir)
        for item in combo_list
    ]
    return tagged_list, [str(bundle_dir)], []


def _combo_key_for_metrics(combo: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(combo["target_family"]),
        str(combo["feature_set_id"]),
        str(combo["model_id"]),
        str(combo["output_mode"]),
    )


def _load_source_cv_metrics_for_combo(combo: dict[str, object]) -> pd.DataFrame:
    source_run_dir = str(combo.get("source_run_dir", "")).strip()
    if not source_run_dir:
        raise RuntimeError(
            f"Combo '{combo.get('combo_id', 'unknown')}' does not contain source_run_dir metadata."
        )
    run_dir = Path(source_run_dir)
    per_target_path = run_dir / "feature_set_screening_per_target.csv"
    repeat_metrics_path = run_dir / "feature_set_screening_repeat_metrics.csv"
    if per_target_path.exists():
        frame = pd.read_csv(per_target_path)
    elif repeat_metrics_path.exists():
        frame = pd.read_csv(repeat_metrics_path)
    else:
        raise RuntimeError(
            f"No per-target screening metrics found in source run dir '{run_dir}'. "
            "Expected feature_set_screening_per_target.csv or feature_set_screening_repeat_metrics.csv."
        )
    return frame


def _source_cv_per_target_metrics_for_combo(combo: dict[str, object]) -> pd.DataFrame:
    metrics = _load_source_cv_metrics_for_combo(combo)
    for required_column in (
        "target_family",
        "feature_set_id",
        "model_id",
        "output_mode",
        "target_id",
        "r2",
        "rmse",
        "mae",
    ):
        if required_column not in metrics.columns:
            raise RuntimeError(
                f"Missing '{required_column}' column in source run metrics for combo "
                f"'{combo.get('combo_id', 'unknown')}'."
            )
    frame = metrics.copy()
    if "split_id" in frame.columns:
        frame = frame[frame["split_id"].astype(str) == "oof_overall"]
    if "stage" in frame.columns:
        frame = frame[frame["stage"].astype(str) == "screening"]
    frame = frame[
        (frame["target_family"].astype(str) == str(combo["target_family"]))
        & (frame["feature_set_id"].astype(str) == str(combo["feature_set_id"]))
        & (frame["model_id"].astype(str) == str(combo["model_id"]))
        & (frame["output_mode"].astype(str) == str(combo["output_mode"]))
    ]
    if frame.empty:
        raise RuntimeError(
            f"No matching per-target CV metrics found for combo '{combo.get('combo_id', 'unknown')}'."
        )
    grouped = (
        frame.groupby("target_id", as_index=False)
        .agg(
            r2=("r2", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
        )
        .sort_values("target_id")
        .reset_index(drop=True)
    )
    return grouped


def _summarize_per_target_reasoning_metrics(per_target: pd.DataFrame) -> pd.DataFrame:
    summary = (
        per_target.groupby("model_set_id", as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", lambda values: float(np.std(np.asarray(values, dtype=float), ddof=0))),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
        )
        .reset_index(drop=True)
    )
    if not summary.empty:
        order = {model_id: idx for idx, model_id in enumerate(FULL_TRANSFER_REPORT_MODEL_ORDER)}
        summary = (
            summary.assign(_order=summary["model_set_id"].map(order).fillna(len(order)).astype(int))
            .sort_values(["_order", "r2_mean"], ascending=[True, False])
            .drop(columns=["_order"])
            .reset_index(drop=True)
        )
    return summary


def _build_source_cv_transfer_metrics(
    *,
    combos: list[dict[str, object]],
    assignment_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combo_per_target: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for combo in combos:
        model_set_id = str(combo["model_id"])
        per_target = _source_cv_per_target_metrics_for_combo(combo)
        combo_per_target[model_set_id] = per_target
        for row in per_target.itertuples(index=False):
            rows.append(
                {
                    "model_set_id": model_set_id,
                    "target_id": str(row.target_id),
                    "r2": float(row.r2),
                    "rmse": float(row.rmse),
                    "mae": float(row.mae),
                }
            )

    missing_models = [model_id for model_id in FULL_TRANSFER_MODEL_ORDER if model_id not in combo_per_target]
    if missing_models:
        raise RuntimeError(
            "Missing source CV per-target metrics for required model ids: "
            + ", ".join(missing_models)
        )

    for row in assignment_frame.itertuples(index=False):
        model_set_id = str(row.selected_model_id)
        target_id = str(row.target_id)
        source_frame = combo_per_target.get(model_set_id)
        if source_frame is None:
            raise RuntimeError(
                f"combined_best assignment references unknown source model_id '{model_set_id}'."
            )
        selected = source_frame[source_frame["target_id"].astype(str) == target_id]
        if len(selected) != 1:
            raise RuntimeError(
                f"Could not resolve source CV metrics for combined_best target '{target_id}' "
                f"from model_id '{model_set_id}'."
            )
        selected_row = selected.iloc[0]
        rows.append(
            {
                "model_set_id": "combined_best",
                "target_id": target_id,
                "r2": float(selected_row["r2"]),
                "rmse": float(selected_row["rmse"]),
                "mae": float(selected_row["mae"]),
            }
        )

    per_target = pd.DataFrame(rows)
    if not per_target.empty:
        order = {model_id: idx for idx, model_id in enumerate(FULL_TRANSFER_REPORT_MODEL_ORDER)}
        per_target = (
            per_target.assign(_order=per_target["model_set_id"].map(order).fillna(len(order)).astype(int))
            .sort_values(["_order", "target_id"])
            .drop(columns=["_order"])
            .reset_index(drop=True)
        )
    return per_target, _summarize_per_target_reasoning_metrics(per_target)


def _build_best_r2_assignment(
    *,
    combos: list[dict[str, object]],
    target_family: LoadedTargetFamily,
    tie_break_model_priority: dict[str, int] | None = None,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    regression_combos = [
        combo
        for combo in combos
        if str(combo.get("target_family")) == target_family.family_id
        and str(combo.get("task_kind")) == "regression"
    ]
    if not regression_combos:
        raise RuntimeError(
            f"Per-target best-R^2 composite requested, but no regression combos were selected "
            f"for target family '{target_family.family_id}'."
        )

    combo_r2_maps: dict[str, dict[str, float]] = {}
    for combo in regression_combos:
        metrics = _load_source_cv_metrics_for_combo(combo)
        for required_column in ("target_family", "feature_set_id", "model_id", "output_mode", "target_id"):
            if required_column not in metrics.columns:
                raise RuntimeError(
                    f"Missing '{required_column}' column in source run metrics for combo "
                    f"'{combo.get('combo_id', 'unknown')}'."
                )
        if "r2" not in metrics.columns:
            raise RuntimeError(
                f"Missing 'r2' column in source run metrics for combo '{combo.get('combo_id', 'unknown')}'."
            )
        frame = metrics.copy()
        if "split_id" in frame.columns:
            frame = frame[frame["split_id"].astype(str) == "oof_overall"]
        if "stage" in frame.columns:
            frame = frame[frame["stage"].astype(str) == "screening"]
        frame = frame[
            (frame["target_family"].astype(str) == str(combo["target_family"]))
            & (frame["feature_set_id"].astype(str) == str(combo["feature_set_id"]))
            & (frame["model_id"].astype(str) == str(combo["model_id"]))
            & (frame["output_mode"].astype(str) == str(combo["output_mode"]))
        ]
        if frame.empty:
            raise RuntimeError(
                f"No matching per-target CV metrics found for combo '{combo.get('combo_id', 'unknown')}'."
            )
        grouped = (
            frame.groupby("target_id", as_index=False)["r2"]
            .mean()
            .rename(columns={"r2": "cv_r2"})
        )
        combo_r2_maps[str(combo["combo_id"])] = {
            str(row.target_id): float(row.cv_r2)
            for row in grouped.itertuples(index=False)
        }

    assignment: dict[str, dict[str, object]] = {}
    assignment_rows: list[dict[str, object]] = []
    for target_id in target_family.target_columns:
        best_combo: dict[str, object] | None = None
        best_r2 = float("-inf")
        for combo in regression_combos:
            combo_id = str(combo["combo_id"])
            target_r2 = combo_r2_maps.get(combo_id, {}).get(target_id)
            if target_r2 is None:
                continue
            if target_r2 > best_r2:
                best_r2 = target_r2
                best_combo = combo
            elif target_r2 == best_r2 and best_combo is not None:
                if tie_break_model_priority is not None:
                    combo_priority = int(
                        tie_break_model_priority.get(str(combo.get("model_id", "")), 10_000)
                    )
                    best_priority = int(
                        tie_break_model_priority.get(str(best_combo.get("model_id", "")), 10_000)
                    )
                    if combo_priority < best_priority:
                        best_combo = combo
                    elif combo_priority == best_priority and str(combo["combo_id"]) < str(best_combo["combo_id"]):
                        best_combo = combo
                elif str(combo["combo_id"]) < str(best_combo["combo_id"]):
                    best_combo = combo
        if best_combo is None:
            raise RuntimeError(
                f"Could not assign a best-R^2 combo for target '{target_id}' in family "
                f"'{target_family.family_id}'."
            )
        assignment[target_id] = best_combo
        assignment_rows.append(
            {
                "target_family": target_family.family_id,
                "target_id": target_id,
                "selected_combo_id": str(best_combo["combo_id"]),
                "selected_feature_set_id": str(best_combo["feature_set_id"]),
                "selected_model_id": str(best_combo["model_id"]),
                "selected_output_mode": str(best_combo["output_mode"]),
                "selected_bundle_dir": str(best_combo["bundle_dir"]),
                "cv_r2": float(best_r2),
            }
        )
    return assignment, pd.DataFrame(assignment_rows)


def _predict_best_r2_composite(
    *,
    assignment: dict[str, dict[str, object]],
    feature_sets_by_id: dict[str, object],
    is_private: bool,
) -> pd.DataFrame:
    unique_combos = {str(combo["combo_id"]): combo for combo in assignment.values()}
    predictions_by_combo: dict[str, pd.DataFrame] = {}
    for combo_id, combo in unique_combos.items():
        feature_set = feature_sets_by_id[str(combo["feature_set_id"])]
        feature_frame = feature_set.private_frame if is_private else feature_set.public_frame
        predictions_by_combo[combo_id] = _predict_combo_on_frame(
            bundle_dir=Path(str(combo["bundle_dir"])),
            combo=combo,
            feature_frame=feature_frame,
        )

    first_combo = next(iter(unique_combos.values()))
    first_feature_set = feature_sets_by_id[str(first_combo["feature_set_id"])]
    founder_frame = first_feature_set.private_frame if is_private else first_feature_set.public_frame
    founder_ids = founder_frame["founder_uuid"].astype(str).reset_index(drop=True)
    result = pd.DataFrame(index=range(len(founder_ids)))
    for target_id, combo in assignment.items():
        combo_id = str(combo["combo_id"])
        result[target_id] = predictions_by_combo[combo_id][target_id].to_numpy(dtype=float)
    return result


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
    combos: list[dict[str, object]],
    feature_sets_by_id: dict[str, object],
    per_target_best_r2: bool,
    logger: Logger | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_map = {spec.family_id: spec for spec in config.target_families}
    loaded_targets: dict[str, LoadedTargetFamily] = {}
    rows: list[dict[str, object]] = []
    assignment_frames: list[pd.DataFrame] = []

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
        combo_bundle_dir = Path(str(combo["bundle_dir"]))
        predictions = _predict_combo_on_frame(
            bundle_dir=combo_bundle_dir,
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
                    "bundle_dir": str(combo_bundle_dir),
                    **metrics,
                }
            )
        _log(logger, f"Evaluated reasoning test metrics for combo '{combo_id}'.")

    if per_target_best_r2:
        family_ids = sorted({str(combo["target_family"]) for combo in combos})
        for family_id in family_ids:
            target_family = loaded_targets.get(family_id)
            if target_family is None:
                target_family = load_target_family(family_map[family_id])
                loaded_targets[family_id] = target_family
            if target_family.task_kind != "regression":
                _log(
                    logger,
                    f"Skipping best-R^2 composite for non-regression family '{family_id}'.",
                )
                continue
            if target_family.test_frame is None:
                raise RuntimeError(
                    f"Target family '{family_id}' has no test frame for composite evaluation."
                )
            assignment, assignment_frame = _build_best_r2_assignment(
                combos=combos,
                target_family=target_family,
            )
            assignment_frames.append(assignment_frame)
            composite_predictions = _predict_best_r2_composite(
                assignment=assignment,
                feature_sets_by_id=feature_sets_by_id,
                is_private=True,
            )
            any_combo = next(iter(assignment.values()))
            feature_set = feature_sets_by_id[str(any_combo["feature_set_id"])]
            founder_ids = feature_set.private_frame["founder_uuid"].astype(str)
            aligned_test = _align_by_founder_uuid(
                founder_ids,
                target_family.test_frame,
                frame_label=f"target family '{family_id}' test frame",
            )
            for target_name in target_family.target_columns:
                y_true = aligned_test[target_name].to_numpy(dtype=float)
                y_pred = composite_predictions[target_name].to_numpy(dtype=float)
                metrics = regression_metrics(y_true, y_pred)
                selected_combo = assignment[target_name]
                rows.append(
                    {
                        "combo_id": f"composite_best_r2__{family_id}",
                        "target_family": family_id,
                        "feature_set_id": "composite_best_r2",
                        "model_id": "best_r2_composite",
                        "output_mode": "composite",
                        "task_kind": "regression",
                        "target_id": target_name,
                        "bundle_dir": str(selected_combo["bundle_dir"]),
                        **metrics,
                    }
                )
            _log(
                logger,
                f"Evaluated per-target best-R^2 composite for family '{family_id}'.",
            )

    per_target = pd.DataFrame(rows)
    if per_target.empty:
        return per_target, pd.DataFrame(), pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    for (combo_id, family_id, feature_set_id, model_id, output_mode, task_kind, bundle_dir), frame in per_target.groupby(
        ["combo_id", "target_family", "feature_set_id", "model_id", "output_mode", "task_kind", "bundle_dir"],
        as_index=False,
    ):
        row: dict[str, object] = {
            "combo_id": combo_id,
            "target_family": family_id,
            "feature_set_id": feature_set_id,
            "model_id": model_id,
            "output_mode": output_mode,
            "task_kind": task_kind,
            "bundle_dir": bundle_dir,
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
    assignment_frame = (
        pd.concat(assignment_frames, ignore_index=True)
        if assignment_frames
        else pd.DataFrame()
    )
    return per_target, per_combo, assignment_frame


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
    combos: list[dict[str, object]],
    feature_sets_by_id: dict[str, object],
    repository_splits: LoadedFeatureRepositorySplits,
    repository_banks: dict[str, object],
    hq_exit_override_mode: str,
    per_target_best_r2: bool,
    logger: Logger | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    assignment_frames: list[pd.DataFrame] = []

    for combo in combos:
        feature_set_id = str(combo["feature_set_id"])
        combo_id = str(combo["combo_id"])
        model_id = str(combo["model_id"])
        feature_set = feature_sets_by_id[feature_set_id]
        combo_bundle_dir = Path(str(combo["bundle_dir"]))

        pred_train = _predict_combo_on_frame(
            bundle_dir=combo_bundle_dir,
            combo=combo,
            feature_frame=feature_set.public_frame,
        )
        pred_test = _predict_combo_on_frame(
            bundle_dir=combo_bundle_dir,
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
                        "bundle_dir": str(combo_bundle_dir),
                        "base_bank": base_bank_id,
                        "hq_exit_override_branch": branch_label,
                        **metrics,
                    }
                )
        _log(logger, f"Evaluated success test metrics for combo '{combo_id}'.")

    if per_target_best_r2:
        family_map = {spec.family_id: spec for spec in config.target_families}
        family_ids = sorted({str(combo["target_family"]) for combo in combos})
        for family_id in family_ids:
            target_family_spec = family_map.get(family_id)
            if target_family_spec is None:
                continue
            if target_family_spec.task_kind != "regression":
                continue
            target_family = load_target_family(target_family_spec)
            assignment, assignment_frame = _build_best_r2_assignment(
                combos=combos,
                target_family=target_family,
            )
            assignment_frames.append(assignment_frame)
            pred_train = _predict_best_r2_composite(
                assignment=assignment,
                feature_sets_by_id=feature_sets_by_id,
                is_private=False,
            )
            pred_test = _predict_best_r2_composite(
                assignment=assignment,
                feature_sets_by_id=feature_sets_by_id,
                is_private=True,
            )
            if len(pred_train) != len(repository_splits.train_ids) or len(pred_test) != len(repository_splits.test_ids):
                raise RuntimeError(
                    f"Composite prediction row count mismatch for family '{family_id}'."
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
                            "combo_id": f"composite_best_r2__{family_id}",
                            "target_family": family_id,
                            "feature_set_id": "composite_best_r2",
                            "model_id": "best_r2_composite",
                            "bundle_dir": "multi_bundle",
                            "base_bank": base_bank_id,
                            "hq_exit_override_branch": branch_label,
                            **metrics,
                        }
                    )
            _log(
                logger,
                f"Evaluated success test metrics for per-target best-R^2 composite family '{family_id}'.",
            )

    assignment_frame = (
        pd.concat(assignment_frames, ignore_index=True)
        if assignment_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(rows), assignment_frame


def _validate_full_transfer_combo_refs(
    combos: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not combos:
        raise RuntimeError("full_transfer_report requires selected combo refs.")

    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for combo in combos:
        family_id = str(combo.get("target_family", ""))
        feature_set_id = str(combo.get("feature_set_id", ""))
        task_kind = str(combo.get("task_kind", ""))
        model_id = str(combo.get("model_id", ""))
        if family_id != FULL_TRANSFER_TARGET_FAMILY:
            raise RuntimeError(
                "full_transfer_report requires target_family=v25_policies for every selected combo."
            )
        if feature_set_id != FULL_TRANSFER_FEATURE_SET_ID:
            raise RuntimeError(
                "full_transfer_report requires feature_set_id=lambda_policies_plus_sentence_bundle "
                "for every selected combo."
            )
        if task_kind != "regression":
            raise RuntimeError("full_transfer_report supports regression reasoning combos only.")
        by_model[model_id].append(combo)

    extras = sorted(set(by_model.keys()) - set(FULL_TRANSFER_MODEL_ORDER))
    if extras:
        raise RuntimeError(
            "full_transfer_report only supports model_ids "
            f"{FULL_TRANSFER_MODEL_ORDER}. Received extras: {extras}"
        )
    missing = [model_id for model_id in FULL_TRANSFER_MODEL_ORDER if model_id not in by_model]
    if missing:
        raise RuntimeError(
            "full_transfer_report requires one combo per model id "
            f"{FULL_TRANSFER_MODEL_ORDER}. Missing: {missing}"
        )
    duplicates = [model_id for model_id in FULL_TRANSFER_MODEL_ORDER if len(by_model[model_id]) != 1]
    if duplicates:
        raise RuntimeError(
            "full_transfer_report requires exactly one combo per model id. "
            f"Found duplicates for: {duplicates}"
        )
    return [by_model[model_id][0] for model_id in FULL_TRANSFER_MODEL_ORDER]


def _build_prediction_frames_by_model_set(
    *,
    combos: list[dict[str, object]],
    feature_sets_by_id: dict[str, object],
    target_family: LoadedTargetFamily,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    predictions_by_model_set: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for combo in combos:
        feature_set = feature_sets_by_id[str(combo["feature_set_id"])]
        combo_bundle_dir = Path(str(combo["bundle_dir"]))
        model_set_id = str(combo["model_id"])
        pred_train = _predict_combo_on_frame(
            bundle_dir=combo_bundle_dir,
            combo=combo,
            feature_frame=feature_set.public_frame,
        )
        pred_test = _predict_combo_on_frame(
            bundle_dir=combo_bundle_dir,
            combo=combo,
            feature_frame=feature_set.private_frame,
        )
        predictions_by_model_set[model_set_id] = (pred_train, pred_test)

    assignment, assignment_frame = _build_best_r2_assignment(
        combos=combos,
        target_family=target_family,
        tie_break_model_priority=FULL_TRANSFER_TIEBREAK_PRIORITY,
    )
    composite_train = _predict_best_r2_composite(
        assignment=assignment,
        feature_sets_by_id=feature_sets_by_id,
        is_private=False,
    )
    composite_test = _predict_best_r2_composite(
        assignment=assignment,
        feature_sets_by_id=feature_sets_by_id,
        is_private=True,
    )
    predictions_by_model_set["combined_best"] = (composite_train, composite_test)
    return predictions_by_model_set, assignment_frame


def _evaluate_reasoning_transfer(
    *,
    target_family: LoadedTargetFamily,
    feature_sets_by_id: dict[str, object],
    combos: list[dict[str, object]],
    predictions_by_model_set: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_set = feature_sets_by_id[FULL_TRANSFER_FEATURE_SET_ID]
    founder_ids = feature_set.private_frame["founder_uuid"].astype(str)
    aligned_test = _align_by_founder_uuid(
        founder_ids,
        target_family.test_frame,
        frame_label=f"target family '{target_family.family_id}' test frame",
    )
    model_combo_map = {str(combo["model_id"]): combo for combo in combos}
    rows: list[dict[str, object]] = []
    for model_set_id, (_, pred_test) in predictions_by_model_set.items():
        for target_id in target_family.target_columns:
            y_true = aligned_test[target_id].to_numpy(dtype=float)
            y_pred = pred_test[target_id].to_numpy(dtype=float)
            metrics = regression_metrics(y_true, y_pred)
            source_combo = model_combo_map.get(model_set_id)
            rows.append(
                {
                    "model_set_id": model_set_id,
                    "target_id": target_id,
                    "source_model_id": str(source_combo["model_id"]) if source_combo else "combined_best",
                    "source_combo_id": str(source_combo["combo_id"]) if source_combo else "combined_best",
                    "feature_set_id": FULL_TRANSFER_FEATURE_SET_ID,
                    **metrics,
                }
            )
    per_target = pd.DataFrame(rows)
    summary = (
        per_target.groupby("model_set_id", as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", lambda values: float(np.std(np.asarray(values, dtype=float), ddof=0))),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
        )
        .sort_values(["r2_mean", "rmse_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return per_target, summary


def _evaluate_success_transfer(
    *,
    config: ExperimentConfig,
    predictions_by_model_set: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    repository_splits: LoadedFeatureRepositorySplits,
    repository_banks: dict[str, object],
) -> pd.DataFrame:
    hq_bank = repository_banks["hq_baseline"]
    llm_bank = repository_banks["llm_engineering"]
    hq_binary = set(getattr(hq_bank, "binary_feature_columns", []))
    llm_binary = set(getattr(llm_bank, "binary_feature_columns", []))

    train_labels = (
        repository_splits.train_labels.set_index("founder_uuid")
        .reindex(repository_splits.train_ids)["success"]
        .astype(int)
        .to_numpy(dtype=int)
    )
    test_labels = (
        repository_splits.test_labels.set_index("founder_uuid")
        .reindex(repository_splits.test_ids)["success"]
        .astype(int)
        .to_numpy(dtype=int)
    )

    hq_train = (
        hq_bank.public_frame.set_index("founder_uuid")
        .reindex(repository_splits.train_ids)
        .drop(columns=["founder_uuid"], errors="ignore")
    )
    hq_test = (
        hq_bank.private_frame.set_index("founder_uuid")
        .reindex(repository_splits.test_ids)
        .drop(columns=["founder_uuid"], errors="ignore")
    )
    llm_train = (
        llm_bank.public_frame.set_index("founder_uuid")
        .reindex(repository_splits.train_ids)
        .drop(columns=["founder_uuid"], errors="ignore")
    )
    llm_test = (
        llm_bank.private_frame.set_index("founder_uuid")
        .reindex(repository_splits.test_ids)
        .drop(columns=["founder_uuid"], errors="ignore")
    )

    hq_exit_train = hq_train["exit_count"].to_numpy(dtype=float) if "exit_count" in hq_train.columns else None
    hq_exit_test = hq_test["exit_count"].to_numpy(dtype=float) if "exit_count" in hq_test.columns else None

    rows: list[dict[str, object]] = []
    branch_ids = [
        "pred_reasoning_only",
        "hq_plus_pred_reasoning",
        "llm_engineering_plus_pred_reasoning",
    ]
    for model_set_id, (pred_train_raw, pred_test_raw) in predictions_by_model_set.items():
        if len(pred_train_raw) != len(repository_splits.train_ids) or len(pred_test_raw) != len(repository_splits.test_ids):
            raise RuntimeError(
                f"Prediction row count mismatch for model_set_id='{model_set_id}'. "
                f"Expected train/test rows {len(repository_splits.train_ids)}/{len(repository_splits.test_ids)}, "
                f"got {len(pred_train_raw)}/{len(pred_test_raw)}."
            )
        pred_train = pred_train_raw.copy()
        pred_test = pred_test_raw.copy()
        pred_train.index = repository_splits.train_ids
        pred_test.index = repository_splits.test_ids

        branch_frames = {
            "pred_reasoning_only": (
                pred_train.reset_index(drop=True),
                pred_test.reset_index(drop=True),
                [],
                False,
                None,
                None,
            ),
            "hq_plus_pred_reasoning": (
                pd.concat([hq_train.reset_index(drop=True), pred_train.reset_index(drop=True)], axis=1),
                pd.concat([hq_test.reset_index(drop=True), pred_test.reset_index(drop=True)], axis=1),
                list(hq_binary),
                True,
                hq_exit_train,
                hq_exit_test,
            ),
            "llm_engineering_plus_pred_reasoning": (
                pd.concat([llm_train.reset_index(drop=True), pred_train.reset_index(drop=True)], axis=1),
                pd.concat([llm_test.reset_index(drop=True), pred_test.reset_index(drop=True)], axis=1),
                list(llm_binary),
                False,
                None,
                None,
            ),
        }
        for branch_id in branch_ids:
            train_features, test_features, binary_columns, use_override, train_exit, test_exit = branch_frames[branch_id]
            train_features_use = train_features.copy()
            test_features_use = test_features.copy()
            y_train_use = np.asarray(train_labels, dtype=int)
            train_exit_use = None if train_exit is None else np.asarray(train_exit, dtype=float)

            # Mirror reproduction's non-seed handling: if a base bank is missing train rows
            # (e.g. llm_engineering seed exclusions), drop those train rows for this branch.
            if train_features_use.isna().any(axis=1).any():
                valid_mask = ~train_features_use.isna().any(axis=1)
                train_features_use = train_features_use.loc[valid_mask].reset_index(drop=True)
                y_train_use = y_train_use[valid_mask.to_numpy()]
                if train_exit_use is not None:
                    train_exit_use = train_exit_use[valid_mask.to_numpy()]

            if test_features_use.isna().any(axis=1).any():
                raise RuntimeError(
                    f"Success transfer branch '{branch_id}' produced NaNs on held-out test features. "
                    "This indicates a feature alignment issue in the selected base bank."
                )

            column_names = list(train_features_use.columns)
            protocol = run_nested_l2_success_protocol(
                X_train=train_features_use.to_numpy(dtype=float),
                y_train=y_train_use,
                X_test=test_features_use.to_numpy(dtype=float),
                y_test=test_labels,
                continuous_indices=continuous_indices(column_names, binary_columns),
                outer_n_splits=config.reproduction.outer_cv.n_splits,
                outer_shuffle=config.reproduction.outer_cv.shuffle,
                outer_random_state=config.reproduction.outer_cv.random_state,
                inner_n_splits=config.reproduction.inner_cv.n_splits,
                inner_shuffle=config.reproduction.inner_cv.shuffle,
                inner_random_state=config.reproduction.inner_cv.random_state,
                c_grid=config.reproduction.logistic_c_grid,
                threshold_start=config.reproduction.threshold_grid.start,
                threshold_stop=config.reproduction.threshold_grid.stop,
                threshold_step=config.reproduction.threshold_grid.step,
                use_nested=True,
                use_exit_override=use_override,
                train_exit_counts=train_exit_use,
                test_exit_counts=test_exit,
            )
            rows.append(
                {
                    "model_set_id": model_set_id,
                    "branch_id": branch_id,
                    "selected_c_final": float(protocol["selected_c_final"]),
                    "selected_c_oof_mean": (
                        float(protocol["selected_c_oof_mean"])
                        if protocol["selected_c_oof_mean"] is not None
                        else None
                    ),
                    **dict(protocol["test_metrics"]),
                }
            )
    return pd.DataFrame(rows)


def _load_reproduction_consistency_reference(
    *,
    config: ExperimentConfig,
    logger: Logger | None,
) -> tuple[pd.DataFrame, str]:
    experiment_dir = RUNS_DIR / config.experiment_id
    candidate_dirs = sorted(
        [
            path
            for path in experiment_dir.glob("*_success_reproduction")
            if path.is_dir() and (path / "reproduction_results.csv").exists()
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in candidate_dirs:
        summary_path = run_dir / "run_summary.md"
        if summary_path.exists():
            summary_text = summary_path.read_text(encoding="utf-8")
            if "Nested hyperparameter CV enabled: True" not in summary_text:
                continue
        _log(logger, f"Using existing reproduction reference run: {run_dir}")
        return pd.read_csv(run_dir / "reproduction_results.csv"), str(run_dir)

    _log(logger, "No compatible reproduction reference run found. Running reproduction now.")
    reproduction_run_dir = run_reproduction_mode(
        config,
        use_nested_hyperparameter_cv=True,
        logger=logger,
    )
    return (
        pd.read_csv(reproduction_run_dir / "reproduction_results.csv"),
        str(reproduction_run_dir),
    )


def _build_reproduction_consistency_table(
    *,
    reproduction_results: pd.DataFrame,
    source_run_dir: str,
    tolerance: float,
) -> pd.DataFrame:
    if "experiment_id" not in reproduction_results.columns or "test_f0_5" not in reproduction_results.columns:
        raise RuntimeError("Reproduction results are missing required columns: experiment_id, test_f0_5.")
    rows: list[dict[str, object]] = []
    by_id = {
        str(row.experiment_id): row
        for row in reproduction_results.itertuples(index=False)
    }
    for experiment_id in FULL_TRANSFER_REPRO_EXPERIMENT_IDS:
        headline = float(FULL_TRANSFER_REPRO_HEADLINE_F05[experiment_id])
        result_row = by_id.get(experiment_id)
        if result_row is None:
            raise RuntimeError(
                f"Reproduction consistency check could not find experiment '{experiment_id}' in reproduction results."
            )
        reproduced = float(getattr(result_row, "test_f0_5"))
        delta = reproduced - headline
        abs_delta = abs(delta)
        rows.append(
            {
                "experiment_id": experiment_id,
                "headline_target_f0_5": headline,
                "reproduced_test_f0_5": reproduced,
                "delta_f0_5": delta,
                "abs_delta_f0_5": abs_delta,
                "within_tolerance": bool(abs_delta <= tolerance),
                "tolerance": tolerance,
                "source_reproduction_run_dir": source_run_dir,
            }
        )
    return pd.DataFrame(rows)


def _render_full_transfer_report(
    *,
    run_dir: Path,
    combo_refs_used: list[str],
    source_cv_summary: pd.DataFrame,
    reasoning_summary: pd.DataFrame,
    reasoning_per_target: pd.DataFrame,
    assignment_frame: pd.DataFrame,
    success_metrics: pd.DataFrame,
    reproduction_consistency: pd.DataFrame,
    reproduction_source_run_dir: str,
) -> str:
    branch_sections = [
        ("pred_reasoning_only", "reasoning_pred-only"),
        ("hq_plus_pred_reasoning", "HQ + reasoning_pred"),
        ("llm_engineering_plus_pred_reasoning", "LLM-eng + reasoning_pred"),
    ]

    lines: list[str] = [
        "# Lambda Bundle Full Transfer Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Target family: `{FULL_TRANSFER_TARGET_FAMILY}`",
        f"- Feature set: `{FULL_TRANSFER_FEATURE_SET_ID}`",
        f"- Combo refs: {combo_refs_used}",
        f"- Reproduction reference run: `{reproduction_source_run_dir}`",
        "",
        "## CV Validation Performance (Source Runs)",
        "",
        "These are CV metrics taken from the source model-testing runs used to build the saved model bundles.",
        "",
        "| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |",
        "|---|---:|---:|---:|---:|",
    ]
    cv_order = {model_id: idx for idx, model_id in enumerate(FULL_TRANSFER_REPORT_MODEL_ORDER)}
    source_cv_sorted = source_cv_summary.copy()
    if not source_cv_sorted.empty:
        source_cv_sorted = source_cv_sorted.assign(
            _order=source_cv_sorted["model_set_id"].astype(str).map(cv_order).fillna(len(cv_order)).astype(int)
        ).sort_values(["_order", "r2_mean"], ascending=[True, False]).drop(columns=["_order"])
    for row in source_cv_sorted.itertuples(index=False):
        lines.append(
            f"| {row.model_set_id} | {row.r2_mean:.4f} | {row.r2_std:.4f} | {row.rmse_mean:.4f} | {row.mae_mean:.4f} |"
        )
    lines.extend(
        [
            "",
            "`combined_best` is defined per target by highest source CV R² across ridge/xgb/mlp "
            "(tie-break: ridge > xgb3_regressor > mlp_regressor).",
            "",
            "## Held-out Test Performance (Reasoning Agreement)",
            "",
            "| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in reasoning_summary.itertuples(index=False):
        lines.append(
            f"| {row.model_set_id} | {row.r2_mean:.4f} | {row.r2_std:.4f} | {row.rmse_mean:.4f} | {row.mae_mean:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Held-out Test Performance (Success Transfer)",
            "",
        ]
    )
    success = success_metrics.copy()
    for branch_id, branch_title in branch_sections:
        lines.extend(
            [
                f"### {branch_title}",
                "",
                "| model_set_id | f0_5 | roc_auc | pr_auc | precision | recall | threshold |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        branch_rows = success[success["branch_id"].astype(str) == branch_id].copy()
        if not branch_rows.empty:
            branch_rows = branch_rows.sort_values(["f0_5", "roc_auc"], ascending=[False, False]).reset_index(
                drop=True
            )
            for row in branch_rows.itertuples(index=False):
                lines.append(
                    f"| {row.model_set_id} | {row.f0_5:.4f} | {row.roc_auc:.4f} | "
                    f"{row.pr_auc:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.threshold:.4f} |"
                )
        else:
            lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        lines.append("")

    lines.extend(
        [
            "## Reproduction Consistency Check",
            "",
            f"- Tolerance: ±{FULL_TRANSFER_REPRO_TOLERANCE:.3f} F0.5",
            "",
            "| experiment_id | headline_target_f0_5 | reproduced_test_f0_5 | delta_f0_5 | abs_delta_f0_5 | within_tolerance |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in reproduction_consistency.itertuples(index=False):
        lines.append(
            f"| {row.experiment_id} | {row.headline_target_f0_5:.4f} | {row.reproduced_test_f0_5:.4f} | "
            f"{row.delta_f0_5:+.4f} | {row.abs_delta_f0_5:.4f} | {bool(row.within_tolerance)} |"
        )

    lines.extend(
        [
            "",
            "## Combined Best Assignment (CV-R2 Source)",
            "",
            "| target_id | selected_model_id | selected_combo_id | cv_r2 |",
            "|---|---|---|---:|",
        ]
    )
    for row in assignment_frame.sort_values("target_id").itertuples(index=False):
        lines.append(
            f"| {row.target_id} | {row.selected_model_id} | {row.selected_combo_id} | {row.cv_r2:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Data Tables (CSV Artifacts)",
            "",
            "- `reasoning_transfer_cv_summary.csv`",
            "- `reasoning_transfer_cv_per_target.csv`",
            "- `reasoning_transfer_per_target.csv`",
            "- `reasoning_transfer_summary.csv`",
            "- `success_transfer_metrics.csv`",
            "- `combined_best_assignment.csv`",
            "- `reproduction_consistency_check.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_full_transfer_docs(
    *,
    run_dir: Path,
    report_markdown: str,
) -> None:
    docs_root = DOCS_DIR / "key-experiment-summaries" / "custom_reports" / "testing_models"
    docs_root.mkdir(parents=True, exist_ok=True)
    write_markdown(docs_root / "full_transfer_report_latest.md", report_markdown)
    write_markdown(docs_root / f"full_transfer_report_{run_dir.name}.md", report_markdown)


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

    combos, bundle_dirs_used, selected_ref_lines = _load_requested_combos(
        bundle_dir_or_id=resolved.saved_config_bundle_path,
        selected_combo_ids=resolved.saved_eval_combo_ids,
        selected_combo_refs=resolved.saved_eval_combo_refs,
    )
    if not combos:
        raise RuntimeError("Selected saved-config bundle has no persisted combos.")
    if resolved.saved_eval_combo_refs:
        _log(
            logger,
            f"Loaded {len(combos)} cross-bundle combos from {len(bundle_dirs_used)} bundle(s).",
        )
    else:
        _log(logger, f"Loaded saved model bundle from {bundle_dirs_used[0]}.")
    _log(logger, f"Saved-config evaluation combo count: {len(combos)}.")

    extra_bank_ids = (
        {"hq_baseline", "llm_engineering"}
        if resolved.saved_eval_mode in {"success_with_pred_reasoning", "full_transfer_report"}
        else None
    )
    feature_sets_by_id, repository_splits, repository_banks = _load_feature_sets_for_bundle(
        config=config,
        combos=combos,
        required_extra_feature_bank_ids=extra_bank_ids,
        logger=logger,
    )

    if resolved.saved_eval_mode == "full_transfer_report":
        validated_combos = _validate_full_transfer_combo_refs(combos)
        family_map = {spec.family_id: spec for spec in config.target_families}
        target_family = load_target_family(family_map[FULL_TRANSFER_TARGET_FAMILY])
        if target_family.test_frame is None:
            raise RuntimeError(
                f"Target family '{FULL_TRANSFER_TARGET_FAMILY}' has no held-out test targets."
            )

        predictions_by_model_set, assignment_frame = _build_prediction_frames_by_model_set(
            combos=validated_combos,
            feature_sets_by_id=feature_sets_by_id,
            target_family=target_family,
        )
        reasoning_per_target, reasoning_summary = _evaluate_reasoning_transfer(
            target_family=target_family,
            feature_sets_by_id=feature_sets_by_id,
            combos=validated_combos,
            predictions_by_model_set=predictions_by_model_set,
        )
        source_cv_per_target, source_cv_summary = _build_source_cv_transfer_metrics(
            combos=validated_combos,
            assignment_frame=assignment_frame,
        )
        success_transfer = _evaluate_success_transfer(
            config=config,
            predictions_by_model_set=predictions_by_model_set,
            repository_splits=repository_splits,
            repository_banks=repository_banks,
        )
        reproduction_results, reproduction_source_run_dir = _load_reproduction_consistency_reference(
            config=config,
            logger=logger,
        )
        reproduction_consistency = _build_reproduction_consistency_table(
            reproduction_results=reproduction_results,
            source_run_dir=reproduction_source_run_dir,
            tolerance=FULL_TRANSFER_REPRO_TOLERANCE,
        )

        write_csv(run_dir / "reasoning_transfer_cv_per_target.csv", source_cv_per_target)
        write_csv(run_dir / "reasoning_transfer_cv_summary.csv", source_cv_summary)
        write_csv(run_dir / "reasoning_transfer_per_target.csv", reasoning_per_target)
        write_csv(run_dir / "reasoning_transfer_summary.csv", reasoning_summary)
        write_csv(run_dir / "success_transfer_metrics.csv", success_transfer)
        write_csv(run_dir / "reproduction_consistency_check.csv", reproduction_consistency)
        write_csv(run_dir / "combined_best_assignment.csv", assignment_frame)

        combo_refs_used = list(resolved.saved_eval_combo_refs or selected_ref_lines)
        report_markdown = _render_full_transfer_report(
            run_dir=run_dir,
            combo_refs_used=combo_refs_used,
            source_cv_summary=source_cv_summary,
            reasoning_summary=reasoning_summary,
            reasoning_per_target=reasoning_per_target,
            assignment_frame=assignment_frame,
            success_metrics=success_transfer,
            reproduction_consistency=reproduction_consistency,
            reproduction_source_run_dir=reproduction_source_run_dir,
        )
        write_markdown(run_dir / "full_transfer_report.md", report_markdown)
        write_markdown(run_dir / "saved_config_eval_summary.md", report_markdown)
        _write_full_transfer_docs(run_dir=run_dir, report_markdown=report_markdown)
        _log(logger, f"Saved-config full transfer report complete. Artifacts written to {run_dir}.")
        return run_dir

    if resolved.saved_eval_mode == "reasoning_test_metrics":
        per_target, per_combo, assignment_frame = _evaluate_reasoning_test_metrics(
            config=config,
            combos=combos,
            feature_sets_by_id=feature_sets_by_id,
            per_target_best_r2=resolved.saved_eval_per_target_best_r2,
            logger=logger,
        )
        write_csv(run_dir / "reasoning_test_metrics_per_target.csv", per_target)
        write_csv(run_dir / "reasoning_test_metrics_per_combo.csv", per_combo)
        if not assignment_frame.empty:
            write_csv(run_dir / "best_r2_target_assignment.csv", assignment_frame)
        summary_lines = [
            "# Saved Config Evaluation Summary",
            "",
            "- Mode: reasoning_test_metrics",
            f"- Bundles: {bundle_dirs_used}",
            f"- Combo count: {len(per_combo)}",
            (
                f"- Requested combo filter: {resolved.saved_eval_combo_ids}"
                if resolved.saved_eval_combo_ids is not None
                else "- Requested combo filter: all combos in bundle"
            ),
            (
                f"- Requested combo refs: {selected_ref_lines}"
                if selected_ref_lines
                else "- Requested combo refs: none"
            ),
            f"- Per-target best R^2 composite enabled: {resolved.saved_eval_per_target_best_r2}",
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
        if not assignment_frame.empty:
            summary_lines.extend(
                [
                    "",
                    "## Per-target best-R^2 assignment",
                    "",
                    "| target_family | target_id | selected_combo_id | selected_feature_set_id | selected_model_id | cv_r2 |",
                    "|---|---|---|---|---|---:|",
                ]
            )
            for row in assignment_frame.sort_values(["target_family", "target_id"]).itertuples(index=False):
                summary_lines.append(
                    f"| {row.target_family} | {row.target_id} | {row.selected_combo_id} | "
                    f"{row.selected_feature_set_id} | {row.selected_model_id} | {row.cv_r2:.4f} |"
                )
        write_markdown(run_dir / "saved_config_eval_summary.md", "\n".join(summary_lines))
        _log(logger, f"Saved-config reasoning evaluation complete. Artifacts written to {run_dir}.")
        return run_dir

    success_results, assignment_frame = _evaluate_success_with_pred_reasoning(
        config=config,
        combos=combos,
        feature_sets_by_id=feature_sets_by_id,
        repository_splits=repository_splits,
        repository_banks=repository_banks,
        hq_exit_override_mode=resolved.hq_exit_override_mode,
        per_target_best_r2=resolved.saved_eval_per_target_best_r2,
        logger=logger,
    )
    write_csv(run_dir / "success_with_pred_reasoning_metrics.csv", success_results)
    if not assignment_frame.empty:
        write_csv(run_dir / "best_r2_target_assignment.csv", assignment_frame)

    summary_lines = [
        "# Saved Config Evaluation Summary",
        "",
        "- Mode: success_with_pred_reasoning",
        f"- Bundles: {bundle_dirs_used}",
        f"- HQ override mode: {resolved.hq_exit_override_mode}",
        f"- Per-target best R^2 composite enabled: {resolved.saved_eval_per_target_best_r2}",
        (
            f"- Requested combo filter: {resolved.saved_eval_combo_ids}"
            if resolved.saved_eval_combo_ids is not None
            else "- Requested combo filter: all combos in bundle"
        ),
        (
            f"- Requested combo refs: {selected_ref_lines}"
            if selected_ref_lines
            else "- Requested combo refs: none"
        ),
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
    if not assignment_frame.empty:
        summary_lines.extend(
            [
                "",
                "## Per-target best-R^2 assignment",
                "",
                "| target_family | target_id | selected_combo_id | selected_feature_set_id | selected_model_id | cv_r2 |",
                "|---|---|---|---|---|---:|",
            ]
        )
        for row in assignment_frame.sort_values(["target_family", "target_id"]).itertuples(index=False):
            summary_lines.append(
                f"| {row.target_family} | {row.target_id} | {row.selected_combo_id} | "
                f"{row.selected_feature_set_id} | {row.selected_model_id} | {row.cv_r2:.4f} |"
            )
    write_markdown(run_dir / "saved_config_eval_summary.md", "\n".join(summary_lines))
    _log(logger, f"Saved-config success evaluation complete. Artifacts written to {run_dir}.")
    return run_dir
