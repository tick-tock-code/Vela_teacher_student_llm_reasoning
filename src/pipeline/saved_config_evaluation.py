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
from src.pipeline.success_protocol import (
    continuous_indices,
    default_l2_c,
    run_nested_l2_soft_ensemble_success_protocol,
    run_nested_l2_success_cv_only,
    run_nested_l2_success_protocol,
    sweep_threshold_grid,
)
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

COMBINATION_TRANSFER_TARGET_FAMILY = "v25_policies"
COMBINATION_TRANSFER_ALLOWED_FEATURE_SET_IDS = [
    "hq_plus_sentence_bundle",
    "llm_engineering_plus_sentence_bundle",
    "lambda_policies_plus_sentence_bundle",
    "hq_plus_llm_engineering_plus_sentence_bundle",
    "hq_plus_lambda_policies_plus_sentence_bundle",
    "llm_engineering_plus_lambda_policies_plus_sentence_bundle",
    "hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle",
]
COMBINATION_TRANSFER_BASE_SUCCESS_COMBOS = [
    ("hq_baseline", ["hq_baseline"]),
    ("llm_engineering", ["llm_engineering"]),
    ("lambda_policies", ["lambda_policies"]),
    ("hq_plus_llm_engineering", ["hq_baseline", "llm_engineering"]),
    ("hq_plus_lambda_policies", ["hq_baseline", "lambda_policies"]),
    ("llm_engineering_plus_lambda_policies", ["llm_engineering", "lambda_policies"]),
    ("hq_plus_llm_engineering_plus_lambda_policies", ["hq_baseline", "llm_engineering", "lambda_policies"]),
]
TRANSFER_DEFAULT_MODEL_ID = "ridge"
TRANSFER_DEFAULT_OUTPUT_MODE = "single_target"
TRANSFER_DEFAULT_TASK_KIND = "regression"
SUCCESS_MODEL_VARIANT_ORDER = {
    "single_model": 0,
    "soft_avg_model": 1,
    "soft_avg_weighted_model": 2,
}
DEFAULT_SUCCESS_MODEL_VARIANTS = tuple(SUCCESS_MODEL_VARIANT_ORDER.keys())


def _resolve_success_model_variants(
    success_model_variants: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if success_model_variants is None:
        return DEFAULT_SUCCESS_MODEL_VARIANTS
    requested: list[str] = []
    for value in success_model_variants:
        variant = str(value).strip()
        if not variant or variant in requested:
            continue
        if variant not in SUCCESS_MODEL_VARIANT_ORDER:
            raise RuntimeError(
                f"Unsupported success model variant '{variant}'. "
                f"Supported: {sorted(SUCCESS_MODEL_VARIANT_ORDER)}"
            )
        requested.append(variant)
    if not requested:
        raise RuntimeError("At least one success model variant must be selected.")
    return tuple(requested)


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


def _resolve_success_override_variants(
    *,
    hq_exit_override_mode: str,
    default_use_override: bool,
) -> list[tuple[str, bool]]:
    if hq_exit_override_mode == "with_override":
        return [("with_override", True)] if default_use_override else [("without_override", False)]
    if hq_exit_override_mode == "both_with_and_without":
        if default_use_override:
            return [("with_override", True), ("without_override", False)]
        return [("without_override", False)]
    if hq_exit_override_mode == "force_off_all_branches":
        return [("without_override", False)]
    if hq_exit_override_mode == "force_on_all_branches":
        return [("with_override", True)]
    if hq_exit_override_mode == "both_force_off_and_on_all_branches":
        return [("without_override", False), ("with_override", True)]
    raise RuntimeError(f"Unsupported hq_exit_override_mode '{hq_exit_override_mode}'.")


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

    hq_exit_train_series = hq_train["exit_count"].astype(float) if "exit_count" in hq_train.columns else None
    hq_exit_test_series = hq_test["exit_count"].astype(float) if "exit_count" in hq_test.columns else None
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
        train_founders = [str(value) for value in feature_set.public_frame["founder_uuid"].tolist()]
        test_founders = [str(value) for value in feature_set.private_frame["founder_uuid"].tolist()]
        if len(pred_train) != len(train_founders) or len(pred_test) != len(test_founders):
            raise RuntimeError(
                f"Prediction row count mismatch for combo '{combo_id}'. "
                f"Expected {len(train_founders)}/{len(test_founders)}, got {len(pred_train)}/{len(pred_test)}."
            )
        pred_train.index = train_founders
        pred_test.index = test_founders
        train_ids = [founder_id for founder_id in repository_splits.train_ids if founder_id in set(pred_train.index)]
        missing_test = [founder_id for founder_id in repository_splits.test_ids if founder_id not in set(pred_test.index)]
        if missing_test:
            raise RuntimeError(
                f"Combo '{combo_id}' is missing held-out prediction rows for {len(missing_test)} founders. "
                f"Examples: {missing_test[:5]}"
            )
        test_ids = [founder_id for founder_id in repository_splits.test_ids if founder_id in set(pred_test.index)]
        y_train_combo = train_labels.reindex(train_ids).to_numpy(dtype=int)
        y_test_combo = test_labels.reindex(test_ids).to_numpy(dtype=int)
        hq_exit_train_combo = (
            hq_exit_train_series.reindex(train_ids).to_numpy(dtype=float)
            if hq_exit_train_series is not None
            else None
        )
        hq_exit_test_combo = (
            hq_exit_test_series.reindex(test_ids).to_numpy(dtype=float)
            if hq_exit_test_series is not None
            else None
        )
        pred_train_frame = pred_train.reindex(train_ids).reset_index(drop=True)
        pred_test_frame = pred_test.reindex(test_ids).reset_index(drop=True)
        branch_specs = [
            ("pred_reasoning_only", None, None, set(), False),
            ("hq_plus_pred_reasoning", hq_train, hq_test, hq_binary, True),
            ("llm_engineering_plus_pred_reasoning", llm_train, llm_test, llm_binary, False),
        ]
        for branch_id, base_train, base_test, binary_columns, default_use_override in branch_specs:
            if base_train is None or base_test is None:
                base_train_frame = pd.DataFrame(index=range(len(pred_train_frame)))
                base_test_frame = pd.DataFrame(index=range(len(pred_test_frame)))
            else:
                base_columns = [column for column in base_train.columns if column != "founder_uuid"]
                base_train_frame = base_train.reindex(train_ids)[base_columns].reset_index(drop=True)
                base_test_frame = base_test.reindex(test_ids)[base_columns].reset_index(drop=True)
            train_features = pd.concat([base_train_frame, pred_train_frame], axis=1)
            test_features = pd.concat([base_test_frame, pred_test_frame], axis=1)
            continuous_columns = [column for column in train_features.columns if column not in binary_columns]
            branch_variants = _resolve_success_override_variants(
                hq_exit_override_mode=hq_exit_override_mode,
                default_use_override=default_use_override,
            )
            for branch_label, use_override in branch_variants:
                metrics = _fit_success_branch(
                    train_features=train_features,
                    test_features=test_features,
                    y_train=y_train_combo,
                    y_test=y_test_combo,
                    train_exit_counts=hq_exit_train_combo,
                    test_exit_counts=hq_exit_test_combo,
                    use_hq_exit_override=use_override,
                    config=config,
                )
                rows.append(
                    {
                        "combo_id": combo_id,
                        "target_family": str(combo["target_family"]),
                        "feature_set_id": feature_set_id,
                        "model_id": model_id,
                        "bundle_dir": str(combo_bundle_dir),
                        "base_bank": branch_id,
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

            train_ids = [str(founder_id) for founder_id in repository_splits.train_ids]
            test_ids = [str(founder_id) for founder_id in repository_splits.test_ids]
            y_train_combo = train_labels.reindex(train_ids).to_numpy(dtype=int)
            y_test_combo = test_labels.reindex(test_ids).to_numpy(dtype=int)
            hq_exit_train_combo = (
                hq_exit_train_series.reindex(train_ids).to_numpy(dtype=float)
                if hq_exit_train_series is not None
                else None
            )
            hq_exit_test_combo = (
                hq_exit_test_series.reindex(test_ids).to_numpy(dtype=float)
                if hq_exit_test_series is not None
                else None
            )
            pred_train_frame = pred_train.reindex(train_ids).reset_index(drop=True)
            pred_test_frame = pred_test.reindex(test_ids).reset_index(drop=True)
            branch_specs = [
                ("pred_reasoning_only", None, None, set(), False),
                ("hq_plus_pred_reasoning", hq_train, hq_test, hq_binary, True),
                ("llm_engineering_plus_pred_reasoning", llm_train, llm_test, llm_binary, False),
            ]
            for branch_id, base_train, base_test, binary_columns, default_use_override in branch_specs:
                if base_train is None or base_test is None:
                    base_train_frame = pd.DataFrame(index=range(len(pred_train_frame)))
                    base_test_frame = pd.DataFrame(index=range(len(pred_test_frame)))
                else:
                    base_columns = [column for column in base_train.columns if column != "founder_uuid"]
                    base_train_frame = base_train.reindex(train_ids)[base_columns].reset_index(drop=True)
                    base_test_frame = base_test.reindex(test_ids)[base_columns].reset_index(drop=True)

                train_features = pd.concat([base_train_frame, pred_train_frame], axis=1)
                test_features = pd.concat([base_test_frame, pred_test_frame], axis=1)
                continuous_columns = [column for column in train_features.columns if column not in binary_columns]
                branch_variants = _resolve_success_override_variants(
                    hq_exit_override_mode=hq_exit_override_mode,
                    default_use_override=default_use_override,
                )

                for branch_label, use_override in branch_variants:
                    metrics = _fit_success_branch(
                        train_features=train_features,
                        test_features=test_features,
                        y_train=y_train_combo,
                        y_test=y_test_combo,
                        train_exit_counts=hq_exit_train_combo,
                        test_exit_counts=hq_exit_test_combo,
                        use_hq_exit_override=use_override,
                        config=config,
                    )
                    rows.append(
                        {
                            "combo_id": f"composite_best_r2__{family_id}",
                            "target_family": family_id,
                            "feature_set_id": "composite_best_r2",
                            "model_id": "best_r2_composite",
                            "bundle_dir": "multi_bundle",
                            "base_bank": branch_id,
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
    if not by_model:
        raise RuntimeError(
            "full_transfer_report requires at least one selected combo from "
            f"{FULL_TRANSFER_MODEL_ORDER}."
        )
    duplicates = [model_id for model_id in FULL_TRANSFER_MODEL_ORDER if len(by_model.get(model_id, [])) > 1]
    if duplicates:
        raise RuntimeError(
            "full_transfer_report allows at most one combo per model id. "
            f"Found duplicates for: {duplicates}"
        )
    return [by_model[model_id][0] for model_id in FULL_TRANSFER_MODEL_ORDER if len(by_model.get(model_id, [])) == 1]


def _select_default_transfer_combo(
    combos: list[dict[str, object]],
) -> dict[str, object]:
    candidates = [
        dict(combo)
        for combo in combos
        if str(combo.get("target_family", "")) == FULL_TRANSFER_TARGET_FAMILY
        and str(combo.get("feature_set_id", "")) == FULL_TRANSFER_FEATURE_SET_ID
        and str(combo.get("model_id", "")) == TRANSFER_DEFAULT_MODEL_ID
        and str(combo.get("output_mode", "")) == TRANSFER_DEFAULT_OUTPUT_MODE
        and str(combo.get("task_kind", "")) == TRANSFER_DEFAULT_TASK_KIND
    ]
    if not candidates:
        raise RuntimeError(
            "Could not resolve transfer default combo. Expected one saved combo matching: "
            f"target_family={FULL_TRANSFER_TARGET_FAMILY}, "
            f"feature_set_id={FULL_TRANSFER_FEATURE_SET_ID}, "
            f"model_id={TRANSFER_DEFAULT_MODEL_ID}, "
            f"output_mode={TRANSFER_DEFAULT_OUTPUT_MODE}, "
            f"task_kind={TRANSFER_DEFAULT_TASK_KIND}."
        )
    candidates.sort(key=lambda item: str(item.get("combo_id", "")))
    return candidates[-1]


def _validate_combination_transfer_combo_refs(
    combos: list[dict[str, object]],
) -> dict[str, object]:
    if not combos:
        raise RuntimeError("combination_transfer_report requires exactly one selected combo ref.")
    if len(combos) != 1:
        raise RuntimeError(
            "combination_transfer_report requires exactly one selected combo ref. "
            f"Received {len(combos)}."
        )
    combo = dict(combos[0])
    family_id = str(combo.get("target_family", ""))
    task_kind = str(combo.get("task_kind", ""))
    feature_set_id = str(combo.get("feature_set_id", ""))
    if family_id != COMBINATION_TRANSFER_TARGET_FAMILY:
        raise RuntimeError(
            "combination_transfer_report requires target_family=v25_policies for the selected combo."
        )
    if task_kind != "regression":
        raise RuntimeError("combination_transfer_report supports regression reasoning combos only.")
    if feature_set_id not in set(COMBINATION_TRANSFER_ALLOWED_FEATURE_SET_IDS):
        raise RuntimeError(
            "Selected combo feature_set_id is not allowed for combination_transfer_report. "
            f"Allowed: {COMBINATION_TRANSFER_ALLOWED_FEATURE_SET_IDS}"
        )
    return combo


def _assert_no_policy_target_columns(
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    leaked = [
        str(column)
        for column in frame.columns
        if str(column).lower().startswith("v25_") or str(column).lower().startswith("taste_")
    ]
    if leaked:
        raise RuntimeError(
            f"Leakage guard failed for {label}: base success features contain policy target columns {leaked[:8]}."
        )


def _metric_values(
    metrics_list: list[dict[str, float]],
    key: str,
) -> np.ndarray:
    if not metrics_list:
        return np.asarray([], dtype=float)
    return np.asarray([float(metrics[key]) for metrics in metrics_list], dtype=float)


def _metric_mean_std(
    metrics_list: list[dict[str, float]],
    key: str,
) -> tuple[float, float]:
    values = _metric_values(metrics_list, key)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _aggregate_train_threshold_sweep(
    *,
    y_true: np.ndarray,
    scores_by_repeat: list[np.ndarray],
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
) -> list[dict[str, float]]:
    if not scores_by_repeat:
        return []
    by_threshold: dict[float, dict[str, list[float]]] = {}
    for scores in scores_by_repeat:
        sweep_rows = sweep_threshold_grid(
            y_true,
            np.asarray(scores, dtype=float),
            start=threshold_start,
            stop=threshold_stop,
            step=threshold_step,
        )
        for row in sweep_rows:
            threshold = float(row["threshold"])
            bucket = by_threshold.setdefault(
                threshold,
                {"f0_5": [], "precision": [], "recall": [], "roc_auc": [], "pr_auc": []},
            )
            bucket["f0_5"].append(float(row["f0_5"]))
            bucket["precision"].append(float(row["precision"]))
            bucket["recall"].append(float(row["recall"]))
            bucket["roc_auc"].append(float(row["roc_auc"]))
            bucket["pr_auc"].append(float(row["pr_auc"]))
    rows: list[dict[str, float]] = []
    for threshold in sorted(by_threshold):
        bucket = by_threshold[threshold]
        rows.append(
            {
                "threshold": float(threshold),
                "train_f0_5": float(np.mean(bucket["f0_5"])),
                "train_precision": float(np.mean(bucket["precision"])),
                "train_recall": float(np.mean(bucket["recall"])),
                "train_roc_auc": float(np.mean(bucket["roc_auc"])),
                "train_pr_auc": float(np.mean(bucket["pr_auc"])),
            }
        )
    return rows


def _build_combination_source_cv_summary(
    *,
    combo: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_target = _source_cv_per_target_metrics_for_combo(combo).copy()
    if per_target.empty:
        raise RuntimeError("No source CV per-target metrics found for selected combination-transfer combo.")
    summary = pd.DataFrame(
        [
            {
                "combo_id": str(combo["combo_id"]),
                "target_family": str(combo["target_family"]),
                "feature_set_id": str(combo["feature_set_id"]),
                "model_id": str(combo["model_id"]),
                "output_mode": str(combo["output_mode"]),
                "r2_mean": float(per_target["r2"].mean()),
                "r2_std": float(np.std(per_target["r2"].to_numpy(dtype=float), ddof=0)),
                "rmse_mean": float(per_target["rmse"].mean()),
                "mae_mean": float(per_target["mae"].mean()),
            }
        ]
    )
    return per_target, summary


def _evaluate_combination_reasoning_transfer(
    *,
    combo: dict[str, object],
    target_family: LoadedTargetFamily,
    feature_set: object,
    pred_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    founder_ids = feature_set.private_frame["founder_uuid"].astype(str)
    aligned_test = _align_by_founder_uuid(
        founder_ids,
        target_family.test_frame,
        frame_label=f"target family '{target_family.family_id}' test frame",
    )
    if len(pred_test) != len(aligned_test):
        raise RuntimeError(
            "Reasoning prediction row count mismatch in combination_transfer_report: "
            f"predictions={len(pred_test)}, targets={len(aligned_test)}."
        )
    rows: list[dict[str, object]] = []
    for target_id in target_family.target_columns:
        y_true = aligned_test[target_id].to_numpy(dtype=float)
        y_pred = pred_test[target_id].to_numpy(dtype=float)
        metrics = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "combo_id": str(combo["combo_id"]),
                "feature_set_id": str(combo["feature_set_id"]),
                "model_id": str(combo["model_id"]),
                "output_mode": str(combo["output_mode"]),
                "target_id": str(target_id),
                **metrics,
            }
        )
    per_target = pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "combo_id": str(combo["combo_id"]),
                "feature_set_id": str(combo["feature_set_id"]),
                "model_id": str(combo["model_id"]),
                "output_mode": str(combo["output_mode"]),
                "r2_mean": float(per_target["r2"].mean()),
                "r2_std": float(np.std(per_target["r2"].to_numpy(dtype=float), ddof=0)),
                "rmse_mean": float(per_target["rmse"].mean()),
                "mae_mean": float(per_target["mae"].mean()),
            }
        ]
    )
    return per_target, summary


def _evaluate_combination_success_transfer(
    *,
    config: ExperimentConfig,
    combo: dict[str, object],
    pred_train: pd.DataFrame,
    pred_test: pd.DataFrame,
    repository_splits: LoadedFeatureRepositorySplits,
    repository_banks: dict[str, object],
    hq_exit_override_mode: str,
    evaluate_test: bool,
    selected_success_branch_ids: set[str] | None = None,
    success_model_variants: list[str] | tuple[str, ...] | None = None,
    use_nested_success: bool = True,
    repeat_cv_with_new_seeds: bool = False,
    cv_seed_repeat_count: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    success_model_variants_use = _resolve_success_model_variants(success_model_variants)
    selected_variant_set = set(success_model_variants_use)
    required_banks = {"hq_baseline", "llm_engineering", "lambda_policies"}
    missing = sorted(bank_id for bank_id in required_banks if bank_id not in repository_banks)
    if missing:
        raise RuntimeError(
            "combination_transfer_report requires repository feature banks "
            f"{sorted(required_banks)}. Missing: {missing}"
        )

    train_ids_all = [str(founder_id) for founder_id in repository_splits.train_ids]
    test_ids = [str(founder_id) for founder_id in repository_splits.test_ids]

    pred_train_use = pred_train.copy()
    pred_test_use = pred_test.copy()
    pred_train_use.index = pred_train_use.index.astype(str)
    pred_test_use.index = pred_test_use.index.astype(str)

    # Predicted reasoning can be available for a strict subset of train founders
    # (for example when llm_engineering is part of the selected reasoning combo).
    train_ids = [founder_id for founder_id in train_ids_all if founder_id in set(pred_train_use.index)]
    if not train_ids:
        raise RuntimeError(
            "combination_transfer_report has zero overlapping train founders between "
            "repository splits and selected reasoning predictions."
        )
    train_labels = (
        repository_splits.train_labels.set_index("founder_uuid")
        .reindex(train_ids)["success"]
        .astype(int)
        .to_numpy(dtype=int)
    )

    pred_train_use = pred_train_use.reindex(train_ids)
    if evaluate_test:
        missing_test_ids = [founder_id for founder_id in test_ids if founder_id not in set(pred_test_use.index)]
        if missing_test_ids:
            raise RuntimeError(
                "combination_transfer_report missing held-out prediction rows for "
                f"{len(missing_test_ids)} test founders. Examples: {missing_test_ids[:5]}"
            )
        pred_test_use = pred_test_use.reindex(test_ids)
        test_labels = (
            repository_splits.test_labels.set_index("founder_uuid")
            .reindex(test_ids)["success"]
            .astype(int)
            .to_numpy(dtype=int)
        )
    else:
        pred_test_use = pred_test_use.iloc[0:0]
        test_labels = np.asarray([], dtype=int)

    prepared_banks: dict[str, tuple[pd.DataFrame, pd.DataFrame, set[str]]] = {}
    for bank_id in required_banks:
        bank = repository_banks[bank_id]
        train_frame = (
            bank.public_frame.set_index("founder_uuid")
            .reindex(train_ids)
            .drop(columns=["founder_uuid"], errors="ignore")
        )
        if evaluate_test:
            test_frame = (
                bank.private_frame.set_index("founder_uuid")
                .reindex(test_ids)
                .drop(columns=["founder_uuid"], errors="ignore")
            )
        else:
            test_frame = pd.DataFrame(index=range(0))
        _assert_no_policy_target_columns(train_frame, label=f"{bank_id} train base bank")
        if evaluate_test:
            _assert_no_policy_target_columns(test_frame, label=f"{bank_id} test base bank")
        prepared_banks[bank_id] = (
            train_frame,
            test_frame,
            set(getattr(bank, "binary_feature_columns", [])),
        )

    hq_train_frame = prepared_banks["hq_baseline"][0]
    hq_test_frame = prepared_banks["hq_baseline"][1]
    hq_exit_train = (
        hq_train_frame["exit_count"].to_numpy(dtype=float)
        if "exit_count" in hq_train_frame.columns
        else None
    )
    hq_exit_test = (
        hq_test_frame["exit_count"].to_numpy(dtype=float)
        if "exit_count" in hq_test_frame.columns
        else None
    )

    repeat_count = int(cv_seed_repeat_count) if repeat_cv_with_new_seeds else 1
    if repeat_count < 1:
        repeat_count = 1
    success_lr_fixed_c = (
        None
        if use_nested_success
        else float(default_l2_c(config.reproduction.logistic_c_grid))
    )

    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for base_combo_id, bank_ids in COMBINATION_TRANSFER_BASE_SUCCESS_COMBOS:
        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        binary_columns: list[str] = []
        for bank_id in bank_ids:
            base_train, base_test, bank_binary = prepared_banks[bank_id]
            train_parts.append(base_train.reset_index(drop=True))
            test_parts.append(base_test.reset_index(drop=True))
            binary_columns.extend(list(bank_binary))
        train_parts.append(pred_train_use.reset_index(drop=True))
        if evaluate_test:
            test_parts.append(pred_test_use.reset_index(drop=True))
        train_features = pd.concat(train_parts, axis=1)
        test_features = pd.concat(test_parts, axis=1) if evaluate_test else pd.DataFrame(index=range(0))

        include_hq = "hq_baseline" in set(bank_ids)
        branch_variants = _resolve_success_override_variants(
            hq_exit_override_mode=hq_exit_override_mode,
            default_use_override=include_hq,
        )
        y_train_use = np.asarray(train_labels, dtype=int)
        train_exit_base = np.asarray(hq_exit_train, dtype=float) if hq_exit_train is not None else None
        if train_features.isna().any(axis=1).any():
            valid_mask = ~train_features.isna().any(axis=1)
            train_features = train_features.loc[valid_mask].reset_index(drop=True)
            y_train_use = y_train_use[valid_mask.to_numpy()]
            if train_exit_base is not None:
                train_exit_base = train_exit_base[valid_mask.to_numpy()]
        if len(train_features) == 0:
            raise RuntimeError(
                f"combination_transfer_report branch '{base_combo_id}' has zero train rows after alignment."
            )
        if evaluate_test and test_features.isna().any(axis=1).any():
            raise RuntimeError(
                f"combination_transfer_report branch '{base_combo_id}' has NaNs on held-out test features."
            )
        for branch_label, use_override in branch_variants:
            success_branch_id = f"{base_combo_id}__{branch_label}"
            if selected_success_branch_ids is not None and success_branch_id not in selected_success_branch_ids:
                continue

            train_exit_use = np.asarray(train_exit_base, dtype=float) if train_exit_base is not None else None
            test_exit_use = (
                np.asarray(hq_exit_test, dtype=float)
                if (use_override and hq_exit_test is not None and evaluate_test)
                else None
            )
            X_train = train_features.to_numpy(dtype=float)
            X_test = test_features.to_numpy(dtype=float) if evaluate_test else None
            column_names = list(train_features.columns)
            cont_idx = continuous_indices(column_names, binary_columns)

            common_row: dict[str, object] = {
                "combo_id": str(combo["combo_id"]),
                "selected_feature_set_id": str(combo["feature_set_id"]),
                "selected_model_id": str(combo["model_id"]),
                "selected_output_mode": str(combo["output_mode"]),
                "base_combo_id": str(base_combo_id),
                "base_bank_ids": ",".join(bank_ids),
                "hq_exit_override_branch": branch_label,
                "success_branch_id": success_branch_id,
                "hq_exit_override_applied": bool(use_override),
                "repeat_cv_with_new_seeds": bool(repeat_cv_with_new_seeds),
                "cv_seed_repeat_count": int(repeat_count),
                "success_lr_nested_cv": bool(use_nested_success),
                "success_lr_fixed_c": (
                    float(success_lr_fixed_c)
                    if success_lr_fixed_c is not None
                    else float("nan")
                ),
            }

            # Variant 1: existing single-model protocol.
            train_f05_values: list[float] = []
            train_roc_values: list[float] = []
            train_pr_values: list[float] = []
            train_precision_values: list[float] = []
            train_recall_values: list[float] = []
            train_oof_scores: list[np.ndarray] = []
            selected_c_values: list[float] = []
            threshold_values: list[float] = []
            test_f05_values: list[float] = []
            test_roc_values: list[float] = []
            test_pr_values: list[float] = []
            test_precision_values: list[float] = []
            test_recall_values: list[float] = []
            for repeat_index in range(repeat_count):
                seed_offset = repeat_index * 10_000
                outer_random_state = int(config.reproduction.outer_cv.random_state) + seed_offset
                inner_random_state = int(config.reproduction.inner_cv.random_state) + seed_offset
                if evaluate_test:
                    protocol = run_nested_l2_success_protocol(
                        X_train=X_train,
                        y_train=y_train_use,
                        X_test=X_test if X_test is not None else np.asarray([], dtype=float),
                        y_test=test_labels,
                        continuous_indices=cont_idx,
                        outer_n_splits=config.reproduction.outer_cv.n_splits,
                        outer_shuffle=config.reproduction.outer_cv.shuffle,
                        outer_random_state=outer_random_state,
                        inner_n_splits=config.reproduction.inner_cv.n_splits,
                        inner_shuffle=config.reproduction.inner_cv.shuffle,
                        inner_random_state=inner_random_state,
                        c_grid=config.reproduction.logistic_c_grid,
                        threshold_start=config.reproduction.threshold_grid.start,
                        threshold_stop=config.reproduction.threshold_grid.stop,
                        threshold_step=config.reproduction.threshold_grid.step,
                        use_nested=use_nested_success,
                        fixed_c_value=success_lr_fixed_c,
                        use_exit_override=use_override,
                        train_exit_counts=train_exit_use,
                        test_exit_counts=test_exit_use,
                    )
                else:
                    protocol = run_nested_l2_success_cv_only(
                        X_train=X_train,
                        y_train=y_train_use,
                        continuous_indices=cont_idx,
                        outer_n_splits=config.reproduction.outer_cv.n_splits,
                        outer_shuffle=config.reproduction.outer_cv.shuffle,
                        outer_random_state=outer_random_state,
                        inner_n_splits=config.reproduction.inner_cv.n_splits,
                        inner_shuffle=config.reproduction.inner_cv.shuffle,
                        inner_random_state=inner_random_state,
                        c_grid=config.reproduction.logistic_c_grid,
                        threshold_start=config.reproduction.threshold_grid.start,
                        threshold_stop=config.reproduction.threshold_grid.stop,
                        threshold_step=config.reproduction.threshold_grid.step,
                        use_nested=use_nested_success,
                        fixed_c_value=success_lr_fixed_c,
                        use_exit_override=use_override,
                        train_exit_counts=train_exit_use,
                    )
                cv_metrics = dict(protocol["cv_metrics"])
                train_f05_values.append(float(cv_metrics["f0_5"]))
                train_roc_values.append(float(cv_metrics["roc_auc"]))
                train_pr_values.append(float(cv_metrics["pr_auc"]))
                train_precision_values.append(float(cv_metrics["precision"]))
                train_recall_values.append(float(cv_metrics["recall"]))
                train_oof_scores.append(np.asarray(protocol["oof_scores"], dtype=float))
                selected_c_values.append(float(protocol["selected_c_final"]))
                threshold_values.append(float(protocol["threshold"]))
                if evaluate_test:
                    test_metrics = dict(protocol["test_metrics"])
                    test_f05_values.append(float(test_metrics["f0_5"]))
                    test_roc_values.append(float(test_metrics["roc_auc"]))
                    test_pr_values.append(float(test_metrics["pr_auc"]))
                    test_precision_values.append(float(test_metrics["precision"]))
                    test_recall_values.append(float(test_metrics["recall"]))

            single_row: dict[str, object] = {
                **common_row,
                "model_variant": "single_model",
                "train_cv_f0_5": float(np.mean(train_f05_values)),
                "train_cv_f0_5_std": float(np.std(train_f05_values, ddof=0)),
                "train_cv_roc_auc": float(np.mean(train_roc_values)),
                "train_cv_roc_auc_std": float(np.std(train_roc_values, ddof=0)),
                "train_cv_pr_auc": float(np.mean(train_pr_values)),
                "train_cv_pr_auc_std": float(np.std(train_pr_values, ddof=0)),
                "train_cv_precision": float(np.mean(train_precision_values)),
                "train_cv_precision_std": float(np.std(train_precision_values, ddof=0)),
                "train_cv_recall": float(np.mean(train_recall_values)),
                "train_cv_recall_std": float(np.std(train_recall_values, ddof=0)),
                "selected_c_final": float(np.mean(selected_c_values)),
                "selected_c_final_std": float(np.std(selected_c_values, ddof=0)),
                "threshold": float(np.mean(threshold_values)),
                "threshold_std": float(np.std(threshold_values, ddof=0)),
            }
            if evaluate_test:
                single_row.update(
                    {
                        "test_f0_5": float(np.mean(test_f05_values)),
                        "test_f0_5_std": float(np.std(test_f05_values, ddof=0)),
                        "test_roc_auc": float(np.mean(test_roc_values)),
                        "test_roc_auc_std": float(np.std(test_roc_values, ddof=0)),
                        "test_pr_auc": float(np.mean(test_pr_values)),
                        "test_pr_auc_std": float(np.std(test_pr_values, ddof=0)),
                        "test_precision": float(np.mean(test_precision_values)),
                        "test_precision_std": float(np.std(test_precision_values, ddof=0)),
                        "test_recall": float(np.mean(test_recall_values)),
                        "test_recall_std": float(np.std(test_recall_values, ddof=0)),
                    }
                )
            rows.append(single_row)

            single_sweep_rows = _aggregate_train_threshold_sweep(
                y_true=y_train_use,
                scores_by_repeat=train_oof_scores,
                threshold_start=config.reproduction.threshold_grid.start,
                threshold_stop=config.reproduction.threshold_grid.stop,
                threshold_step=config.reproduction.threshold_grid.step,
            )
            for sweep_row in single_sweep_rows:
                threshold_rows.append(
                    {
                        **common_row,
                        "model_variant": "single_model",
                        "threshold": float(sweep_row["threshold"]),
                        "train_f0_5": float(sweep_row["train_f0_5"]),
                        "train_precision": float(sweep_row["train_precision"]),
                        "train_recall": float(sweep_row["train_recall"]),
                        "train_roc_auc": float(sweep_row["train_roc_auc"]),
                        "train_pr_auc": float(sweep_row["train_pr_auc"]),
                        "selected_threshold": float(single_row["threshold"]),
                        "selected_train_f0_5": float(single_row["train_cv_f0_5"]),
                        "selected_test_f0_5": float(single_row.get("test_f0_5", float("nan"))),
                        "selected_test_roc_auc": float(single_row.get("test_roc_auc", float("nan"))),
                        "selected_test_pr_auc": float(single_row.get("test_pr_auc", float("nan"))),
                        "selected_test_precision": float(single_row.get("test_precision", float("nan"))),
                        "selected_test_recall": float(single_row.get("test_recall", float("nan"))),
                    }
                )

            # Variants 2/3: soft-avg ensembles over all outer-fold voters.
            if {"soft_avg_model", "soft_avg_weighted_model"} & selected_variant_set:
                soft_ensemble = run_nested_l2_soft_ensemble_success_protocol(
                    X_train=X_train,
                    y_train=y_train_use,
                    X_test=X_test,
                    y_test=test_labels if evaluate_test else None,
                    continuous_indices=cont_idx,
                    outer_n_splits=config.reproduction.outer_cv.n_splits,
                    outer_shuffle=config.reproduction.outer_cv.shuffle,
                    outer_random_state=config.reproduction.outer_cv.random_state,
                    inner_n_splits=config.reproduction.inner_cv.n_splits,
                    inner_shuffle=config.reproduction.inner_cv.shuffle,
                    inner_random_state=config.reproduction.inner_cv.random_state,
                    c_grid=config.reproduction.logistic_c_grid,
                    use_nested=use_nested_success,
                    fixed_c_value=success_lr_fixed_c,
                    use_exit_override=use_override,
                    train_exit_counts=train_exit_use,
                    test_exit_counts=test_exit_use,
                    repeat_count=repeat_count,
                    threshold_start=config.reproduction.threshold_grid.start,
                    threshold_stop=config.reproduction.threshold_grid.stop,
                    threshold_step=config.reproduction.threshold_grid.step,
                )
            else:
                soft_ensemble = None
            for model_variant in ("soft_avg_model", "soft_avg_weighted_model"):
                if model_variant not in selected_variant_set:
                    continue
                if soft_ensemble is None:
                    raise RuntimeError("Soft success variant requested but ensemble results were not built.")
                variant = dict(soft_ensemble["variants"][model_variant])
                train_metrics_repeat = list(variant["repeat_train_metrics"])
                train_cv_f0_5, train_cv_f0_5_std = _metric_mean_std(train_metrics_repeat, "f0_5")
                train_cv_roc_auc, train_cv_roc_auc_std = _metric_mean_std(train_metrics_repeat, "roc_auc")
                train_cv_pr_auc, train_cv_pr_auc_std = _metric_mean_std(train_metrics_repeat, "pr_auc")
                train_cv_precision, train_cv_precision_std = _metric_mean_std(train_metrics_repeat, "precision")
                train_cv_recall, train_cv_recall_std = _metric_mean_std(train_metrics_repeat, "recall")
                variant_row: dict[str, object] = {
                    **common_row,
                    "model_variant": model_variant,
                    "train_cv_f0_5": train_cv_f0_5,
                    "train_cv_f0_5_std": train_cv_f0_5_std,
                    "train_cv_roc_auc": train_cv_roc_auc,
                    "train_cv_roc_auc_std": train_cv_roc_auc_std,
                    "train_cv_pr_auc": train_cv_pr_auc,
                    "train_cv_pr_auc_std": train_cv_pr_auc_std,
                    "train_cv_precision": train_cv_precision,
                    "train_cv_precision_std": train_cv_precision_std,
                    "train_cv_recall": train_cv_recall,
                    "train_cv_recall_std": train_cv_recall_std,
                    "selected_c_final": (
                        float(soft_ensemble["selected_c_final"])
                        if soft_ensemble["selected_c_final"] is not None
                        else float("nan")
                    ),
                    "selected_c_final_std": float("nan"),
                    "threshold": float(variant["threshold"]),
                    "threshold_std": 0.0,
                    "voter_count": int(soft_ensemble["voter_count"]),
                }
                if evaluate_test:
                    variant_test_metrics = dict(variant["test_metrics"] or {})
                    variant_test_repeat_metrics = list(variant["repeat_test_metrics"])
                    variant_row.update(
                        {
                            "test_f0_5": float(variant_test_metrics.get("f0_5", float("nan"))),
                            "test_f0_5_std": _metric_mean_std(variant_test_repeat_metrics, "f0_5")[1],
                            "test_roc_auc": float(variant_test_metrics.get("roc_auc", float("nan"))),
                            "test_roc_auc_std": _metric_mean_std(variant_test_repeat_metrics, "roc_auc")[1],
                            "test_pr_auc": float(variant_test_metrics.get("pr_auc", float("nan"))),
                            "test_pr_auc_std": _metric_mean_std(variant_test_repeat_metrics, "pr_auc")[1],
                            "test_precision": float(variant_test_metrics.get("precision", float("nan"))),
                            "test_precision_std": _metric_mean_std(variant_test_repeat_metrics, "precision")[1],
                            "test_recall": float(variant_test_metrics.get("recall", float("nan"))),
                            "test_recall_std": _metric_mean_std(variant_test_repeat_metrics, "recall")[1],
                        }
                    )
                rows.append(variant_row)

                for sweep_row in list(variant["threshold_sweep"]):
                    threshold_rows.append(
                        {
                            **common_row,
                            "model_variant": model_variant,
                            "threshold": float(sweep_row["threshold"]),
                            "train_f0_5": float(sweep_row["f0_5"]),
                            "train_precision": float(sweep_row["precision"]),
                            "train_recall": float(sweep_row["recall"]),
                            "train_roc_auc": float(sweep_row["roc_auc"]),
                            "train_pr_auc": float(sweep_row["pr_auc"]),
                            "selected_threshold": float(variant["threshold"]),
                            "selected_train_f0_5": float(variant["selected_train_f0_5"]),
                            "selected_test_f0_5": float(variant_row.get("test_f0_5", float("nan"))),
                            "selected_test_roc_auc": float(variant_row.get("test_roc_auc", float("nan"))),
                            "selected_test_pr_auc": float(variant_row.get("test_pr_auc", float("nan"))),
                            "selected_test_precision": float(variant_row.get("test_precision", float("nan"))),
                            "selected_test_recall": float(variant_row.get("test_recall", float("nan"))),
                        }
                    )

    metrics = pd.DataFrame(rows)
    threshold_sweep = pd.DataFrame(threshold_rows)
    if metrics.empty:
        return metrics, threshold_sweep
    metrics = metrics[metrics["model_variant"].astype(str).isin(selected_variant_set)].reset_index(drop=True)
    if not threshold_sweep.empty:
        threshold_sweep = threshold_sweep[
            threshold_sweep["model_variant"].astype(str).isin(selected_variant_set)
        ].reset_index(drop=True)
    if metrics.empty:
        return metrics, threshold_sweep

    metrics = metrics.assign(
        _variant_order=metrics["model_variant"].astype(str).map(SUCCESS_MODEL_VARIANT_ORDER).fillna(999).astype(int)
    )
    if evaluate_test:
        metrics = metrics.sort_values(
            ["success_branch_id", "hq_exit_override_branch", "_variant_order", "test_f0_5", "test_roc_auc"],
            ascending=[True, True, True, False, False],
        )
    else:
        metrics = metrics.sort_values(
            ["success_branch_id", "hq_exit_override_branch", "_variant_order", "train_cv_f0_5", "train_cv_roc_auc"],
            ascending=[True, True, True, False, False],
        )
    metrics = metrics.drop(columns=["_variant_order"]).reset_index(drop=True)

    if not threshold_sweep.empty:
        threshold_sweep = threshold_sweep.assign(
            _variant_order=(
                threshold_sweep["model_variant"]
                .astype(str)
                .map(SUCCESS_MODEL_VARIANT_ORDER)
                .fillna(999)
                .astype(int)
            )
        ).sort_values(
            ["success_branch_id", "hq_exit_override_branch", "_variant_order", "threshold"],
            ascending=[True, True, True, True],
        )
        threshold_sweep = threshold_sweep.drop(columns=["_variant_order"]).reset_index(drop=True)

    return metrics, threshold_sweep


def _render_combination_transfer_report(
    *,
    run_dir: Path,
    combo_ref_used: str,
    source_cv_summary: pd.DataFrame,
    source_cv_per_target: pd.DataFrame,
    reasoning_summary: pd.DataFrame,
    reasoning_per_target: pd.DataFrame,
    success_metrics: pd.DataFrame,
    success_threshold_sweep: pd.DataFrame,
) -> str:
    lines: list[str] = [
        "# Combination Transfer Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Selected combo ref: `{combo_ref_used}`",
        f"- Target family: `{COMBINATION_TRANSFER_TARGET_FAMILY}`",
        "",
        "## Source CV Validation (Selected Combo)",
        "",
        "These metrics come from the source Stage-A model-testing run (train-only CV).",
        "",
        "| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in source_cv_summary.itertuples(index=False):
        lines.append(
            f"| {row.combo_id} | {row.feature_set_id} | {row.model_id} | {row.output_mode} | "
            f"{row.r2_mean:.4f} | {row.r2_std:.4f} | {row.rmse_mean:.4f} | {row.mae_mean:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Held-out Test Reasoning Agreement",
            "",
            "| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in reasoning_summary.itertuples(index=False):
        lines.append(
            f"| {row.combo_id} | {row.feature_set_id} | {row.model_id} | {row.output_mode} | "
            f"{row.r2_mean:.4f} | {row.r2_std:.4f} | {row.rmse_mean:.4f} | {row.mae_mean:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Success Transfer (Predicted Reasoning Appended)",
            "",
            "| success_branch_id | model_variant | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | test_f0_5 | train_cv_roc_auc | test_roc_auc | train_cv_pr_auc | test_pr_auc | selected_c_final | threshold |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    success_rows = success_metrics.assign(
        _variant_order=(
            success_metrics["model_variant"].astype(str).map(SUCCESS_MODEL_VARIANT_ORDER).fillna(999).astype(int)
        )
    ).sort_values(
        ["success_branch_id", "hq_exit_override_branch", "_variant_order", "test_f0_5", "test_roc_auc"],
        ascending=[True, True, True, False, False],
    )
    for row in success_rows.itertuples(index=False):
        lines.append(
            f"| {row.success_branch_id} | {row.model_variant} | {row.base_combo_id} | {row.hq_exit_override_branch} | "
            f"{row.train_cv_f0_5:.4f} | {row.test_f0_5:.4f} | "
            f"{row.train_cv_roc_auc:.4f} | {row.test_roc_auc:.4f} | "
            f"{row.train_cv_pr_auc:.4f} | {row.test_pr_auc:.4f} | {row.selected_c_final:.4f} | {row.threshold:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Train Threshold Sweep (F0.5)",
            "",
            "Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in "
            "`combination_transfer_success_train_threshold_sweep.csv`.",
            "",
            "| success_branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if success_threshold_sweep.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        sweep_rows = success_threshold_sweep.assign(
            _variant_order=(
                success_threshold_sweep["model_variant"]
                .astype(str)
                .map(SUCCESS_MODEL_VARIANT_ORDER)
                .fillna(999)
                .astype(int)
            )
        ).sort_values(
            [
                "success_branch_id",
                "hq_exit_override_branch",
                "_variant_order",
                "train_f0_5",
                "threshold",
            ],
            ascending=[True, True, True, False, True],
        )
        for _, group in sweep_rows.groupby(
            ["success_branch_id", "hq_exit_override_branch", "model_variant"], sort=True
        ):
            for row in group.head(3).itertuples(index=False):
                lines.append(
                    f"| {row.success_branch_id} | {row.model_variant} | {row.hq_exit_override_branch} | "
                    f"{row.threshold:.4f} | {row.train_f0_5:.4f} | {row.train_precision:.4f} | {row.train_recall:.4f} | "
                    f"{row.selected_threshold:.4f} | {row.selected_train_f0_5:.4f} | {row.selected_test_f0_5:.4f} |"
                )

    lines.extend(
        [
            "",
            "## Detailed Data Tables (CSV Artifacts)",
            "",
            "- `combination_transfer_source_cv_per_target.csv`",
            "- `combination_transfer_source_cv_summary.csv`",
            "- `combination_transfer_reasoning_per_target.csv`",
            "- `combination_transfer_reasoning_summary.csv`",
            "- `combination_transfer_success_metrics.csv`",
            "- `combination_transfer_success_train_threshold_sweep.csv`",
            "",
            "## Per-Target Source CV Metrics",
            "",
            "| target_id | r2 | rmse | mae |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in source_cv_per_target.sort_values("target_id").itertuples(index=False):
        lines.append(f"| {row.target_id} | {row.r2:.4f} | {row.rmse:.4f} | {row.mae:.4f} |")

    lines.extend(
        [
            "",
            "## Per-Target Held-out Test Reasoning Metrics",
            "",
            "| target_id | r2 | rmse | mae |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in reasoning_per_target.sort_values("target_id").itertuples(index=False):
        lines.append(f"| {row.target_id} | {row.r2:.4f} | {row.rmse:.4f} | {row.mae:.4f} |")
    return "\n".join(lines) + "\n"


def _render_combination_success_cv_report(
    *,
    run_dir: Path,
    combo_ref_used: str,
    source_cv_summary: pd.DataFrame,
    success_metrics: pd.DataFrame,
    success_threshold_sweep: pd.DataFrame,
) -> str:
    repeat_count = int(success_metrics["cv_seed_repeat_count"].max()) if not success_metrics.empty else 1
    repeat_enabled = bool(success_metrics["repeat_cv_with_new_seeds"].max()) if not success_metrics.empty else False
    success_lr_nested_cv = (
        bool(success_metrics["success_lr_nested_cv"].iloc[0])
        if "success_lr_nested_cv" in success_metrics.columns and not success_metrics.empty
        else True
    )
    if "success_lr_fixed_c" in success_metrics.columns and not success_metrics.empty:
        fixed_c_values = (
            pd.to_numeric(success_metrics["success_lr_fixed_c"], errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )
        fixed_c_text = ", ".join(f"{float(value):.4g}" for value in sorted(fixed_c_values)) or "n/a"
    else:
        fixed_c_text = "n/a"
    lines: list[str] = [
        "# Combination Success Screening Report (Train CV Only)",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Selected combo ref: `{combo_ref_used}`",
        f"- Target family: `{COMBINATION_TRANSFER_TARGET_FAMILY}`",
        "- Held-out evaluation: disabled",
        f"- Repeat CV with new seeds: {repeat_enabled} (n_runs={repeat_count})",
        f"- Success LR nested C CV: {success_lr_nested_cv}",
        f"- Success LR fixed C when nested CV disabled: `{fixed_c_text}`",
        "",
        "## Source CV Validation (Selected Combo)",
        "",
        "| combo_id | feature_set_id | model_id | output_mode | r2_mean | r2_std | rmse_mean | mae_mean |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in source_cv_summary.itertuples(index=False):
        lines.append(
            f"| {row.combo_id} | {row.feature_set_id} | {row.model_id} | {row.output_mode} | "
            f"{row.r2_mean:.4f} | {row.r2_std:.4f} | {row.rmse_mean:.4f} | {row.mae_mean:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Success Screening (L2 Logistic Regression, CV only)",
            "",
            "| rank | success_branch_id | model_variant | base_combo_id | hq_exit_override_branch | train_cv_f0_5 | train_cv_f0_5_std | train_cv_roc_auc | train_cv_pr_auc | selected_c_final | threshold |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if success_metrics.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        success_rows = success_metrics.assign(
            _variant_order=(
                success_metrics["model_variant"]
                .astype(str)
                .map(SUCCESS_MODEL_VARIANT_ORDER)
                .fillna(999)
                .astype(int)
            )
        ).sort_values(
            [
                "train_cv_f0_5",
                "train_cv_roc_auc",
                "train_cv_pr_auc",
                "success_branch_id",
                "_variant_order",
            ],
            ascending=[False, False, False, True, True],
        )
        for rank, row in enumerate(success_rows.itertuples(index=False), start=1):
            lines.append(
                f"| {rank} | {row.success_branch_id} | {row.model_variant} | {row.base_combo_id} | {row.hq_exit_override_branch} | "
                f"{row.train_cv_f0_5:.4f} | {row.train_cv_f0_5_std:.4f} | {row.train_cv_roc_auc:.4f} | {row.train_cv_pr_auc:.4f} | "
                f"{row.selected_c_final:.4f} | {row.threshold:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Train Threshold Sweep (F0.5)",
            "",
            "Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in "
            "`combination_success_cv_train_threshold_sweep.csv`.",
            "",
            "| success_branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if success_threshold_sweep.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        sweep_rows = success_threshold_sweep.assign(
            _variant_order=(
                success_threshold_sweep["model_variant"]
                .astype(str)
                .map(SUCCESS_MODEL_VARIANT_ORDER)
                .fillna(999)
                .astype(int)
            )
        ).sort_values(
            ["success_branch_id", "hq_exit_override_branch", "_variant_order", "train_f0_5", "threshold"],
            ascending=[True, True, True, False, True],
        )
        for _, group in sweep_rows.groupby(
            ["success_branch_id", "hq_exit_override_branch", "model_variant"], sort=True
        ):
            for row in group.head(3).itertuples(index=False):
                lines.append(
                    f"| {row.success_branch_id} | {row.model_variant} | {row.hq_exit_override_branch} | "
                    f"{row.threshold:.4f} | {row.train_f0_5:.4f} | {row.train_precision:.4f} | {row.train_recall:.4f} | "
                    f"{row.selected_threshold:.4f} | {row.selected_train_f0_5:.4f} |"
                )
    return "\n".join(lines) + "\n"


def _write_combination_transfer_docs(
    *,
    run_dir: Path,
    report_markdown: str,
) -> None:
    docs_root = DOCS_DIR / "experiment-archive" / "generated-reports" / "testing_models"
    docs_root.mkdir(parents=True, exist_ok=True)
    write_markdown(docs_root / "combination_transfer_report_latest.md", report_markdown)
    write_markdown(docs_root / f"combination_transfer_report_{run_dir.name}.md", report_markdown)


def _write_combination_success_cv_docs(
    *,
    run_dir: Path,
    report_markdown: str,
) -> None:
    docs_root = DOCS_DIR / "experiment-archive" / "generated-reports" / "testing_models"
    docs_root.mkdir(parents=True, exist_ok=True)
    write_markdown(docs_root / "combination_success_cv_report_latest.md", report_markdown)
    write_markdown(docs_root / f"combination_success_cv_report_{run_dir.name}.md", report_markdown)


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

    model_set_ids = sorted(predictions_by_model_set.keys())
    # Only build/use combined_best when at least two model families are present.
    if len(model_set_ids) < 2:
        return predictions_by_model_set, pd.DataFrame()

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
    hq_exit_override_mode: str,
    success_model_variants: list[str] | tuple[str, ...] | None = None,
    use_nested_success: bool = True,
    repeat_cv_with_new_seeds: bool = False,
    cv_seed_repeat_count: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    success_model_variants_use = _resolve_success_model_variants(success_model_variants)
    selected_variant_set = set(success_model_variants_use)
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

    repeat_count = int(cv_seed_repeat_count) if repeat_cv_with_new_seeds else 1
    if repeat_count < 1:
        repeat_count = 1
    success_lr_fixed_c = (
        None
        if use_nested_success
        else float(default_l2_c(config.reproduction.logistic_c_grid))
    )

    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
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
            ),
            "hq_plus_pred_reasoning": (
                pd.concat([hq_train.reset_index(drop=True), pred_train.reset_index(drop=True)], axis=1),
                pd.concat([hq_test.reset_index(drop=True), pred_test.reset_index(drop=True)], axis=1),
                list(hq_binary),
                True,
            ),
            "llm_engineering_plus_pred_reasoning": (
                pd.concat([llm_train.reset_index(drop=True), pred_train.reset_index(drop=True)], axis=1),
                pd.concat([llm_test.reset_index(drop=True), pred_test.reset_index(drop=True)], axis=1),
                list(llm_binary),
                False,
            ),
        }
        for branch_id in branch_ids:
            train_features, test_features, binary_columns, default_use_override = branch_frames[branch_id]
            branch_variants = _resolve_success_override_variants(
                hq_exit_override_mode=hq_exit_override_mode,
                default_use_override=default_use_override,
            )
            for branch_label, use_override in branch_variants:
                train_features_use = train_features.copy()
                test_features_use = test_features.copy()
                y_train_use = np.asarray(train_labels, dtype=int)
                train_exit_use = np.asarray(hq_exit_train, dtype=float) if hq_exit_train is not None else None
                test_exit_use = np.asarray(hq_exit_test, dtype=float) if (use_override and hq_exit_test is not None) else None

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

                X_train = train_features_use.to_numpy(dtype=float)
                X_test = test_features_use.to_numpy(dtype=float)
                column_names = list(train_features_use.columns)
                cont_idx = continuous_indices(column_names, binary_columns)
                common = {
                    "model_set_id": model_set_id,
                    "branch_id": branch_id,
                    "hq_exit_override_branch": branch_label,
                    "repeat_cv_with_new_seeds": bool(repeat_cv_with_new_seeds),
                    "cv_seed_repeat_count": int(repeat_count),
                    "success_lr_nested_cv": bool(use_nested_success),
                    "success_lr_fixed_c": (
                        float(success_lr_fixed_c)
                        if success_lr_fixed_c is not None
                        else float("nan")
                    ),
                }

                # Variant 1: existing single model.
                train_f05_values: list[float] = []
                train_oof_scores: list[np.ndarray] = []
                selected_c_values: list[float] = []
                threshold_values: list[float] = []
                for repeat_index in range(repeat_count):
                    seed_offset = repeat_index * 10_000
                    protocol = run_nested_l2_success_protocol(
                        X_train=X_train,
                        y_train=y_train_use,
                        X_test=X_test,
                        y_test=test_labels,
                        continuous_indices=cont_idx,
                        outer_n_splits=config.reproduction.outer_cv.n_splits,
                        outer_shuffle=config.reproduction.outer_cv.shuffle,
                        outer_random_state=config.reproduction.outer_cv.random_state + seed_offset,
                        inner_n_splits=config.reproduction.inner_cv.n_splits,
                        inner_shuffle=config.reproduction.inner_cv.shuffle,
                        inner_random_state=config.reproduction.inner_cv.random_state + seed_offset,
                        c_grid=config.reproduction.logistic_c_grid,
                        threshold_start=config.reproduction.threshold_grid.start,
                        threshold_stop=config.reproduction.threshold_grid.stop,
                        threshold_step=config.reproduction.threshold_grid.step,
                        use_nested=use_nested_success,
                        fixed_c_value=success_lr_fixed_c,
                        use_exit_override=use_override,
                        train_exit_counts=train_exit_use,
                        test_exit_counts=test_exit_use,
                        random_state_offset=seed_offset,
                    )
                    train_f05_values.append(float(dict(protocol["cv_metrics"])["f0_5"]))
                    train_oof_scores.append(np.asarray(protocol["oof_scores"], dtype=float))
                    selected_c_values.append(float(protocol["selected_c_final"]))
                    threshold_values.append(float(protocol["threshold"]))

                selected_c_final = float(np.median(np.asarray(selected_c_values, dtype=float)))
                selected_threshold = float(np.median(np.asarray(threshold_values, dtype=float)))
                final_protocol = run_nested_l2_success_protocol(
                    X_train=X_train,
                    y_train=y_train_use,
                    X_test=X_test,
                    y_test=test_labels,
                    continuous_indices=cont_idx,
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
                    use_nested=False,
                    fixed_c_value=selected_c_final,
                    use_exit_override=use_override,
                    train_exit_counts=train_exit_use,
                    test_exit_counts=test_exit_use,
                )
                test_scores = np.asarray(final_protocol["test_scores"], dtype=float)
                test_metrics = binary_classification_metrics(
                    test_labels,
                    test_scores,
                    threshold=selected_threshold,
                )
                rows.append(
                    {
                        **common,
                        "model_variant": "single_model",
                        "train_cv_f0_5": float(np.mean(np.asarray(train_f05_values, dtype=float))),
                        "train_cv_f0_5_std": float(np.std(np.asarray(train_f05_values, dtype=float), ddof=0)),
                        "selected_c_final": selected_c_final,
                        "selected_c_oof_mean": float(np.mean(np.asarray(selected_c_values, dtype=float))),
                        "threshold": selected_threshold,
                        **dict(test_metrics),
                    }
                )

                single_sweep_rows = _aggregate_train_threshold_sweep(
                    y_true=y_train_use,
                    scores_by_repeat=train_oof_scores,
                    threshold_start=config.reproduction.threshold_grid.start,
                    threshold_stop=config.reproduction.threshold_grid.stop,
                    threshold_step=config.reproduction.threshold_grid.step,
                )
                for sweep_row in single_sweep_rows:
                    threshold_rows.append(
                        {
                            **common,
                            "model_variant": "single_model",
                            "threshold": float(sweep_row["threshold"]),
                            "train_f0_5": float(sweep_row["train_f0_5"]),
                            "train_precision": float(sweep_row["train_precision"]),
                            "train_recall": float(sweep_row["train_recall"]),
                            "train_roc_auc": float(sweep_row["train_roc_auc"]),
                            "train_pr_auc": float(sweep_row["train_pr_auc"]),
                            "selected_threshold": float(selected_threshold),
                            "selected_train_f0_5": float(np.mean(np.asarray(train_f05_values, dtype=float))),
                            "selected_test_f0_5": float(test_metrics.get("f0_5", float("nan"))),
                            "selected_test_roc_auc": float(test_metrics.get("roc_auc", float("nan"))),
                            "selected_test_pr_auc": float(test_metrics.get("pr_auc", float("nan"))),
                            "selected_test_precision": float(test_metrics.get("precision", float("nan"))),
                            "selected_test_recall": float(test_metrics.get("recall", float("nan"))),
                        }
                    )

                # Variants 2/3: soft-avg ensembles over all outer-fold models.
                if {"soft_avg_model", "soft_avg_weighted_model"} & selected_variant_set:
                    soft_ensemble = run_nested_l2_soft_ensemble_success_protocol(
                        X_train=X_train,
                        y_train=y_train_use,
                        X_test=X_test,
                        y_test=test_labels,
                        continuous_indices=cont_idx,
                        outer_n_splits=config.reproduction.outer_cv.n_splits,
                        outer_shuffle=config.reproduction.outer_cv.shuffle,
                        outer_random_state=config.reproduction.outer_cv.random_state,
                        inner_n_splits=config.reproduction.inner_cv.n_splits,
                    inner_shuffle=config.reproduction.inner_cv.shuffle,
                    inner_random_state=config.reproduction.inner_cv.random_state,
                    c_grid=config.reproduction.logistic_c_grid,
                    use_nested=use_nested_success,
                    fixed_c_value=success_lr_fixed_c,
                    use_exit_override=use_override,
                    train_exit_counts=train_exit_use,
                    test_exit_counts=test_exit_use,
                        repeat_count=repeat_count,
                        threshold_start=config.reproduction.threshold_grid.start,
                        threshold_stop=config.reproduction.threshold_grid.stop,
                        threshold_step=config.reproduction.threshold_grid.step,
                    )
                else:
                    soft_ensemble = None
                for model_variant in ("soft_avg_model", "soft_avg_weighted_model"):
                    if model_variant not in selected_variant_set:
                        continue
                    if soft_ensemble is None:
                        raise RuntimeError("Soft success variant requested but ensemble results were not built.")
                    variant = dict(soft_ensemble["variants"][model_variant])
                    train_repeat_metrics = list(variant["repeat_train_metrics"])
                    train_cv_f0_5, train_cv_f0_5_std = _metric_mean_std(train_repeat_metrics, "f0_5")
                    variant_test_metrics = dict(variant["test_metrics"] or {})
                    rows.append(
                        {
                            **common,
                            "model_variant": model_variant,
                            "train_cv_f0_5": train_cv_f0_5,
                            "train_cv_f0_5_std": train_cv_f0_5_std,
                            "selected_c_final": (
                                float(soft_ensemble["selected_c_final"])
                                if soft_ensemble["selected_c_final"] is not None
                                else float("nan")
                            ),
                            "selected_c_oof_mean": (
                                float(soft_ensemble["selected_c_oof_mean"])
                                if soft_ensemble["selected_c_oof_mean"] is not None
                                else float("nan")
                            ),
                            "threshold": float(variant["threshold"]),
                            **variant_test_metrics,
                        }
                    )
                    for sweep_row in list(variant["threshold_sweep"]):
                        threshold_rows.append(
                            {
                                **common,
                                "model_variant": model_variant,
                                "threshold": float(sweep_row["threshold"]),
                                "train_f0_5": float(sweep_row["f0_5"]),
                                "train_precision": float(sweep_row["precision"]),
                                "train_recall": float(sweep_row["recall"]),
                                "train_roc_auc": float(sweep_row["roc_auc"]),
                                "train_pr_auc": float(sweep_row["pr_auc"]),
                                "selected_threshold": float(variant["threshold"]),
                                "selected_train_f0_5": float(variant["selected_train_f0_5"]),
                                "selected_test_f0_5": float(variant_test_metrics.get("f0_5", float("nan"))),
                                "selected_test_roc_auc": float(variant_test_metrics.get("roc_auc", float("nan"))),
                                "selected_test_pr_auc": float(variant_test_metrics.get("pr_auc", float("nan"))),
                                "selected_test_precision": float(variant_test_metrics.get("precision", float("nan"))),
                                "selected_test_recall": float(variant_test_metrics.get("recall", float("nan"))),
                            }
                        )
    metrics = pd.DataFrame(rows)
    threshold_sweep = pd.DataFrame(threshold_rows)
    if not metrics.empty:
        metrics = metrics[metrics["model_variant"].astype(str).isin(selected_variant_set)].reset_index(drop=True)
    if not threshold_sweep.empty:
        threshold_sweep = threshold_sweep[
            threshold_sweep["model_variant"].astype(str).isin(selected_variant_set)
        ].reset_index(drop=True)
    if not metrics.empty:
        metrics = metrics.assign(
            _variant_order=(
                metrics["model_variant"].astype(str).map(SUCCESS_MODEL_VARIANT_ORDER).fillna(999).astype(int)
            )
        ).sort_values(
            ["branch_id", "hq_exit_override_branch", "_variant_order", "f0_5", "roc_auc"],
            ascending=[True, True, True, False, False],
        )
        metrics = metrics.drop(columns=["_variant_order"]).reset_index(drop=True)
    if not threshold_sweep.empty:
        threshold_sweep = threshold_sweep.assign(
            _variant_order=(
                threshold_sweep["model_variant"]
                .astype(str)
                .map(SUCCESS_MODEL_VARIANT_ORDER)
                .fillna(999)
                .astype(int)
            )
        ).sort_values(
            ["branch_id", "hq_exit_override_branch", "_variant_order", "threshold"],
            ascending=[True, True, True, True],
        )
        threshold_sweep = threshold_sweep.drop(columns=["_variant_order"]).reset_index(drop=True)
    return metrics, threshold_sweep


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
    success_threshold_sweep: pd.DataFrame,
    reproduction_consistency: pd.DataFrame,
    reproduction_source_run_dir: str,
) -> str:
    branch_sections = [
        ("pred_reasoning_only", "reasoning_pred-only"),
        ("hq_plus_pred_reasoning", "HQ + reasoning_pred"),
        ("llm_engineering_plus_pred_reasoning", "LLM-eng + reasoning_pred"),
    ]

    include_combined_best = "combined_best" in set(source_cv_summary["model_set_id"].astype(str).tolist())

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
    if not success_metrics.empty and "cv_seed_repeat_count" in success_metrics.columns:
        repeat_count = int(success_metrics["cv_seed_repeat_count"].max())
        repeat_enabled = bool(success_metrics["repeat_cv_with_new_seeds"].max())
        cv_heading_idx = lines.index("## CV Validation Performance (Source Runs)")
        lines[cv_heading_idx:cv_heading_idx] = [
            f"- Success CV repeats: `{repeat_count}` (enabled={repeat_enabled})",
            "",
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
    lines.append("")
    if include_combined_best:
        lines.extend(
            [
                "`combined_best` is defined per target by highest source CV R² across selected source models "
                "(tie-break: ridge > xgb3_regressor > mlp_regressor).",
                "",
            ]
        )
    lines.extend(
        [
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
                "| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        branch_rows = success[success["branch_id"].astype(str) == branch_id].copy()
        if not branch_rows.empty:
            branch_rows = branch_rows.assign(
                _variant_order=(
                    branch_rows["model_variant"].astype(str).map(SUCCESS_MODEL_VARIANT_ORDER).fillna(999).astype(int)
                )
            ).sort_values(
                ["hq_exit_override_branch", "_variant_order", "f0_5", "roc_auc"],
                ascending=[True, True, False, False],
            ).reset_index(drop=True)
            for row in branch_rows.itertuples(index=False):
                train_cv_text = (
                    f"{row.train_cv_f0_5:.4f} +/- {row.train_cv_f0_5_std:.4f}"
                    if hasattr(row, "train_cv_f0_5") and hasattr(row, "train_cv_f0_5_std")
                    else "n/a"
                )
                lines.append(
                    f"| {row.model_set_id} | {row.model_variant} | {row.hq_exit_override_branch} | {train_cv_text} | "
                    f"{row.f0_5:.4f} | {row.roc_auc:.4f} | "
                    f"{row.pr_auc:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.threshold:.4f} |"
                )
        else:
            lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        lines.append("")

    lines.extend(
        [
            "## Train Threshold Sweep (F0.5)",
            "",
            "Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in "
            "`success_transfer_train_threshold_sweep.csv`.",
            "",
            "| branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if success_threshold_sweep.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        sweep_rows = success_threshold_sweep.assign(
            _variant_order=(
                success_threshold_sweep["model_variant"]
                .astype(str)
                .map(SUCCESS_MODEL_VARIANT_ORDER)
                .fillna(999)
                .astype(int)
            )
        ).sort_values(
            ["branch_id", "hq_exit_override_branch", "_variant_order", "train_f0_5", "threshold"],
            ascending=[True, True, True, False, True],
        )
        for _, group in sweep_rows.groupby(["branch_id", "hq_exit_override_branch", "model_variant"], sort=True):
            for row in group.head(3).itertuples(index=False):
                lines.append(
                    f"| {row.branch_id} | {row.model_variant} | {row.hq_exit_override_branch} | "
                    f"{row.threshold:.4f} | {row.train_f0_5:.4f} | {row.train_precision:.4f} | {row.train_recall:.4f} | "
                    f"{row.selected_threshold:.4f} | {row.selected_train_f0_5:.4f} | {row.selected_test_f0_5:.4f} |"
                )
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

    if include_combined_best and not assignment_frame.empty:
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
            "- `success_transfer_train_threshold_sweep.csv`",
        ]
    )
    if include_combined_best and not assignment_frame.empty:
        lines.append("- `combined_best_assignment.csv`")
    lines.append("- `reproduction_consistency_check.csv`")
    return "\n".join(lines) + "\n"


def _write_full_transfer_docs(
    *,
    run_dir: Path,
    report_markdown: str,
) -> None:
    docs_root = DOCS_DIR / "experiment-archive" / "generated-reports" / "testing_models"
    docs_root.mkdir(parents=True, exist_ok=True)
    write_markdown(docs_root / "full_transfer_report_latest.md", report_markdown)
    write_markdown(docs_root / f"full_transfer_report_{run_dir.name}.md", report_markdown)


def _coerce_success_eval_result(
    result: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(result, tuple) and len(result) == 2:
        metrics, threshold_sweep = result
        metrics_df = pd.DataFrame(metrics)
        threshold_df = pd.DataFrame(threshold_sweep)
    else:
        # Backward-compatible path for older call sites/tests that still return only metrics.
        metrics_df = pd.DataFrame(result)
        threshold_df = pd.DataFrame()

    if not metrics_df.empty and "model_variant" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["model_variant"] = "single_model"
    if not metrics_df.empty and "hq_exit_override_branch" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["hq_exit_override_branch"] = "with_override"
    if not metrics_df.empty and "base_combo_id" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["base_combo_id"] = "unknown"
    if not metrics_df.empty and "success_branch_id" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["success_branch_id"] = metrics_df["base_combo_id"].astype(str) + "__with_override"
    if not metrics_df.empty and "branch_id" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["branch_id"] = "pred_reasoning_only"
    if not metrics_df.empty and "model_set_id" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["model_set_id"] = "unknown"
    if not metrics_df.empty and "f0_5" not in metrics_df.columns and "test_f0_5" in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["f0_5"] = metrics_df["test_f0_5"]
    if not metrics_df.empty and "roc_auc" not in metrics_df.columns and "test_roc_auc" in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["roc_auc"] = metrics_df["test_roc_auc"]
    if not metrics_df.empty and "pr_auc" not in metrics_df.columns and "test_pr_auc" in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["pr_auc"] = metrics_df["test_pr_auc"]
    if not metrics_df.empty and "precision" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["precision"] = float("nan")
    if not metrics_df.empty and "recall" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["recall"] = float("nan")
    if not metrics_df.empty and "train_cv_f0_5_std" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["train_cv_f0_5_std"] = float("nan")
    if not threshold_df.empty and "model_variant" not in threshold_df.columns:
        threshold_df = threshold_df.copy()
        threshold_df["model_variant"] = "single_model"
    if not threshold_df.empty and "hq_exit_override_branch" not in threshold_df.columns:
        threshold_df = threshold_df.copy()
        threshold_df["hq_exit_override_branch"] = "with_override"
    if not threshold_df.empty and "success_branch_id" not in threshold_df.columns:
        threshold_df = threshold_df.copy()
        threshold_df["success_branch_id"] = "unknown__with_override"
    if not threshold_df.empty and "branch_id" not in threshold_df.columns:
        threshold_df = threshold_df.copy()
        threshold_df["branch_id"] = "pred_reasoning_only"
    return metrics_df, threshold_df


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
    if (
        resolved.saved_eval_mode in {"full_transfer_report", "combination_transfer_report"}
        and not resolved.saved_eval_combo_refs
        and resolved.saved_eval_combo_ids is None
    ):
        default_combo = _select_default_transfer_combo(combos)
        combos = [default_combo]
        selected_ref_lines = [f"{default_combo.get('bundle_dir')}::{default_combo.get('combo_id')}"]
        bundle_dirs_used = [str(default_combo.get("bundle_dir", ""))]
        _log(
            logger,
            "No explicit combo refs provided for transfer mode. "
            f"Using default ridge combo: {default_combo.get('combo_id')}.",
        )

    extra_bank_ids = (
        {"hq_baseline", "llm_engineering", "lambda_policies"}
        if resolved.saved_eval_mode in {
            "success_with_pred_reasoning",
            "full_transfer_report",
            "combination_transfer_report",
        }
        else None
    )
    feature_sets_by_id, repository_splits, repository_banks = _load_feature_sets_for_bundle(
        config=config,
        combos=combos,
        required_extra_feature_bank_ids=extra_bank_ids,
        logger=logger,
    )

    if resolved.saved_eval_mode == "combination_transfer_report":
        selected_combo = _validate_combination_transfer_combo_refs(combos)
        family_map = {spec.family_id: spec for spec in config.target_families}
        target_family = load_target_family(family_map[COMBINATION_TRANSFER_TARGET_FAMILY])
        if resolved.heldout_evaluation and target_family.test_frame is None:
            raise RuntimeError(
                f"Target family '{COMBINATION_TRANSFER_TARGET_FAMILY}' has no held-out test targets."
            )
        feature_set_id = str(selected_combo["feature_set_id"])
        feature_set = feature_sets_by_id.get(feature_set_id)
        if feature_set is None:
            raise RuntimeError(
                f"Feature set '{feature_set_id}' required by selected combo is not available for evaluation."
            )
        combo_bundle_dir = Path(str(selected_combo["bundle_dir"]))
        pred_train = _predict_combo_on_frame(
            bundle_dir=combo_bundle_dir,
            combo=selected_combo,
            feature_frame=feature_set.public_frame,
        )
        pred_test = _predict_combo_on_frame(
            bundle_dir=combo_bundle_dir,
            combo=selected_combo,
            feature_frame=feature_set.private_frame,
        )
        # Preserve founder alignment for downstream success-transfer assembly.
        pred_train.index = feature_set.public_frame["founder_uuid"].astype(str).to_list()
        pred_test.index = feature_set.private_frame["founder_uuid"].astype(str).to_list()
        source_cv_per_target, source_cv_summary = _build_combination_source_cv_summary(
            combo=selected_combo,
        )
        selected_success_branch_ids = (
            set(resolved.saved_eval_success_branch_ids)
            if resolved.saved_eval_success_branch_ids is not None
            else None
        )
        success_metrics, success_threshold_sweep = _coerce_success_eval_result(
            _evaluate_combination_success_transfer(
                config=config,
                combo=selected_combo,
                pred_train=pred_train,
                pred_test=pred_test,
                repository_splits=repository_splits,
                repository_banks=repository_banks,
                hq_exit_override_mode=resolved.hq_exit_override_mode,
                evaluate_test=bool(resolved.heldout_evaluation),
                selected_success_branch_ids=selected_success_branch_ids,
                success_model_variants=resolved.success_model_variants,
                use_nested_success=resolved.distillation_nested_sweep,
                repeat_cv_with_new_seeds=resolved.repeat_cv_with_new_seeds,
                cv_seed_repeat_count=resolved.cv_seed_repeat_count,
            )
        )
        if success_metrics.empty:
            raise RuntimeError(
                "No success branches were evaluated. Check saved_eval_success_branch_ids selection."
            )
        write_csv(run_dir / "combination_transfer_source_cv_per_target.csv", source_cv_per_target)
        write_csv(run_dir / "combination_transfer_source_cv_summary.csv", source_cv_summary)
        combo_ref_used = (
            str(resolved.saved_eval_combo_refs[0])
            if resolved.saved_eval_combo_refs
            else f"{selected_combo.get('bundle_dir')}::{selected_combo.get('combo_id')}"
        )
        if resolved.heldout_evaluation:
            reasoning_per_target, reasoning_summary = _evaluate_combination_reasoning_transfer(
                combo=selected_combo,
                target_family=target_family,
                feature_set=feature_set,
                pred_test=pred_test,
            )
            write_csv(run_dir / "combination_transfer_reasoning_per_target.csv", reasoning_per_target)
            write_csv(run_dir / "combination_transfer_reasoning_summary.csv", reasoning_summary)
            write_csv(run_dir / "combination_transfer_success_metrics.csv", success_metrics)
            write_csv(
                run_dir / "combination_transfer_success_train_threshold_sweep.csv",
                success_threshold_sweep,
            )
            report_markdown = _render_combination_transfer_report(
                run_dir=run_dir,
                combo_ref_used=combo_ref_used,
                source_cv_summary=source_cv_summary,
                source_cv_per_target=source_cv_per_target,
                reasoning_summary=reasoning_summary,
                reasoning_per_target=reasoning_per_target,
                success_metrics=success_metrics,
                success_threshold_sweep=success_threshold_sweep,
            )
            write_markdown(run_dir / "combination_transfer_report.md", report_markdown)
            write_markdown(run_dir / "saved_config_eval_summary.md", report_markdown)
            _write_combination_transfer_docs(run_dir=run_dir, report_markdown=report_markdown)
            _log(logger, f"Saved-config combination transfer report complete. Artifacts written to {run_dir}.")
            return run_dir

        write_csv(run_dir / "combination_success_cv_metrics.csv", success_metrics)
        write_csv(run_dir / "combination_success_cv_train_threshold_sweep.csv", success_threshold_sweep)
        report_markdown = _render_combination_success_cv_report(
            run_dir=run_dir,
            combo_ref_used=combo_ref_used,
            source_cv_summary=source_cv_summary,
            success_metrics=success_metrics,
            success_threshold_sweep=success_threshold_sweep,
        )
        write_markdown(run_dir / "combination_success_cv_report.md", report_markdown)
        write_markdown(run_dir / "saved_config_eval_summary.md", report_markdown)
        _write_combination_success_cv_docs(run_dir=run_dir, report_markdown=report_markdown)
        _log(logger, f"Saved-config combination success screening complete. Artifacts written to {run_dir}.")
        return run_dir

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
        success_transfer, success_threshold_sweep = _coerce_success_eval_result(
            _evaluate_success_transfer(
                config=config,
                predictions_by_model_set=predictions_by_model_set,
                repository_splits=repository_splits,
                repository_banks=repository_banks,
                hq_exit_override_mode=resolved.hq_exit_override_mode,
                success_model_variants=resolved.success_model_variants,
                use_nested_success=resolved.distillation_nested_sweep,
                repeat_cv_with_new_seeds=resolved.repeat_cv_with_new_seeds,
                cv_seed_repeat_count=resolved.cv_seed_repeat_count,
            )
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
        write_csv(run_dir / "success_transfer_train_threshold_sweep.csv", success_threshold_sweep)
        write_csv(run_dir / "reproduction_consistency_check.csv", reproduction_consistency)
        if not assignment_frame.empty:
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
            success_threshold_sweep=success_threshold_sweep,
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
