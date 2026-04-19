from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.feature_repository import load_feature_repository_splits, load_repository_feature_banks
from src.data.raw_datasets import load_raw_datasets
from src.data.splits import build_stratified_reasoning_cv_splits
from src.data.targets import load_target_family
from src.evaluation.metrics import binary_classification_metrics, regression_metrics, select_f05_threshold
from src.intermediary_features.registry import assemble_feature_sets, prepare_intermediary_banks
from src.pipeline.config import ExperimentConfig, FeatureSetSpec, TargetFamilySpec, load_experiment_config
from src.student.models import build_reasoning_classifier, build_reasoning_regressor


DEFAULT_HIDDEN_SIZE_TOKENS = [
    "(2,)",
    "(4,)",
    "(8,)",
    "(16,)",
    "(32,)",
    "(64,)",
    "(128,)",
    "(32,4)",
    "(64,16)",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an MLP hidden-size sweep using stratified CV on the public train pool "
            "for a single feature set."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/teacher_student_distillation_v1.json",
        help="Experiment config path.",
    )
    parser.add_argument(
        "--target-family",
        default="v25_policies",
        choices=["v25_policies", "taste_policies", "v25_and_taste"],
        help="Target family to evaluate.",
    )
    parser.add_argument(
        "--feature-set",
        required=True,
        help="Feature set id to evaluate (must exist in distillation_feature_sets).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for CV splitting and model initialization.",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="*",
        default=DEFAULT_HIDDEN_SIZE_TOKENS,
        help=(
            "Hidden-layer specs like '(32,)' '(64,16)' or '32' '64,16'. "
            "Defaults to the requested 9-size sweep."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="MLP alpha (L2 regularization).",
    )
    parser.add_argument(
        "--learning-rate-init",
        type=float,
        default=1e-3,
        help="MLP learning_rate_init.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="MLP max_iter.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="MLP optimizer tolerance.",
    )
    parser.add_argument(
        "--n-iter-no-change",
        type=int,
        default=20,
        help="MLP n_iter_no_change (used when early stopping is on).",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable sklearn MLP early stopping.",
    )
    parser.add_argument(
        "--force-rebuild-intermediary-features",
        action="store_true",
        help="Rebuild intermediary features instead of reusing cached artifacts.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional explicit output CSV path. Defaults under tmp/benchmarks/.",
    )
    return parser


def _resolve_feature_set(config: ExperimentConfig, feature_set_id: str) -> FeatureSetSpec:
    for feature_set in config.distillation_feature_sets:
        if feature_set.feature_set_id == feature_set_id:
            return feature_set
    available = ", ".join(spec.feature_set_id for spec in config.distillation_feature_sets)
    raise RuntimeError(
        f"Unknown feature_set '{feature_set_id}'. Available feature sets: {available}"
    )


def _resolve_target_spec(config: ExperimentConfig, family_id: str) -> TargetFamilySpec:
    for spec in config.target_families:
        if spec.family_id == family_id:
            return spec
    available = ", ".join(spec.family_id for spec in config.target_families)
    raise RuntimeError(f"Unknown target family '{family_id}'. Available: {available}")


def _family_sequence(target_family: str) -> list[str]:
    if target_family == "v25_and_taste":
        return ["v25_policies", "taste_policies"]
    return [target_family]


def _parse_hidden_size_token(token: str) -> tuple[int, ...]:
    text = token.strip()
    if not text:
        raise RuntimeError("Hidden-size token cannot be empty.")
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    text = text.replace(" ", "")
    if text.endswith(","):
        text = text[:-1]
    parts = [part for part in text.split(",") if part]
    if not parts:
        raise RuntimeError(f"Could not parse hidden-size token '{token}'.")
    values = tuple(int(part) for part in parts)
    if any(value < 1 for value in values):
        raise RuntimeError(f"Hidden-size token '{token}' contains non-positive values.")
    return values


def _parse_hidden_sizes(tokens: list[str]) -> list[tuple[int, ...]]:
    parsed: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for token in tokens:
        value = _parse_hidden_size_token(token)
        if value in seen:
            continue
        seen.add(value)
        parsed.append(value)
    if not parsed:
        raise RuntimeError("At least one hidden size must be provided.")
    return parsed


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


def _as_probability_matrix(probs_raw: object, n_targets: int) -> np.ndarray:
    if isinstance(probs_raw, list):
        columns: list[np.ndarray] = []
        for probs in probs_raw:
            arr = np.asarray(probs, dtype=float)
            if arr.ndim == 1:
                columns.append(arr)
            elif arr.shape[1] == 1:
                columns.append(arr[:, 0])
            else:
                columns.append(arr[:, 1])
        probs = np.column_stack(columns)
    else:
        probs = np.asarray(probs_raw, dtype=float)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
    if probs.shape[1] != n_targets:
        raise RuntimeError(
            f"MLP probability output has {probs.shape[1]} columns; expected {n_targets}."
        )
    return probs


def _evaluate_hidden_size(
    *,
    task_kind: str,
    hidden_layer_sizes: tuple[int, ...],
    X: np.ndarray,
    y: np.ndarray,
    target_columns: list[str],
    cv_splits: int,
    split_seed: int,
    alpha: float,
    learning_rate_init: float,
    max_iter: int,
    tol: float,
    n_iter_no_change: int,
    early_stopping: bool,
    scale_min: float | None,
    scale_max: float | None,
) -> dict[str, float]:
    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "tol": tol,
        "n_iter_no_change": n_iter_no_change,
        "early_stopping": early_stopping,
    }

    splits = build_stratified_reasoning_cv_splits(
        pd.DataFrame(y, columns=target_columns),
        n_splits=cv_splits,
        shuffle=True,
        random_state=split_seed,
    )

    if task_kind == "regression":
        fold_r2_means: list[float] = []
        fold_rmse_means: list[float] = []
        fold_mae_means: list[float] = []
        for fold_offset, split in enumerate(splits):
            seed = split_seed + fold_offset
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
            per_target = [
                regression_metrics(y[split.test_idx][:, target_idx], preds[:, target_idx])
                for target_idx in range(y.shape[1])
            ]
            fold_r2_means.append(float(np.mean([entry["r2"] for entry in per_target])))
            fold_rmse_means.append(float(np.mean([entry["rmse"] for entry in per_target])))
            fold_mae_means.append(float(np.mean([entry["mae"] for entry in per_target])))
        return {
            "primary_mean": float(np.mean(fold_r2_means)),
            "primary_std": float(np.std(fold_r2_means)),
            "r2_mean": float(np.mean(fold_r2_means)),
            "rmse_mean": float(np.mean(fold_rmse_means)),
            "mae_mean": float(np.mean(fold_mae_means)),
        }

    fold_f05_means: list[float] = []
    fold_roc_auc_means: list[float] = []
    fold_pr_auc_means: list[float] = []
    for fold_offset, split in enumerate(splits):
        seed = split_seed + fold_offset
        model = build_reasoning_classifier(
            "mlp_classifier",
            random_state=seed,
            param_overrides=params,
        )
        model.fit(X[split.train_idx], y[split.train_idx].astype(int))
        probs = _as_probability_matrix(model.predict_proba(X[split.test_idx]), n_targets=y.shape[1])
        per_target_metrics: list[dict[str, float]] = []
        for target_idx in range(y.shape[1]):
            y_target = y[split.test_idx][:, target_idx].astype(int)
            threshold = select_f05_threshold(y_target, probs[:, target_idx])
            per_target_metrics.append(
                binary_classification_metrics(y_target, probs[:, target_idx], threshold=threshold)
            )
        fold_f05_means.append(float(np.mean([entry["f0_5"] for entry in per_target_metrics])))
        fold_roc_auc_means.append(float(np.mean([entry["roc_auc"] for entry in per_target_metrics])))
        fold_pr_auc_means.append(float(np.mean([entry["pr_auc"] for entry in per_target_metrics])))

    return {
        "primary_mean": float(np.mean(fold_f05_means)),
        "primary_std": float(np.std(fold_f05_means)),
        "f0_5_mean": float(np.mean(fold_f05_means)),
        "roc_auc_mean": float(np.mean(fold_roc_auc_means)),
        "pr_auc_mean": float(np.mean(fold_pr_auc_means)),
    }


def _default_output_csv() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("tmp") / "benchmarks" / f"mlp_hidden_size_cv_sweep_{stamp}.csv"


def main() -> None:
    args = build_parser().parse_args()
    if args.cv_splits < 2:
        raise RuntimeError("cv-splits must be >= 2.")
    if args.alpha <= 0:
        raise RuntimeError("alpha must be > 0.")
    if args.learning_rate_init <= 0:
        raise RuntimeError("learning-rate-init must be > 0.")
    if args.max_iter < 1:
        raise RuntimeError("max-iter must be >= 1.")
    if args.tol <= 0:
        raise RuntimeError("tol must be > 0.")
    if args.n_iter_no_change < 1:
        raise RuntimeError("n-iter-no-change must be >= 1.")

    hidden_sizes = _parse_hidden_sizes(list(args.hidden_sizes))
    config = load_experiment_config(args.config)
    selected_feature_set = _resolve_feature_set(config, args.feature_set)

    raw_datasets = load_raw_datasets(
        Path(config.datasets.public_train_csv),
        Path(config.datasets.private_test_csv),
    )
    repository_splits = load_feature_repository_splits(config.feature_repository)
    enabled_repository_specs = [spec for spec in config.repository_feature_banks if spec.enabled]
    enabled_intermediary_specs = [spec for spec in config.intermediary_features if spec.enabled]

    repository_banks = load_repository_feature_banks(
        repository_splits=repository_splits,
        specs=enabled_repository_specs,
    )
    intermediary_banks = prepare_intermediary_banks(
        public_raw=raw_datasets.public_frame,
        private_raw=raw_datasets.private_frame,
        feature_specs=enabled_intermediary_specs,
        force_rebuild=bool(args.force_rebuild_intermediary_features),
        logger=print,
    )
    assembled_sets = assemble_feature_sets(
        public_founder_ids=raw_datasets.public_frame["founder_uuid"],
        private_founder_ids=raw_datasets.private_frame["founder_uuid"],
        banks_by_id={**repository_banks, **intermediary_banks},
        feature_sets=[selected_feature_set],
    )
    if len(assembled_sets) != 1:
        raise RuntimeError("Expected exactly one assembled feature set.")
    feature_set = assembled_sets[0]

    rows: list[dict[str, object]] = []
    for family_id in _family_sequence(args.target_family):
        target_spec = _resolve_target_spec(config, family_id)
        target_family = load_target_family(target_spec)
        aligned_targets = _require_full_overlap(
            feature_set.public_frame[["founder_uuid"]],
            target_family.train_frame,
            on="founder_uuid",
            left_name=f"feature set '{feature_set.feature_set_id}' public rows",
            right_name=f"target family '{family_id}' public targets",
        )

        X_all = feature_set.public_frame[feature_set.feature_columns].to_numpy(dtype=float)
        target_columns = list(target_family.target_columns)
        y_all = aligned_targets[target_columns].to_numpy(
            dtype=float if target_family.task_kind == "regression" else int
        )

        primary_metric = "r2" if target_family.task_kind == "regression" else "f0_5"
        print(
            f"\n=== family={family_id} feature_set={feature_set.feature_set_id} "
            f"rows={len(y_all)} targets={len(target_columns)} "
            f"primary_metric={primary_metric} cv_splits={args.cv_splits} ==="
        )

        for hidden_layer_sizes in hidden_sizes:
            started = time.perf_counter()
            metrics = _evaluate_hidden_size(
                task_kind=target_family.task_kind,
                hidden_layer_sizes=hidden_layer_sizes,
                X=X_all,
                y=y_all,
                target_columns=target_columns,
                cv_splits=int(args.cv_splits),
                split_seed=int(args.split_seed),
                alpha=float(args.alpha),
                learning_rate_init=float(args.learning_rate_init),
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                n_iter_no_change=int(args.n_iter_no_change),
                early_stopping=bool(args.early_stopping),
                scale_min=target_family.scale_min,
                scale_max=target_family.scale_max,
            )
            elapsed_seconds = time.perf_counter() - started
            row = {
                "target_family": family_id,
                "task_kind": target_family.task_kind,
                "feature_set_id": feature_set.feature_set_id,
                "hidden_layer_sizes": str(hidden_layer_sizes),
                "primary_metric": primary_metric,
                "primary_mean": round(float(metrics["primary_mean"]), 6),
                "primary_std": round(float(metrics.get("primary_std", 0.0)), 6),
                "r2_mean": round(float(metrics.get("r2_mean", np.nan)), 6),
                "rmse_mean": round(float(metrics.get("rmse_mean", np.nan)), 6),
                "mae_mean": round(float(metrics.get("mae_mean", np.nan)), 6),
                "f0_5_mean": round(float(metrics.get("f0_5_mean", np.nan)), 6),
                "roc_auc_mean": round(float(metrics.get("roc_auc_mean", np.nan)), 6),
                "pr_auc_mean": round(float(metrics.get("pr_auc_mean", np.nan)), 6),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "row_count": len(y_all),
                "feature_count": len(feature_set.feature_columns),
                "target_count": len(target_columns),
                "cv_splits": int(args.cv_splits),
                "split_seed": int(args.split_seed),
                "alpha": float(args.alpha),
                "learning_rate_init": float(args.learning_rate_init),
                "max_iter": int(args.max_iter),
                "tol": float(args.tol),
                "n_iter_no_change": int(args.n_iter_no_change),
                "early_stopping": bool(args.early_stopping),
            }
            rows.append(row)
            print(
                f"hidden_layer_sizes={hidden_layer_sizes} "
                f"{primary_metric}={row['primary_mean']:.6f} +/- {row['primary_std']:.6f} "
                f"elapsed={elapsed_seconds:.2f}s"
            )

    if not rows:
        raise RuntimeError("Hidden-size sweep produced no rows.")

    rows_sorted = sorted(
        rows,
        key=lambda item: (
            str(item["target_family"]),
            -float(item["primary_mean"]),
            float(item["elapsed_seconds"]),
        ),
    )

    output_path = Path(args.output_csv) if args.output_csv else _default_output_csv()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_family",
        "task_kind",
        "feature_set_id",
        "hidden_layer_sizes",
        "primary_metric",
        "primary_mean",
        "primary_std",
        "r2_mean",
        "rmse_mean",
        "mae_mean",
        "f0_5_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "elapsed_seconds",
        "row_count",
        "feature_count",
        "target_count",
        "cv_splits",
        "split_seed",
        "alpha",
        "learning_rate_init",
        "max_iter",
        "tol",
        "n_iter_no_change",
        "early_stopping",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    print("\n=== Ranking by family (best primary metric first) ===")
    for family_id in _family_sequence(args.target_family):
        family_rows = [row for row in rows if row["target_family"] == family_id]
        if not family_rows:
            continue
        ranked = sorted(
            family_rows,
            key=lambda item: (-float(item["primary_mean"]), float(item["elapsed_seconds"])),
        )
        metric_name = str(ranked[0]["primary_metric"])
        print(f"\n[{family_id}] metric={metric_name}")
        for rank, row in enumerate(ranked, start=1):
            print(
                f"{rank}. hidden_layer_sizes={row['hidden_layer_sizes']} "
                f"{metric_name}={row['primary_mean']:.6f} +/- {row['primary_std']:.6f} "
                f"elapsed={row['elapsed_seconds']}s"
            )

    print(f"\nWrote hidden-size CV sweep CSV to: {output_path}")


if __name__ == "__main__":
    main()
