from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.pipeline.model_testing as model_testing
from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_pipeline
from src.pipeline.run_options import RunOverrides
from src.student.models import XGB_CLASSIFIER_PARAMS, XGB_REGRESSOR_PARAMS


def _parse_float_csv(value: str) -> list[float]:
    values = [float(token.strip()) for token in value.split(",") if token.strip()]
    if not values:
        raise RuntimeError("Expected at least one numeric value.")
    return values


def _default_reg_alpha_sweep() -> str:
    alpha = float(XGB_REGRESSOR_PARAMS.get("reg_alpha", XGB_CLASSIFIER_PARAMS.get("reg_alpha", 0.73)))
    return f"0.1,{alpha:.4g},2.0"


def _default_reg_lambda_sweep() -> str:
    reg_lambda = float(XGB_REGRESSOR_PARAMS.get("reg_lambda", XGB_CLASSIFIER_PARAMS.get("reg_lambda", 15.0)))
    return f"5.0,{reg_lambda:.4g},30.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run model_testing_mode for XGBoost with nested CV over a coarse regularization grid "
            "on one feature set."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/teacher_student_distillation_v1.json",
        help="Experiment config path.",
    )
    parser.add_argument(
        "--feature-set",
        default="lambda_policies_plus_sentence_prose",
        help="Single feature set id to evaluate.",
    )
    parser.add_argument(
        "--target-family",
        default="v25_and_taste",
        choices=["v25_policies", "taste_policies", "v25_and_taste"],
        help="Target family selection.",
    )
    parser.add_argument(
        "--outer-cv-splits",
        type=int,
        default=5,
        help="Outer CV splits used by nested model testing.",
    )
    parser.add_argument(
        "--inner-cv-splits",
        type=int,
        default=3,
        help="Inner CV splits used by nested model testing.",
    )
    parser.add_argument(
        "--reg-alpha-sweep",
        default=_default_reg_alpha_sweep(),
        help="Comma-separated reg_alpha values for coarse nested sweep.",
    )
    parser.add_argument(
        "--reg-lambda-sweep",
        default=_default_reg_lambda_sweep(),
        help="Comma-separated reg_lambda values for coarse nested sweep.",
    )
    parser.add_argument(
        "--use-latest-xgb-calibration",
        action="store_true",
        help="Use selected n_estimators from the latest XGB calibration artifact.",
    )
    parser.add_argument(
        "--max-parallel-workers",
        type=int,
        default=2,
        help="Max fold/task workers for model testing.",
    )
    parser.add_argument(
        "--per-fit-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads per fit.",
    )
    parser.add_argument(
        "--no-nested-grid",
        action="store_true",
        help=(
            "Run an explicit grid of fixed reg_alpha/reg_lambda settings with nested tuning disabled. "
            "One model-testing run is executed per grid point."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional explicit CSV output path for no-nested grid mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.outer_cv_splits < 2:
        raise RuntimeError("outer-cv-splits must be >= 2.")
    if args.inner_cv_splits < 2:
        raise RuntimeError("inner-cv-splits must be >= 2.")
    if args.max_parallel_workers < 1:
        raise RuntimeError("max-parallel-workers must be >= 1.")
    if args.per_fit_threads < 1:
        raise RuntimeError("per-fit-threads must be >= 1.")

    reg_alpha_values = _parse_float_csv(args.reg_alpha_sweep)
    reg_lambda_values = _parse_float_csv(args.reg_lambda_sweep)
    coarse_grid = [
        {"reg_alpha": float(reg_alpha), "reg_lambda": float(reg_lambda)}
        for reg_alpha in reg_alpha_values
        for reg_lambda in reg_lambda_values
    ]

    original_nested_param_grid = getattr(model_testing, "_nested_param_grid", None)
    original_select_best_params_regression = getattr(
        model_testing,
        "_select_best_params_regression",
        None,
    )
    original_select_best_params_classification = getattr(
        model_testing,
        "_select_best_params_classification",
        None,
    )
    selected_param_calls: list[dict[str, object]] = []

    def patched_nested_param_grid(model_kind: str, task_kind: str) -> list[dict[str, float | int]]:
        if model_kind in {"xgb1_regressor", "xgb1_classifier"}:
            return [dict(item) for item in coarse_grid]
        return original_nested_param_grid(model_kind, task_kind)

    def patched_select_best_params_regression(
        X_train,
        y_train,
        *,
        model_kind: str,
        random_state: int,
        inner_n_splits: int,
        inner_shuffle: bool,
    ) -> dict[str, float | int]:
        best_params = original_select_best_params_regression(
            X_train,
            y_train,
            model_kind=model_kind,
            random_state=random_state,
            inner_n_splits=inner_n_splits,
            inner_shuffle=inner_shuffle,
        )
        if model_kind == "xgb1_regressor":
            selected_param_calls.append(
                {
                    "task_kind": "regression",
                    "model_kind": model_kind,
                    "random_state": int(random_state),
                    "inner_n_splits": int(inner_n_splits),
                    "reg_alpha": float(best_params.get("reg_alpha", 0.0)),
                    "reg_lambda": float(best_params.get("reg_lambda", 0.0)),
                }
            )
        return best_params

    def patched_select_best_params_classification(
        X_train,
        y_train,
        *,
        model_kind: str,
        random_state: int,
        inner_n_splits: int,
        inner_shuffle: bool,
    ) -> dict[str, float | int]:
        best_params = original_select_best_params_classification(
            X_train,
            y_train,
            model_kind=model_kind,
            random_state=random_state,
            inner_n_splits=inner_n_splits,
            inner_shuffle=inner_shuffle,
        )
        if model_kind == "xgb1_classifier":
            selected_param_calls.append(
                {
                    "task_kind": "classification",
                    "model_kind": model_kind,
                    "random_state": int(random_state),
                    "inner_n_splits": int(inner_n_splits),
                    "reg_alpha": float(best_params.get("reg_alpha", 0.0)),
                    "reg_lambda": float(best_params.get("reg_lambda", 0.0)),
                }
            )
        return best_params

    config = load_experiment_config(args.config)
    config = replace(
        config,
        distillation_cv=replace(config.distillation_cv, n_splits=args.outer_cv_splits),
        reproduction=replace(
            config.reproduction,
            outer_cv=replace(config.reproduction.outer_cv, n_splits=args.outer_cv_splits),
            inner_cv=replace(config.reproduction.inner_cv, n_splits=args.inner_cv_splits),
        ),
    )

    overrides = RunOverrides(
        run_mode="model_testing_mode",
        target_family=args.target_family,
        candidate_feature_sets=[args.feature_set],
        model_families=["xgb1"],
        model_family_output_modes={"xgb1": ["single_target"]},
        repeat_cv_with_new_seeds=False,
        cv_seed_repeat_count=1,
        distillation_nested_sweep=True,
        run_advanced_models=False,
        use_latest_xgb_calibration=bool(args.use_latest_xgb_calibration),
        max_parallel_workers=args.max_parallel_workers,
        model_testing_per_fit_threads=args.per_fit_threads,
    )

    if args.no_nested_grid:
        print("Running XGB no-nested explicit regularization grid with settings:")
        print(f"  target_family={args.target_family}")
        print(f"  feature_set={args.feature_set}")
        print(f"  outer_cv_splits={args.outer_cv_splits}")
        print("  nested=False")
        print("  repeats=1")
        print(f"  use_latest_xgb_calibration={bool(args.use_latest_xgb_calibration)}")
        print(f"  reg_alpha_values={reg_alpha_values}")
        print(f"  reg_lambda_values={reg_lambda_values}")

        rows: list[dict[str, object]] = []
        for grid_idx, grid_params in enumerate(coarse_grid, start=1):
            print(
                f"\n[{grid_idx}/{len(coarse_grid)}] "
                f"reg_alpha={grid_params['reg_alpha']}, reg_lambda={grid_params['reg_lambda']}"
            )
            run_overrides = replace(
                overrides,
                distillation_nested_sweep=False,
                xgb_model_param_overrides_by_model_id={
                    "xgb1_regressor": dict(grid_params),
                    "xgb1_classifier": dict(grid_params),
                },
            )
            started = time.perf_counter()
            run_dir = run_pipeline(config, run_overrides, logger=print)
            elapsed_seconds = time.perf_counter() - started

            screening_path = Path(run_dir) / "feature_set_screening.csv"
            if not screening_path.exists():
                raise RuntimeError(f"Expected screening CSV missing: {screening_path}")

            with screening_path.open("r", encoding="utf-8", newline="") as handle:
                screening_rows = list(csv.DictReader(handle))

            row: dict[str, object] = {
                "reg_alpha": float(grid_params["reg_alpha"]),
                "reg_lambda": float(grid_params["reg_lambda"]),
                "elapsed_seconds": round(float(elapsed_seconds), 3),
                "run_dir": str(run_dir),
            }
            for screening_row in screening_rows:
                if str(screening_row.get("feature_set_id", "")) != args.feature_set:
                    continue
                family = str(screening_row["target_family"])
                row[f"{family}_primary_metric"] = screening_row.get("primary_metric", "")
                row[f"{family}_primary_mean"] = screening_row.get("primary_mean", "")
                row[f"{family}_primary_std"] = screening_row.get("primary_std", "")
            rows.append(row)

            print(
                f"Completed in {elapsed_seconds:.1f}s | "
                f"run_dir={run_dir}"
            )

        output_dir = Path("tmp") / "benchmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.output_csv:
            output_path = Path(args.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"xgb_reg_grid_no_nested_{stamp}.csv"

        fieldnames = sorted({key for row in rows for key in row.keys()})
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print("\nNo-nested grid complete.")
        print(f"GRID_ROWS={len(rows)}")
        print(f"GRID_CSV={output_path}")
        return

    print("Running XGB nested regularization sweep with settings:")
    print(f"  target_family={args.target_family}")
    print(f"  feature_set={args.feature_set}")
    print(f"  outer_cv_splits={args.outer_cv_splits}")
    print(f"  inner_cv_splits={args.inner_cv_splits}")
    print(f"  use_latest_xgb_calibration={bool(args.use_latest_xgb_calibration)}")
    print(f"  reg_alpha_values={reg_alpha_values}")
    print(f"  reg_lambda_values={reg_lambda_values}")

    if (
        original_nested_param_grid is None
        or original_select_best_params_regression is None
        or original_select_best_params_classification is None
    ):
        raise RuntimeError(
            "Nested tracking hooks are unavailable in src.pipeline.model_testing. "
            "Use --no-nested-grid for explicit fixed-parameter sweeps."
        )

    start = time.perf_counter()
    model_testing._nested_param_grid = patched_nested_param_grid
    model_testing._select_best_params_regression = patched_select_best_params_regression
    model_testing._select_best_params_classification = patched_select_best_params_classification
    try:
        run_dir = run_pipeline(config, overrides, logger=print)
    finally:
        model_testing._nested_param_grid = original_nested_param_grid
        model_testing._select_best_params_regression = original_select_best_params_regression
        model_testing._select_best_params_classification = original_select_best_params_classification

    elapsed_seconds = time.perf_counter() - start

    count_by_combo: dict[tuple[str, float, float], int] = {}
    for row in selected_param_calls:
        key = (
            str(row["task_kind"]),
            float(row["reg_alpha"]),
            float(row["reg_lambda"]),
        )
        count_by_combo[key] = int(count_by_combo.get(key, 0) + 1)

    selection_counts: list[dict[str, object]] = []
    for (task_kind, reg_alpha, reg_lambda), count in sorted(
        count_by_combo.items(),
        key=lambda item: (item[0][0], -item[1], item[0][1], item[0][2]),
    ):
        selection_counts.append(
            {
                "task_kind": task_kind,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "count": int(count),
            }
        )

    most_frequent_by_task_kind: dict[str, dict[str, object]] = {}
    for row in selection_counts:
        task_kind = str(row["task_kind"])
        if task_kind in most_frequent_by_task_kind:
            continue
        most_frequent_by_task_kind[task_kind] = {
            "reg_alpha": float(row["reg_alpha"]),
            "reg_lambda": float(row["reg_lambda"]),
            "count": int(row["count"]),
        }

    metadata = {
        "target_family": args.target_family,
        "feature_set": args.feature_set,
        "outer_cv_splits": int(args.outer_cv_splits),
        "inner_cv_splits": int(args.inner_cv_splits),
        "use_latest_xgb_calibration": bool(args.use_latest_xgb_calibration),
        "max_parallel_workers": int(args.max_parallel_workers),
        "per_fit_threads": int(args.per_fit_threads),
        "reg_alpha_values": reg_alpha_values,
        "reg_lambda_values": reg_lambda_values,
        "coarse_grid": coarse_grid,
        "selected_param_call_count": int(len(selected_param_calls)),
        "selected_regularization_counts": selection_counts,
        "selected_regularization_most_frequent_by_task_kind": most_frequent_by_task_kind,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "run_dir": str(run_dir),
    }
    metadata_path = Path(run_dir) / "xgb_nested_regularization_sweep.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    call_rows_path = Path(run_dir) / "xgb_nested_selected_param_calls.csv"
    with call_rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_kind",
                "model_kind",
                "random_state",
                "inner_n_splits",
                "reg_alpha",
                "reg_lambda",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_param_calls)

    count_rows_path = Path(run_dir) / "xgb_nested_selected_param_counts.csv"
    with count_rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_kind",
                "reg_alpha",
                "reg_lambda",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(selection_counts)

    print(f"\nELAPSED_SECONDS={elapsed_seconds:.3f}")
    print(f"RUN_DIR={run_dir}")
    print(f"SWEEP_METADATA={metadata_path}")
    print(f"SWEEP_SELECTED_CALLS={call_rows_path}")
    print(f"SWEEP_SELECTED_COUNTS={count_rows_path}")


if __name__ == "__main__":
    main()
