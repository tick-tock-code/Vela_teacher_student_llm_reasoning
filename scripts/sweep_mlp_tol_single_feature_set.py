from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_pipeline
from src.pipeline.run_options import RunOverrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 2-fold CV tolerance sweep for a single MLP model on a single feature set "
            "and record elapsed fit time plus mean OOF R2."
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
        choices=["v25_policies", "taste_policies"],
        help="Target family to evaluate.",
    )
    parser.add_argument(
        "--feature-set",
        default="hq_plus_sentence_prose",
        help="Single feature set id to evaluate.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=2,
        help="Outer CV split count.",
    )
    parser.add_argument(
        "--tol-start",
        type=float,
        default=1e-3,
        help="Largest tol value in the log-spaced sweep.",
    )
    parser.add_argument(
        "--tol-end",
        type=float,
        default=1e-5,
        help="Smallest tol value in the log-spaced sweep.",
    )
    parser.add_argument(
        "--tol-points",
        type=int,
        default=6,
        help="Number of log-spaced tol points.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional explicit output CSV path. Defaults under tmp/benchmarks/.",
    )
    return parser


def _resolve_model_id(target_family: str) -> str:
    return "mlp_regressor" if target_family == "v25_policies" else "mlp_classifier"


def _read_mean_oof_r2(run_dir: Path, model_id: str) -> float:
    metrics_path = run_dir / "reasoning_metrics.csv"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing reasoning metrics artifact: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    rows = frame[(frame["split_id"] == "oof_overall") & (frame["model_id"] == model_id)]
    if rows.empty:
        raise RuntimeError(
            f"No oof_overall rows found for model_id={model_id} in reasoning_metrics.csv"
        )
    if "r2" not in rows.columns:
        raise RuntimeError("reasoning_metrics.csv does not contain an r2 column.")
    return float(rows["r2"].mean())


def main() -> None:
    args = build_parser().parse_args()
    if args.cv_splits < 2:
        raise RuntimeError("cv-splits must be >= 2.")
    if args.tol_points < 2:
        raise RuntimeError("tol-points must be >= 2.")
    if args.tol_start <= 0 or args.tol_end <= 0:
        raise RuntimeError("tol-start and tol-end must be > 0.")

    config = load_experiment_config(args.config)
    config_use = replace(
        config,
        distillation_cv=replace(config.distillation_cv, n_splits=args.cv_splits),
    )

    model_id = _resolve_model_id(args.target_family)
    tol_values = np.logspace(
        np.log10(args.tol_start),
        np.log10(args.tol_end),
        num=args.tol_points,
    )

    results: list[dict[str, object]] = []
    for tol in tol_values:
        tol_value = float(tol)
        print(
            f"\\n=== tol sweep run: target_family={args.target_family}, "
            f"feature_set={args.feature_set}, tol={tol_value:.8f}, cv_splits={args.cv_splits} ==="
        )
        overrides = RunOverrides(
            run_mode="reasoning_distillation_mode",
            target_family=args.target_family,
            candidate_feature_sets=[args.feature_set],
            reasoning_models=[model_id],
            output_modes=["multi_output"],
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            xgb_model_param_overrides_by_model_id={
                model_id: {"tol": tol_value},
            },
        )

        started = time.perf_counter()
        run_dir = run_pipeline(config_use, overrides, logger=print)
        elapsed_seconds = time.perf_counter() - started
        mean_oof_r2 = _read_mean_oof_r2(run_dir, model_id=model_id)

        row = {
            "target_family": args.target_family,
            "feature_set": args.feature_set,
            "model_id": model_id,
            "cv_splits": args.cv_splits,
            "tol": f"{tol_value:.8g}",
            "elapsed_seconds": round(elapsed_seconds, 3),
            "mean_oof_r2": round(mean_oof_r2, 6),
            "run_dir": str(run_dir),
        }
        results.append(row)
        print(
            f"Completed tol={tol_value:.8g} | elapsed={elapsed_seconds:.1f}s | "
            f"mean_oof_r2={mean_oof_r2:.6f}"
        )

    results_sorted = sorted(results, key=lambda item: float(item["tol"]))
    output_dir = Path("tmp") / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"mlp_tol_sweep_{stamp}.csv"

    fieldnames = [
        "target_family",
        "feature_set",
        "model_id",
        "cv_splits",
        "tol",
        "elapsed_seconds",
        "mean_oof_r2",
        "run_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)

    print("\\n=== R2 ranking (best first) ===")
    for rank, row in enumerate(
        sorted(results, key=lambda item: float(item["mean_oof_r2"]), reverse=True),
        start=1,
    ):
        print(
            f"{rank}. tol={row['tol']} | mean_oof_r2={row['mean_oof_r2']} | "
            f"elapsed={row['elapsed_seconds']}s"
        )
    print(f"\\nWrote tol sweep CSV to: {output_path}")


if __name__ == "__main__":
    main()
