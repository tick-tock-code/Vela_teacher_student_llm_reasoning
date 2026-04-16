from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_pipeline
from src.pipeline.run_options import RunOverrides
from src.utils.parallel import resolve_max_parallel_workers


DEFAULT_FEATURE_SETS = [
    "hq_plus_sentence_prose",
    "hq_plus_sentence_bundle",
    "llm_engineering_plus_sentence_prose",
    "llm_engineering_plus_sentence_bundle",
    "lambda_policies_plus_sentence_prose",
    "lambda_policies_plus_sentence_bundle",
]


@dataclass(frozen=True)
class BenchmarkCase:
    mode_name: str
    max_parallel_workers: int
    model_testing_per_fit_threads: int
    description: str


def _parse_int_list(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise RuntimeError("Expected at least one integer value.")
    if any(value < 1 for value in values):
        raise RuntimeError("All integer values must be >= 1.")
    return values


def _default_single_fit_threads() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count < 2:
        return 1
    return max(2, min(8, cpu_count // 2))


def _dedupe_cases(cases: list[BenchmarkCase]) -> list[BenchmarkCase]:
    seen: set[tuple[int, int]] = set()
    deduped: list[BenchmarkCase] = []
    for case in cases:
        key = (case.max_parallel_workers, case.model_testing_per_fit_threads)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def _build_quick_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    single_fit_threads = args.single_fit_threads or _default_single_fit_threads()
    parallel_workers = resolve_max_parallel_workers(args.parallel_workers)
    return _dedupe_cases(
        [
            BenchmarkCase(
                mode_name="single_fit_1thread",
                max_parallel_workers=1,
                model_testing_per_fit_threads=1,
                description="One fit at a time with BLAS/OpenMP limited to 1 thread.",
            ),
            BenchmarkCase(
                mode_name=f"single_fit_{single_fit_threads}threads",
                max_parallel_workers=1,
                model_testing_per_fit_threads=single_fit_threads,
                description="One fit at a time with moderate BLAS/OpenMP threading.",
            ),
            BenchmarkCase(
                mode_name=f"parallel_fits_{parallel_workers}workers_1thread",
                max_parallel_workers=parallel_workers,
                model_testing_per_fit_threads=1,
                description="Many fits in parallel with each fit limited to 1 thread.",
            ),
        ]
    )


def _build_grid_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    workers_grid = _parse_int_list(args.workers_grid)
    per_fit_threads_grid = _parse_int_list(args.per_fit_threads_grid)
    cases: list[BenchmarkCase] = []
    for workers in workers_grid:
        for per_fit_threads in per_fit_threads_grid:
            cases.append(
                BenchmarkCase(
                    mode_name=f"grid_w{workers}_t{per_fit_threads}",
                    max_parallel_workers=workers,
                    model_testing_per_fit_threads=per_fit_threads,
                    description="Cartesian grid benchmark point.",
                )
            )
    return _dedupe_cases(cases)


def _read_estimated_fits(run_dir: Path) -> int:
    summary_path = run_dir / "run_summary.md"
    if not summary_path.exists():
        return 0
    text = summary_path.read_text(encoding="utf-8")
    match = re.search(r"Estimated Stage A outer fits:\s*(\d+)", text)
    if not match:
        return 0
    return int(match.group(1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark model-testing throughput across a small set of compute layouts "
            "or an exhaustive workers/thread grid."
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
        help="Target family for model-testing benchmark.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="CV repeat count for model-testing benchmark (1 disables repeat-seed mode).",
    )
    parser.add_argument(
        "--benchmark-profile",
        choices=["quick", "grid"],
        default="quick",
        help=(
            "Use 'quick' for three short comparison modes or 'grid' for the full "
            "workers/thread Cartesian product."
        ),
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help=(
            "Worker count for the quick parallel-fits comparison. Defaults to the "
            "resolved max_parallel_workers value for the current machine."
        ),
    )
    parser.add_argument(
        "--single-fit-threads",
        type=int,
        default=None,
        help=(
            "BLAS/OpenMP threads for the quick single-fit threaded comparison. "
            "Defaults to a moderate fraction of logical cores."
        ),
    )
    parser.add_argument(
        "--feature-set-limit",
        type=int,
        default=1,
        help=(
            "Limit the benchmark to the first N feature sets so each run stays small. "
            "Defaults to 1."
        ),
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=2,
        help=(
            "Outer CV split count used for the benchmark-local config override. "
            "Defaults to 2 so the quick profile stays short."
        ),
    )
    parser.add_argument(
        "--workers-grid",
        default="4,6,8",
        help="Comma-separated max_parallel_workers values (e.g. 4,6,8).",
    )
    parser.add_argument(
        "--per-fit-threads-grid",
        default="1,2",
        help="Comma-separated model-testing per-fit BLAS thread values (e.g. 1,2).",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=DEFAULT_FEATURE_SETS,
        help="Candidate feature sets to benchmark.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional explicit output CSV path. Defaults under tmp/benchmarks/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeat_count < 1:
        raise RuntimeError("repeat-count must be >= 1.")
    if not args.feature_sets:
        raise RuntimeError("At least one feature set is required.")
    if args.feature_set_limit < 1:
        raise RuntimeError("feature-set-limit must be >= 1.")
    if args.cv_splits < 2:
        raise RuntimeError("cv-splits must be >= 2.")
    if args.single_fit_threads is not None and args.single_fit_threads < 1:
        raise RuntimeError("single-fit-threads must be >= 1.")

    config = load_experiment_config(args.config)
    benchmark_config = replace(
        config,
        distillation_cv=replace(config.distillation_cv, n_splits=args.cv_splits),
    )
    selected_feature_sets = list(args.feature_sets[: args.feature_set_limit])
    if not selected_feature_sets:
        raise RuntimeError("No feature sets were selected after applying the limit.")

    if len(selected_feature_sets) < len(args.feature_sets):
        print(
            f"Limiting benchmark feature sets to the first {len(selected_feature_sets)} "
            f"of {len(args.feature_sets)} requested entries."
        )

    if args.benchmark_profile == "quick":
        benchmark_cases = _build_quick_cases(args)
    else:
        benchmark_cases = _build_grid_cases(args)

    results: list[dict[str, object]] = []
    for case in benchmark_cases:
        print(
            f"\n=== Benchmark run: mode={case.mode_name}, workers={case.max_parallel_workers}, "
            f"per_fit_threads={case.model_testing_per_fit_threads}, target_family={args.target_family}, "
            f"repeat_count={args.repeat_count}, feature_sets={len(selected_feature_sets)} ==="
        )
        overrides = RunOverrides(
            run_mode="model_testing_mode",
            target_family=args.target_family,
            candidate_feature_sets=selected_feature_sets,
            model_families=["mlp"],
            model_family_output_modes={"mlp": ["multi_output"]},
            repeat_cv_with_new_seeds=(args.repeat_count > 1),
            cv_seed_repeat_count=args.repeat_count,
            distillation_nested_sweep=False,
            run_advanced_models=False,
            max_parallel_workers=case.max_parallel_workers,
            model_testing_per_fit_threads=case.model_testing_per_fit_threads,
        )
        started = time.perf_counter()
        run_dir = run_pipeline(benchmark_config, overrides, logger=print)
        elapsed_seconds = time.perf_counter() - started
        estimated_fits = _read_estimated_fits(run_dir)
        fits_per_min = (estimated_fits / elapsed_seconds) * 60.0 if estimated_fits > 0 else 0.0
        row = {
            "mode_name": case.mode_name,
            "mode_description": case.description,
            "workers": case.max_parallel_workers,
            "per_fit_threads": case.model_testing_per_fit_threads,
            "target_family": args.target_family,
            "repeat_count": args.repeat_count,
            "feature_set_count": len(selected_feature_sets),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "estimated_fits": estimated_fits,
            "fits_per_minute": round(fits_per_min, 3),
            "run_dir": str(run_dir),
        }
        results.append(row)
        print(
            f"Completed mode={case.mode_name} | elapsed={elapsed_seconds:.1f}s | "
            f"estimated_fits={estimated_fits} | fits_per_min={fits_per_min:.2f}"
        )

    results_sorted = sorted(results, key=lambda item: float(item["elapsed_seconds"]))
    output_dir = Path("tmp") / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"model_testing_parallel_{args.benchmark_profile}_{stamp}.csv"

    fieldnames = [
        "mode_name",
        "mode_description",
        "workers",
        "per_fit_threads",
        "target_family",
        "repeat_count",
        "feature_set_count",
        "elapsed_seconds",
        "estimated_fits",
        "fits_per_minute",
        "run_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)

    print("\n=== Ranking (fastest first) ===")
    for rank, row in enumerate(results_sorted, start=1):
        print(
            f"{rank}. mode={row['mode_name']}, workers={row['workers']}, per_fit_threads={row['per_fit_threads']} | "
            f"elapsed={row['elapsed_seconds']}s | fits_per_min={row['fits_per_minute']}"
        )
    print(f"\nWrote benchmark CSV to: {output_path}")


if __name__ == "__main__":
    main()
