from __future__ import annotations

import argparse

from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_distillation_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the teacher-student reasoning distillation pipeline.")
    parser.add_argument(
        "--config",
        default="experiments/teacher_student_distillation_v1.json",
        help="Path to the experiment config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_experiment_config(args.config)
    run_dir = run_distillation_experiment(config)
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
