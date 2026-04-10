from __future__ import annotations

import argparse

from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_reasoning_reconstruction
from src.pipeline.run_options import DEFAULT_CONFIG_PATH, RunOverrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reasoning-reconstruction pipeline on VCBench."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--run-reasoning-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the reasoning reconstruction pipeline.",
    )
    parser.add_argument(
        "--run-success-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dormant future option. This currently fails immediately if enabled.",
    )
    parser.add_argument(
        "--active-intermediary-features",
        nargs="*",
        help="Optional subset of intermediary feature ids to activate.",
    )
    parser.add_argument(
        "--force-rebuild-intermediary-features",
        action="store_true",
        help="Rebuild intermediary feature banks even if cached artifacts already exist.",
    )
    parser.add_argument(
        "--reasoning-targets",
        nargs="*",
        help="Optional subset of configured reasoning targets to run.",
    )
    parser.add_argument(
        "--reasoning-models",
        nargs="*",
        help="Optional subset of configured reasoning model ids to run.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Override the embedding model name for sentence-transformer intermediary features.",
    )
    return parser


def parse_run_overrides(argv: list[str] | None = None) -> RunOverrides:
    args = build_parser().parse_args(argv)
    return RunOverrides(
        config_path=args.config,
        run_reasoning_predictions=bool(args.run_reasoning_predictions),
        run_success_predictions=bool(args.run_success_predictions),
        active_intermediary_features=(
            [str(item) for item in args.active_intermediary_features]
            if args.active_intermediary_features
            else None
        ),
        force_rebuild_intermediary_features=bool(args.force_rebuild_intermediary_features),
        reasoning_targets=(
            [str(item) for item in args.reasoning_targets]
            if args.reasoning_targets
            else None
        ),
        reasoning_models=(
            [str(item) for item in args.reasoning_models]
            if args.reasoning_models
            else None
        ),
        embedding_model_name=str(args.embedding_model) if args.embedding_model else None,
    )


def main(argv: list[str] | None = None) -> None:
    overrides = parse_run_overrides(argv)
    config = load_experiment_config(overrides.config_path)
    run_dir = run_reasoning_reconstruction(config, overrides)
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
