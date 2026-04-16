from __future__ import annotations

import argparse

from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_pipeline
from src.pipeline.run_options import DEFAULT_CONFIG_PATH, RunOverrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Feature Repository reproduction or reasoning-distillation pipeline."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--run-mode",
        choices=["reproduction_mode", "reasoning_distillation_mode", "model_testing_mode", "xgb_calibration_mode", "rf_calibration_mode", "mlp_calibration_mode"],
        help="Pipeline mode. Defaults to the config default.",
    )
    parser.add_argument(
        "--target-family",
        help="Optional target family override for reasoning distillation (v25_policies, taste_policies, v25_and_taste).",
    )
    parser.add_argument(
        "--heldout-evaluation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt in or out of held-out evaluation for reasoning-distillation mode.",
    )
    parser.add_argument(
        "--active-feature-banks",
        nargs="*",
        help="Optional subset of repository/intermediary feature banks for reasoning distillation.",
    )
    parser.add_argument(
        "--force-rebuild-intermediary-features",
        action="store_true",
        help="Rebuild sentence-transformer intermediary banks even if cached artifacts already exist.",
    )
    parser.add_argument(
        "--reasoning-models",
        nargs="*",
        help="Optional subset of distillation model ids to run.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Override the embedding model name for sentence-transformer intermediary features.",
    )
    parser.add_argument(
        "--repeat-cv-with-new-seeds",
        action="store_true",
        help="Run distillation CV repeatedly with different random seeds and average metrics.",
    )
    parser.add_argument(
        "--cv-seed-repeat-count",
        type=int,
        help="Number of repeated stratified CV runs when --repeat-cv-with-new-seeds is enabled.",
    )
    parser.add_argument(
        "--nested-hyperparameter-cv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable nested hyperparameter tuning CV (shared across reproduction and distillation).",
    )
    parser.add_argument(
        "--save-reasoning-predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable writing reasoning OOF/held-out prediction CSVs.",
    )
    parser.add_argument(
        "--candidate-feature-sets",
        nargs="*",
        help="Feature-set ids used by model_testing_mode screening/advanced runs.",
    )
    parser.add_argument(
        "--model-families",
        nargs="*",
        help="Model families for model_testing_mode: linear_l2, xgb1, mlp, elasticnet, randomforest.",
    )
    parser.add_argument(
        "--output-modes",
        nargs="*",
        help="Output modes for model_testing_mode: single_target, multi_output.",
    )
    parser.add_argument(
        "--run-advanced-models",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="In model_testing_mode, run stage-B advanced model comparisons on shortlisted feature sets.",
    )
    parser.add_argument(
        "--xgb-calibration-estimators",
        nargs="*",
        type=int,
        help="Coarse n_estimators sweep values for xgb_calibration_mode.",
    )
    parser.add_argument(
        "--use-latest-xgb-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="In model_testing_mode, load latest xgb calibration and override xgb n_estimators.",
    )
    parser.add_argument(
        "--rf-calibration-min-samples-leaf",
        nargs="*",
        type=int,
        help="Sweep values for min_samples_leaf in rf_calibration_mode.",
    )
    parser.add_argument(
        "--rf-calibration-max-depth",
        nargs="*",
        type=str,
        help="Sweep values for max_depth in rf_calibration_mode; use 'none' for unbounded depth.",
    )
    parser.add_argument(
        "--rf-calibration-max-features",
        nargs="*",
        type=str,
        help="Sweep values for max_features in rf_calibration_mode (e.g. sqrt 0.5).",
    )
    parser.add_argument(
        "--use-latest-rf-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="In model_testing_mode, load latest rf calibration and override rf hyperparameters.",
    )
    parser.add_argument(
        "--mlp-calibration-hidden-layer-sizes",
        nargs="*",
        type=str,
        help="Sweep hidden layer sizes for mlp_calibration_mode, e.g. 8 16 32 16,8",
    )
    parser.add_argument(
        "--mlp-calibration-alpha",
        nargs="*",
        type=float,
        help="Sweep alpha values for mlp_calibration_mode.",
    )
    parser.add_argument(
        "--use-latest-mlp-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="In model_testing_mode, load latest mlp calibration and override mlp hyperparameters.",
    )
    parser.add_argument(
        "--mlp-hidden-layer-sizes",
        type=str,
        help=(
            "Direct fixed MLP hidden layer sizes for reasoning/model-testing runs, "
            "for example '32' or '16,8'."
        ),
    )
    parser.add_argument(
        "--mlp-alpha",
        type=float,
        help="Direct fixed MLP alpha for reasoning/model-testing runs.",
    )
    parser.add_argument(
        "--model-testing-per-fit-threads",
        type=int,
        help="Per-fit BLAS/OpenMP thread count used inside model_testing_mode (default 1).",
    )
    parser.add_argument(
        "--max-parallel-workers",
        type=int,
        help="Maximum worker threads for parallel model training (default auto: min(7, cpu_count-1)).",
    )
    parser.add_argument(
        "--distillation-nested-sweep",
        dest="nested_hyperparameter_cv",
        action="store_const",
        const=True,
        help=argparse.SUPPRESS,
    )
    return parser


def parse_run_overrides(argv: list[str] | None = None) -> RunOverrides:
    args = build_parser().parse_args(argv)
    rf_max_depth_values = None
    if args.rf_calibration_max_depth:
        rf_max_depth_values = []
        for raw in args.rf_calibration_max_depth:
            token = str(raw).strip().lower()
            if token in {"none", "null"}:
                rf_max_depth_values.append(None)
            else:
                rf_max_depth_values.append(int(token))

    rf_max_features_values = None
    if args.rf_calibration_max_features:
        rf_max_features_values = []
        for raw in args.rf_calibration_max_features:
            token = str(raw).strip()
            try:
                rf_max_features_values.append(float(token))
            except ValueError:
                rf_max_features_values.append(token)

    mlp_hidden_sizes = None
    if args.mlp_calibration_hidden_layer_sizes:
        mlp_hidden_sizes = []
        for raw in args.mlp_calibration_hidden_layer_sizes:
            token = str(raw).strip()
            if not token:
                continue
            if "," in token:
                mlp_hidden_sizes.append([int(v.strip()) for v in token.split(",") if v.strip()])
            else:
                mlp_hidden_sizes.append([int(token)])

    mlp_hidden_layer_sizes_override = None
    if args.mlp_hidden_layer_sizes:
        token = str(args.mlp_hidden_layer_sizes).strip()
        if token:
            if "," in token:
                mlp_hidden_layer_sizes_override = [int(v.strip()) for v in token.split(",") if v.strip()]
            else:
                mlp_hidden_layer_sizes_override = [int(token)]

    return RunOverrides(
        config_path=args.config,
        run_mode=str(args.run_mode) if args.run_mode else None,
        target_family=str(args.target_family) if args.target_family else None,
        heldout_evaluation=args.heldout_evaluation,
        active_feature_banks=(
            [str(item) for item in args.active_feature_banks]
            if args.active_feature_banks
            else None
        ),
        force_rebuild_intermediary_features=bool(args.force_rebuild_intermediary_features),
        reasoning_models=(
            [str(item) for item in args.reasoning_models]
            if args.reasoning_models
            else None
        ),
        embedding_model_name=str(args.embedding_model) if args.embedding_model else None,
        repeat_cv_with_new_seeds=bool(args.repeat_cv_with_new_seeds),
        cv_seed_repeat_count=args.cv_seed_repeat_count,
        distillation_nested_sweep=args.nested_hyperparameter_cv,
        save_reasoning_predictions=args.save_reasoning_predictions,
        candidate_feature_sets=(
            [str(item) for item in args.candidate_feature_sets]
            if args.candidate_feature_sets
            else None
        ),
        model_families=(
            [str(item) for item in args.model_families]
            if args.model_families
            else None
        ),
        output_modes=(
            [str(item) for item in args.output_modes]
            if args.output_modes
            else None
        ),
        run_advanced_models=args.run_advanced_models,
        xgb_calibration_estimators=(
            [int(item) for item in args.xgb_calibration_estimators]
            if args.xgb_calibration_estimators
            else None
        ),
        use_latest_xgb_calibration=args.use_latest_xgb_calibration,
        rf_calibration_min_samples_leaf=(
            [int(item) for item in args.rf_calibration_min_samples_leaf]
            if args.rf_calibration_min_samples_leaf
            else None
        ),
        rf_calibration_max_depth=rf_max_depth_values,
        rf_calibration_max_features=rf_max_features_values,
        use_latest_rf_calibration=args.use_latest_rf_calibration,
        mlp_calibration_hidden_layer_sizes=mlp_hidden_sizes,
        mlp_calibration_alpha=(
            [float(item) for item in args.mlp_calibration_alpha]
            if args.mlp_calibration_alpha
            else None
        ),
        use_latest_mlp_calibration=args.use_latest_mlp_calibration,
        mlp_hidden_layer_sizes=mlp_hidden_layer_sizes_override,
        mlp_alpha=args.mlp_alpha,
        model_testing_per_fit_threads=args.model_testing_per_fit_threads,
        max_parallel_workers=args.max_parallel_workers,
    )


def main(argv: list[str] | None = None) -> None:
    overrides = parse_run_overrides(argv)
    config = load_experiment_config(overrides.config_path)
    run_dir = run_pipeline(config, overrides)
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
