from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data.input_features import build_input_features
from src.data.raw_datasets import load_raw_datasets
from src.data.splits import build_public_cv_splits
from src.data.targets import load_reasoning_target_bank, target_manifest_payload
from src.downstream.routes import (
    evaluate_public_downstream_routes,
    extract_predicted_reasoning_features,
    predict_private_downstream_routes,
)
from src.evaluation.metrics import regression_metrics
from src.pipeline.config import ExperimentConfig
from src.student.reasoning_regression import (
    fit_reasoning_regressors_full,
    predict_reasoning_regressors_full,
    train_reasoning_regressors_oof,
)
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.paths import RUNS_DIR


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
        preview = missing[:5]
        raise RuntimeError(
            f"{right_name} is missing {len(missing)} ids required by {left_name}. "
            f"Examples: {preview}"
        )
    return merged


def _evaluate_private_reasoning_predictions(
    private_targets: pd.DataFrame,
    private_predictions: pd.DataFrame,
    *,
    target_columns: list[str],
    model_specs: list,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_column in target_columns:
        for model_spec in model_specs:
            pred_column = f"{target_column}__{model_spec.model_id}"
            metrics = regression_metrics(
                private_targets[target_column].to_numpy(dtype=float),
                private_predictions[pred_column].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "target_id": target_column,
                    "model_id": model_spec.model_id,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_distillation_experiment(config: ExperimentConfig) -> Path:
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "distillation")
    write_json(run_dir / "resolved_config.json", asdict(config))

    raw_datasets = load_raw_datasets(
        Path(config.datasets.public_train_csv),
        Path(config.datasets.private_test_csv),
    )
    input_features = build_input_features(
        public_raw=raw_datasets.public_frame,
        private_raw=raw_datasets.private_frame,
        spec=config.input_features,
    )
    write_json(run_dir / "input_feature_manifest.json", input_features.manifest)
    reasoning_targets = load_reasoning_target_bank(
        config.reasoning_target_bank,
        selected_targets=[item.target_id for item in config.reasoning_targets],
    )
    write_json(run_dir / "reasoning_target_manifest.json", target_manifest_payload(reasoning_targets))

    public_feature_rows = _require_full_overlap(
        raw_datasets.public_frame[["founder_uuid", "success"]],
        input_features.public_frame,
        on="founder_uuid",
        left_name="public raw dataset",
        right_name="public input features",
    )
    private_feature_rows = _require_full_overlap(
        raw_datasets.private_frame[["founder_uuid"]],
        input_features.private_frame,
        on="founder_uuid",
        left_name="private raw dataset",
        right_name="private input features",
    )

    public_model_rows = _require_full_overlap(
        public_feature_rows,
        reasoning_targets.train_frame,
        on="founder_uuid",
        left_name="public input features",
        right_name="public reasoning target bank",
    )

    feature_columns = input_features.feature_columns
    target_columns = reasoning_targets.train_target_columns
    splits = build_public_cv_splits(
        public_model_rows["success"],
        n_splits=config.cv.n_splits,
        shuffle=config.cv.shuffle,
        random_state=config.cv.random_state,
    )

    reasoning_outputs = train_reasoning_regressors_oof(
        public_model_rows[feature_columns],
        public_model_rows[target_columns],
        founder_ids=public_model_rows["founder_uuid"],
        target_columns=target_columns,
        model_specs=config.reasoning_models,
        splits=splits,
        random_state=config.cv.random_state,
        scale_min=reasoning_targets.scale_min,
        scale_max=reasoning_targets.scale_max,
    )
    write_csv(run_dir / "reasoning_oof_predictions.csv", reasoning_outputs.oof_predictions)
    write_csv(run_dir / "reasoning_metrics.csv", reasoning_outputs.metrics)

    predicted_reasoning_by_model = {
        model_spec.model_id: extract_predicted_reasoning_features(
            reasoning_outputs.oof_predictions,
            model_id=model_spec.model_id,
        )[target_columns]
        for model_spec in config.reasoning_models
    }

    downstream_public = evaluate_public_downstream_routes(
        base_features=public_model_rows[feature_columns],
        labels=public_model_rows["success"],
        true_reasoning=public_model_rows[target_columns],
        predicted_reasoning_by_model=predicted_reasoning_by_model,
        model_specs=config.downstream_models,
        splits=splits,
        random_state=config.cv.random_state,
    )
    write_csv(run_dir / "downstream_public_summary.csv", downstream_public.summary)
    write_csv(run_dir / "downstream_public_fold_metrics.csv", downstream_public.fold_metrics)

    promotion_status = {
        "mode": config.promotion.mode,
        "approved": config.promotion.approved,
        "private_predictions_written": False,
    }

    if not config.promotion.approved:
        write_json(run_dir / "promotion_status.json", promotion_status)
        write_markdown(
            run_dir / "run_summary.md",
            "\n".join(
                [
                    "# Run Summary",
                    "",
                    "- Public-stage outputs were written successfully.",
                    "- Promotion is manual and currently blocked.",
                    "- Private reasoning and downstream prediction artifacts were not generated.",
                ]
            ),
        )
        return run_dir

    fitted_models = fit_reasoning_regressors_full(
        public_model_rows[feature_columns],
        public_model_rows[target_columns],
        target_columns=target_columns,
        model_specs=config.reasoning_models,
        random_state=config.cv.random_state,
    )
    private_reasoning_predictions = predict_reasoning_regressors_full(
        private_feature_rows[feature_columns],
        founder_ids=private_feature_rows["founder_uuid"],
        target_columns=target_columns,
        model_specs=config.reasoning_models,
        fitted_models=fitted_models,
        scale_min=reasoning_targets.scale_min,
        scale_max=reasoning_targets.scale_max,
    )
    write_csv(run_dir / "reasoning_private_predictions.csv", private_reasoning_predictions)

    private_reasoning_metrics = None
    if reasoning_targets.test_frame is not None:
        private_targets = _require_full_overlap(
            private_feature_rows[["founder_uuid"]],
            reasoning_targets.test_frame,
            on="founder_uuid",
            left_name="private input features",
            right_name="held-out reasoning target bank",
        )
        private_reasoning_metrics = _evaluate_private_reasoning_predictions(
            private_targets,
            private_reasoning_predictions,
            target_columns=target_columns,
            model_specs=config.reasoning_models,
        )
        write_csv(run_dir / "reasoning_private_metrics.csv", private_reasoning_metrics)

    predicted_private_reasoning_by_model = {
        model_spec.model_id: extract_predicted_reasoning_features(
            private_reasoning_predictions,
            model_id=model_spec.model_id,
        )[target_columns]
        for model_spec in config.reasoning_models
    }
    true_reasoning_private = None
    if reasoning_targets.test_frame is not None:
        true_reasoning_private = _require_full_overlap(
            private_feature_rows[["founder_uuid"]],
            reasoning_targets.test_frame,
            on="founder_uuid",
            left_name="private input features",
            right_name="held-out reasoning target bank",
        )[target_columns]

    downstream_private_predictions = predict_private_downstream_routes(
        public_base_features=public_model_rows[feature_columns],
        public_labels=public_model_rows["success"],
        private_founder_ids=private_feature_rows["founder_uuid"],
        private_base_features=private_feature_rows[feature_columns],
        true_reasoning_public=public_model_rows[target_columns],
        true_reasoning_private=true_reasoning_private,
        predicted_reasoning_public_by_model=predicted_reasoning_by_model,
        predicted_reasoning_private_by_model=predicted_private_reasoning_by_model,
        model_specs=config.downstream_models,
        random_state=config.cv.random_state,
    )
    write_csv(run_dir / "downstream_private_predictions.csv", downstream_private_predictions)

    promotion_status["private_predictions_written"] = True
    write_json(run_dir / "promotion_status.json", promotion_status)

    summary_lines = [
        "# Run Summary",
        "",
        f"- Feature count: {len(feature_columns)}",
        f"- Configured reasoning targets: {len(target_columns)}",
        f"- Public reasoning metric rows: {len(reasoning_outputs.metrics)}",
        f"- Public downstream summary rows: {len(downstream_public.summary)}",
        "- Private reasoning predictions were generated.",
    ]
    if private_reasoning_metrics is not None:
        summary_lines.append("- Private reasoning agreement metrics were generated.")
    else:
        summary_lines.append("- Private reasoning agreement metrics were skipped because no private target tables were configured.")
    write_markdown(run_dir / "run_summary.md", "\n".join(summary_lines))
    return run_dir
