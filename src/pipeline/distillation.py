from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data.raw_datasets import load_raw_datasets
from src.data.splits import build_stratified_reasoning_cv_splits
from src.data.targets import load_reasoning_target_bank, target_manifest_payload
from src.evaluation.metrics import regression_metrics
from src.intermediary_features.registry import assemble_feature_sets, prepare_intermediary_banks
from src.pipeline.config import ExperimentConfig
from src.pipeline.run_options import RunOverrides, resolve_run_options
from src.student.reasoning_regression import (
    fit_reasoning_regressors_full,
    predict_reasoning_regressors_full,
    train_reasoning_regressors_oof,
)
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.paths import RUNS_DIR


Logger = Callable[[str], None]


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


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
            f"{right_name} is missing {len(missing)} ids required by {left_name}. Examples: {preview}"
        )
    return merged


def _prefix_prediction_columns(frame: pd.DataFrame, *, feature_set_id: str) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [
        column if column == "founder_uuid" else f"{feature_set_id}__{column}"
        for column in renamed.columns
    ]
    return renamed


def _merge_prediction_tables(
    current: pd.DataFrame | None,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    if current is None:
        return incoming
    return current.merge(incoming, on="founder_uuid", how="outer", validate="one_to_one")


def _evaluate_heldout_predictions(
    *,
    feature_set_id: str,
    heldout_targets: pd.DataFrame,
    heldout_predictions: pd.DataFrame,
    target_columns: list[str],
    model_specs: list,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_column in target_columns:
        for model_spec in model_specs:
            pred_column = f"{target_column}__{model_spec.model_id}"
            metrics = regression_metrics(
                heldout_targets[target_column].to_numpy(dtype=float),
                heldout_predictions[pred_column].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "feature_set_id": feature_set_id,
                    "target_id": target_column,
                    "model_id": model_spec.model_id,
                    "split_id": "heldout_overall",
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_distillation_experiment(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    resolved_run = resolve_run_options(config, overrides)
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "reasoning_reconstruction")

    write_json(run_dir / "resolved_config.json", asdict(config))
    write_json(run_dir / "resolved_run_options.json", asdict(resolved_run))

    _log(logger, "Loading VCBench public and held-out datasets.")
    raw_datasets = load_raw_datasets(
        Path(config.datasets.public_train_csv),
        Path(config.datasets.private_test_csv),
    )

    _log(logger, "Loading reasoning target bank.")
    reasoning_targets = load_reasoning_target_bank(
        config.reasoning_target_bank,
        selected_targets=[item.target_id for item in resolved_run.reasoning_targets],
    )
    write_json(run_dir / "reasoning_target_manifest.json", target_manifest_payload(reasoning_targets))

    _log(logger, "Preparing intermediary feature banks.")
    banks_by_id = prepare_intermediary_banks(
        public_raw=raw_datasets.public_frame,
        private_raw=raw_datasets.private_frame,
        feature_specs=resolved_run.intermediary_features,
        force_rebuild=resolved_run.force_rebuild_intermediary_features,
        logger=logger,
    )
    write_json(
        run_dir / "intermediary_feature_manifests.json",
        {feature_id: bank.manifest for feature_id, bank in banks_by_id.items()},
    )

    feature_sets = assemble_feature_sets(
        public_founder_ids=raw_datasets.public_frame["founder_uuid"],
        private_founder_ids=raw_datasets.private_frame["founder_uuid"],
        banks_by_id=banks_by_id,
        feature_sets=resolved_run.feature_sets,
    )
    write_json(
        run_dir / "feature_set_manifest.json",
        {feature_set.feature_set_id: feature_set.manifest for feature_set in feature_sets},
    )

    public_target_rows = _require_full_overlap(
        raw_datasets.public_frame[["founder_uuid"]],
        reasoning_targets.train_frame,
        on="founder_uuid",
        left_name="public raw dataset",
        right_name="public reasoning target bank",
    )

    heldout_target_rows = None
    if resolved_run.run_heldout_reasoning_predictions and reasoning_targets.test_frame is not None:
        heldout_target_rows = _require_full_overlap(
            raw_datasets.private_frame[["founder_uuid"]],
            reasoning_targets.test_frame,
            on="founder_uuid",
            left_name="held-out raw dataset",
            right_name="held-out reasoning target bank",
        )

    target_columns = reasoning_targets.train_target_columns
    splits = build_stratified_reasoning_cv_splits(
        public_target_rows[target_columns],
        n_splits=config.cv.n_splits,
        shuffle=config.cv.shuffle,
        random_state=config.cv.random_state,
    )

    combined_oof_predictions: pd.DataFrame | None = None
    combined_heldout_predictions: pd.DataFrame | None = None
    public_metric_frames: list[pd.DataFrame] = []
    heldout_metric_frames: list[pd.DataFrame] = []

    for feature_set in feature_sets:
        _log(logger, f"Training reasoning models for feature set '{feature_set.feature_set_id}'.")
        public_feature_rows = _require_full_overlap(
            public_target_rows[["founder_uuid"]],
            feature_set.public_frame,
            on="founder_uuid",
            left_name=f"public reasoning targets for '{feature_set.feature_set_id}'",
            right_name=f"public feature set '{feature_set.feature_set_id}'",
        )

        reasoning_outputs = train_reasoning_regressors_oof(
            public_feature_rows[feature_set.feature_columns],
            public_target_rows[target_columns],
            founder_ids=public_feature_rows["founder_uuid"],
            target_columns=target_columns,
            model_specs=resolved_run.reasoning_models,
            splits=splits,
            random_state=config.cv.random_state,
            scale_min=reasoning_targets.scale_min,
            scale_max=reasoning_targets.scale_max,
        )
        public_metrics = reasoning_outputs.metrics.copy()
        public_metrics.insert(0, "feature_set_id", feature_set.feature_set_id)
        public_metric_frames.append(public_metrics)
        combined_oof_predictions = _merge_prediction_tables(
            combined_oof_predictions,
            _prefix_prediction_columns(
                reasoning_outputs.oof_predictions,
                feature_set_id=feature_set.feature_set_id,
            ),
        )

        if resolved_run.run_heldout_reasoning_predictions:
            heldout_feature_rows = _require_full_overlap(
                raw_datasets.private_frame[["founder_uuid"]],
                feature_set.private_frame,
                on="founder_uuid",
                left_name=f"held-out ids for '{feature_set.feature_set_id}'",
                right_name=f"held-out feature set '{feature_set.feature_set_id}'",
            )
            fitted_models = fit_reasoning_regressors_full(
                public_feature_rows[feature_set.feature_columns],
                public_target_rows[target_columns],
                target_columns=target_columns,
                model_specs=resolved_run.reasoning_models,
                random_state=config.cv.random_state,
            )
            heldout_predictions = predict_reasoning_regressors_full(
                heldout_feature_rows[feature_set.feature_columns],
                founder_ids=heldout_feature_rows["founder_uuid"],
                target_columns=target_columns,
                model_specs=resolved_run.reasoning_models,
                fitted_models=fitted_models,
                scale_min=reasoning_targets.scale_min,
                scale_max=reasoning_targets.scale_max,
            )
            combined_heldout_predictions = _merge_prediction_tables(
                combined_heldout_predictions,
                _prefix_prediction_columns(
                    heldout_predictions,
                    feature_set_id=feature_set.feature_set_id,
                ),
            )

            if heldout_target_rows is not None:
                heldout_metric_frames.append(
                    _evaluate_heldout_predictions(
                        feature_set_id=feature_set.feature_set_id,
                        heldout_targets=heldout_target_rows,
                        heldout_predictions=heldout_predictions,
                        target_columns=target_columns,
                        model_specs=resolved_run.reasoning_models,
                    )
                )

    write_csv(
        run_dir / "reasoning_oof_predictions.csv",
        combined_oof_predictions if combined_oof_predictions is not None else pd.DataFrame({"founder_uuid": []}),
    )
    write_csv(
        run_dir / "reasoning_metrics.csv",
        pd.concat(public_metric_frames, ignore_index=True) if public_metric_frames else pd.DataFrame(),
    )
    if resolved_run.run_heldout_reasoning_predictions:
        write_csv(
            run_dir / "reasoning_heldout_predictions.csv",
            combined_heldout_predictions
            if combined_heldout_predictions is not None
            else pd.DataFrame({"founder_uuid": []}),
        )
        if heldout_metric_frames:
            write_csv(
                run_dir / "reasoning_heldout_metrics.csv",
                pd.concat(heldout_metric_frames, ignore_index=True),
            )

    summary_lines = [
        "# Run Summary",
        "",
        f"- Selected reasoning targets: {len(target_columns)}",
        f"- Intermediary banks prepared: {len(banks_by_id)}",
        f"- Feature-set comparisons run: {len(feature_sets)}",
        f"- Reasoning models run per target: {len(resolved_run.reasoning_models)}",
        "- Public CV strategy: stratified on quantile buckets of the row-wise mean selected target score.",
        f"- OOF prediction columns written: {0 if combined_oof_predictions is None else len(combined_oof_predictions.columns) - 1}",
        f"- Held-out prediction columns written: {0 if combined_heldout_predictions is None else len(combined_heldout_predictions.columns) - 1}",
    ]
    if not resolved_run.run_heldout_reasoning_predictions:
        summary_lines.append("- Held-out reasoning prediction was skipped because it was not requested.")
    write_markdown(run_dir / "run_summary.md", "\n".join(summary_lines))
    _log(logger, f"Run complete. Artifacts written to {run_dir}.")
    return run_dir


def run_reasoning_reconstruction(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    return run_distillation_experiment(config, overrides, logger=logger)
