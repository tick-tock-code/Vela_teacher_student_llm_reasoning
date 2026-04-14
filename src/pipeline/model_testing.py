from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import pandas as pd

from src.pipeline.config import ExperimentConfig
from src.pipeline.run_options import MODEL_FAMILY_TO_MODEL_ID, RunOverrides, resolve_run_options
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.paths import RUNS_DIR


Logger = Callable[[str], None]


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _load_reasoning_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "reasoning_metrics.csv"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing reasoning metrics artifact: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    return frame[frame["split_id"] == "oof_overall"].copy()


def _repeat_seeds(config: ExperimentConfig, repeat_count: int) -> list[int]:
    base = config.distillation_cv.random_state
    return [base + (index * 10_000) for index in range(repeat_count)]


def _resolve_model_ids(
    config: ExperimentConfig,
    *,
    task_kind: str,
    model_families: list[str],
) -> list[str]:
    available_by_id = {spec.model_id: spec for spec in config.distillation_models}
    mapped_ids = [MODEL_FAMILY_TO_MODEL_ID[task_kind][family] for family in model_families]
    missing = [model_id for model_id in mapped_ids if model_id not in available_by_id]
    if missing:
        raise RuntimeError(
            f"model_testing_mode requested model ids not present in distillation_models: {missing}"
        )
    return mapped_ids


def _group_repeat_metrics(
    metrics_frame: pd.DataFrame,
    *,
    repeat_index: int,
    repeat_seed: int,
    target_family: str,
    stage: str,
) -> pd.DataFrame:
    grouped = (
        metrics_frame.groupby(["feature_set_id", "model_id"], as_index=False)
        .mean(numeric_only=True)
    )
    grouped["repeat_index"] = repeat_index
    grouped["repeat_seed"] = repeat_seed
    grouped["target_family"] = target_family
    grouped["stage"] = stage
    return grouped


def _aggregate_screening_metrics(
    repeat_model_metrics: pd.DataFrame,
    *,
    task_kind: str,
    score_delta: float,
    max_recommended: int,
) -> pd.DataFrame:
    if repeat_model_metrics.empty:
        return pd.DataFrame()
    for column in ("r2", "rmse", "mae", "f0_5", "roc_auc", "pr_auc"):
        if column not in repeat_model_metrics.columns:
            repeat_model_metrics[column] = float("nan")

    primary_column = "r2" if task_kind == "regression" else "f0_5"

    per_repeat_feature = (
        repeat_model_metrics.groupby(["target_family", "feature_set_id", "repeat_index"], as_index=False)
        .mean(numeric_only=True)
    )
    screening = (
        per_repeat_feature.groupby(["target_family", "feature_set_id"], as_index=False)
        .agg(
            primary_mean=(primary_column, "mean"),
            primary_std=(primary_column, "std"),
            r2_mean=("r2", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            f0_5_mean=("f0_5", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
        )
    )
    screening["primary_std"] = screening["primary_std"].fillna(0.0)
    screening["screen_score"] = screening["primary_mean"] - (0.5 * screening["primary_std"])
    screening["primary_metric"] = primary_column
    screening["recommended_take_forward"] = False
    screening["rank"] = 0

    output_rows: list[pd.DataFrame] = []
    for family_id, family_frame in screening.groupby("target_family", sort=False):
        ranked = family_frame.sort_values("screen_score", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        best_score = float(ranked.iloc[0]["screen_score"])
        threshold = best_score - float(score_delta)
        mask = ranked["screen_score"] >= threshold
        recommended_indices = ranked.index[mask].tolist()
        if not recommended_indices:
            recommended_indices = [0]
        recommended_indices = recommended_indices[:max_recommended]
        ranked.loc[recommended_indices, "recommended_take_forward"] = True
        output_rows.append(ranked)
        _ = family_id

    output = pd.concat(output_rows, ignore_index=True)
    columns = [
        "target_family",
        "feature_set_id",
        "rank",
        "primary_metric",
        "primary_mean",
        "primary_std",
        "screen_score",
        "recommended_take_forward",
        "r2_mean",
        "rmse_mean",
        "mae_mean",
        "f0_5_mean",
        "roc_auc_mean",
        "pr_auc_mean",
    ]
    return output[columns].sort_values(["target_family", "rank"]).reset_index(drop=True)


def _aggregate_model_results(
    repeat_model_metrics: pd.DataFrame,
    *,
    task_kind: str,
) -> pd.DataFrame:
    if repeat_model_metrics.empty:
        return pd.DataFrame()
    for column in ("r2", "rmse", "mae", "f0_5", "roc_auc", "pr_auc"):
        if column not in repeat_model_metrics.columns:
            repeat_model_metrics[column] = float("nan")
    primary_column = "r2" if task_kind == "regression" else "f0_5"
    grouped = (
        repeat_model_metrics.groupby(["target_family", "feature_set_id", "model_id"], as_index=False)
        .agg(
            primary_mean=(primary_column, "mean"),
            primary_std=(primary_column, "std"),
            r2_mean=("r2", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            f0_5_mean=("f0_5", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
        )
    )
    grouped["primary_std"] = grouped["primary_std"].fillna(0.0)
    grouped["primary_metric"] = primary_column
    columns = [
        "target_family",
        "feature_set_id",
        "model_id",
        "primary_metric",
        "primary_mean",
        "primary_std",
        "r2_mean",
        "rmse_mean",
        "mae_mean",
        "f0_5_mean",
        "roc_auc_mean",
        "pr_auc_mean",
    ]
    return grouped[columns].sort_values(
        ["target_family", "feature_set_id", "primary_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _render_screening_markdown(
    screening: pd.DataFrame,
    *,
    repeat_count: int,
    score_delta: float,
    max_recommended: int,
) -> str:
    lines = [
        "# Feature-Set Screening Report",
        "",
        f"- Repeats: {repeat_count}",
        "- Stage A models: `linear_l2` + `xgb1` only",
        "- Held-out features/targets: not used",
        f"- Recommendation rule: top score + any within `best - {score_delta}` (max {max_recommended}).",
        "",
    ]
    if screening.empty:
        lines.append("No screening results were produced.")
        return "\n".join(lines)

    for target_family, frame in screening.groupby("target_family", sort=False):
        lines.extend(
            [
                f"## {target_family}",
                "",
                "| rank | feature_set_id | primary_mean | primary_std | screen_score | recommended |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in frame.sort_values("rank").itertuples(index=False):
            lines.append(
                f"| {row.rank} | {row.feature_set_id} | {row.primary_mean:.4f} | "
                f"{row.primary_std:.4f} | {row.screen_score:.4f} | {bool(row.recommended_take_forward)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _render_model_testing_markdown(
    results: pd.DataFrame,
    *,
    repeat_count: int,
) -> str:
    lines = [
        "# Model Testing Report",
        "",
        f"- Repeats: {repeat_count}",
        "- This report compares shortlisted feature sets by model family.",
        "",
    ]
    if results.empty:
        lines.append("Advanced model stage was skipped or produced no rows.")
        return "\n".join(lines)

    for (target_family, feature_set_id), frame in results.groupby(
        ["target_family", "feature_set_id"], sort=False
    ):
        lines.extend(
            [
                f"## {target_family} | {feature_set_id}",
                "",
                "| model_id | primary_mean | primary_std |",
                "|---|---:|---:|",
            ]
        )
        for row in frame.sort_values("primary_mean", ascending=False).itertuples(index=False):
            lines.append(
                f"| {row.model_id} | {row.primary_mean:.4f} | {row.primary_std:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _resolve_family_sequence(
    *,
    overrides: RunOverrides,
    resolved_target_family: str,
) -> list[str]:
    if overrides.target_family == "v25_and_taste":
        return ["v25_policies", "taste_policies"]
    return [resolved_target_family]


def _run_stage(
    config: ExperimentConfig,
    *,
    base_overrides: RunOverrides,
    family_id: str,
    feature_set_ids: list[str],
    model_ids: list[str],
    repeat_count: int,
    force_rebuild_intermediary_features: bool,
    nested_sweep: bool,
    logger: Logger | None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    from src.pipeline.distillation import run_reasoning_distillation_mode

    seeds = _repeat_seeds(config, repeat_count)
    stage_rows: list[pd.DataFrame] = []
    child_runs: list[dict[str, object]] = []
    for repeat_index, seed in enumerate(seeds):
        if repeat_index > 0:
            _log(logger, f"{family_id}: repeat {repeat_index + 1}/{repeat_count} (seed={seed})")
        config_seeded = replace(
            config,
            distillation_cv=replace(config.distillation_cv, random_state=seed),
            reproduction=replace(
                config.reproduction,
                outer_cv=replace(config.reproduction.outer_cv, random_state=seed),
                inner_cv=replace(config.reproduction.inner_cv, random_state=seed),
            ),
        )
        run_overrides = replace(
            base_overrides,
            run_mode="reasoning_distillation_mode",
            target_family=family_id,
            heldout_evaluation=False,
            repeat_cv_with_new_seeds=False,
            cv_seed_repeat_count=1,
            distillation_nested_sweep=nested_sweep,
            save_reasoning_predictions=False,
            force_rebuild_intermediary_features=(
                force_rebuild_intermediary_features if repeat_index == 0 else False
            ),
            reasoning_models=model_ids,
            candidate_feature_sets=feature_set_ids,
        )
        run_dir = run_reasoning_distillation_mode(config_seeded, run_overrides, logger=logger)
        child_runs.append(
            {
                "target_family": family_id,
                "repeat_index": repeat_index,
                "repeat_seed": seed,
                "run_dir": str(run_dir),
            }
        )
        metrics = _load_reasoning_metrics(run_dir)
        stage_rows.append(
            _group_repeat_metrics(
                metrics,
                repeat_index=repeat_index,
                repeat_seed=seed,
                target_family=family_id,
                stage="screening",
            )
        )
    return pd.concat(stage_rows, ignore_index=True), child_runs


def run_model_testing_mode(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
    *,
    logger: Logger | None = None,
) -> Path:
    overrides_use = overrides or RunOverrides()
    if overrides_use.heldout_evaluation is True:
        raise RuntimeError("model_testing_mode is training-only. heldout_evaluation must be false.")

    resolved = resolve_run_options(config, replace(overrides_use, run_mode="model_testing_mode"))
    run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "model_testing")
    write_json(run_dir / "resolved_config.json", asdict(config))
    write_json(run_dir / "resolved_run_options.json", asdict(resolved))

    feature_set_ids = [spec.feature_set_id for spec in resolved.distillation_feature_sets]
    if not feature_set_ids:
        raise RuntimeError("No candidate feature sets are available for model_testing_mode.")

    repeat_count = resolved.cv_seed_repeat_count if resolved.repeat_cv_with_new_seeds else 1
    family_sequence = _resolve_family_sequence(
        overrides=overrides_use,
        resolved_target_family=resolved.target_family.family_id,
    )
    family_map = {spec.family_id: spec for spec in config.target_families}

    stage_a_rows: list[pd.DataFrame] = []
    stage_a_child_runs: list[dict[str, object]] = []
    for family_id in family_sequence:
        if family_id not in family_map:
            raise RuntimeError(f"Unknown target family '{family_id}'.")
        task_kind = family_map[family_id].task_kind
        stage_a_model_ids = _resolve_model_ids(
            config,
            task_kind=task_kind,
            model_families=["linear_l2", "xgb1"],
        )
        _log(logger, f"Stage A screening for '{family_id}' with feature sets: {feature_set_ids}.")
        stage_metrics, child_runs = _run_stage(
            config,
            base_overrides=overrides_use,
            family_id=family_id,
            feature_set_ids=feature_set_ids,
            model_ids=stage_a_model_ids,
            repeat_count=repeat_count,
            force_rebuild_intermediary_features=resolved.force_rebuild_intermediary_features,
            nested_sweep=resolved.distillation_nested_sweep,
            logger=logger,
        )
        stage_a_rows.append(stage_metrics)
        stage_a_child_runs.extend(child_runs)

    stage_a_repeat_metrics = pd.concat(stage_a_rows, ignore_index=True) if stage_a_rows else pd.DataFrame()
    write_csv(run_dir / "feature_set_screening_repeat_metrics.csv", stage_a_repeat_metrics)
    write_json(run_dir / "feature_set_screening_child_runs.json", stage_a_child_runs)

    screening_rows: list[pd.DataFrame] = []
    recommended_sets: dict[str, list[str]] = {}
    for family_id in family_sequence:
        task_kind = family_map[family_id].task_kind
        family_metrics = stage_a_repeat_metrics[stage_a_repeat_metrics["target_family"] == family_id].copy()
        screening_family = _aggregate_screening_metrics(
            family_metrics,
            task_kind=task_kind,
            score_delta=config.model_testing.screening_score_delta,
            max_recommended=config.model_testing.max_recommended_feature_sets,
        )
        screening_rows.append(screening_family)
        recommended_sets[family_id] = (
            screening_family.loc[
                screening_family["recommended_take_forward"],
                "feature_set_id",
            ]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )

    screening_frame = pd.concat(screening_rows, ignore_index=True) if screening_rows else pd.DataFrame()
    write_csv(run_dir / "feature_set_screening.csv", screening_frame)
    write_markdown(
        run_dir / "feature_set_screening_report.md",
        _render_screening_markdown(
            screening_frame,
            repeat_count=repeat_count,
            score_delta=config.model_testing.screening_score_delta,
            max_recommended=config.model_testing.max_recommended_feature_sets,
        ),
    )

    model_testing_results = pd.DataFrame()
    model_testing_runs: list[dict[str, object]] = []
    if resolved.run_advanced_models:
        stage_b_rows: list[pd.DataFrame] = []
        for family_id in family_sequence:
            shortlisted = recommended_sets.get(family_id, [])
            if not shortlisted:
                continue
            task_kind = family_map[family_id].task_kind
            model_ids = _resolve_model_ids(
                config,
                task_kind=task_kind,
                model_families=resolved.model_families,
            )
            _log(logger, f"Stage B model testing for '{family_id}' on shortlisted sets: {shortlisted}.")
            stage_metrics, child_runs = _run_stage(
                config,
                base_overrides=overrides_use,
                family_id=family_id,
                feature_set_ids=shortlisted,
                model_ids=model_ids,
                repeat_count=repeat_count,
                force_rebuild_intermediary_features=False,
                nested_sweep=resolved.distillation_nested_sweep,
                logger=logger,
            )
            stage_metrics["stage"] = "advanced"
            stage_b_rows.append(stage_metrics)
            model_testing_runs.extend(child_runs)

        if stage_b_rows:
            stage_b_repeat_metrics = pd.concat(stage_b_rows, ignore_index=True)
            write_csv(run_dir / "model_testing_repeat_metrics.csv", stage_b_repeat_metrics)
            write_json(run_dir / "model_testing_child_runs.json", model_testing_runs)

            result_rows: list[pd.DataFrame] = []
            for family_id in family_sequence:
                family_task_kind = family_map[family_id].task_kind
                family_frame = stage_b_repeat_metrics[
                    stage_b_repeat_metrics["target_family"] == family_id
                ].copy()
                result_rows.append(
                    _aggregate_model_results(family_frame, task_kind=family_task_kind)
                )
            model_testing_results = pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()
    write_csv(run_dir / "model_testing_results.csv", model_testing_results)
    write_markdown(
        run_dir / "model_testing_report.md",
        _render_model_testing_markdown(
            model_testing_results,
            repeat_count=repeat_count,
        ),
    )

    summary_lines = [
        "# Model Testing Summary",
        "",
        f"- Candidate feature sets: {len(feature_set_ids)}",
        f"- Repeats: {repeat_count}",
        "- Stage A uses only Linear L2 + XGB1.",
        f"- Stage B enabled: {resolved.run_advanced_models}",
        "- Held-out/test features or targets are not used in this mode.",
    ]
    write_markdown(run_dir / "run_summary.md", "\n".join(summary_lines))
    _log(logger, f"Model testing run complete. Artifacts written to {run_dir}.")
    return run_dir
