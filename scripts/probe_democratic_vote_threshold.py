from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_experiment_config
from src.pipeline.saved_config_evaluation import (
    _load_feature_sets_for_bundle,
    _predict_combo_on_frame,
)
from src.pipeline.saved_model_configs import list_saved_bundle_dirs, load_bundle_manifest
from src.pipeline.success_protocol import (
    DEMOCRATIC_VOTE_THRESHOLD_START,
    DEMOCRATIC_VOTE_THRESHOLD_STEP,
    DEMOCRATIC_VOTE_THRESHOLD_STOP,
    continuous_indices,
    run_nested_l2_democratic_success_protocol,
)
from src.utils.artifact_io import timestamped_run_dir, write_csv, write_json, write_markdown
from src.utils.paths import RUNS_DIR


TARGET_FAMILY = "v25_policies"
FEATURE_SET_ID = "lambda_policies_plus_sentence_bundle"
MODEL_ID = "ridge"
OUTPUT_MODE = "single_target"
SUCCESS_BRANCH_ID = "llm_engineering_plus_pred_reasoning"
HQ_OVERRIDE_BRANCH = "without_override"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe democratic vote-threshold sensitivity for a fixed transfer config: "
            "lambda bundle predicted reasoning + llm_engineering success branch + no HQ override."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/teacher_student_distillation_v1.json",
        help="Experiment config path.",
    )
    parser.add_argument(
        "--saved-config-bundle-path",
        default="",
        help=(
            "Saved model-config bundle directory (or run id). "
            "Defaults to latest directory under data/saved_model_configs/."
        ),
    )
    parser.add_argument(
        "--combo-id",
        default="",
        help="Optional explicit combo_id. Defaults to first combo matching the fixed filter.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=16,
        help="Number of outer-CV repeat seeds used to build democratic voters.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to a timestamped run dir under tmp/runs/<experiment_id>/.",
    )
    return parser


def _resolve_bundle_path(path_or_id: str) -> Path:
    if str(path_or_id).strip():
        bundle_dir, _ = load_bundle_manifest(path_or_id)
        return bundle_dir
    bundle_dirs = list_saved_bundle_dirs()
    if not bundle_dirs:
        raise RuntimeError("No saved model-config bundles found under data/saved_model_configs.")
    return bundle_dirs[0]


def _select_combo(
    *,
    combos: list[dict[str, object]],
    requested_combo_id: str | None,
) -> dict[str, object]:
    if requested_combo_id:
        for combo in combos:
            if str(combo.get("combo_id", "")) == requested_combo_id:
                return dict(combo)
        raise RuntimeError(f"Requested combo_id '{requested_combo_id}' not found in selected bundle.")
    filtered = [
        dict(combo)
        for combo in combos
        if str(combo.get("target_family", "")) == TARGET_FAMILY
        and str(combo.get("feature_set_id", "")) == FEATURE_SET_ID
        and str(combo.get("model_id", "")) == MODEL_ID
        and str(combo.get("output_mode", "")) == OUTPUT_MODE
        and str(combo.get("task_kind", "")) == "regression"
    ]
    if not filtered:
        raise RuntimeError(
            "No combo matched fixed probe filter: "
            f"target_family={TARGET_FAMILY}, feature_set_id={FEATURE_SET_ID}, "
            f"model_id={MODEL_ID}, output_mode={OUTPUT_MODE}."
        )
    filtered.sort(key=lambda item: str(item.get("combo_id", "")))
    return filtered[0]


def main() -> None:
    args = _build_parser().parse_args()
    if args.repeat_count < 1:
        raise RuntimeError("--repeat-count must be >= 1.")

    config = load_experiment_config(args.config)
    bundle_dir = _resolve_bundle_path(args.saved_config_bundle_path)
    bundle_dir, manifest = load_bundle_manifest(bundle_dir)
    combos = [dict(item) for item in list(manifest.get("combos", []))]
    selected_combo = _select_combo(
        combos=combos,
        requested_combo_id=str(args.combo_id).strip() or None,
    )
    selected_combo["bundle_dir"] = str(bundle_dir)

    feature_sets_by_id, repository_splits, repository_banks = _load_feature_sets_for_bundle(
        config=config,
        combos=[selected_combo],
        required_extra_feature_bank_ids={"llm_engineering"},
        logger=print,
    )
    if "llm_engineering" not in repository_banks:
        raise RuntimeError("llm_engineering feature bank is required for this probe.")

    feature_set = feature_sets_by_id[str(selected_combo["feature_set_id"])]
    pred_train = _predict_combo_on_frame(
        bundle_dir=bundle_dir,
        combo=selected_combo,
        feature_frame=feature_set.public_frame,
    )
    pred_test = _predict_combo_on_frame(
        bundle_dir=bundle_dir,
        combo=selected_combo,
        feature_frame=feature_set.private_frame,
    )
    pred_train.index = feature_set.public_frame["founder_uuid"].astype(str).to_list()
    pred_test.index = feature_set.private_frame["founder_uuid"].astype(str).to_list()

    train_ids_all = [str(founder_id) for founder_id in repository_splits.train_ids]
    test_ids = [str(founder_id) for founder_id in repository_splits.test_ids]
    train_ids = [founder_id for founder_id in train_ids_all if founder_id in set(pred_train.index)]
    if not train_ids:
        raise RuntimeError("No overlapping train founders between repository splits and selected combo predictions.")
    missing_test = [founder_id for founder_id in test_ids if founder_id not in set(pred_test.index)]
    if missing_test:
        raise RuntimeError(
            f"Selected combo is missing held-out prediction rows for {len(missing_test)} founders. "
            f"Examples: {missing_test[:5]}"
        )

    y_train = (
        repository_splits.train_labels.set_index("founder_uuid")
        .reindex(train_ids)["success"]
        .astype(int)
        .to_numpy(dtype=int)
    )
    y_test = (
        repository_splits.test_labels.set_index("founder_uuid")
        .reindex(test_ids)["success"]
        .astype(int)
        .to_numpy(dtype=int)
    )

    llm_bank = repository_banks["llm_engineering"]
    llm_train = (
        llm_bank.public_frame.set_index("founder_uuid")
        .reindex(train_ids)
        .drop(columns=["founder_uuid"], errors="ignore")
    )
    llm_test = (
        llm_bank.private_frame.set_index("founder_uuid")
        .reindex(test_ids)
        .drop(columns=["founder_uuid"], errors="ignore")
    )
    llm_binary = list(getattr(llm_bank, "binary_feature_columns", []))

    train_features = pd.concat([llm_train.reset_index(drop=True), pred_train.reindex(train_ids).reset_index(drop=True)], axis=1)
    test_features = pd.concat([llm_test.reset_index(drop=True), pred_test.reindex(test_ids).reset_index(drop=True)], axis=1)

    if train_features.isna().any(axis=1).any():
        valid_mask = ~train_features.isna().any(axis=1)
        train_features = train_features.loc[valid_mask].reset_index(drop=True)
        y_train = y_train[valid_mask.to_numpy()]
    if len(train_features) == 0:
        raise RuntimeError("Train features are empty after alignment.")
    if test_features.isna().any(axis=1).any():
        raise RuntimeError("Test features contain NaNs after alignment.")

    X_train = train_features.to_numpy(dtype=float)
    X_test = test_features.to_numpy(dtype=float)
    cont_idx = continuous_indices(list(train_features.columns), llm_binary)

    protocol = run_nested_l2_democratic_success_protocol(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        continuous_indices=cont_idx,
        outer_n_splits=config.reproduction.outer_cv.n_splits,
        outer_shuffle=config.reproduction.outer_cv.shuffle,
        outer_random_state=config.reproduction.outer_cv.random_state,
        inner_n_splits=config.reproduction.inner_cv.n_splits,
        inner_shuffle=config.reproduction.inner_cv.shuffle,
        inner_random_state=config.reproduction.inner_cv.random_state,
        c_grid=config.reproduction.logistic_c_grid,
        use_nested=True,
        use_exit_override=False,
        train_exit_counts=None,
        test_exit_counts=None,
        repeat_count=int(args.repeat_count),
        vote_threshold_start=DEMOCRATIC_VOTE_THRESHOLD_START,
        vote_threshold_stop=DEMOCRATIC_VOTE_THRESHOLD_STOP,
        vote_threshold_step=DEMOCRATIC_VOTE_THRESHOLD_STEP,
    )

    sweep_rows = pd.DataFrame(list(protocol["vote_threshold_sweep"])).sort_values("threshold").reset_index(drop=True)
    sweep_rows["vote_threshold_pct"] = (sweep_rows["threshold"].astype(float) * 100.0).round(1)
    best_threshold = float(protocol["threshold"])
    best_train_metrics = dict(protocol["cv_metrics"])
    test_metrics = dict(protocol["test_metrics"] or {})

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = timestamped_run_dir(RUNS_DIR / config.experiment_id, "democratic_vote_threshold_probe")

    write_csv(run_dir / "democratic_vote_threshold_sweep.csv", sweep_rows)
    summary_payload = {
        "probe_config": {
            "target_family": TARGET_FAMILY,
            "reasoning_feature_set_id": FEATURE_SET_ID,
            "reasoning_model_id": MODEL_ID,
            "reasoning_output_mode": OUTPUT_MODE,
            "success_branch_id": SUCCESS_BRANCH_ID,
            "hq_override_branch": HQ_OVERRIDE_BRANCH,
            "bundle_dir": str(bundle_dir),
            "combo_id": str(selected_combo["combo_id"]),
            "repeat_count": int(args.repeat_count),
            "vote_threshold_grid": {
                "start": DEMOCRATIC_VOTE_THRESHOLD_START,
                "stop": DEMOCRATIC_VOTE_THRESHOLD_STOP,
                "step": DEMOCRATIC_VOTE_THRESHOLD_STEP,
            },
            "voter_count": int(protocol["voter_count"]),
        },
        "best_threshold": best_threshold,
        "best_train_metrics": best_train_metrics,
        "test_metrics_at_best_threshold": test_metrics,
    }
    write_json(run_dir / "democratic_probe_summary.json", summary_payload)

    report_lines = [
        "# Democratic Vote Threshold Probe",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Bundle dir: `{bundle_dir}`",
        f"- Combo id: `{selected_combo['combo_id']}`",
        f"- Success branch: `{SUCCESS_BRANCH_ID}`",
        f"- HQ override branch: `{HQ_OVERRIDE_BRANCH}`",
        f"- Repeat count: `{int(args.repeat_count)}`",
        f"- Voter count: `{int(protocol['voter_count'])}`",
        "",
        "## Best Train Threshold",
        "",
        f"- Threshold: `{best_threshold:.2f}`",
        (
            f"- Train metrics at best threshold: "
            f"F0.5={float(best_train_metrics.get('f0_5', float('nan'))):.4f}, "
            f"ROC AUC={float(best_train_metrics.get('roc_auc', float('nan'))):.4f}, "
            f"PR AUC={float(best_train_metrics.get('pr_auc', float('nan'))):.4f}, "
            f"Precision={float(best_train_metrics.get('precision', float('nan'))):.4f}, "
            f"Recall={float(best_train_metrics.get('recall', float('nan'))):.4f}"
        ),
        "",
        "## Held-out Test @ Best Threshold",
        "",
        (
            f"- Test metrics: "
            f"F0.5={float(test_metrics.get('f0_5', float('nan'))):.4f}, "
            f"ROC AUC={float(test_metrics.get('roc_auc', float('nan'))):.4f}, "
            f"PR AUC={float(test_metrics.get('pr_auc', float('nan'))):.4f}, "
            f"Precision={float(test_metrics.get('precision', float('nan'))):.4f}, "
            f"Recall={float(test_metrics.get('recall', float('nan'))):.4f}"
        ),
        "",
        "## Artifacts",
        "",
        "- `democratic_vote_threshold_sweep.csv`",
        "- `democratic_probe_summary.json`",
    ]
    write_markdown(run_dir / "democratic_probe_report.md", "\n".join(report_lines))
    print(f"Probe complete. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
