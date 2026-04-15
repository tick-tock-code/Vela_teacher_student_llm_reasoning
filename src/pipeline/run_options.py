from __future__ import annotations

from dataclasses import dataclass, replace

from src.pipeline.config import (
    DistillationModelSpec,
    ExperimentConfig,
    FeatureSetSpec,
    IntermediaryFeatureSpec,
    RepositoryFeatureBankSpec,
    SUPPORTED_MODEL_TESTING_FAMILIES,
    TargetFamilySpec,
)
from src.utils.parallel import resolve_max_parallel_workers


DEFAULT_CONFIG_PATH = "experiments/teacher_student_distillation_v1.json"
SUPPORTED_OUTPUT_MODES = {"single_target", "multi_output"}


MODEL_FAMILY_TO_MODEL_ID: dict[str, dict[str, str]] = {
    "regression": {
        "linear_l2": "ridge",
        "xgb1": "xgb1_regressor",
        "mlp": "mlp_regressor",
        "elasticnet": "elasticnet_regressor",
        "randomforest": "randomforest_regressor",
    },
    "classification": {
        "linear_l2": "logreg_classifier",
        "xgb1": "xgb1_classifier",
        "mlp": "mlp_classifier",
        "elasticnet": "elasticnet_logreg_classifier",
        "randomforest": "randomforest_classifier",
    },
}


@dataclass(frozen=True)
class RunOverrides:
    config_path: str = DEFAULT_CONFIG_PATH
    run_mode: str | None = None
    target_family: str | None = None
    heldout_evaluation: bool | None = None
    active_feature_banks: list[str] | None = None
    force_rebuild_intermediary_features: bool = False
    reasoning_models: list[str] | None = None
    embedding_model_name: str | None = None
    repeat_cv_with_new_seeds: bool = False
    cv_seed_repeat_count: int | None = None
    distillation_nested_sweep: bool | None = None
    save_reasoning_predictions: bool | None = None
    candidate_feature_sets: list[str] | None = None
    model_families: list[str] | None = None
    output_modes: list[str] | None = None
    run_advanced_models: bool | None = None
    xgb_calibration_estimators: list[int] | None = None
    use_latest_xgb_calibration: bool | None = None
    xgb_model_param_overrides_by_model_id: dict[str, dict[str, float | int]] | None = None
    max_parallel_workers: int | None = None


@dataclass(frozen=True)
class ResolvedRunOptions:
    config_path: str
    run_mode: str
    target_family: TargetFamilySpec
    heldout_evaluation: bool
    active_feature_banks: list[str]
    force_rebuild_intermediary_features: bool
    repository_feature_banks: list[RepositoryFeatureBankSpec]
    intermediary_features: list[IntermediaryFeatureSpec]
    distillation_feature_sets: list[FeatureSetSpec]
    distillation_models: list[DistillationModelSpec]
    repeat_cv_with_new_seeds: bool
    cv_seed_repeat_count: int
    distillation_nested_sweep: bool
    save_reasoning_predictions: bool
    candidate_feature_sets: list[str]
    model_families: list[str]
    output_modes: list[str]
    run_advanced_models: bool
    xgb_calibration_estimators: list[int]
    use_latest_xgb_calibration: bool
    xgb_model_param_overrides_by_model_id: dict[str, dict[str, float | int]]
    max_parallel_workers: int


def _require_known_subset(
    requested: list[str],
    *,
    available: set[str],
    label: str,
) -> None:
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise RuntimeError(f"Unknown {label}: {unknown}")


def _feature_set_requested_ids(
    *,
    config: ExperimentConfig,
    overrides: RunOverrides,
    run_mode: str,
) -> list[str]:
    if overrides.candidate_feature_sets is not None:
        return [value for value in overrides.candidate_feature_sets]
    if run_mode in {"model_testing_mode", "xgb_calibration_mode"}:
        return list(config.model_testing.candidate_feature_sets)
    return [spec.feature_set_id for spec in config.distillation_feature_sets]


def _resolve_requested_feature_banks(
    *,
    config: ExperimentConfig,
    requested_feature_set_ids: list[str],
    overrides: RunOverrides,
    run_mode: str,
    available_feature_bank_ids: set[str],
) -> list[str]:
    if overrides.active_feature_banks is not None:
        return [value for value in overrides.active_feature_banks]
    if run_mode not in {"model_testing_mode", "xgb_calibration_mode"}:
        return sorted(available_feature_bank_ids)

    feature_set_map = {spec.feature_set_id: spec for spec in config.distillation_feature_sets}
    required: set[str] = set()
    for feature_set_id in requested_feature_set_ids:
        required.update(feature_set_map[feature_set_id].feature_bank_ids)
    return sorted(required)


def _resolve_model_families(
    *,
    config: ExperimentConfig,
    overrides: RunOverrides,
    run_mode: str,
) -> list[str]:
    if run_mode == "xgb_calibration_mode":
        return ["xgb1"]
    if run_mode != "model_testing_mode":
        return []
    requested = (
        list(overrides.model_families)
        if overrides.model_families is not None
        else list(config.model_testing.default_model_families)
    )
    _require_known_subset(
        requested,
        available=SUPPORTED_MODEL_TESTING_FAMILIES,
        label="model families",
    )
    if not requested:
        raise RuntimeError("At least one model family must be selected for model_testing_mode.")
    return requested


def _resolve_output_modes(
    *,
    overrides: RunOverrides,
    run_mode: str,
) -> list[str]:
    if run_mode == "xgb_calibration_mode":
        return ["single_target"]
    requested = list(overrides.output_modes) if overrides.output_modes is not None else ["single_target"]
    _require_known_subset(
        requested,
        available=SUPPORTED_OUTPUT_MODES,
        label="output modes",
    )
    if not requested:
        raise RuntimeError("At least one output mode must be selected.")
    if run_mode != "model_testing_mode" and len(requested) > 1:
        raise RuntimeError(
            "Only one output mode is supported outside model_testing_mode."
        )
    return requested


def _resolve_distillation_models(
    *,
    run_mode: str,
    target_family: TargetFamilySpec,
    config: ExperimentConfig,
    overrides: RunOverrides,
    model_families: list[str],
) -> list[DistillationModelSpec]:
    if run_mode == "reproduction_mode":
        return []

    available_model_specs = [
        spec
        for spec in config.distillation_models
        if target_family.task_kind in set(spec.supported_task_kinds)
    ]
    if not available_model_specs:
        raise RuntimeError(
            f"No distillation models support target family '{target_family.family_id}' "
            f"with task_kind '{target_family.task_kind}'."
        )
    available_ids = {spec.model_id for spec in available_model_specs}

    if run_mode == "reasoning_distillation_mode":
        if overrides.reasoning_models is None:
            requested_ids = [spec.model_id for spec in available_model_specs]
        else:
            requested_ids = [value for value in overrides.reasoning_models]
            _require_known_subset(
                requested_ids,
                available=available_ids,
                label="distillation models",
            )
    elif run_mode == "xgb_calibration_mode":
        requested_ids = [
            "xgb1_regressor" if target_family.task_kind == "regression" else "xgb1_classifier"
        ]
        missing = [model_id for model_id in requested_ids if model_id not in available_ids]
        if missing:
            raise RuntimeError(
                "xgb_calibration_mode requested model ids that are not configured in distillation_models: "
                f"{missing}"
            )
    else:
        model_map = MODEL_FAMILY_TO_MODEL_ID[target_family.task_kind]
        requested_ids = [model_map[family] for family in model_families]
        missing = [model_id for model_id in requested_ids if model_id not in available_ids]
        if missing:
            raise RuntimeError(
                "model_testing_mode requested model ids that are not configured in distillation_models: "
                f"{missing}"
            )

    selected = [spec for spec in available_model_specs if spec.model_id in set(requested_ids)]
    if not selected:
        raise RuntimeError(
            f"No compatible distillation models were selected for target family '{target_family.family_id}'."
        )
    return selected


def resolve_run_options(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
) -> ResolvedRunOptions:
    overrides_use = overrides or RunOverrides()

    run_mode = overrides_use.run_mode or config.defaults.run_mode
    if run_mode not in {"reproduction_mode", "reasoning_distillation_mode", "model_testing_mode", "xgb_calibration_mode"}:
        raise RuntimeError(f"Unsupported run_mode '{run_mode}'.")

    target_family_id = overrides_use.target_family or config.defaults.target_family
    if target_family_id == "v25_and_taste":
        # Composite orchestration is handled above this resolver.
        target_family_id = config.defaults.target_family
    target_family_map = {spec.family_id: spec for spec in config.target_families}
    if target_family_id not in target_family_map:
        raise RuntimeError(f"Unknown target family '{target_family_id}'.")
    target_family = target_family_map[target_family_id]

    heldout_evaluation = (
        config.defaults.heldout_evaluation
        if overrides_use.heldout_evaluation is None
        else bool(overrides_use.heldout_evaluation)
    )
    if run_mode == "reproduction_mode":
        heldout_evaluation = True
    if run_mode == "model_testing_mode":
        heldout_evaluation = False
    if run_mode == "xgb_calibration_mode":
        heldout_evaluation = False

    available_repository_banks = [spec for spec in config.repository_feature_banks if spec.enabled]
    available_intermediary_banks = [spec for spec in config.intermediary_features if spec.enabled]
    available_feature_bank_ids = {
        *(spec.feature_bank_id for spec in available_repository_banks),
        *(spec.feature_bank_id for spec in available_intermediary_banks),
    }

    feature_set_map = {spec.feature_set_id: spec for spec in config.distillation_feature_sets}
    requested_feature_set_ids = _feature_set_requested_ids(
        config=config,
        overrides=overrides_use,
        run_mode=run_mode,
    )
    _require_known_subset(
        requested_feature_set_ids,
        available=set(feature_set_map.keys()),
        label="feature sets",
    )
    if not requested_feature_set_ids:
        raise RuntimeError("At least one feature set must be selected.")

    requested_feature_banks = _resolve_requested_feature_banks(
        config=config,
        requested_feature_set_ids=requested_feature_set_ids,
        overrides=overrides_use,
        run_mode=run_mode,
        available_feature_bank_ids=available_feature_bank_ids,
    )
    _require_known_subset(
        requested_feature_banks,
        available=available_feature_bank_ids,
        label="feature banks",
    )

    selected_repository_feature_banks = [
        spec for spec in available_repository_banks if spec.feature_bank_id in set(requested_feature_banks)
    ]

    selected_intermediary_features: list[IntermediaryFeatureSpec] = []
    for spec in available_intermediary_banks:
        if spec.feature_bank_id not in set(requested_feature_banks):
            continue
        if spec.kind == "llm_engineered_v1":
            raise RuntimeError(
                "The llm_engineered intermediary feature family is scaffolded but inactive. "
                "Provide the custom prompt assets before selecting it."
            )
        if overrides_use.embedding_model_name and spec.kind.startswith("sentence_transformer"):
            selected_intermediary_features.append(
                replace(spec, embedding_model_name=overrides_use.embedding_model_name)
            )
            continue
        selected_intermediary_features.append(spec)

    selected_feature_bank_ids = {
        *(spec.feature_bank_id for spec in selected_repository_feature_banks),
        *(spec.feature_bank_id for spec in selected_intermediary_features),
    }
    selected_feature_sets = [
        feature_set
        for feature_set in config.distillation_feature_sets
        if feature_set.feature_set_id in set(requested_feature_set_ids)
        and set(feature_set.feature_bank_ids).issubset(selected_feature_bank_ids)
    ]
    if not selected_feature_sets and run_mode != "reproduction_mode":
        raise RuntimeError(
            "No distillation/model-testing feature sets can be built from the selected feature banks."
        )

    model_families = _resolve_model_families(
        config=config,
        overrides=overrides_use,
        run_mode=run_mode,
    )
    output_modes = _resolve_output_modes(
        overrides=overrides_use,
        run_mode=run_mode,
    )
    selected_models = _resolve_distillation_models(
        run_mode=run_mode,
        target_family=target_family,
        config=config,
        overrides=overrides_use,
        model_families=model_families,
    )

    if run_mode == "reasoning_distillation_mode":
        if target_family.family_id == "v25_policies" and "policy_v25" in selected_feature_bank_ids:
            raise RuntimeError(
                "policy_v25 cannot be used as an input feature bank when v25_policies is the target family."
            )
        if target_family.family_id == "taste_policies" and "policy_v25" in selected_feature_bank_ids:
            raise RuntimeError(
                "policy_v25 is reserved for the success-reproduction track and is not a distillation input bank."
            )

    cv_seed_repeat_count = (
        int(overrides_use.cv_seed_repeat_count)
        if overrides_use.cv_seed_repeat_count is not None
        else (
            config.model_testing.screening_repeat_cv_count
            if (run_mode == "model_testing_mode" and overrides_use.repeat_cv_with_new_seeds)
            else 1
        )
    )
    repeat_cv_with_new_seeds = bool(overrides_use.repeat_cv_with_new_seeds)
    if repeat_cv_with_new_seeds and cv_seed_repeat_count < 2:
        raise RuntimeError("cv_seed_repeat_count must be >= 2 when repeat CV is enabled.")
    if not repeat_cv_with_new_seeds:
        cv_seed_repeat_count = 1

    run_advanced_models = (
        config.model_testing.run_advanced_models_default
        if overrides_use.run_advanced_models is None
        else bool(overrides_use.run_advanced_models)
    )
    if run_mode == "xgb_calibration_mode":
        run_advanced_models = False

    xgb_calibration_estimators = (
        [int(value) for value in overrides_use.xgb_calibration_estimators]
        if overrides_use.xgb_calibration_estimators is not None
        else list(config.model_testing.xgb_calibration_estimators)
    )
    if not xgb_calibration_estimators or any(value <= 0 for value in xgb_calibration_estimators):
        raise RuntimeError("xgb calibration estimators must be a non-empty list of positive integers.")

    use_latest_xgb_calibration = (
        config.model_testing.use_latest_xgb_calibration_default
        if overrides_use.use_latest_xgb_calibration is None
        else bool(overrides_use.use_latest_xgb_calibration)
    )
    if run_mode == "xgb_calibration_mode":
        use_latest_xgb_calibration = False

    return ResolvedRunOptions(
        config_path=overrides_use.config_path,
        run_mode=run_mode,
        target_family=target_family,
        heldout_evaluation=heldout_evaluation,
        active_feature_banks=sorted(selected_feature_bank_ids),
        force_rebuild_intermediary_features=overrides_use.force_rebuild_intermediary_features,
        repository_feature_banks=selected_repository_feature_banks,
        intermediary_features=selected_intermediary_features,
        distillation_feature_sets=selected_feature_sets,
        distillation_models=selected_models,
        repeat_cv_with_new_seeds=repeat_cv_with_new_seeds,
        cv_seed_repeat_count=cv_seed_repeat_count,
        distillation_nested_sweep=(
            True if overrides_use.distillation_nested_sweep is None else bool(overrides_use.distillation_nested_sweep)
        ),
        save_reasoning_predictions=(
            True if overrides_use.save_reasoning_predictions is None else bool(overrides_use.save_reasoning_predictions)
        ),
        candidate_feature_sets=[spec.feature_set_id for spec in selected_feature_sets],
        model_families=model_families,
        output_modes=output_modes,
        run_advanced_models=run_advanced_models,
        xgb_calibration_estimators=xgb_calibration_estimators,
        use_latest_xgb_calibration=use_latest_xgb_calibration,
        xgb_model_param_overrides_by_model_id=dict(overrides_use.xgb_model_param_overrides_by_model_id or {}),
        max_parallel_workers=resolve_max_parallel_workers(overrides_use.max_parallel_workers),
    )
