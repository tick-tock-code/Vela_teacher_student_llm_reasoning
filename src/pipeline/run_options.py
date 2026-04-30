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
from src.utils.model_ids import (
    XGB_CLASSIFIER_MODEL_KIND,
    XGB_FAMILY_ID,
    XGB_REGRESSOR_MODEL_KIND,
    normalize_xgb_family_id,
    normalize_xgb_model_kind,
)
from src.utils.parallel import resolve_max_parallel_workers


DEFAULT_CONFIG_PATH = "experiments/teacher_student_distillation_v1.json"
SUPPORTED_OUTPUT_MODES = {"single_target", "multi_output"}
SUPPORTED_SAVED_EVAL_MODES = {
    "reasoning_test_metrics",
    "success_with_pred_reasoning",
    "full_transfer_report",
    "combination_transfer_report",
}
SUPPORTED_HQ_EXIT_OVERRIDE_MODES = {
    "with_override",
    "both_with_and_without",
    "force_off_all_branches",
    "force_on_all_branches",
    "both_force_off_and_on_all_branches",
}
DEFAULT_SUCCESS_MODEL_VARIANTS: tuple[str, ...] = (
    "single_model",
    "soft_avg_model",
    "soft_avg_weighted_model",
)
SUPPORTED_SUCCESS_MODEL_VARIANTS = set(DEFAULT_SUCCESS_MODEL_VARIANTS)

ABLATION_V25_19SET_LINEAR_FEATURE_SET_IDS: tuple[str, ...] = (
    "hq_baseline",
    "llm_engineering",
    "lambda_policies",
    "sentence_prose",
    "sentence_structured",
    "sentence_bundle",
    "hq_plus_sentence_prose",
    "hq_plus_sentence_structured",
    "hq_plus_sentence_bundle",
    "llm_engineering_plus_sentence_prose",
    "llm_engineering_plus_sentence_structured",
    "llm_engineering_plus_sentence_bundle",
    "lambda_policies_plus_sentence_prose",
    "lambda_policies_plus_sentence_structured",
    "lambda_policies_plus_sentence_bundle",
    "hq_plus_llm_engineering_plus_sentence_bundle",
    "hq_plus_lambda_policies_plus_sentence_bundle",
    "llm_engineering_plus_lambda_policies_plus_sentence_bundle",
    "hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle",
)


MODEL_FAMILY_TO_MODEL_ID: dict[str, dict[str, str]] = {
    "regression": {
        "linear_l2": "ridge",
        "linear_svm": "linear_svr_regressor",
        XGB_FAMILY_ID: XGB_REGRESSOR_MODEL_KIND,
        "mlp": "mlp_regressor",
        "elasticnet": "elasticnet_regressor",
        "randomforest": "randomforest_regressor",
    },
    "classification": {
        "linear_l2": "logreg_classifier",
        "linear_svm": "linear_svm_classifier",
        XGB_FAMILY_ID: XGB_CLASSIFIER_MODEL_KIND,
        "mlp": "mlp_classifier",
        "elasticnet": "elasticnet_logreg_classifier",
        "randomforest": "randomforest_classifier",
    },
}

DEFAULT_MODEL_FAMILY_OUTPUT_MODES: dict[str, list[str]] = {
    "linear_l2": ["single_target"],
    "linear_svm": ["single_target"],
    XGB_FAMILY_ID: ["single_target"],
    "mlp": ["multi_output"],
    "elasticnet": ["single_target"],
    "randomforest": ["single_target"],
}


def _default_linear_distillation_model_id(task_kind: str) -> str:
    if task_kind == "regression":
        return "ridge"
    if task_kind == "classification":
        return "logreg_classifier"
    raise RuntimeError(f"Unsupported task_kind for linear default model selection: {task_kind}")


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
    repeat_cv_with_new_seeds: bool | None = None
    cv_seed_repeat_count: int | None = None
    distillation_nested_sweep: bool | None = None
    save_reasoning_predictions: bool | None = None
    candidate_feature_sets: list[str] | None = None
    model_families: list[str] | None = None
    output_modes: list[str] | None = None
    model_family_output_modes: dict[str, list[str]] | None = None
    run_advanced_models: bool | None = None
    save_model_configs_after_training: bool | None = None
    saved_config_bundle_path: str | None = None
    saved_eval_mode: str | None = None
    saved_eval_combo_ids: list[str] | None = None
    saved_eval_combo_refs: list[str] | None = None
    saved_eval_success_branch_ids: list[str] | None = None
    success_model_variants: list[str] | None = None
    saved_eval_per_target_best_r2: bool | None = None
    hq_exit_override_mode: str | None = None
    xgb_calibration_estimators: list[int] | None = None
    use_latest_xgb_calibration: bool | None = None
    rf_calibration_min_samples_leaf: list[int] | None = None
    rf_calibration_max_depth: list[int | None] | None = None
    rf_calibration_max_features: list[str | float] | None = None
    use_latest_rf_calibration: bool | None = None
    mlp_calibration_hidden_layer_sizes: list[list[int]] | None = None
    mlp_calibration_alpha: list[float] | None = None
    use_latest_mlp_calibration: bool | None = None
    mlp_hidden_layer_sizes: list[int] | None = None
    mlp_alpha: float | None = None
    xgb_model_param_overrides_by_model_id: dict[str, dict[str, object]] | None = None
    rf_model_param_overrides_by_model_id: dict[str, dict[str, object]] | None = None
    model_testing_per_fit_threads: int | None = None
    max_parallel_workers: int | None = None
    ablation_v25_19set_linear_profile: bool = False


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
    model_family_output_modes: dict[str, list[str]]
    run_advanced_models: bool
    save_model_configs_after_training: bool
    saved_config_bundle_path: str | None
    saved_eval_mode: str | None
    saved_eval_combo_ids: list[str] | None
    saved_eval_combo_refs: list[str] | None
    saved_eval_success_branch_ids: list[str] | None
    success_model_variants: list[str]
    saved_eval_per_target_best_r2: bool
    hq_exit_override_mode: str
    xgb_calibration_estimators: list[int]
    use_latest_xgb_calibration: bool
    rf_calibration_min_samples_leaf: list[int]
    rf_calibration_max_depth: list[int | None]
    rf_calibration_max_features: list[str | float]
    use_latest_rf_calibration: bool
    mlp_calibration_hidden_layer_sizes: list[list[int]]
    mlp_calibration_alpha: list[float]
    use_latest_mlp_calibration: bool
    mlp_hidden_layer_sizes: list[int] | None
    mlp_alpha: float | None
    xgb_model_param_overrides_by_model_id: dict[str, dict[str, object]]
    rf_model_param_overrides_by_model_id: dict[str, dict[str, object]]
    max_parallel_workers: int
    ablation_v25_19set_linear_profile: bool


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
    if run_mode in {"model_testing_mode", "xgb_calibration_mode", "rf_calibration_mode", "mlp_calibration_mode"}:
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
    if run_mode not in {"model_testing_mode", "xgb_calibration_mode", "rf_calibration_mode", "mlp_calibration_mode"}:
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
        return [XGB_FAMILY_ID]
    if run_mode == "rf_calibration_mode":
        return ["randomforest"]
    if run_mode == "mlp_calibration_mode":
        return ["mlp"]
    if run_mode != "model_testing_mode":
        return []
    requested = (
        _normalize_model_families(list(overrides.model_families))
        if overrides.model_families is not None
        else _normalize_model_families(list(config.model_testing.default_model_families))
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
    if run_mode in {"xgb_calibration_mode", "rf_calibration_mode", "mlp_calibration_mode"}:
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


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _normalize_model_families(values: list[str]) -> list[str]:
    return _dedupe_preserve_order([normalize_xgb_family_id(value) for value in values])


def _normalize_model_family_output_modes(
    mapping: dict[str, list[str]] | None,
) -> dict[str, list[str]] | None:
    if mapping is None:
        return None
    normalized: dict[str, list[str]] = {}
    for family_id, modes in mapping.items():
        key = normalize_xgb_family_id(family_id)
        if key in normalized:
            normalized[key] = _dedupe_preserve_order(normalized[key] + list(modes))
        else:
            normalized[key] = list(modes)
    return normalized


def _normalize_model_param_overrides(
    mapping: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for model_id, params in (mapping or {}).items():
        normalized[normalize_xgb_model_kind(model_id)] = dict(params)
    return normalized


def _resolve_model_family_output_modes(
    *,
    overrides: RunOverrides,
    run_mode: str,
    model_families: list[str],
) -> dict[str, list[str]]:
    if run_mode != "model_testing_mode":
        return {}

    normalized_override_mapping = _normalize_model_family_output_modes(
        overrides.model_family_output_modes
    )

    unknown_keys: list[str] = []
    if normalized_override_mapping is not None:
        unknown_keys = [
            key for key in normalized_override_mapping.keys()
            if key not in set(model_families)
        ]
    if unknown_keys:
        raise RuntimeError(
            f"model_family_output_modes references unknown or unselected model families: {unknown_keys}"
        )

    mapping: dict[str, list[str]] = {}
    for family_id in model_families:
        if normalized_override_mapping is not None:
            requested_modes = list(
                normalized_override_mapping.get(
                    family_id,
                    DEFAULT_MODEL_FAMILY_OUTPUT_MODES.get(family_id, ["single_target"]),
                )
            )
        elif overrides.output_modes is not None:
            requested_modes = list(overrides.output_modes)
        else:
            requested_modes = list(DEFAULT_MODEL_FAMILY_OUTPUT_MODES.get(family_id, ["single_target"]))

        requested_modes = _dedupe_preserve_order(requested_modes)
        _require_known_subset(
            requested_modes,
            available=SUPPORTED_OUTPUT_MODES,
            label=f"output modes for model family '{family_id}'",
        )
        if not requested_modes:
            raise RuntimeError(f"Model family '{family_id}' must have at least one output mode selected.")
        if family_id != "mlp" and "multi_output" in requested_modes:
            raise RuntimeError(
                f"Model family '{family_id}' cannot use output mode 'multi_output'. "
                "Only the 'mlp' family supports multi_output."
            )
        mapping[family_id] = requested_modes
    return mapping


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
            requested_ids = [_default_linear_distillation_model_id(target_family.task_kind)]
            _require_known_subset(
                requested_ids,
                available=available_ids,
                label="distillation models",
            )
        else:
            requested_ids = [normalize_xgb_model_kind(value) for value in overrides.reasoning_models]
            _require_known_subset(
                requested_ids,
                available=available_ids,
                label="distillation models",
            )
    elif run_mode == "xgb_calibration_mode":
        requested_ids = [
            XGB_REGRESSOR_MODEL_KIND if target_family.task_kind == "regression" else XGB_CLASSIFIER_MODEL_KIND
        ]
        missing = [model_id for model_id in requested_ids if model_id not in available_ids]
        if missing:
            raise RuntimeError(
                "xgb_calibration_mode requested model ids that are not configured in distillation_models: "
                f"{missing}"
            )
    elif run_mode == "rf_calibration_mode":
        requested_ids = [
            "randomforest_regressor" if target_family.task_kind == "regression" else "randomforest_classifier"
        ]
        missing = [model_id for model_id in requested_ids if model_id not in available_ids]
        if missing:
            raise RuntimeError(
                "rf_calibration_mode requested model ids that are not configured in distillation_models: "
                f"{missing}"
            )
    elif run_mode == "mlp_calibration_mode":
        requested_ids = [
            "mlp_regressor" if target_family.task_kind == "regression" else "mlp_classifier"
        ]
        missing = [model_id for model_id in requested_ids if model_id not in available_ids]
        if missing:
            raise RuntimeError(
                "mlp_calibration_mode requested model ids that are not configured in distillation_models: "
                f"{missing}"
            )
    elif run_mode == "saved_config_evaluation_mode":
        if overrides.reasoning_models is None:
            requested_ids = [_default_linear_distillation_model_id(target_family.task_kind)]
            _require_known_subset(
                requested_ids,
                available=available_ids,
                label="distillation models",
            )
        else:
            requested_ids = [normalize_xgb_model_kind(value) for value in overrides.reasoning_models]
            _require_known_subset(
                requested_ids,
                available=available_ids,
                label="distillation models",
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
    if overrides_use.ablation_v25_19set_linear_profile:
        # Locked profile for train-only reasoning ablation recalculation.
        overrides_use = replace(
            overrides_use,
            run_mode="model_testing_mode",
            target_family="v25_policies",
            heldout_evaluation=False,
            active_feature_banks=None,
            reasoning_models=None,
            candidate_feature_sets=list(ABLATION_V25_19SET_LINEAR_FEATURE_SET_IDS),
            model_families=["linear_l2"],
            output_modes=["single_target"],
            model_family_output_modes={"linear_l2": ["single_target"]},
            repeat_cv_with_new_seeds=False,
            cv_seed_repeat_count=1,
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            save_model_configs_after_training=False,
            use_latest_xgb_calibration=False,
            use_latest_rf_calibration=False,
            use_latest_mlp_calibration=False,
            xgb_model_param_overrides_by_model_id={},
            rf_model_param_overrides_by_model_id={},
        )

    run_mode = overrides_use.run_mode or config.defaults.run_mode
    if run_mode not in {
        "reproduction_mode",
        "reasoning_distillation_mode",
        "model_testing_mode",
        "saved_config_evaluation_mode",
        "xgb_calibration_mode",
        "rf_calibration_mode",
        "mlp_calibration_mode",
    }:
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
    if run_mode == "rf_calibration_mode":
        heldout_evaluation = False
    if run_mode == "mlp_calibration_mode":
        heldout_evaluation = False
    if run_mode == "saved_config_evaluation_mode":
        heldout_evaluation = (
            True
            if overrides_use.heldout_evaluation is None
            else bool(overrides_use.heldout_evaluation)
        )

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
    selected_models = _resolve_distillation_models(
        run_mode=run_mode,
        target_family=target_family,
        config=config,
        overrides=overrides_use,
        model_families=model_families,
    )
    model_family_output_modes = _resolve_model_family_output_modes(
        overrides=overrides_use,
        run_mode=run_mode,
        model_families=model_families,
    )
    if run_mode == "model_testing_mode":
        output_modes = _dedupe_preserve_order(
            [
                mode
                for family_id in model_families
                for mode in model_family_output_modes.get(family_id, [])
            ]
        )
        if not output_modes:
            raise RuntimeError("At least one output mode must be active across selected model families.")
    else:
        output_modes = _resolve_output_modes(
            overrides=overrides_use,
            run_mode=run_mode,
        )
        if run_mode == "reasoning_distillation_mode" and "multi_output" in output_modes:
            selected_model_ids = {spec.model_id for spec in selected_models}
            mlp_model_ids = {"mlp_regressor", "mlp_classifier"}
            if not selected_model_ids.issubset(mlp_model_ids):
                raise RuntimeError(
                    "output_mode 'multi_output' is only supported when all selected distillation models "
                    "are MLP models."
                )
        if run_mode == "reasoning_distillation_mode" and overrides_use.output_modes is None:
            selected_model_ids = {spec.model_id for spec in selected_models}
            mlp_model_ids = {"mlp_regressor", "mlp_classifier"}
            if selected_model_ids and selected_model_ids.issubset(mlp_model_ids):
                output_modes = ["multi_output"]

    if run_mode == "reasoning_distillation_mode":
        if target_family.family_id == "v25_policies" and "policy_v25" in selected_feature_bank_ids:
            raise RuntimeError(
                "policy_v25 cannot be used as an input feature bank when v25_policies is the target family."
            )
        if target_family.family_id == "taste_policies" and "policy_v25" in selected_feature_bank_ids:
            raise RuntimeError(
                "policy_v25 is reserved for the success-reproduction track and is not a distillation input bank."
            )

    repeat_cv_override = overrides_use.repeat_cv_with_new_seeds
    repeat_cv_with_new_seeds = False
    cv_seed_repeat_count = 1

    # Deprecated: Stage-B advanced model-testing is inactive.
    run_advanced_models = False

    save_model_configs_after_training = (
        config.model_testing.save_model_configs_after_training_default
        if overrides_use.save_model_configs_after_training is None
        else bool(overrides_use.save_model_configs_after_training)
    )
    if run_mode != "model_testing_mode":
        save_model_configs_after_training = False

    saved_config_bundle_path = None
    saved_eval_mode = None
    saved_eval_combo_ids: list[str] | None = None
    saved_eval_combo_refs: list[str] | None = None
    saved_eval_success_branch_ids: list[str] | None = None
    success_model_variants = list(DEFAULT_SUCCESS_MODEL_VARIANTS)
    saved_eval_per_target_best_r2 = False
    hq_exit_override_mode = "with_override"
    if run_mode == "saved_config_evaluation_mode":
        saved_config_bundle_path = (
            str(overrides_use.saved_config_bundle_path).strip()
            if overrides_use.saved_config_bundle_path
            else None
        )
        if overrides_use.saved_eval_combo_refs is not None:
            saved_eval_combo_refs = [
                str(value).strip()
                for value in overrides_use.saved_eval_combo_refs
                if str(value).strip()
            ]
            if not saved_eval_combo_refs:
                raise RuntimeError(
                    "saved_eval_combo_refs was provided but no non-empty combo refs were supplied."
                )
        if not saved_config_bundle_path and not saved_eval_combo_refs:
            raise RuntimeError(
                "saved_config_evaluation_mode requires saved_config_bundle_path "
                "or saved_eval_combo_refs."
            )
        saved_eval_mode = (
            str(overrides_use.saved_eval_mode).strip()
            if overrides_use.saved_eval_mode
            else "reasoning_test_metrics"
        )
        if saved_eval_mode not in SUPPORTED_SAVED_EVAL_MODES:
            raise RuntimeError(
                f"Unsupported saved_eval_mode '{saved_eval_mode}'. "
                f"Supported: {sorted(SUPPORTED_SAVED_EVAL_MODES)}"
            )
        if (
            saved_eval_mode == "combination_transfer_report"
            and saved_eval_combo_refs is not None
            and len(saved_eval_combo_refs) != 1
        ):
            raise RuntimeError(
                "saved_eval_mode=combination_transfer_report requires exactly one combo ref "
                "(<bundle>::<combo_id>)."
            )
        if saved_eval_mode in {"full_transfer_report", "combination_transfer_report"}:
            if not saved_eval_combo_refs:
                raise RuntimeError(
                    "saved_eval_mode requires saved_eval_combo_refs for transfer evaluation."
                )
        if overrides_use.saved_eval_combo_ids is not None:
            saved_eval_combo_ids = [
                str(value).strip()
                for value in overrides_use.saved_eval_combo_ids
                if str(value).strip()
            ]
            if not saved_eval_combo_ids:
                raise RuntimeError(
                    "saved_eval_combo_ids was provided but no non-empty combo ids were supplied."
                )
        if overrides_use.saved_eval_success_branch_ids is not None:
            saved_eval_success_branch_ids = [
                str(value).strip()
                for value in overrides_use.saved_eval_success_branch_ids
                if str(value).strip()
            ]
            if not saved_eval_success_branch_ids:
                raise RuntimeError(
                    "saved_eval_success_branch_ids was provided but no non-empty branch ids were supplied."
                )
            if saved_eval_mode != "combination_transfer_report":
                raise RuntimeError(
                    "saved_eval_success_branch_ids is only supported for "
                    "saved_eval_mode=combination_transfer_report."
                )
        if overrides_use.success_model_variants is not None:
            success_model_variants = _dedupe_preserve_order(
                [
                    str(value).strip()
                    for value in overrides_use.success_model_variants
                    if str(value).strip()
                ]
            )
            if not success_model_variants:
                raise RuntimeError(
                    "success_model_variants was provided but no non-empty variants were supplied."
                )
            _require_known_subset(
                success_model_variants,
                available=SUPPORTED_SUCCESS_MODEL_VARIANTS,
                label="success model variants",
            )
        saved_eval_per_target_best_r2 = bool(overrides_use.saved_eval_per_target_best_r2)
        hq_exit_override_mode = (
            str(overrides_use.hq_exit_override_mode).strip()
            if overrides_use.hq_exit_override_mode
            else "with_override"
        )
        if hq_exit_override_mode not in SUPPORTED_HQ_EXIT_OVERRIDE_MODES:
            raise RuntimeError(
                f"Unsupported hq_exit_override_mode '{hq_exit_override_mode}'. "
                f"Supported: {sorted(SUPPORTED_HQ_EXIT_OVERRIDE_MODES)}"
            )

    default_repeat_enabled = False
    default_repeat_count = 1
    if run_mode == "model_testing_mode":
        default_repeat_enabled = True
        default_repeat_count = int(config.model_testing.screening_repeat_cv_count)
    elif run_mode == "saved_config_evaluation_mode" and saved_eval_mode == "full_transfer_report":
        default_repeat_enabled = True
        default_repeat_count = int(config.model_testing.screening_repeat_cv_count)

    repeat_cv_with_new_seeds = (
        default_repeat_enabled if repeat_cv_override is None else bool(repeat_cv_override)
    )
    cv_seed_repeat_count = (
        int(overrides_use.cv_seed_repeat_count)
        if overrides_use.cv_seed_repeat_count is not None
        else (default_repeat_count if repeat_cv_with_new_seeds else 1)
    )
    if repeat_cv_with_new_seeds and cv_seed_repeat_count < 2:
        raise RuntimeError("cv_seed_repeat_count must be >= 2 when repeat CV is enabled.")
    if not repeat_cv_with_new_seeds:
        cv_seed_repeat_count = 1

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

    rf_calibration_min_samples_leaf = (
        [int(value) for value in overrides_use.rf_calibration_min_samples_leaf]
        if overrides_use.rf_calibration_min_samples_leaf is not None
        else list(config.model_testing.rf_calibration_min_samples_leaf)
    )
    if not rf_calibration_min_samples_leaf or any(value <= 0 for value in rf_calibration_min_samples_leaf):
        raise RuntimeError("rf calibration min_samples_leaf must be a non-empty list of positive integers.")

    rf_calibration_max_depth = (
        [value for value in overrides_use.rf_calibration_max_depth]
        if overrides_use.rf_calibration_max_depth is not None
        else list(config.model_testing.rf_calibration_max_depth)
    )
    if not rf_calibration_max_depth:
        raise RuntimeError("rf calibration max_depth must be a non-empty list.")

    rf_calibration_max_features = (
        [value for value in overrides_use.rf_calibration_max_features]
        if overrides_use.rf_calibration_max_features is not None
        else list(config.model_testing.rf_calibration_max_features)
    )
    if not rf_calibration_max_features:
        raise RuntimeError("rf calibration max_features must be a non-empty list.")

    use_latest_rf_calibration = (
        config.model_testing.use_latest_rf_calibration_default
        if overrides_use.use_latest_rf_calibration is None
        else bool(overrides_use.use_latest_rf_calibration)
    )
    if run_mode == "rf_calibration_mode":
        use_latest_rf_calibration = False

    mlp_calibration_hidden_layer_sizes = (
        [
            [int(v) for v in layer]
            for layer in overrides_use.mlp_calibration_hidden_layer_sizes
        ]
        if getattr(overrides_use, "mlp_calibration_hidden_layer_sizes", None) is not None
        else [list(layer) for layer in config.model_testing.mlp_calibration_hidden_layer_sizes]
    )
    if not mlp_calibration_hidden_layer_sizes:
        raise RuntimeError("mlp calibration hidden_layer_sizes must be a non-empty list.")
    mlp_calibration_alpha = (
        [float(value) for value in overrides_use.mlp_calibration_alpha]
        if getattr(overrides_use, "mlp_calibration_alpha", None) is not None
        else list(config.model_testing.mlp_calibration_alpha)
    )
    if not mlp_calibration_alpha or any(value <= 0 for value in mlp_calibration_alpha):
        raise RuntimeError("mlp calibration alpha must be a non-empty list of > 0 values.")
    use_latest_mlp_calibration = (
        config.model_testing.use_latest_mlp_calibration_default
        if getattr(overrides_use, "use_latest_mlp_calibration", None) is None
        else bool(overrides_use.use_latest_mlp_calibration)
    )
    if run_mode == "mlp_calibration_mode":
        use_latest_mlp_calibration = False

    mlp_hidden_layer_sizes = (
        [int(value) for value in overrides_use.mlp_hidden_layer_sizes]
        if overrides_use.mlp_hidden_layer_sizes is not None
        else None
    )
    if mlp_hidden_layer_sizes is not None:
        if not mlp_hidden_layer_sizes or any(value <= 0 for value in mlp_hidden_layer_sizes):
            raise RuntimeError("mlp_hidden_layer_sizes must be a non-empty list of positive integers.")

    mlp_alpha = (
        float(overrides_use.mlp_alpha)
        if overrides_use.mlp_alpha is not None
        else None
    )
    if mlp_alpha is not None and mlp_alpha <= 0:
        raise RuntimeError("mlp_alpha must be > 0 when provided.")

    nested_sweep_enabled = (
        True
        if (
            run_mode == "saved_config_evaluation_mode"
            and saved_eval_mode in {"full_transfer_report", "combination_transfer_report"}
            and overrides_use.distillation_nested_sweep is None
        )
        else (
            False
            if overrides_use.distillation_nested_sweep is None
            else bool(overrides_use.distillation_nested_sweep)
        )
    )

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
        distillation_nested_sweep=nested_sweep_enabled,
        save_reasoning_predictions=(
            True if overrides_use.save_reasoning_predictions is None else bool(overrides_use.save_reasoning_predictions)
        ),
        candidate_feature_sets=[spec.feature_set_id for spec in selected_feature_sets],
        model_families=model_families,
        output_modes=output_modes,
        model_family_output_modes=model_family_output_modes,
        run_advanced_models=run_advanced_models,
        save_model_configs_after_training=save_model_configs_after_training,
        saved_config_bundle_path=saved_config_bundle_path,
        saved_eval_mode=saved_eval_mode,
        saved_eval_combo_ids=saved_eval_combo_ids,
        saved_eval_combo_refs=saved_eval_combo_refs,
        saved_eval_success_branch_ids=saved_eval_success_branch_ids,
        success_model_variants=success_model_variants,
        saved_eval_per_target_best_r2=saved_eval_per_target_best_r2,
        hq_exit_override_mode=hq_exit_override_mode,
        xgb_calibration_estimators=xgb_calibration_estimators,
        use_latest_xgb_calibration=use_latest_xgb_calibration,
        rf_calibration_min_samples_leaf=rf_calibration_min_samples_leaf,
        rf_calibration_max_depth=rf_calibration_max_depth,
        rf_calibration_max_features=rf_calibration_max_features,
        use_latest_rf_calibration=use_latest_rf_calibration,
        mlp_calibration_hidden_layer_sizes=mlp_calibration_hidden_layer_sizes,
        mlp_calibration_alpha=mlp_calibration_alpha,
        use_latest_mlp_calibration=use_latest_mlp_calibration,
        mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
        mlp_alpha=mlp_alpha,
        xgb_model_param_overrides_by_model_id=_normalize_model_param_overrides(
            overrides_use.xgb_model_param_overrides_by_model_id
        ),
        rf_model_param_overrides_by_model_id=dict(overrides_use.rf_model_param_overrides_by_model_id or {}),
        max_parallel_workers=resolve_max_parallel_workers(overrides_use.max_parallel_workers),
        ablation_v25_19set_linear_profile=bool(overrides_use.ablation_v25_19set_linear_profile),
    )
