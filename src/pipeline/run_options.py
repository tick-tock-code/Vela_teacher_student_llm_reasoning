from __future__ import annotations

from dataclasses import dataclass, replace

from src.pipeline.config import ExperimentConfig, FeatureSetSpec, IntermediaryFeatureSpec, ModelSpec, ReasoningTargetSpec


DEFAULT_CONFIG_PATH = "experiments/teacher_student_distillation_v1.json"


@dataclass(frozen=True)
class RunOverrides:
    config_path: str = DEFAULT_CONFIG_PATH
    run_reasoning_predictions: bool = True
    run_success_predictions: bool = False
    active_intermediary_features: list[str] | None = None
    force_rebuild_intermediary_features: bool = False
    reasoning_targets: list[str] | None = None
    reasoning_models: list[str] | None = None
    embedding_model_name: str | None = None


@dataclass(frozen=True)
class ResolvedRunOptions:
    config_path: str
    run_reasoning_predictions: bool
    active_intermediary_features: list[str]
    force_rebuild_intermediary_features: bool
    reasoning_targets: list[ReasoningTargetSpec]
    reasoning_models: list[ModelSpec]
    intermediary_features: list[IntermediaryFeatureSpec]
    feature_sets: list[FeatureSetSpec]


def _require_known_subset(
    requested: list[str],
    *,
    available: set[str],
    label: str,
) -> None:
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise RuntimeError(f"Unknown {label}: {unknown}")


def resolve_run_options(
    config: ExperimentConfig,
    overrides: RunOverrides | None = None,
) -> ResolvedRunOptions:
    overrides_use = overrides or RunOverrides()

    if overrides_use.run_success_predictions:
        raise RuntimeError(
            "run_success_predictions=true is inactive in this repo. "
            "The active pipeline only reconstructs reasoning targets."
        )
    if not overrides_use.run_reasoning_predictions:
        raise RuntimeError("run_reasoning_predictions=false leaves no active pipeline stages to run.")

    config_feature_ids = [spec.feature_id for spec in config.intermediary_features if spec.enabled]
    active_feature_ids = overrides_use.active_intermediary_features or config_feature_ids
    if not active_feature_ids:
        raise RuntimeError("No active intermediary features were selected.")
    _require_known_subset(
        active_feature_ids,
        available={spec.feature_id for spec in config.intermediary_features},
        label="intermediary features",
    )

    selected_feature_specs: list[IntermediaryFeatureSpec] = []
    for spec in config.intermediary_features:
        if spec.feature_id not in active_feature_ids:
            continue
        if spec.kind == "llm_engineered_v1":
            raise RuntimeError(
                "The llm_engineered intermediary feature family is scaffolded but inactive. "
                "Provide the custom prompt assets before selecting it."
            )
        if overrides_use.embedding_model_name and spec.kind.startswith("sentence_transformer"):
            selected_feature_specs.append(
                replace(spec, embedding_model_name=overrides_use.embedding_model_name)
            )
            continue
        selected_feature_specs.append(spec)

    if not selected_feature_specs:
        raise RuntimeError("No enabled intermediary feature builders remain after applying overrides.")

    selected_feature_ids = {spec.feature_id for spec in selected_feature_specs}
    selected_feature_sets = [
        feature_set
        for feature_set in config.feature_sets
        if set(feature_set.feature_ids).issubset(selected_feature_ids)
    ]
    if not selected_feature_sets:
        raise RuntimeError(
            "No configured feature-set comparisons can be built from the selected intermediary features."
        )

    available_target_ids = [spec.target_id for spec in config.reasoning_targets]
    requested_target_ids = overrides_use.reasoning_targets or available_target_ids
    _require_known_subset(
        requested_target_ids,
        available=set(available_target_ids),
        label="reasoning targets",
    )
    selected_targets = [
        spec for spec in config.reasoning_targets if spec.target_id in set(requested_target_ids)
    ]

    available_model_ids = [spec.model_id for spec in config.reasoning_models]
    requested_model_ids = overrides_use.reasoning_models or available_model_ids
    _require_known_subset(
        requested_model_ids,
        available=set(available_model_ids),
        label="reasoning models",
    )
    selected_models = [
        spec for spec in config.reasoning_models if spec.model_id in set(requested_model_ids)
    ]

    return ResolvedRunOptions(
        config_path=overrides_use.config_path,
        run_reasoning_predictions=True,
        active_intermediary_features=[spec.feature_id for spec in selected_feature_specs],
        force_rebuild_intermediary_features=overrides_use.force_rebuild_intermediary_features,
        reasoning_targets=selected_targets,
        reasoning_models=selected_models,
        intermediary_features=selected_feature_specs,
        feature_sets=selected_feature_sets,
    )
