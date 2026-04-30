from __future__ import annotations

from dataclasses import dataclass

from src.utils.artifact_io import read_json
from src.utils.model_ids import (
    LEGACY_XGB_CLASSIFIER_MODEL_KIND,
    LEGACY_XGB_FAMILY_ID,
    LEGACY_XGB_REGRESSOR_MODEL_KIND,
    XGB_CLASSIFIER_MODEL_KIND,
    XGB_FAMILY_ID,
    XGB_REGRESSOR_MODEL_KIND,
    normalize_xgb_family_id,
    normalize_xgb_model_kind,
)
from src.utils.paths import resolve_repo_path


SUPPORTED_RUN_MODES = {
    "reproduction_mode",
    "reasoning_distillation_mode",
    "model_testing_mode",
    "saved_config_evaluation_mode",
    "xgb_calibration_mode",
    "rf_calibration_mode",
    "mlp_calibration_mode",
}

SUPPORTED_INTERMEDIARY_FEATURE_KINDS = {
    "sentence_transformer_prose_v1",
    "sentence_transformer_structured_v1",
    "llm_engineered_v1",
}

SUPPORTED_DISTILLATION_MODEL_KINDS = {
    "ridge",
    LEGACY_XGB_REGRESSOR_MODEL_KIND,
    XGB_REGRESSOR_MODEL_KIND,
    "linear_svr_regressor",
    "logreg_classifier",
    LEGACY_XGB_CLASSIFIER_MODEL_KIND,
    XGB_CLASSIFIER_MODEL_KIND,
    "linear_svm_classifier",
    "mlp_regressor",
    "elasticnet_regressor",
    "randomforest_regressor",
    "mlp_classifier",
    "elasticnet_logreg_classifier",
    "randomforest_classifier",
}

SUPPORTED_REPRODUCTION_MODEL_KINDS = {
    "nested_l2_logreg",
    "xgb_joel_classifier",
    "xgb_autoresearch_classifier",
}

SUPPORTED_TARGET_TASK_KINDS = {
    "regression",
    "classification",
}

SUPPORTED_MODEL_TESTING_FAMILIES = {
    "linear_l2",
    "linear_svm",
    LEGACY_XGB_FAMILY_ID,
    XGB_FAMILY_ID,
    "mlp",
    "elasticnet",
    "randomforest",
}


@dataclass(frozen=True)
class DatasetPaths:
    public_train_csv: str
    private_test_csv: str


@dataclass(frozen=True)
class FeatureRepositoryPaths:
    root_dir: str
    labels_path: str
    train_uuids_path: str
    test_uuids_path: str


@dataclass(frozen=True)
class DefaultRunSpec:
    run_mode: str
    target_family: str
    heldout_evaluation: bool


@dataclass(frozen=True)
class RepositoryFeatureBankSpec:
    feature_bank_id: str
    train_path: str
    test_path: str
    source_id_column: str
    enabled: bool
    feature_prefixes: list[str]
    exclude_columns: list[str]
    label_column: str | None
    all_features_binary: bool
    binary_feature_columns: list[str]


@dataclass(frozen=True)
class IntermediaryFeatureSpec:
    feature_bank_id: str
    kind: str
    enabled: bool
    embedding_model_name: str | None


@dataclass(frozen=True)
class FeatureSetSpec:
    feature_set_id: str
    feature_bank_ids: list[str]


@dataclass(frozen=True)
class TargetFamilySpec:
    family_id: str
    train_path: str
    test_path: str | None
    source_id_column: str
    target_id_column: str
    target_prefixes: list[str]
    task_kind: str
    scale_min: float | None
    scale_max: float | None
    enabled_by_default: bool


@dataclass(frozen=True)
class DistillationModelSpec:
    model_id: str
    kind: str
    supported_task_kinds: list[str]


@dataclass(frozen=True)
class CVSpec:
    n_splits: int
    shuffle: bool
    random_state: int


@dataclass(frozen=True)
class ThresholdGridSpec:
    start: float
    stop: float
    step: float


@dataclass(frozen=True)
class LambdaRankingSpec:
    c: float
    max_iter: int
    solver: str
    class_weight: str
    random_state: int


@dataclass(frozen=True)
class ReproductionExperimentSpec:
    experiment_id: str
    title: str
    feature_bank_ids: list[str]
    training_pool: str
    model_kind: str
    use_exit_override: bool
    lambda_top_k: int | None
    lambda_rank_base_bank_id: str | None
    standardize: bool


@dataclass(frozen=True)
class ReproductionSpec:
    outer_cv: CVSpec
    inner_cv: CVSpec
    threshold_grid: ThresholdGridSpec
    logistic_c_grid: list[float]
    lambda_ranking: LambdaRankingSpec
    experiments: list[ReproductionExperimentSpec]


@dataclass(frozen=True)
class ModelTestingSpec:
    candidate_feature_sets: list[str]
    default_model_families: list[str]
    save_model_configs_after_training_default: bool
    screening_repeat_cv_count: int
    screening_score_delta: float
    max_recommended_feature_sets: int
    xgb_calibration_estimators: list[int]
    use_latest_xgb_calibration_default: bool
    rf_calibration_min_samples_leaf: list[int]
    rf_calibration_max_depth: list[int | None]
    rf_calibration_max_features: list[str | float]
    use_latest_rf_calibration_default: bool
    mlp_calibration_hidden_layer_sizes: list[list[int]]
    mlp_calibration_alpha: list[float]
    use_latest_mlp_calibration_default: bool


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    description: str
    datasets: DatasetPaths
    feature_repository: FeatureRepositoryPaths
    defaults: DefaultRunSpec
    repository_feature_banks: list[RepositoryFeatureBankSpec]
    intermediary_features: list[IntermediaryFeatureSpec]
    distillation_feature_sets: list[FeatureSetSpec]
    target_families: list[TargetFamilySpec]
    distillation_models: list[DistillationModelSpec]
    reproduction: ReproductionSpec
    distillation_cv: CVSpec
    model_testing: ModelTestingSpec


def _validate_cv_spec(label: str, spec: CVSpec) -> None:
    if spec.n_splits < 2:
        raise RuntimeError(f"{label} requires n_splits >= 2.")


def _validate_defaults(
    defaults: DefaultRunSpec,
    *,
    target_family_ids: set[str],
) -> None:
    if defaults.run_mode not in SUPPORTED_RUN_MODES:
        raise RuntimeError(f"Unsupported defaults.run_mode '{defaults.run_mode}'.")
    if defaults.target_family not in target_family_ids:
        raise RuntimeError(
            f"defaults.target_family '{defaults.target_family}' is not declared in target_families."
        )


def _validate_repository_feature_banks(specs: list[RepositoryFeatureBankSpec]) -> None:
    if not specs:
        raise RuntimeError("At least one repository feature bank must be declared.")
    seen: set[str] = set()
    for spec in specs:
        feature_bank_id = spec.feature_bank_id.strip()
        if not feature_bank_id:
            raise RuntimeError("repository_feature_banks entries require a non-empty feature_bank_id.")
        if feature_bank_id in seen:
            raise RuntimeError(f"Duplicate repository feature_bank_id '{feature_bank_id}'.")
        seen.add(feature_bank_id)


def _validate_intermediary_features(specs: list[IntermediaryFeatureSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        feature_bank_id = spec.feature_bank_id.strip()
        if not feature_bank_id:
            raise RuntimeError("intermediary_features entries require a non-empty feature_bank_id.")
        if feature_bank_id in seen:
            raise RuntimeError(f"Duplicate intermediary feature_bank_id '{feature_bank_id}'.")
        if spec.kind not in SUPPORTED_INTERMEDIARY_FEATURE_KINDS:
            raise RuntimeError(f"Unsupported intermediary feature kind '{spec.kind}'.")
        if spec.kind.startswith("sentence_transformer") and not spec.embedding_model_name:
            raise RuntimeError(
                f"Intermediary feature '{feature_bank_id}' requires embedding_model_name."
            )
        seen.add(feature_bank_id)


def _validate_feature_sets(
    specs: list[FeatureSetSpec],
    *,
    available_feature_bank_ids: set[str],
) -> None:
    if not specs:
        raise RuntimeError("At least one distillation feature set must be declared.")
    seen: set[str] = set()
    for spec in specs:
        feature_set_id = spec.feature_set_id.strip()
        if not feature_set_id:
            raise RuntimeError("distillation_feature_sets entries require a non-empty feature_set_id.")
        if feature_set_id in seen:
            raise RuntimeError(f"Duplicate distillation feature_set_id '{feature_set_id}'.")
        if not spec.feature_bank_ids:
            raise RuntimeError(f"Feature set '{feature_set_id}' must include at least one feature_bank_id.")
        missing = [
            feature_bank_id
            for feature_bank_id in spec.feature_bank_ids
            if feature_bank_id not in available_feature_bank_ids
        ]
        if missing:
            raise RuntimeError(
                f"Feature set '{feature_set_id}' references unknown feature banks: {missing}"
            )
        seen.add(feature_set_id)


def _validate_target_families(specs: list[TargetFamilySpec]) -> None:
    if not specs:
        raise RuntimeError("At least one target family must be declared.")
    seen: set[str] = set()
    for spec in specs:
        family_id = spec.family_id.strip()
        if not family_id:
            raise RuntimeError("target_families entries require a non-empty family_id.")
        if family_id in seen:
            raise RuntimeError(f"Duplicate target family '{family_id}'.")
        if spec.task_kind not in SUPPORTED_TARGET_TASK_KINDS:
            raise RuntimeError(
                f"Target family '{family_id}' has unsupported task_kind '{spec.task_kind}'."
            )
        if not spec.target_prefixes:
            raise RuntimeError(f"Target family '{family_id}' must declare target_prefixes.")
        if spec.task_kind == "regression":
            if spec.scale_min is None or spec.scale_max is None:
                raise RuntimeError(
                    f"Regression target family '{family_id}' requires scale_min and scale_max."
                )
            if spec.scale_min >= spec.scale_max:
                raise RuntimeError(
                    f"Regression target family '{family_id}' requires scale_min < scale_max."
                )
        seen.add(family_id)


def _validate_distillation_models(specs: list[DistillationModelSpec]) -> None:
    if not specs:
        raise RuntimeError("At least one distillation model must be declared.")
    seen: set[str] = set()
    for spec in specs:
        model_id = spec.model_id.strip()
        if not model_id:
            raise RuntimeError("distillation_models entries require a non-empty model_id.")
        if model_id in seen:
            raise RuntimeError(f"Duplicate distillation model_id '{model_id}'.")
        if spec.kind not in SUPPORTED_DISTILLATION_MODEL_KINDS:
            raise RuntimeError(f"Unsupported distillation model kind '{spec.kind}'.")
        if not spec.supported_task_kinds:
            raise RuntimeError(
                f"Distillation model '{model_id}' must declare supported_task_kinds."
            )
        unsupported = [
            task_kind
            for task_kind in spec.supported_task_kinds
            if task_kind not in SUPPORTED_TARGET_TASK_KINDS
        ]
        if unsupported:
            raise RuntimeError(
                f"Distillation model '{model_id}' has unsupported task kinds: {unsupported}"
            )
        seen.add(model_id)


def _validate_reproduction(
    spec: ReproductionSpec,
    *,
    available_feature_bank_ids: set[str],
) -> None:
    _validate_cv_spec("reproduction.outer_cv", spec.outer_cv)
    _validate_cv_spec("reproduction.inner_cv", spec.inner_cv)
    if not spec.logistic_c_grid:
        raise RuntimeError("reproduction.logistic_c_grid must contain at least one value.")
    if spec.threshold_grid.step <= 0:
        raise RuntimeError("reproduction.threshold_grid.step must be > 0.")
    if spec.threshold_grid.start >= spec.threshold_grid.stop:
        raise RuntimeError("reproduction.threshold_grid requires start < stop.")
    if not spec.experiments:
        raise RuntimeError("At least one reproduction experiment must be declared.")

    seen: set[str] = set()
    for experiment in spec.experiments:
        experiment_id = experiment.experiment_id.strip()
        if not experiment_id:
            raise RuntimeError("reproduction.experiments entries require a non-empty experiment_id.")
        if experiment_id in seen:
            raise RuntimeError(f"Duplicate reproduction experiment_id '{experiment_id}'.")
        if not experiment.feature_bank_ids:
            raise RuntimeError(
                f"Reproduction experiment '{experiment_id}' must declare feature_bank_ids."
            )
        missing = [
            feature_bank_id
            for feature_bank_id in experiment.feature_bank_ids
            if feature_bank_id not in available_feature_bank_ids
        ]
        if missing:
            raise RuntimeError(
                f"Reproduction experiment '{experiment_id}' references unknown feature banks: {missing}"
            )
        if experiment.training_pool not in {"full", "llm_engineering_non_seed"}:
            raise RuntimeError(
                f"Reproduction experiment '{experiment_id}' has unsupported training_pool "
                f"'{experiment.training_pool}'."
            )
        if experiment.model_kind not in SUPPORTED_REPRODUCTION_MODEL_KINDS:
            raise RuntimeError(
                f"Reproduction experiment '{experiment_id}' has unsupported model_kind "
                f"'{experiment.model_kind}'."
            )
        if experiment.lambda_top_k is not None:
            if experiment.lambda_top_k <= 0:
                raise RuntimeError(
                    f"Reproduction experiment '{experiment_id}' requires lambda_top_k > 0."
                )
            if experiment.lambda_rank_base_bank_id not in available_feature_bank_ids:
                raise RuntimeError(
                    f"Reproduction experiment '{experiment_id}' references unknown "
                    f"lambda_rank_base_bank_id '{experiment.lambda_rank_base_bank_id}'."
                )
        seen.add(experiment_id)


def _validate_model_testing(
    spec: ModelTestingSpec,
    *,
    available_feature_set_ids: set[str],
) -> None:
    if not spec.candidate_feature_sets:
        raise RuntimeError("model_testing.candidate_feature_sets must include at least one feature set.")
    unknown_feature_sets = [
        feature_set_id
        for feature_set_id in spec.candidate_feature_sets
        if feature_set_id not in available_feature_set_ids
    ]
    if unknown_feature_sets:
        raise RuntimeError(
            "model_testing.candidate_feature_sets references unknown feature sets: "
            f"{unknown_feature_sets}"
        )
    if not spec.default_model_families:
        raise RuntimeError("model_testing.default_model_families must include at least one model family.")
    unknown_model_families = [
        family for family in spec.default_model_families if family not in SUPPORTED_MODEL_TESTING_FAMILIES
    ]
    if unknown_model_families:
        raise RuntimeError(
            "model_testing.default_model_families contains unsupported values: "
            f"{unknown_model_families}"
        )
    if spec.screening_repeat_cv_count < 1:
        raise RuntimeError("model_testing.screening_repeat_cv_count must be >= 1.")
    if spec.max_recommended_feature_sets < 1:
        raise RuntimeError("model_testing.max_recommended_feature_sets must be >= 1.")
    if spec.screening_score_delta < 0.0:
        raise RuntimeError("model_testing.screening_score_delta must be >= 0.")
    if not spec.xgb_calibration_estimators:
        raise RuntimeError("model_testing.xgb_calibration_estimators must include at least one value.")
    if any(value <= 0 for value in spec.xgb_calibration_estimators):
        raise RuntimeError("model_testing.xgb_calibration_estimators values must all be > 0.")
    if not spec.rf_calibration_min_samples_leaf:
        raise RuntimeError("model_testing.rf_calibration_min_samples_leaf must include at least one value.")
    if any(value <= 0 for value in spec.rf_calibration_min_samples_leaf):
        raise RuntimeError("model_testing.rf_calibration_min_samples_leaf values must all be > 0.")
    if not spec.rf_calibration_max_depth:
        raise RuntimeError("model_testing.rf_calibration_max_depth must include at least one value.")
    for depth in spec.rf_calibration_max_depth:
        if depth is not None and depth <= 0:
            raise RuntimeError("model_testing.rf_calibration_max_depth values must be positive integers or null.")
    if not spec.rf_calibration_max_features:
        raise RuntimeError("model_testing.rf_calibration_max_features must include at least one value.")
    for value in spec.rf_calibration_max_features:
        if isinstance(value, str):
            if value not in {"sqrt", "log2"}:
                raise RuntimeError(
                    "model_testing.rf_calibration_max_features string values must be one of: sqrt, log2."
                )
        elif isinstance(value, (int, float)):
            float_value = float(value)
            if not (0.0 < float_value <= 1.0):
                raise RuntimeError(
                    "model_testing.rf_calibration_max_features numeric values must be in (0, 1]."
                )
        else:
            raise RuntimeError(
                "model_testing.rf_calibration_max_features values must be string or numeric."
            )
    if not spec.mlp_calibration_hidden_layer_sizes:
        raise RuntimeError("model_testing.mlp_calibration_hidden_layer_sizes must include at least one value.")
    for layers in spec.mlp_calibration_hidden_layer_sizes:
        if not layers or any(int(v) <= 0 for v in layers):
            raise RuntimeError(
                "model_testing.mlp_calibration_hidden_layer_sizes entries must be non-empty positive integer lists."
            )
    if not spec.mlp_calibration_alpha:
        raise RuntimeError("model_testing.mlp_calibration_alpha must include at least one value.")
    if any(float(v) <= 0 for v in spec.mlp_calibration_alpha):
        raise RuntimeError("model_testing.mlp_calibration_alpha values must be > 0.")


def _resolve_cv_spec(payload: dict[str, object], *, default_n_splits: int) -> CVSpec:
    return CVSpec(
        n_splits=int(payload.get("n_splits", default_n_splits)),
        shuffle=bool(payload.get("shuffle", True)),
        random_state=int(payload.get("random_state", 42)),
    )


def load_experiment_config(path: str) -> ExperimentConfig:
    payload = read_json(resolve_repo_path(path))

    repository_feature_banks = [
        RepositoryFeatureBankSpec(
            feature_bank_id=str(item["feature_bank_id"]),
            train_path=str(resolve_repo_path(str(item["train_path"]))),
            test_path=str(resolve_repo_path(str(item["test_path"]))),
            source_id_column=str(item.get("source_id_column", "founder_uuid")),
            enabled=bool(item.get("enabled", True)),
            feature_prefixes=[str(value) for value in item.get("feature_prefixes", [])],
            exclude_columns=[str(value) for value in item.get("exclude_columns", [])],
            label_column=str(item["label_column"]) if item.get("label_column") else None,
            all_features_binary=bool(item.get("all_features_binary", False)),
            binary_feature_columns=[str(value) for value in item.get("binary_feature_columns", [])],
        )
        for item in payload.get("repository_feature_banks", [])
    ]
    _validate_repository_feature_banks(repository_feature_banks)

    intermediary_features = [
        IntermediaryFeatureSpec(
            feature_bank_id=str(item["feature_bank_id"]),
            kind=str(item["kind"]),
            enabled=bool(item.get("enabled", True)),
            embedding_model_name=(
                str(item["embedding_model_name"])
                if item.get("embedding_model_name") is not None
                else None
            ),
        )
        for item in payload.get("intermediary_features", [])
    ]
    _validate_intermediary_features(intermediary_features)

    target_families = [
        TargetFamilySpec(
            family_id=str(item["family_id"]),
            train_path=str(resolve_repo_path(str(item["train_path"]))),
            test_path=(
                str(resolve_repo_path(str(item["test_path"])))
                if item.get("test_path")
                else None
            ),
            source_id_column=str(item.get("source_id_column", "founder_uuid")),
            target_id_column=str(item.get("target_id_column", "founder_uuid")),
            target_prefixes=[str(value) for value in item.get("target_prefixes", [])],
            task_kind=str(item["task_kind"]),
            scale_min=(
                float(item["scale_min"])
                if item.get("scale_min") is not None
                else None
            ),
            scale_max=(
                float(item["scale_max"])
                if item.get("scale_max") is not None
                else None
            ),
            enabled_by_default=bool(item.get("enabled_by_default", False)),
        )
        for item in payload.get("target_families", [])
    ]
    _validate_target_families(target_families)

    distillation_models = [
        DistillationModelSpec(
            model_id=normalize_xgb_model_kind(str(item["model_id"])),
            kind=normalize_xgb_model_kind(str(item["kind"])),
            supported_task_kinds=[str(value) for value in item.get("supported_task_kinds", [])],
        )
        for item in payload.get("distillation_models", [])
    ]
    _validate_distillation_models(distillation_models)

    all_feature_bank_ids = {
        *(spec.feature_bank_id for spec in repository_feature_banks),
        *(spec.feature_bank_id for spec in intermediary_features),
    }
    distillation_feature_sets = [
        FeatureSetSpec(
            feature_set_id=str(item["feature_set_id"]),
            feature_bank_ids=[str(value) for value in item.get("feature_bank_ids", [])],
        )
        for item in payload.get("distillation_feature_sets", [])
    ]
    _validate_feature_sets(
        distillation_feature_sets,
        available_feature_bank_ids=all_feature_bank_ids,
    )
    distillation_feature_set_ids = {spec.feature_set_id for spec in distillation_feature_sets}

    defaults_payload = payload.get("defaults", {})
    defaults = DefaultRunSpec(
        run_mode=str(defaults_payload.get("run_mode", "reproduction_mode")),
        target_family=str(defaults_payload.get("target_family", target_families[0].family_id)),
        heldout_evaluation=bool(defaults_payload.get("heldout_evaluation", False)),
    )
    _validate_defaults(
        defaults,
        target_family_ids={spec.family_id for spec in target_families},
    )

    reproduction_payload = payload["reproduction"]
    reproduction = ReproductionSpec(
        outer_cv=_resolve_cv_spec(
            reproduction_payload.get("outer_cv", {}),
            default_n_splits=5,
        ),
        inner_cv=_resolve_cv_spec(
            reproduction_payload.get("inner_cv", {}),
            default_n_splits=3,
        ),
        threshold_grid=ThresholdGridSpec(
            start=float(reproduction_payload.get("threshold_grid", {}).get("start", 0.05)),
            stop=float(reproduction_payload.get("threshold_grid", {}).get("stop", 0.95)),
            step=float(reproduction_payload.get("threshold_grid", {}).get("step", 0.01)),
        ),
        logistic_c_grid=[
            float(value) for value in reproduction_payload.get("logistic_c_grid", [])
        ],
        lambda_ranking=LambdaRankingSpec(
            c=float(reproduction_payload.get("lambda_ranking", {}).get("c", 0.05)),
            max_iter=int(reproduction_payload.get("lambda_ranking", {}).get("max_iter", 2000)),
            solver=str(reproduction_payload.get("lambda_ranking", {}).get("solver", "liblinear")),
            class_weight=str(
                reproduction_payload.get("lambda_ranking", {}).get("class_weight", "balanced")
            ),
            random_state=int(
                reproduction_payload.get("lambda_ranking", {}).get("random_state", 42)
            ),
        ),
        experiments=[
            ReproductionExperimentSpec(
                experiment_id=str(item["experiment_id"]),
                title=str(item.get("title", item["experiment_id"])),
                feature_bank_ids=[str(value) for value in item.get("feature_bank_ids", [])],
                training_pool=str(item.get("training_pool", "full")),
                model_kind=str(item["model_kind"]),
                use_exit_override=bool(item.get("use_exit_override", False)),
                lambda_top_k=(
                    int(item["lambda_top_k"])
                    if item.get("lambda_top_k") is not None
                    else None
                ),
                lambda_rank_base_bank_id=(
                    str(item["lambda_rank_base_bank_id"])
                    if item.get("lambda_rank_base_bank_id") is not None
                    else None
                ),
                standardize=bool(item.get("standardize", True)),
            )
            for item in reproduction_payload.get("experiments", [])
        ],
    )
    _validate_reproduction(
        reproduction,
        available_feature_bank_ids=all_feature_bank_ids,
    )

    distillation_cv = _resolve_cv_spec(
        payload.get("distillation_cv", {}),
        default_n_splits=3,
    )
    _validate_cv_spec("distillation_cv", distillation_cv)

    model_testing_payload = payload.get("model_testing", {})
    default_candidate_feature_sets = [
        spec.feature_set_id for spec in distillation_feature_sets[:4]
    ] or [spec.feature_set_id for spec in distillation_feature_sets]
    model_testing = ModelTestingSpec(
        candidate_feature_sets=[
            str(value)
            for value in model_testing_payload.get(
                "candidate_feature_sets",
                default_candidate_feature_sets,
            )
        ],
        default_model_families=[
            normalize_xgb_family_id(str(value))
            for value in model_testing_payload.get(
                "default_model_families",
                ["linear_l2"],
            )
        ],
        save_model_configs_after_training_default=bool(
            model_testing_payload.get(
                "save_model_configs_after_training_default",
                model_testing_payload.get("run_advanced_models_default", False),
            )
        ),
        screening_repeat_cv_count=int(
            model_testing_payload.get("screening_repeat_cv_count", 16)
        ),
        screening_score_delta=float(model_testing_payload.get("screening_score_delta", 0.005)),
        max_recommended_feature_sets=int(
            model_testing_payload.get("max_recommended_feature_sets", 3)
        ),
        xgb_calibration_estimators=[
            int(value)
            for value in model_testing_payload.get(
                "xgb_calibration_estimators",
                [40, 80, 120, 180, 240, 320],
            )
        ],
        use_latest_xgb_calibration_default=bool(
            model_testing_payload.get("use_latest_xgb_calibration_default", False)
        ),
        rf_calibration_min_samples_leaf=[
            int(value)
            for value in model_testing_payload.get(
                "rf_calibration_min_samples_leaf",
                [2, 3, 4, 5],
            )
        ],
        rf_calibration_max_depth=[
            (None if value is None else int(value))
            for value in model_testing_payload.get(
                "rf_calibration_max_depth",
                [None, 10, 20, 30],
            )
        ],
        rf_calibration_max_features=[
            (str(value) if isinstance(value, str) else float(value))
            for value in model_testing_payload.get(
                "rf_calibration_max_features",
                ["sqrt", 0.5],
            )
        ],
        use_latest_rf_calibration_default=bool(
            model_testing_payload.get("use_latest_rf_calibration_default", False)
        ),
        mlp_calibration_hidden_layer_sizes=[
            [int(v) for v in layer]
            for layer in model_testing_payload.get(
                "mlp_calibration_hidden_layer_sizes",
                [[8], [16], [32], [16, 8]],
            )
        ],
        mlp_calibration_alpha=[
            float(value)
            for value in model_testing_payload.get(
                "mlp_calibration_alpha",
                [0.001, 0.01, 0.1],
            )
        ],
        use_latest_mlp_calibration_default=bool(
            model_testing_payload.get("use_latest_mlp_calibration_default", False)
        ),
    )
    _validate_model_testing(
        model_testing,
        available_feature_set_ids=distillation_feature_set_ids,
    )

    return ExperimentConfig(
        experiment_id=str(payload["experiment_id"]),
        description=str(payload.get("description", "")),
        datasets=DatasetPaths(
            public_train_csv=str(resolve_repo_path(str(payload["datasets"]["public_train_csv"]))),
            private_test_csv=str(resolve_repo_path(str(payload["datasets"]["private_test_csv"]))),
        ),
        feature_repository=FeatureRepositoryPaths(
            root_dir=str(resolve_repo_path(str(payload["feature_repository"]["root_dir"]))),
            labels_path=str(resolve_repo_path(str(payload["feature_repository"]["labels_path"]))),
            train_uuids_path=str(
                resolve_repo_path(str(payload["feature_repository"]["train_uuids_path"]))
            ),
            test_uuids_path=str(
                resolve_repo_path(str(payload["feature_repository"]["test_uuids_path"]))
            ),
        ),
        defaults=defaults,
        repository_feature_banks=repository_feature_banks,
        intermediary_features=intermediary_features,
        distillation_feature_sets=distillation_feature_sets,
        target_families=target_families,
        distillation_models=distillation_models,
        reproduction=reproduction,
        distillation_cv=distillation_cv,
        model_testing=model_testing,
    )
