from __future__ import annotations

from dataclasses import dataclass

from src.utils.artifact_io import read_json
from src.utils.paths import resolve_repo_path


SUPPORTED_INTERMEDIARY_FEATURE_KINDS = {
    "vcbench_mirror_baseline_v1",
    "sentence_transformer_prose_v1",
    "sentence_transformer_structured_v1",
    "llm_engineered_v1",
}

SUPPORTED_REASONING_MODEL_KINDS = {
    "ridge",
    "xgb1_regressor",
}


@dataclass(frozen=True)
class DatasetPaths:
    public_train_csv: str
    private_test_csv: str


@dataclass(frozen=True)
class IntermediaryFeatureSpec:
    feature_id: str
    kind: str
    enabled: bool
    embedding_model_name: str | None


@dataclass(frozen=True)
class FeatureSetSpec:
    feature_set_id: str
    feature_ids: list[str]


@dataclass(frozen=True)
class ReasoningTargetBankSpec:
    train_path: str
    test_path: str | None
    source_id_column: str
    target_id_column: str
    target_regex: str
    scale_min: float
    scale_max: float
    prediction_mode: str


@dataclass(frozen=True)
class ReasoningTargetSpec:
    target_id: str


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    kind: str


@dataclass(frozen=True)
class CVSpec:
    n_splits: int
    shuffle: bool
    random_state: int


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    description: str
    datasets: DatasetPaths
    reasoning_target_bank: ReasoningTargetBankSpec
    reasoning_targets: list[ReasoningTargetSpec]
    reasoning_models: list[ModelSpec]
    intermediary_features: list[IntermediaryFeatureSpec]
    feature_sets: list[FeatureSetSpec]
    cv: CVSpec


def _validate_reasoning_target_bank_spec(spec: ReasoningTargetBankSpec) -> None:
    if spec.prediction_mode != "regression":
        raise RuntimeError(
            f"Unsupported reasoning_target_bank.prediction_mode '{spec.prediction_mode}'. "
            "v1 only supports 'regression'."
        )
    if spec.scale_min >= spec.scale_max:
        raise RuntimeError("reasoning_target_bank requires scale_min < scale_max.")
    if not spec.target_regex.strip():
        raise RuntimeError("reasoning_target_bank requires a non-empty target_regex.")


def _validate_reasoning_targets(specs: list[ReasoningTargetSpec]) -> None:
    if not specs:
        raise RuntimeError("At least one reasoning target must be declared.")
    seen: set[str] = set()
    for spec in specs:
        target_id = spec.target_id.strip()
        if not target_id:
            raise RuntimeError("Reasoning target entries require a non-empty target_id.")
        if target_id in seen:
            raise RuntimeError(f"Duplicate reasoning target '{target_id}' in config.")
        seen.add(target_id)


def _validate_reasoning_models(specs: list[ModelSpec]) -> None:
    if not specs:
        raise RuntimeError("At least one reasoning model must be declared.")
    seen: set[str] = set()
    for spec in specs:
        if not spec.model_id.strip():
            raise RuntimeError("Reasoning model entries require a non-empty model_id.")
        if spec.model_id in seen:
            raise RuntimeError(f"Duplicate reasoning model_id '{spec.model_id}' in config.")
        if spec.kind not in SUPPORTED_REASONING_MODEL_KINDS:
            raise RuntimeError(f"Unsupported reasoning model kind '{spec.kind}'.")
        seen.add(spec.model_id)


def _validate_intermediary_features(specs: list[IntermediaryFeatureSpec]) -> None:
    if not specs:
        raise RuntimeError("At least one intermediary feature builder must be declared.")
    seen: set[str] = set()
    for spec in specs:
        if not spec.feature_id.strip():
            raise RuntimeError("Intermediary feature entries require a non-empty feature_id.")
        if spec.feature_id in seen:
            raise RuntimeError(f"Duplicate intermediary feature_id '{spec.feature_id}' in config.")
        if spec.kind not in SUPPORTED_INTERMEDIARY_FEATURE_KINDS:
            raise RuntimeError(f"Unsupported intermediary feature kind '{spec.kind}'.")
        if spec.kind.startswith("sentence_transformer") and not spec.embedding_model_name:
            raise RuntimeError(
                f"Intermediary feature '{spec.feature_id}' requires embedding_model_name."
            )
        seen.add(spec.feature_id)


def _validate_feature_sets(
    specs: list[FeatureSetSpec],
    *,
    available_feature_ids: set[str],
) -> None:
    if not specs:
        raise RuntimeError("At least one feature-set combination must be declared.")
    seen: set[str] = set()
    for spec in specs:
        feature_set_id = spec.feature_set_id.strip()
        if not feature_set_id:
            raise RuntimeError("Feature-set entries require a non-empty feature_set_id.")
        if feature_set_id in seen:
            raise RuntimeError(f"Duplicate feature_set_id '{feature_set_id}' in config.")
        if not spec.feature_ids:
            raise RuntimeError(f"Feature set '{feature_set_id}' must include at least one feature_id.")
        missing = [feature_id for feature_id in spec.feature_ids if feature_id not in available_feature_ids]
        if missing:
            raise RuntimeError(
                f"Feature set '{feature_set_id}' references unknown intermediary features: {missing}"
            )
        seen.add(feature_set_id)


def load_experiment_config(path: str) -> ExperimentConfig:
    payload = read_json(resolve_repo_path(path))

    datasets_payload = payload["datasets"]
    target_bank_payload = payload["reasoning_target_bank"]
    cv_payload = payload["cv"]

    reasoning_models = [
        ModelSpec(model_id=str(item["model_id"]), kind=str(item["kind"]))
        for item in payload.get("reasoning_models", [])
    ]
    _validate_reasoning_models(reasoning_models)

    reasoning_targets = [
        ReasoningTargetSpec(target_id=str(item["target_id"]))
        for item in payload.get("reasoning_targets", [])
    ]
    _validate_reasoning_targets(reasoning_targets)

    intermediary_features = [
        IntermediaryFeatureSpec(
            feature_id=str(item["feature_id"]),
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

    feature_sets = [
        FeatureSetSpec(
            feature_set_id=str(item["feature_set_id"]),
            feature_ids=[str(feature_id) for feature_id in item.get("feature_ids", [])],
        )
        for item in payload.get("feature_sets", [])
    ]
    _validate_feature_sets(
        feature_sets,
        available_feature_ids={spec.feature_id for spec in intermediary_features},
    )

    reasoning_target_bank = ReasoningTargetBankSpec(
        train_path=str(resolve_repo_path(str(target_bank_payload["train_path"]))),
        test_path=(
            str(resolve_repo_path(str(target_bank_payload["test_path"])))
            if target_bank_payload.get("test_path")
            else None
        ),
        source_id_column=str(target_bank_payload.get("source_id_column", "founder_uuid")),
        target_id_column=str(target_bank_payload.get("target_id_column", "founder_uuid")),
        target_regex=str(target_bank_payload["target_regex"]),
        scale_min=float(target_bank_payload.get("scale_min", 0.0)),
        scale_max=float(target_bank_payload.get("scale_max", 1.0)),
        prediction_mode=str(target_bank_payload.get("prediction_mode", "regression")),
    )
    _validate_reasoning_target_bank_spec(reasoning_target_bank)

    return ExperimentConfig(
        experiment_id=str(payload["experiment_id"]),
        description=str(payload.get("description", "")),
        datasets=DatasetPaths(
            public_train_csv=str(resolve_repo_path(str(datasets_payload["public_train_csv"]))),
            private_test_csv=str(resolve_repo_path(str(datasets_payload["private_test_csv"]))),
        ),
        reasoning_target_bank=reasoning_target_bank,
        reasoning_targets=reasoning_targets,
        reasoning_models=reasoning_models,
        intermediary_features=intermediary_features,
        feature_sets=feature_sets,
        cv=CVSpec(
            n_splits=int(cv_payload.get("n_splits", 3)),
            shuffle=bool(cv_payload.get("shuffle", True)),
            random_state=int(cv_payload.get("random_state", 42)),
        ),
    )
