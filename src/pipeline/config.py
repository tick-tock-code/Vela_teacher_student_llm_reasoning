from __future__ import annotations

from dataclasses import dataclass

from src.utils.artifact_io import read_json
from src.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class DatasetPaths:
    public_train_csv: str
    private_test_csv: str


@dataclass(frozen=True)
class InputFeatureSpec:
    kind: str
    train_path: str | None
    test_path: str | None
    source_id_column: str
    target_id_column: str
    feature_regex: str | None
    expected_feature_count: int | None


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
class PromotionSpec:
    mode: str
    approved: bool


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    description: str
    datasets: DatasetPaths
    input_features: InputFeatureSpec
    reasoning_target_bank: ReasoningTargetBankSpec
    reasoning_targets: list[ReasoningTargetSpec]
    reasoning_models: list[ModelSpec]
    downstream_models: list[ModelSpec]
    cv: CVSpec
    promotion: PromotionSpec


def _validate_input_feature_spec(spec: InputFeatureSpec) -> None:
    supported_kinds = {
        "founder_baseline_v1",
        "table_bank",
        "llm_engineered_rules",
        "sentence_transformer_embeddings",
    }
    if spec.kind not in supported_kinds:
        raise RuntimeError(f"Unsupported input_features.kind '{spec.kind}'.")
    if spec.kind == "table_bank":
        if not spec.train_path or not spec.test_path:
            raise RuntimeError("input_features.kind 'table_bank' requires both train_path and test_path.")
        if not spec.feature_regex:
            raise RuntimeError("input_features.kind 'table_bank' requires feature_regex.")
        if spec.expected_feature_count is None:
            raise RuntimeError("input_features.kind 'table_bank' requires expected_feature_count.")


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


def load_experiment_config(path: str) -> ExperimentConfig:
    payload = read_json(resolve_repo_path(path))

    datasets_payload = payload["datasets"]
    input_features_payload = payload["input_features"]
    target_bank_payload = payload["reasoning_target_bank"]
    cv_payload = payload["cv"]
    promotion_payload = payload["promotion"]

    reasoning_models = [
        ModelSpec(model_id=str(item["model_id"]), kind=str(item["kind"]))
        for item in payload.get("reasoning_models", [])
    ]
    downstream_models = [
        ModelSpec(model_id=str(item["model_id"]), kind=str(item["kind"]))
        for item in payload.get("downstream_models", [])
    ]
    reasoning_targets = [
        ReasoningTargetSpec(target_id=str(item["target_id"]))
        for item in payload.get("reasoning_targets", [])
    ]
    _validate_reasoning_targets(reasoning_targets)

    promotion = PromotionSpec(
        mode=str(promotion_payload.get("mode", "manual")),
        approved=bool(promotion_payload.get("approved", False)),
    )
    if promotion.mode != "manual":
        raise RuntimeError(f"Unsupported promotion mode '{promotion.mode}'. v1 only supports 'manual'.")

    input_features = InputFeatureSpec(
        kind=str(input_features_payload["kind"]),
        train_path=(
            str(resolve_repo_path(str(input_features_payload["train_path"])))
            if input_features_payload.get("train_path")
            else None
        ),
        test_path=(
            str(resolve_repo_path(str(input_features_payload["test_path"])))
            if input_features_payload.get("test_path")
            else None
        ),
        source_id_column=str(input_features_payload.get("source_id_column", "founder_uuid")),
        target_id_column=str(input_features_payload.get("target_id_column", "founder_uuid")),
        feature_regex=(
            str(input_features_payload["feature_regex"])
            if input_features_payload.get("feature_regex") is not None
            else None
        ),
        expected_feature_count=(
            int(input_features_payload["expected_feature_count"])
            if input_features_payload.get("expected_feature_count") is not None
            else None
        ),
    )
    _validate_input_feature_spec(input_features)

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
        input_features=input_features,
        reasoning_target_bank=reasoning_target_bank,
        reasoning_targets=reasoning_targets,
        reasoning_models=reasoning_models,
        downstream_models=downstream_models,
        cv=CVSpec(
            n_splits=int(cv_payload.get("n_splits", 3)),
            shuffle=bool(cv_payload.get("shuffle", True)),
            random_state=int(cv_payload.get("random_state", 42)),
        ),
        promotion=promotion,
    )
