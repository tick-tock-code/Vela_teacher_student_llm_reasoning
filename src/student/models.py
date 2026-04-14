from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.utils.dependencies import require_dependency


XGB_CLASSIFIER_PARAMS = {
    "n_estimators": 227,
    "max_depth": 1,
    "learning_rate": 0.0674,
    "subsample": 0.949,
    "colsample_bytree": 0.413,
    "scale_pos_weight": 10,
    "min_child_weight": 14,
    "gamma": 4.19,
    "reg_alpha": 0.73,
    "reg_lambda": 15.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": 1,
}

XGB_REGRESSOR_PARAMS = {
    "n_estimators": 227,
    "max_depth": 1,
    "learning_rate": 0.0674,
    "subsample": 0.949,
    "colsample_bytree": 0.413,
    "min_child_weight": 4,
    "reg_alpha": 0.73,
    "reg_lambda": 15.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_jobs": 1,
}

JOEL_XGB_CLASSIFIER_PARAMS = {
    "n_estimators": 227,
    "max_depth": 1,
    "learning_rate": 0.0674,
    "subsample": 0.949,
    "colsample_bytree": 0.413,
    "scale_pos_weight": 10,
    "min_child_weight": 14,
    "gamma": 4.19,
    "reg_alpha": 0.73,
    "reg_lambda": 15.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": 1,
}

AUTORESEARCH_XGB_CLASSIFIER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "scale_pos_weight": 10,
    "min_child_weight": 20,
    "gamma": 4.0,
    "reg_alpha": 1.0,
    "reg_lambda": 15.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": 1,
}


def build_reasoning_regressor(
    model_kind: str,
    *,
    random_state: int,
    param_overrides: dict[str, Any] | None = None,
) -> Any:
    overrides = param_overrides or {}
    if model_kind == "ridge":
        return Ridge(alpha=float(overrides.get("alpha", 1.0)))
    if model_kind == "xgb1_regressor":
        require_dependency("xgboost", "build the xgb1_regressor reasoning model")
        import xgboost as xgb  # type: ignore

        params = {**XGB_REGRESSOR_PARAMS, **overrides}
        return xgb.XGBRegressor(random_state=random_state, **params)
    if model_kind == "mlp_regressor":
        return MLPRegressor(
            hidden_layer_sizes=tuple(overrides.get("hidden_layer_sizes", (128, 64))),
            alpha=float(overrides.get("alpha", 1e-4)),
            learning_rate_init=float(overrides.get("learning_rate_init", 1e-3)),
            max_iter=int(overrides.get("max_iter", 600)),
            early_stopping=bool(overrides.get("early_stopping", True)),
            random_state=random_state,
        )
    if model_kind == "elasticnet_regressor":
        return ElasticNet(
            alpha=float(overrides.get("alpha", 0.01)),
            l1_ratio=float(overrides.get("l1_ratio", 0.5)),
            max_iter=int(overrides.get("max_iter", 5000)),
            random_state=random_state,
        )
    if model_kind == "randomforest_regressor":
        return RandomForestRegressor(
            n_estimators=int(overrides.get("n_estimators", 500)),
            max_depth=(
                int(overrides["max_depth"])
                if overrides.get("max_depth") is not None
                else None
            ),
            min_samples_leaf=int(overrides.get("min_samples_leaf", 1)),
            n_jobs=1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported reasoning model kind: {model_kind}")


def build_downstream_classifier(model_kind: str, *, random_state: int) -> Any:
    if model_kind == "lr_classifier":
        return LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=3000,
            random_state=random_state,
        )
    if model_kind == "xgb1_classifier":
        require_dependency("xgboost", "build the xgb1_classifier downstream model")
        import xgboost as xgb  # type: ignore

        return xgb.XGBClassifier(random_state=random_state, **XGB_CLASSIFIER_PARAMS)
    raise ValueError(f"Unsupported downstream model kind: {model_kind}")


def build_reasoning_classifier(
    model_kind: str,
    *,
    random_state: int,
    param_overrides: dict[str, Any] | None = None,
) -> Any:
    overrides = param_overrides or {}
    if model_kind == "logreg_classifier":
        return LogisticRegression(
            solver="lbfgs",
            C=float(overrides.get("C", 1.0)),
            max_iter=3000,
            random_state=random_state,
        )
    if model_kind == "xgb1_classifier":
        require_dependency("xgboost", "build the xgb1_classifier reasoning model")
        import xgboost as xgb  # type: ignore

        params = {**XGB_CLASSIFIER_PARAMS, **overrides}
        return xgb.XGBClassifier(random_state=random_state, **params)
    if model_kind == "mlp_classifier":
        return MLPClassifier(
            hidden_layer_sizes=tuple(overrides.get("hidden_layer_sizes", (128, 64))),
            alpha=float(overrides.get("alpha", 1e-4)),
            learning_rate_init=float(overrides.get("learning_rate_init", 1e-3)),
            max_iter=int(overrides.get("max_iter", 600)),
            early_stopping=bool(overrides.get("early_stopping", True)),
            random_state=random_state,
        )
    if model_kind == "elasticnet_logreg_classifier":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(overrides.get("l1_ratio", 0.5)),
            C=float(overrides.get("C", 1.0)),
            max_iter=int(overrides.get("max_iter", 5000)),
            random_state=random_state,
        )
    if model_kind == "randomforest_classifier":
        return RandomForestClassifier(
            n_estimators=int(overrides.get("n_estimators", 500)),
            max_depth=(
                int(overrides["max_depth"])
                if overrides.get("max_depth") is not None
                else None
            ),
            min_samples_leaf=int(overrides.get("min_samples_leaf", 1)),
            n_jobs=1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported reasoning classifier kind: {model_kind}")


def build_reproduction_classifier(model_kind: str, *, random_state: int) -> Any:
    require_dependency("xgboost", f"build the reproduction classifier '{model_kind}'")
    import xgboost as xgb  # type: ignore

    if model_kind == "xgb_joel_classifier":
        return xgb.XGBClassifier(random_state=random_state, **JOEL_XGB_CLASSIFIER_PARAMS)
    if model_kind == "xgb_autoresearch_classifier":
        return xgb.XGBClassifier(random_state=random_state, **AUTORESEARCH_XGB_CLASSIFIER_PARAMS)
    raise ValueError(f"Unsupported reproduction classifier kind: {model_kind}")
