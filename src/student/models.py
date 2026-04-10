from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression, Ridge

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


def build_reasoning_regressor(model_kind: str, *, random_state: int) -> Any:
    if model_kind == "ridge":
        return Ridge(alpha=1.0)
    if model_kind == "xgb1_regressor":
        require_dependency("xgboost", "build the xgb1_regressor reasoning model")
        import xgboost as xgb  # type: ignore

        return xgb.XGBRegressor(random_state=random_state, **XGB_REGRESSOR_PARAMS)
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
