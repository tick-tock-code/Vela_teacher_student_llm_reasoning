from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

from src.utils.dependencies import require_dependency
from src.utils.model_ids import (
    DEFAULT_XGB_MAX_DEPTH,
    XGB_CLASSIFIER_MODEL_KIND,
    XGB_REGRESSOR_MODEL_KIND,
    normalize_xgb_model_kind,
)
from src.utils.parallel import preferred_thread_count


DEFAULT_MODEL_THREADS = preferred_thread_count()
DEFAULT_MLP_HIDDEN_LAYER_SIZES = (128,)


class SigmoidLinearSVC(BaseEstimator, ClassifierMixin):
    """LinearSVC with direct sigmoid score mapping for predict_proba.

    This avoids an internal calibration CV while preserving a probability-like
    score in [0, 1] for downstream ranking/threshold code paths.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        loss: str = "squared_hinge",
        dual: str | bool = "auto",
        max_iter: int = 10_000,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.loss = loss
        self.dual = dual
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "SigmoidLinearSVC":
        self.estimator_ = LinearSVC(
            C=float(self.C),
            loss=str(self.loss),
            dual=self.dual,
            max_iter=int(self.max_iter),
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator_.decision_function(X), dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        scores = self.decision_function(X)
        if scores.ndim == 1:
            probs_pos = expit(scores)
            return np.column_stack([1.0 - probs_pos, probs_pos])
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


XGB_CLASSIFIER_PARAMS = {
    "n_estimators": 320,
    "max_depth": DEFAULT_XGB_MAX_DEPTH,
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
    "n_jobs": DEFAULT_MODEL_THREADS,
}

XGB_REGRESSOR_PARAMS = {
    "n_estimators": 320,
    "max_depth": DEFAULT_XGB_MAX_DEPTH,
    "learning_rate": 0.0674,
    "subsample": 0.949,
    "colsample_bytree": 0.413,
    "min_child_weight": 4,
    "reg_alpha": 0.73,
    "reg_lambda": 15.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_jobs": DEFAULT_MODEL_THREADS,
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
    "n_jobs": DEFAULT_MODEL_THREADS,
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
    "n_jobs": DEFAULT_MODEL_THREADS,
}


def build_reasoning_regressor(
    model_kind: str,
    *,
    random_state: int,
    param_overrides: dict[str, Any] | None = None,
) -> Any:
    model_kind = normalize_xgb_model_kind(model_kind)
    overrides = param_overrides or {}
    if model_kind == "ridge":
        return Ridge(alpha=float(overrides.get("alpha", 1.0)))
    if model_kind == XGB_REGRESSOR_MODEL_KIND:
        require_dependency("xgboost", f"build the {XGB_REGRESSOR_MODEL_KIND} reasoning model")
        import xgboost as xgb  # type: ignore

        params = {**XGB_REGRESSOR_PARAMS, **overrides}
        return xgb.XGBRegressor(random_state=random_state, **params)
    if model_kind == "mlp_regressor":
        return MLPRegressor(
            hidden_layer_sizes=tuple(overrides.get("hidden_layer_sizes", DEFAULT_MLP_HIDDEN_LAYER_SIZES)),
            alpha=float(overrides.get("alpha", 0.1)),
            learning_rate_init=float(overrides.get("learning_rate_init", 1e-3)),
            max_iter=int(overrides.get("max_iter", 1000)),
            tol=float(overrides.get("tol", 1e-5)),
            n_iter_no_change=int(overrides.get("n_iter_no_change", 20)),
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
    if model_kind == "linear_svr_regressor":
        return LinearSVR(
            C=float(overrides.get("C", 1.0)),
            epsilon=float(overrides.get("epsilon", 0.1)),
            max_iter=int(overrides.get("max_iter", 10_000)),
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
            max_features=overrides.get("max_features", "sqrt"),
            bootstrap=bool(overrides.get("bootstrap", True)),
            n_jobs=int(overrides.get("n_jobs", DEFAULT_MODEL_THREADS)),
            random_state=random_state,
        )
    raise ValueError(f"Unsupported reasoning model kind: {model_kind}")


def build_downstream_classifier(model_kind: str, *, random_state: int) -> Any:
    model_kind = normalize_xgb_model_kind(model_kind)
    if model_kind == "lr_classifier":
        return LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=3000,
            random_state=random_state,
        )
    if model_kind == XGB_CLASSIFIER_MODEL_KIND:
        require_dependency("xgboost", f"build the {XGB_CLASSIFIER_MODEL_KIND} downstream model")
        import xgboost as xgb  # type: ignore

        return xgb.XGBClassifier(random_state=random_state, **XGB_CLASSIFIER_PARAMS)
    raise ValueError(f"Unsupported downstream model kind: {model_kind}")


def build_reasoning_classifier(
    model_kind: str,
    *,
    random_state: int,
    param_overrides: dict[str, Any] | None = None,
) -> Any:
    model_kind = normalize_xgb_model_kind(model_kind)
    overrides = param_overrides or {}
    if model_kind == "logreg_classifier":
        return LogisticRegression(
            solver="lbfgs",
            C=float(overrides.get("C", 5.0)),
            max_iter=3000,
            random_state=random_state,
        )
    if model_kind == XGB_CLASSIFIER_MODEL_KIND:
        require_dependency("xgboost", f"build the {XGB_CLASSIFIER_MODEL_KIND} reasoning model")
        import xgboost as xgb  # type: ignore

        params = {**XGB_CLASSIFIER_PARAMS, **overrides}
        return xgb.XGBClassifier(random_state=random_state, **params)
    if model_kind == "mlp_classifier":
        return MLPClassifier(
            hidden_layer_sizes=tuple(overrides.get("hidden_layer_sizes", DEFAULT_MLP_HIDDEN_LAYER_SIZES)),
            alpha=float(overrides.get("alpha", 0.1)),
            learning_rate_init=float(overrides.get("learning_rate_init", 1e-3)),
            max_iter=int(overrides.get("max_iter", 1000)),
            tol=float(overrides.get("tol", 1e-5)),
            n_iter_no_change=int(overrides.get("n_iter_no_change", 20)),
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
    if model_kind == "linear_svm_classifier":
        return SigmoidLinearSVC(
            C=float(overrides.get("C", 1.0)),
            loss=str(overrides.get("loss", "squared_hinge")),
            dual=overrides.get("dual", "auto"),
            max_iter=int(overrides.get("max_iter", 10_000)),
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
            max_features=overrides.get("max_features", "sqrt"),
            bootstrap=bool(overrides.get("bootstrap", True)),
            n_jobs=int(overrides.get("n_jobs", DEFAULT_MODEL_THREADS)),
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
