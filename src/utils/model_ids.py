from __future__ import annotations

import re


DEFAULT_XGB_MAX_DEPTH = 3

XGB_FAMILY_ID = f"xgb{DEFAULT_XGB_MAX_DEPTH}"
XGB_FAMILY_UI_LABEL = XGB_FAMILY_ID.upper()
XGB_REGRESSOR_MODEL_KIND = f"{XGB_FAMILY_ID}_regressor"
XGB_CLASSIFIER_MODEL_KIND = f"{XGB_FAMILY_ID}_classifier"

LEGACY_XGB_FAMILY_ID = "xgb1"
LEGACY_XGB_REGRESSOR_MODEL_KIND = "xgb1_regressor"
LEGACY_XGB_CLASSIFIER_MODEL_KIND = "xgb1_classifier"

_XGB_FAMILY_ALIASES = {
    LEGACY_XGB_FAMILY_ID: XGB_FAMILY_ID,
    XGB_FAMILY_ID: XGB_FAMILY_ID,
}

_XGB_MODEL_KIND_ALIASES = {
    LEGACY_XGB_REGRESSOR_MODEL_KIND: XGB_REGRESSOR_MODEL_KIND,
    LEGACY_XGB_CLASSIFIER_MODEL_KIND: XGB_CLASSIFIER_MODEL_KIND,
    XGB_REGRESSOR_MODEL_KIND: XGB_REGRESSOR_MODEL_KIND,
    XGB_CLASSIFIER_MODEL_KIND: XGB_CLASSIFIER_MODEL_KIND,
}


def normalize_xgb_family_id(value: str) -> str:
    if re.fullmatch(r"xgb\d+", value):
        return XGB_FAMILY_ID
    return _XGB_FAMILY_ALIASES.get(value, value)


def normalize_xgb_model_kind(value: str) -> str:
    match = re.fullmatch(r"xgb\d+_(regressor|classifier)", value)
    if match is not None:
        return (
            XGB_REGRESSOR_MODEL_KIND
            if match.group(1) == "regressor"
            else XGB_CLASSIFIER_MODEL_KIND
        )
    return _XGB_MODEL_KIND_ALIASES.get(value, value)
