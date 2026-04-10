from __future__ import annotations

import importlib.util


INSTALL_HINTS = {
    "openai": "pip install openai",
    "think_reason_learn": "Install the think_reason_learn package or make it importable in this environment.",
    "xgboost": "pip install xgboost",
}


def has_dependency(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def require_dependency(module_name: str, purpose: str) -> None:
    if has_dependency(module_name):
        return
    install_hint = INSTALL_HINTS.get(module_name, f"Install the `{module_name}` package.")
    raise RuntimeError(f"`{module_name}` is required to {purpose}. {install_hint}")
