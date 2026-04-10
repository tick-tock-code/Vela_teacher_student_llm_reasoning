from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Iterable

import pandas as pd

from src.llm_engineering.cache import RuleDefinition, load_engineered_rule_set
from src.llm_engineering.custom_prompts import generate_custom_engineered_rule_family
from src.utils.dependencies import require_dependency
from src.utils.paths import LLM_ENGINEERED_ARCHIVES_DIR


def _load_env_if_present(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


async def generate_engineered_features_from_trl(
    *,
    train_records: list[dict[str, Any]],
    train_labels: Iterable[int],
    test_records: list[dict[str, Any]],
    model: str,
    n_features: int,
    all_records: list[dict[str, Any]] | None = None,
    providers: dict[str, bool] | None = None,
    google_model: str | None = None,
    env_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[dict[str, str]]]:
    require_dependency("think_reason_learn", "generate LLM-engineered rules")
    if env_path is not None:
        _load_env_if_present(env_path)

    from think_reason_learn.core.llms import GoogleChoice, OpenAIChoice
    from think_reason_learn.datasets import VCBENCH_HELPERS, VCBENCH_SCHEMA
    from think_reason_learn.features import FeatureEvaluator, FeatureGenerator

    if providers is None:
        providers = {"openai": True, "google": False}

    llm_priority = []
    if providers.get("openai", False):
        llm_priority.append(OpenAIChoice(model=model))
    if providers.get("google", False):
        llm_priority.append(GoogleChoice(model=google_model or "gemini-2.0-flash"))
    if not llm_priority:
        raise RuntimeError("No LLM providers enabled for engineered rule generation.")

    generator = FeatureGenerator(
        schema=VCBENCH_SCHEMA,
        helpers=VCBENCH_HELPERS,
        llm_priority=llm_priority,
        temperature=0.7,
    )
    rules = await generator.generate(
        samples=train_records,
        labels=list(train_labels),
        n_rules=n_features,
        n_samples=min(60, len(train_records)),
    )
    if not rules:
        raise RuntimeError("No LLM-engineered rules were generated.")

    evaluator = FeatureEvaluator(rules=rules, helpers=VCBENCH_HELPERS)
    train_frame = evaluator.evaluate_df(train_records)
    test_frame = evaluator.evaluate_df(test_records)
    all_frame = evaluator.evaluate_df(all_records) if all_records is not None else pd.DataFrame()

    rule_payload = [
        {
            "name": str(rule.name),
            "description": str(rule.description),
            "expression": str(rule.expression),
        }
        for rule in rules
    ]
    feature_names = [rule["name"] for rule in rule_payload]
    return all_frame, train_frame, test_frame, feature_names, rule_payload


def load_archived_rule_set(
    *,
    family_id: str,
    set_id: str,
    archives_root: Path | None = None,
) -> list[RuleDefinition]:
    return load_engineered_rule_set(
        family_id=family_id,
        set_id=set_id,
        archives_root=archives_root or LLM_ENGINEERED_ARCHIVES_DIR,
    )


def generate_custom_rule_family(*args, **kwargs):
    return generate_custom_engineered_rule_family(*args, **kwargs)
