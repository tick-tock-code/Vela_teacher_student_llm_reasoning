from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loading import read_table
from src.utils.artifact_io import ensure_dir


@dataclass(frozen=True)
class RuleDefinition:
    name: str
    description: str
    expression: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "expression": self.expression,
        }


def load_engineered_rules(rules_path: Path) -> list[RuleDefinition]:
    if not rules_path.exists():
        raise RuntimeError(f"Missing engineered rules: {rules_path}")
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Invalid rule format in {rules_path}")

    rules: list[RuleDefinition] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        expression = str(item.get("expression", "")).strip()
        if not name or not expression:
            continue
        try:
            compile(expression, "<rule>", "eval")
        except SyntaxError:
            continue
        rules.append(RuleDefinition(name=name, description=description, expression=expression))
    if not rules:
        raise RuntimeError(f"No usable rules found in {rules_path}")
    return rules


def load_engineered_rule_set(
    *,
    family_id: str,
    set_id: str,
    archives_root: Path,
) -> list[RuleDefinition]:
    rules_path = archives_root / f"family_{family_id}" / set_id / "current" / "llm_rules.json"
    return load_engineered_rules(rules_path)


def _feature_cache_paths(cache_dir: Path) -> list[Path]:
    current_dir = cache_dir / "current"
    return [
        current_dir / "llm_features.parquet",
        current_dir / "llm_features.csv",
        cache_dir / "llm_features.parquet",
        cache_dir / "llm_features.csv",
    ]


def load_feature_cache(
    *,
    cache_dir: Path,
    expected_rows: int | None = None,
    expected_n_features: int | None = None,
    model: str | None = None,
    providers: dict[str, bool] | None = None,
    google_model: str | None = None,
    seed_hash: str | None = None,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    current_dir = cache_dir / "current"
    meta_candidates = [
        current_dir / "llm_features_meta.json",
        cache_dir / "llm_features_meta.json",
    ]
    meta_path = next((path for path in meta_candidates if path.exists()), None)
    if meta_path is None:
        return None, None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if expected_rows is not None and meta.get("n_rows") != expected_rows:
        return None, None
    if expected_n_features is not None and meta.get("n_features") != expected_n_features:
        return None, None
    if model is not None and meta.get("model") != model:
        return None, None
    if providers is not None and meta.get("providers") != providers:
        return None, None
    if google_model is not None and meta.get("google_model") != google_model:
        return None, None
    if seed_hash is not None and meta.get("seed_hash") != seed_hash:
        return None, None
    feature_names = meta.get("feature_names")
    if not isinstance(feature_names, list):
        return None, None

    cache_path = next((path for path in _feature_cache_paths(cache_dir) if path.exists()), None)
    if cache_path is None:
        return None, None

    frame = read_table(cache_path)
    return frame, [str(name) for name in feature_names]


def save_feature_cache(
    *,
    cache_dir: Path,
    frame: pd.DataFrame,
    feature_names: list[str],
    model: str,
    providers: dict[str, bool],
    google_model: str | None,
    n_features: int,
    seed_hash: str | None = None,
    rules: list[dict[str, str]] | None = None,
) -> None:
    current_dir = ensure_dir(cache_dir / "current")
    meta_path = current_dir / "llm_features_meta.json"
    storage_path = current_dir / "llm_features.parquet"
    storage_format = "parquet"
    try:
        frame.to_parquet(storage_path, index=False)
    except Exception:
        storage_path = current_dir / "llm_features.csv"
        frame.to_csv(storage_path, index=False)
        storage_format = "csv"

    meta = {
        "n_rows": int(len(frame)),
        "n_features": int(n_features),
        "feature_names": list(feature_names),
        "model": model,
        "providers": providers,
        "google_model": google_model,
        "seed_hash": seed_hash,
        "storage_format": storage_format,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if rules:
        (current_dir / "llm_rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
