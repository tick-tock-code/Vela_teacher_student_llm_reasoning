from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_engineering.cache import load_engineered_rule_set, load_feature_cache, save_feature_cache


def _workspace_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "tmp" / "test_runs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class LLMEngineeringCompatibilityTests(unittest.TestCase):
    def test_rule_set_loader_reads_old_style_json(self) -> None:
        root = _workspace_temp_dir()
        try:
            rules_dir = root / "family_20260408_000000" / "set_01" / "current"
            rules_dir.mkdir(parents=True, exist_ok=True)
            payload = [
                {
                    "name": "feature_a",
                    "description": "Test rule",
                    "expression": "lambda founder: 1",
                },
                {
                    "name": "bad_rule",
                    "description": "Bad syntax",
                    "expression": "lambda founder:",
                },
            ]
            (rules_dir / "llm_rules.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            rules = load_engineered_rule_set(
                family_id="20260408_000000",
                set_id="set_01",
                archives_root=root,
            )

            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0].name, "feature_a")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_feature_cache_round_trip(self) -> None:
        root = _workspace_temp_dir()
        try:
            cache_dir = root / "family_01" / "set_01"
            frame = pd.DataFrame({"feature_1": [0.1, 0.2], "feature_2": [0.3, 0.4]})
            save_feature_cache(
                cache_dir=cache_dir,
                frame=frame,
                feature_names=["feature_1", "feature_2"],
                model="gpt-4.1-nano",
                providers={"openai": True, "google": False},
                google_model=None,
                n_features=2,
                seed_hash="abc123",
                rules=None,
            )

            loaded_frame, feature_names = load_feature_cache(
                cache_dir=cache_dir,
                expected_rows=2,
                expected_n_features=2,
                model="gpt-4.1-nano",
                providers={"openai": True, "google": False},
                google_model=None,
                seed_hash="abc123",
            )

            self.assertIsNotNone(loaded_frame)
            self.assertEqual(feature_names, ["feature_1", "feature_2"])
            self.assertEqual(list(loaded_frame.columns), ["feature_1", "feature_2"])
        finally:
            shutil.rmtree(root, ignore_errors=True)
