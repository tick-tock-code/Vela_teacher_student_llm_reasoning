from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest
import uuid

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.feature_bank import load_feature_bank
from src.data.input_features import build_input_features
from src.data.targets import load_reasoning_target_bank
from src.evaluation.metrics import binary_classification_metrics, regression_metrics
from src.llm_engineering.custom_prompts import (
    generate_custom_engineered_rule_family,
    load_custom_rule_prompt_bundle,
    postprocess_generated_rules,
    render_custom_rule_prompt,
)
from src.pipeline.config import InputFeatureSpec, ReasoningTargetBankSpec
from src.student.models import build_downstream_classifier, build_reasoning_regressor
from src.utils.dependencies import has_dependency


def _workspace_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "tmp" / "test_runs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ComponentTests(unittest.TestCase):
    def test_feature_bank_restricts_train_to_test_columns(self) -> None:
        root = _workspace_temp_dir()
        try:
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            train_df = pd.DataFrame(
                {
                    "uuid": ["a", "b"],
                    **{f"feature_{idx}": [idx, idx + 1] for idx in range(1, 11)},
                    "feature_11": [11, 12],
                    "ignore_me": [99, 100],
                }
            )
            test_df = pd.DataFrame(
                {
                    "uuid": ["x", "y"],
                    **{f"feature_{idx}": [idx, idx + 1] for idx in range(1, 11)},
                }
            )
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            resolved = load_feature_bank(
                train_path,
                test_path,
                source_id_column="uuid",
                target_id_column="founder_uuid",
                feature_regex=r"^feature_\d+$",
                expected_feature_count=10,
            )

            self.assertEqual(resolved.feature_columns, [f"feature_{idx}" for idx in range(1, 11)])
            self.assertEqual(
                list(resolved.train_frame.columns),
                ["founder_uuid"] + [f"feature_{idx}" for idx in range(1, 11)],
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_feature_bank_raises_if_train_missing_required_column(self) -> None:
        root = _workspace_temp_dir()
        try:
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            train_df = pd.DataFrame(
                {
                    "uuid": ["a", "b"],
                    **{f"feature_{idx}": [idx, idx + 1] for idx in range(1, 10)},
                }
            )
            test_df = pd.DataFrame(
                {
                    "uuid": ["x", "y"],
                    **{f"feature_{idx}": [idx, idx + 1] for idx in range(1, 11)},
                }
            )
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            with self.assertRaises(RuntimeError):
                load_feature_bank(
                    train_path,
                    test_path,
                    source_id_column="uuid",
                    target_id_column="founder_uuid",
                    feature_regex=r"^feature_\d+$",
                    expected_feature_count=10,
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_reasoning_target_bank_loads_public_and_shared_test_targets(self) -> None:
        root = _workspace_temp_dir()
        try:
            train_path = root / "targets_public.csv"
            test_path = root / "targets_test.csv"
            pd.DataFrame(
                {
                    "uuid": ["a", "b"],
                    "Policy_0": [0.1, 0.2],
                    "Policy_1": [0.3, 0.4],
                    "Policy_2": [0.5, 0.6],
                }
            ).to_csv(train_path, index=False)
            pd.DataFrame(
                {
                    "uuid": ["x", "y"],
                    "Policy_1": [0.7, 0.8],
                    "Policy_2": [0.2, 0.1],
                }
            ).to_csv(test_path, index=False)

            loaded = load_reasoning_target_bank(
                ReasoningTargetBankSpec(
                    train_path=str(train_path),
                    test_path=str(test_path),
                    source_id_column="uuid",
                    target_id_column="founder_uuid",
                    target_regex=r"^Policy_\d+$",
                    scale_min=0.0,
                    scale_max=1.0,
                    prediction_mode="regression",
                ),
                selected_targets=["Policy_1", "Policy_2"],
            )

            self.assertEqual(loaded.available_train_target_columns, ["Policy_0", "Policy_1", "Policy_2"])
            self.assertEqual(loaded.train_target_columns, ["Policy_1", "Policy_2"])
            self.assertEqual(list(loaded.test_frame.columns), ["founder_uuid", "Policy_1", "Policy_2"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_founder_baseline_builder_produces_numeric_features(self) -> None:
        public_raw = pd.DataFrame(
            {
                "founder_uuid": ["a", "b"],
                "success": [0, 1],
                "industry": ["Tech", "Health"],
                "ipos": ["[]", "[{'amount_raised_usd': '>500M'}]"],
                "acquisitions": ["[]", "[{'acquired_by_well_known': True}]"],
                "educations_json": [
                    '[{"degree": "BS", "field": "CS", "qs_ranking": "4"}]',
                    '[{"degree": "MBA", "field": "Business", "qs_ranking": "60"}]',
                ],
                "jobs_json": [
                    '[{"role": "CTO", "company_size": "1001-5000 employees", "duration": "3-5"}]',
                    '[{"role": "Founder", "company_size": "2-10 employees", "duration": "<2"}]',
                ],
                "anonymised_prose": ["alpha founder", "beta founder with more words"],
            }
        )
        private_raw = public_raw.drop(columns=["success"]).copy()
        private_raw["founder_uuid"] = ["x", "y"]

        resolved = build_input_features(
            public_raw=public_raw,
            private_raw=private_raw,
            spec=InputFeatureSpec(
                kind="founder_baseline_v1",
                train_path=None,
                test_path=None,
                source_id_column="founder_uuid",
                target_id_column="founder_uuid",
                feature_regex=None,
                expected_feature_count=None,
            ),
        )

        self.assertIn("education_count", resolved.feature_columns)
        self.assertIn("job_count", resolved.feature_columns)
        self.assertIn("ipo_count", resolved.feature_columns)
        self.assertIn("industry__tech", resolved.feature_columns)
        self.assertEqual(list(resolved.public_frame.columns)[0], "founder_uuid")

    def test_metric_helpers_return_expected_keys(self) -> None:
        reg = regression_metrics([0.0, 0.5, 1.0], [0.1, 0.45, 0.95])
        self.assertEqual(set(reg.keys()), {"pearson", "spearman", "mae", "rmse", "r2"})

        cls = binary_classification_metrics([0, 1, 0, 1], [0.2, 0.8, 0.1, 0.9], threshold=0.5)
        self.assertIn("f0_5", cls)
        self.assertIn("roc_auc", cls)

    def test_placeholder_hooks_raise_explicitly(self) -> None:
        for fn, expected_name in [
            (load_custom_rule_prompt_bundle, "load_custom_rule_prompt_bundle"),
            (render_custom_rule_prompt, "render_custom_rule_prompt"),
            (postprocess_generated_rules, "postprocess_generated_rules"),
            (generate_custom_engineered_rule_family, "generate_custom_engineered_rule_family"),
        ]:
            with self.assertRaises(NotImplementedError) as ctx:
                fn()
            self.assertIn(expected_name, str(ctx.exception))

    def test_xgboost_guards_are_explicit(self) -> None:
        if has_dependency("xgboost"):
            build_reasoning_regressor("xgb1_regressor", random_state=42)
            build_downstream_classifier("xgb1_classifier", random_state=42)
            return

        with self.assertRaises(RuntimeError) as reg_ctx:
            build_reasoning_regressor("xgb1_regressor", random_state=42)
        self.assertIn("xgboost", str(reg_ctx.exception))

        with self.assertRaises(RuntimeError) as cls_ctx:
            build_downstream_classifier("xgb1_classifier", random_state=42)
        self.assertIn("xgboost", str(cls_ctx.exception))
