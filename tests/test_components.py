from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tkinter as tk
import unittest
import uuid

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.targets import load_reasoning_target_bank
from src.gui.run_launcher import RunLauncher, LauncherSelections, selections_to_overrides
from src.intermediary_features.mirror import build_vcbench_mirror_frames
from src.intermediary_features.registry import assemble_feature_sets
from src.intermediary_features.sentence_transformer import build_sentence_transformer_frames
from src.intermediary_features.storage import ResolvedIntermediaryBank, bank_exists, load_intermediary_bank, save_intermediary_bank
from src.intermediary_features.structured_text import render_structured_founder_text
from src.llm_engineering.custom_prompts import (
    generate_custom_engineered_rule_family,
    load_custom_rule_prompt_bundle,
    postprocess_generated_rules,
    render_custom_rule_prompt,
)
from src.pipeline.config import FeatureSetSpec, ReasoningTargetBankSpec, load_experiment_config
from src.pipeline.run_distillation import parse_run_overrides
from src.pipeline.run_options import resolve_run_options


def _workspace_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "tmp" / "ut"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"c_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sample_public_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "founder_uuid": ["a", "b"],
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


class ComponentTests(unittest.TestCase):
    def test_default_config_selected_targets_match_heldout_bank(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        heldout = pd.read_csv(config.reasoning_target_bank.test_path, nrows=1)
        expected_targets = heldout.columns.tolist()[1:]

        loaded = load_reasoning_target_bank(
            config.reasoning_target_bank,
            selected_targets=[item.target_id for item in config.reasoning_targets],
        )

        self.assertEqual([item.target_id for item in config.reasoning_targets], expected_targets)
        self.assertEqual(loaded.test_target_columns, expected_targets)

    def test_reasoning_target_bank_raises_when_selected_target_missing_from_heldout(self) -> None:
        root = _workspace_temp_dir()
        try:
            train_path = root / "targets_public.csv"
            test_path = root / "targets_test.csv"
            pd.DataFrame(
                {
                    "uuid": ["a", "b"],
                    "Policy_1": [0.1, 0.2],
                    "Policy_2": [0.3, 0.4],
                }
            ).to_csv(train_path, index=False)
            pd.DataFrame(
                {
                    "uuid": ["x", "y"],
                    "Policy_1": [0.5, 0.6],
                }
            ).to_csv(test_path, index=False)

            with self.assertRaises(RuntimeError):
                load_reasoning_target_bank(
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
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_mirror_feature_builder_produces_prefixed_numeric_features(self) -> None:
        public_raw = _sample_public_raw()
        private_raw = public_raw.copy()
        private_raw["founder_uuid"] = ["x", "y"]

        public_frame, private_frame, feature_columns, manifest = build_vcbench_mirror_frames(public_raw, private_raw)

        self.assertEqual(public_frame.columns[0], "founder_uuid")
        self.assertIn("mirror__education_count", feature_columns)
        self.assertIn("mirror__job_count", feature_columns)
        self.assertIn("mirror__ipo_count", feature_columns)
        self.assertIn("mirror__industry__tech", feature_columns)
        self.assertEqual(manifest["feature_family"], "vcbench_mirror_baseline_v1")
        self.assertEqual(list(private_frame["founder_uuid"]), ["x", "y"])

    def test_structured_text_renderer_is_deterministic(self) -> None:
        row = _sample_public_raw().iloc[0]
        rendered_first = render_structured_founder_text(row)
        rendered_second = render_structured_founder_text(row)

        self.assertEqual(rendered_first, rendered_second)
        self.assertIn("industry: Tech", rendered_first)
        self.assertIn("education_1:", rendered_first)
        self.assertIn("job_1:", rendered_first)

    def test_sentence_transformer_cache_round_trip_uses_manifest(self) -> None:
        root = _workspace_temp_dir()
        try:
            public_text = pd.DataFrame(
                {"founder_uuid": ["a", "b"], "rendered_text": ["alpha", "beta"]}
            )
            private_text = pd.DataFrame(
                {"founder_uuid": ["x", "y"], "rendered_text": ["gamma", "delta"]}
            )

            def fake_encoder(texts: list[str]):
                return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

            public_frame, private_frame, feature_columns, manifest = build_sentence_transformer_frames(
                public_text_frame=public_text,
                private_text_frame=private_text,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                feature_prefix="sentence_prose",
                encode_texts=fake_encoder,
            )
            storage_dir = root / "sentence_bank"
            save_intermediary_bank(
                feature_id="sentence_prose",
                builder_kind="sentence_transformer_prose_v1",
                storage_dir=storage_dir,
                public_frame=public_frame,
                private_frame=private_frame,
                feature_columns=feature_columns,
                manifest=manifest,
                extra_tables={"public_rendered.csv": public_text, "private_rendered.csv": private_text},
            )

            self.assertTrue(bank_exists(storage_dir))
            loaded = load_intermediary_bank(
                feature_id="sentence_prose",
                builder_kind="sentence_transformer_prose_v1",
                storage_dir=storage_dir,
            )
            self.assertEqual(loaded.feature_columns, feature_columns)
            self.assertEqual(loaded.manifest["embedding_dimension"], 2)
            self.assertIn("public_rendered.csv", loaded.manifest["extra_tables"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_feature_set_assembly_preserves_row_order_and_feature_provenance(self) -> None:
        public_ids = pd.Series(["b", "a"])
        private_ids = pd.Series(["y", "x"])
        banks = {
            "mirror": ResolvedIntermediaryBank(
                feature_id="mirror",
                builder_kind="vcbench_mirror_baseline_v1",
                storage_dir=Path("unused"),
                public_frame=pd.DataFrame(
                    {"founder_uuid": ["a", "b"], "mirror__one": [1.0, 2.0]}
                ),
                private_frame=pd.DataFrame(
                    {"founder_uuid": ["x", "y"], "mirror__one": [3.0, 4.0]}
                ),
                feature_columns=["mirror__one"],
                manifest={},
            ),
            "sentence_prose": ResolvedIntermediaryBank(
                feature_id="sentence_prose",
                builder_kind="sentence_transformer_prose_v1",
                storage_dir=Path("unused"),
                public_frame=pd.DataFrame(
                    {"founder_uuid": ["b", "a"], "sentence_prose__dim_000": [10.0, 20.0]}
                ),
                private_frame=pd.DataFrame(
                    {"founder_uuid": ["y", "x"], "sentence_prose__dim_000": [30.0, 40.0]}
                ),
                feature_columns=["sentence_prose__dim_000"],
                manifest={},
            ),
        }

        assembled = assemble_feature_sets(
            public_founder_ids=public_ids,
            private_founder_ids=private_ids,
            banks_by_id=banks,
            feature_sets=[
                FeatureSetSpec(
                    feature_set_id="mirror_plus_sentence_prose",
                    feature_ids=["mirror", "sentence_prose"],
                )
            ],
        )

        feature_set = assembled[0]
        self.assertEqual(feature_set.feature_columns, ["mirror__one", "sentence_prose__dim_000"])
        self.assertEqual(feature_set.public_frame["founder_uuid"].tolist(), ["b", "a"])
        self.assertEqual(feature_set.private_frame["founder_uuid"].tolist(), ["y", "x"])

    def test_cli_override_parsing_and_inactive_flag_validation(self) -> None:
        overrides = parse_run_overrides(
            [
                "--config",
                "experiments/teacher_student_distillation_v1.json",
                "--active-intermediary-features",
                "mirror",
                "sentence_prose",
                "--reasoning-models",
                "ridge",
                "--embedding-model",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--run-success-predictions",
            ]
        )
        self.assertTrue(overrides.run_success_predictions)
        self.assertEqual(overrides.active_intermediary_features, ["mirror", "sentence_prose"])
        self.assertEqual(overrides.reasoning_models, ["ridge"])

        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        with self.assertRaises(RuntimeError) as ctx:
            resolve_run_options(config, overrides)
        self.assertIn("run_success_predictions=true", str(ctx.exception))

    def test_launcher_controls_expose_disabled_future_options(self) -> None:
        try:
            root = tk.Tk()
        except tk.TclError as exc:  # pragma: no cover - depends on desktop availability
            self.skipTest(f"Tk is unavailable in this environment: {exc}")
        root.withdraw()
        try:
            launcher = RunLauncher(root)
            self.assertTrue(launcher.run_success_control.instate(["disabled"]))
            self.assertTrue(launcher.feature_controls["llm_engineered"].instate(["disabled"]))

            selections = LauncherSelections(
                config_path="experiments/teacher_student_distillation_v1.json",
                run_reasoning_predictions=True,
                run_success_predictions=False,
                active_intermediary_features=["mirror", "sentence_prose"],
                force_rebuild_intermediary_features=True,
                reasoning_models=["ridge"],
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            overrides = selections_to_overrides(selections)
            self.assertEqual(overrides.active_intermediary_features, ["mirror", "sentence_prose"])
            self.assertTrue(overrides.force_rebuild_intermediary_features)
            self.assertEqual(overrides.reasoning_models, ["ridge"])
        finally:
            root.destroy()

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
