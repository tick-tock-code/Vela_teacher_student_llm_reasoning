from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tkinter as tk
import unittest
import uuid

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.feature_repository import load_feature_repository_splits, load_repository_feature_bank
from src.data.targets import load_target_family
from src.data.splits import build_stratified_reasoning_cv_splits
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
from src.pipeline.model_testing import (
    _aggregate_screening_metrics,
    _resolve_stage_a_model_families,
    run_model_testing_mode,
)
from src.pipeline.config import FeatureSetSpec, load_experiment_config
from src.pipeline.run_distillation import parse_run_overrides
from src.pipeline.run_options import RunOverrides, resolve_run_options
from src.pipeline.xgb_calibration import load_latest_xgb_calibration


def _workspace_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "tmp" / "ut"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"c_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


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
    def test_default_config_defaults_to_reproduction_mode(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        self.assertEqual(config.defaults.run_mode, "reproduction_mode")
        self.assertEqual(config.defaults.target_family, "v25_policies")
        self.assertEqual(config.reproduction.outer_cv.n_splits, 5)
        self.assertEqual(config.distillation_cv.n_splits, 3)

    def test_v25_and_taste_target_families_load_expected_columns(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        families = {spec.family_id: spec for spec in config.target_families}

        v25 = load_target_family(families["v25_policies"])
        taste = load_target_family(families["taste_policies"])

        self.assertEqual(len(v25.target_columns), 16)
        self.assertTrue(all(column.startswith("v25_") for column in v25.target_columns))
        self.assertEqual(v25.task_kind, "regression")

        self.assertEqual(len(taste.target_columns), 20)
        self.assertTrue(all(column.startswith("taste_") for column in taste.target_columns))
        self.assertEqual(taste.task_kind, "classification")

    def test_repository_feature_bank_loader_preserves_canonical_order(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        repository_splits = load_feature_repository_splits(config.feature_repository)
        hq_spec = next(spec for spec in config.repository_feature_banks if spec.feature_bank_id == "hq_baseline")

        bank = load_repository_feature_bank(repository_splits=repository_splits, spec=hq_spec)

        self.assertEqual(bank.public_frame["founder_uuid"].tolist()[:5], repository_splits.train_ids[:5])
        self.assertEqual(bank.private_frame["founder_uuid"].tolist()[:5], repository_splits.test_ids[:5])
        self.assertIn("exit_count", bank.feature_columns)
        self.assertIn("stem_flag", bank.binary_feature_columns)

    def test_repository_feature_bank_never_selects_success_as_feature(self) -> None:
        root = _workspace_temp_dir()
        try:
            labels = pd.DataFrame(
                {
                    "founder_uuid": ["f1", "f2", "g1", "g2"],
                    "split": ["train", "train", "test", "test"],
                    "success": [1, 0, 1, 0],
                }
            )
            _write_csv(root / "labels.csv", labels)
            (root / "train_uuids.txt").write_text("f1\nf2\n", encoding="utf-8")
            (root / "test_uuids.txt").write_text("g1\ng2\n", encoding="utf-8")

            train = pd.DataFrame(
                {
                    "founder_uuid": ["f1", "f2"],
                    "success": [1, 0],
                    "numeric_feature_a": [0.1, 0.2],
                }
            )
            test = pd.DataFrame(
                {
                    "founder_uuid": ["g1", "g2"],
                    "success": [1, 0],
                    "numeric_feature_a": [0.3, 0.4],
                }
            )
            _write_csv(root / "bank_train.csv", train)
            _write_csv(root / "bank_test.csv", test)

            from src.pipeline.config import FeatureRepositoryPaths, RepositoryFeatureBankSpec

            repository_splits = load_feature_repository_splits(
                FeatureRepositoryPaths(
                    root_dir=str(root),
                    labels_path=str(root / "labels.csv"),
                    train_uuids_path=str(root / "train_uuids.txt"),
                    test_uuids_path=str(root / "test_uuids.txt"),
                )
            )
            spec = RepositoryFeatureBankSpec(
                feature_bank_id="temp_bank",
                train_path=str(root / "bank_train.csv"),
                test_path=str(root / "bank_test.csv"),
                source_id_column="founder_uuid",
                enabled=True,
                feature_prefixes=[],
                exclude_columns=[],
                label_column=None,
                all_features_binary=False,
                binary_feature_columns=[],
            )
            bank = load_repository_feature_bank(repository_splits=repository_splits, spec=spec)
            self.assertNotIn("success", bank.feature_columns)
            self.assertIn("numeric_feature_a", bank.feature_columns)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_llm_engineering_feature_bank_drops_seed_rows_at_feature_set_assembly(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        repository_splits = load_feature_repository_splits(config.feature_repository)
        llm_spec = next(spec for spec in config.repository_feature_banks if spec.feature_bank_id == "llm_engineering")
        bank = load_repository_feature_bank(repository_splits=repository_splits, spec=llm_spec)

        assembled = assemble_feature_sets(
            public_founder_ids=pd.Series(repository_splits.train_ids),
            private_founder_ids=pd.Series(repository_splits.test_ids),
            banks_by_id={"llm_engineering": bank},
            feature_sets=[
                FeatureSetSpec(
                    feature_set_id="llm_only",
                    feature_bank_ids=["llm_engineering"],
                )
            ],
        )

        feature_set = assembled[0]
        self.assertEqual(len(feature_set.public_frame), 4400)
        self.assertEqual(len(feature_set.private_frame), 4500)
        self.assertGreater(feature_set.manifest["dropped_public_ids_count"], 0)

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
            try:
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
            except ImportError as exc:
                self.skipTest(f"Parquet engine unavailable in this environment: {exc}")

            self.assertTrue(bank_exists(storage_dir))
            loaded = load_intermediary_bank(
                feature_id="sentence_prose",
                builder_kind="sentence_transformer_prose_v1",
                storage_dir=storage_dir,
            )
            self.assertEqual(loaded.feature_columns, feature_columns)
            self.assertEqual(loaded.binary_feature_columns, [])
            self.assertEqual(loaded.manifest["embedding_dimension"], 2)
            self.assertIn("public_rendered.csv", loaded.manifest["extra_tables"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_feature_set_assembly_preserves_row_order_and_feature_provenance(self) -> None:
        public_ids = pd.Series(["b", "a"])
        private_ids = pd.Series(["y", "x"])
        banks = {
            "hq_baseline": ResolvedIntermediaryBank(
                feature_id="hq_baseline",
                builder_kind="repository_feature_bank",
                storage_dir=Path("unused"),
                public_frame=pd.DataFrame(
                    {"founder_uuid": ["a", "b"], "hq__one": [1.0, 2.0]}
                ),
                private_frame=pd.DataFrame(
                    {"founder_uuid": ["x", "y"], "hq__one": [3.0, 4.0]}
                ),
                feature_columns=["hq__one"],
                binary_feature_columns=[],
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
                binary_feature_columns=[],
                manifest={},
            ),
        }

        assembled = assemble_feature_sets(
            public_founder_ids=public_ids,
            private_founder_ids=private_ids,
            banks_by_id=banks,
            feature_sets=[
                FeatureSetSpec(
                    feature_set_id="hq_plus_sentence_prose",
                    feature_bank_ids=["hq_baseline", "sentence_prose"],
                )
            ],
        )

        feature_set = assembled[0]
        self.assertEqual(feature_set.feature_columns, ["hq__one", "sentence_prose__dim_000"])
        self.assertEqual(feature_set.public_frame["founder_uuid"].tolist(), ["b", "a"])
        self.assertEqual(feature_set.private_frame["founder_uuid"].tolist(), ["y", "x"])

    def test_stratified_reasoning_splits_cover_all_rows(self) -> None:
        targets = pd.DataFrame(
            {
                "v25_p1": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.15, 0.85, 0.25, 0.75],
                "v25_p11": [0.2, 0.3, 0.35, 0.45, 0.55, 0.65, 0.78, 0.88, 0.18, 0.82, 0.28, 0.72],
            }
        )
        splits = build_stratified_reasoning_cv_splits(
            targets,
            n_splits=3,
            shuffle=True,
            random_state=42,
        )

        self.assertEqual(len(splits), 3)
        covered = sorted(index for split in splits for index in split.test_idx.tolist())
        self.assertEqual(covered, list(range(len(targets))))

    def test_cli_override_parsing_and_resolution_support_new_surface(self) -> None:
        overrides = parse_run_overrides(
            [
                "--config",
                "experiments/teacher_student_distillation_v1.json",
                "--run-mode",
                "reasoning_distillation_mode",
                "--target-family",
                "taste_policies",
                "--heldout-evaluation",
                "--active-feature-banks",
                "hq_baseline",
                "sentence_prose",
                "--reasoning-models",
                "logreg_classifier",
                "--embedding-model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ]
        )
        self.assertEqual(overrides.run_mode, "reasoning_distillation_mode")
        self.assertEqual(overrides.target_family, "taste_policies")
        self.assertTrue(overrides.heldout_evaluation)
        self.assertEqual(overrides.active_feature_banks, ["hq_baseline", "sentence_prose"])
        self.assertEqual(overrides.reasoning_models, ["logreg_classifier"])

        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        resolved = resolve_run_options(config, overrides)
        self.assertEqual(resolved.target_family.family_id, "taste_policies")
        self.assertEqual([spec.model_id for spec in resolved.distillation_models], ["logreg_classifier"])

    def test_launcher_controls_expose_mode_and_target_family(self) -> None:
        try:
            root = tk.Tk()
        except tk.TclError as exc:  # pragma: no cover - depends on desktop availability
            self.skipTest(f"Tk is unavailable in this environment: {exc}")
        root.withdraw()
        try:
            launcher = RunLauncher(root)
            self.assertEqual(launcher.run_mode_var.get(), "reproduction_mode")
            self.assertIn("reproduction_mode", launcher.setup_mode_combo.cget("values"))
            self.assertIn("v25_policies", launcher.setup_target_combo.cget("values"))
            self.assertIn("v25_and_taste", launcher.mt_target_combo.cget("values"))

            selections = LauncherSelections(
                config_path="experiments/teacher_student_distillation_v1.json",
                run_mode="reasoning_distillation_mode",
                target_family="taste_policies",
                heldout_evaluation=True,
                active_feature_banks=["hq_baseline", "sentence_prose"],
                force_rebuild_intermediary_features=True,
                reasoning_models=["logreg_classifier"],
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            overrides = selections_to_overrides(selections)
            self.assertEqual(overrides.run_mode, "reasoning_distillation_mode")
            self.assertEqual(overrides.target_family, "taste_policies")
            self.assertTrue(overrides.heldout_evaluation)
            self.assertEqual(overrides.active_feature_banks, ["hq_baseline", "sentence_prose"])
            self.assertEqual(overrides.reasoning_models, ["logreg_classifier"])
        finally:
            root.destroy()

    def test_model_testing_mode_maps_model_families_by_task_kind(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        resolved_regression = resolve_run_options(
            config,
            RunOverrides(
                run_mode="model_testing_mode",
                target_family="v25_policies",
                candidate_feature_sets=["sentence_bundle"],
                model_families=["linear_l2", "xgb1"],
            ),
        )
        self.assertEqual(
            sorted(spec.model_id for spec in resolved_regression.distillation_models),
            ["ridge", "xgb1_regressor"],
        )
        self.assertEqual(resolved_regression.output_modes, ["single_target"])

        resolved_classification = resolve_run_options(
            config,
            RunOverrides(
                run_mode="model_testing_mode",
                target_family="taste_policies",
                candidate_feature_sets=["sentence_bundle"],
                model_families=["linear_l2", "xgb1"],
            ),
        )
        self.assertEqual(
            sorted(spec.model_id for spec in resolved_classification.distillation_models),
            ["logreg_classifier", "xgb1_classifier"],
        )

    def test_model_testing_mode_accepts_multi_output_selection(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        resolved = resolve_run_options(
            config,
            RunOverrides(
                run_mode="model_testing_mode",
                target_family="v25_policies",
                candidate_feature_sets=["sentence_bundle"],
                model_families=["linear_l2"],
                output_modes=["single_target", "multi_output"],
            ),
        )
        self.assertEqual(resolved.output_modes, ["single_target", "multi_output"])

    def test_xgb_calibration_mode_resolves_xgb_only_and_disables_heldout(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        resolved = resolve_run_options(
            config,
            RunOverrides(
                run_mode="xgb_calibration_mode",
                target_family="v25_policies",
                candidate_feature_sets=["sentence_bundle"],
            ),
        )
        self.assertEqual(resolved.run_mode, "xgb_calibration_mode")
        self.assertFalse(resolved.heldout_evaluation)
        self.assertEqual([spec.model_id for spec in resolved.distillation_models], ["xgb1_regressor"])
        self.assertEqual(resolved.output_modes, ["single_target"])
        self.assertEqual(resolved.model_families, ["xgb1"])

    def test_cli_parsing_supports_xgb_calibration_flags(self) -> None:
        overrides = parse_run_overrides(
            [
                "--config",
                "experiments/teacher_student_distillation_v1.json",
                "--run-mode",
                "xgb_calibration_mode",
                "--target-family",
                "v25_and_taste",
                "--candidate-feature-sets",
                "hq_plus_sentence_bundle",
                "lambda_policies_plus_sentence_bundle",
                "--xgb-calibration-estimators",
                "40",
                "80",
                "120",
                "--use-latest-xgb-calibration",
            ]
        )
        self.assertEqual(overrides.run_mode, "xgb_calibration_mode")
        self.assertEqual(overrides.target_family, "v25_and_taste")
        self.assertEqual(overrides.candidate_feature_sets, ["hq_plus_sentence_bundle", "lambda_policies_plus_sentence_bundle"])
        self.assertEqual(overrides.xgb_calibration_estimators, [40, 80, 120])
        self.assertTrue(overrides.use_latest_xgb_calibration)

    def test_model_testing_mode_can_enable_latest_xgb_calibration_toggle(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        resolved = resolve_run_options(
            config,
            RunOverrides(
                run_mode="model_testing_mode",
                target_family="v25_policies",
                candidate_feature_sets=["sentence_bundle"],
                model_families=["xgb1"],
                use_latest_xgb_calibration=True,
            ),
        )
        self.assertTrue(resolved.use_latest_xgb_calibration)

    def test_load_latest_xgb_calibration_returns_none_when_missing(self) -> None:
        self.assertIsNone(load_latest_xgb_calibration("definitely_missing_experiment_for_test"))

    def test_screening_score_rule_marks_top_set_and_close_runners(self) -> None:
        repeat_metrics = pd.DataFrame(
            {
                "target_family": ["v25_policies"] * 6,
                "output_mode": ["single_target"] * 6,
                "feature_set_id": ["a", "a", "b", "b", "c", "c"],
                "repeat_index": [0, 1, 0, 1, 0, 1],
                "r2": [0.40, 0.42, 0.405, 0.41, 0.30, 0.31],
                "rmse": [0.1] * 6,
                "mae": [0.1] * 6,
                "f0_5": [0.0] * 6,
                "roc_auc": [0.0] * 6,
                "pr_auc": [0.0] * 6,
            }
        )
        screening = _aggregate_screening_metrics(
            repeat_metrics,
            task_kind="regression",
            score_delta=0.005,
            max_recommended=3,
        )
        recommended = screening[screening["recommended_take_forward"]]
        self.assertIn("a", recommended["feature_set_id"].tolist())
        self.assertIn("b", recommended["feature_set_id"].tolist())

    def test_stage_a_model_family_selection_respects_ui_choices(self) -> None:
        self.assertEqual(
            _resolve_stage_a_model_families(["linear_l2", "mlp"]),
            ["linear_l2"],
        )
        self.assertEqual(
            _resolve_stage_a_model_families(["xgb1", "randomforest"]),
            ["xgb1"],
        )
        with self.assertRaises(RuntimeError):
            _resolve_stage_a_model_families(["mlp", "elasticnet", "randomforest"])

    def test_model_testing_mode_rejects_heldout_override(self) -> None:
        config = load_experiment_config("experiments/teacher_student_distillation_v1.json")
        with self.assertRaises(RuntimeError):
            run_model_testing_mode(
                config,
                RunOverrides(
                    run_mode="model_testing_mode",
                    heldout_evaluation=True,
                ),
            )

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
