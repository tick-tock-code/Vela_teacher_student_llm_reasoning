from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid
from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_pipeline
from src.pipeline.run_options import RunOverrides
from src.utils.dependencies import has_dependency


def _workspace_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "tmp" / "ut"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"c_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_raw_rows(prefix: str, count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    industries = ["Tech", "Health", "Finance"]
    for index in range(count):
        rows.append(
            {
                "founder_uuid": f"{prefix}_{index:02d}",
                "industry": industries[index % len(industries)],
                "ipos": "[]",
                "acquisitions": "[]",
                "educations_json": json.dumps(
                    [{"degree": "BS", "field": "CS", "qs_ranking": str(10 + index)}]
                ),
                "jobs_json": json.dumps(
                    [{"role": "Founder" if index % 2 else "CTO", "company_size": "1001-5000 employees", "duration": "3-5"}]
                ),
                "anonymised_prose": ("founder profile " * (5 + index)).strip(),
            }
        )
    return rows


def _success_for_index(index: int) -> int:
    return 1 if index % 2 == 0 else 0


def _write_feature_repository(root: Path, *, train_count: int = 18, test_count: int = 6) -> tuple[Path, Path, Path]:
    repo_root = root / "feature_repository"
    (repo_root / "splits").mkdir(parents=True, exist_ok=True)
    (repo_root / "hq_baseline").mkdir(parents=True, exist_ok=True)
    (repo_root / "llm_engineering").mkdir(parents=True, exist_ok=True)
    (repo_root / "lambda_policies").mkdir(parents=True, exist_ok=True)
    (repo_root / "policies").mkdir(parents=True, exist_ok=True)

    public_rows = _build_raw_rows("train", train_count)
    private_rows = _build_raw_rows("test", test_count)
    public_raw = pd.DataFrame(public_rows)
    private_raw = pd.DataFrame(private_rows)
    public_raw.to_csv(root / "public.csv", index=False)
    private_raw.to_csv(root / "private.csv", index=False)

    train_ids = public_raw["founder_uuid"].tolist()
    test_ids = private_raw["founder_uuid"].tolist()
    (repo_root / "splits" / "train_uuids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (repo_root / "splits" / "test_uuids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")

    labels_rows = [
        {"founder_uuid": founder_id, "split": "train", "success": _success_for_index(index)}
        for index, founder_id in enumerate(train_ids)
    ] + [
        {"founder_uuid": founder_id, "split": "test", "success": _success_for_index(index)}
        for index, founder_id in enumerate(test_ids)
    ]
    pd.DataFrame(labels_rows).to_csv(repo_root / "splits" / "labels.csv", index=False)

    def hq_frame(founder_ids: list[str], *, is_test: bool) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, founder_id in enumerate(founder_ids):
            rows.append(
                {
                    "founder_uuid": founder_id,
                    "success": _success_for_index(index),
                    "has_prior_ipo": int(index % 3 == 0),
                    "has_prior_acquisition": int(index % 4 == 0),
                    "exit_count": int(index % 5 == 0),
                    "max_company_size_before_founding": 100 + index,
                    "prestige_sacrifice_score": float(index) / 10.0,
                    "years_in_large_company": float(index % 6),
                    "comfort_index": float(index % 4),
                    "founding_timing": 3.0 + index / 10.0,
                    "edu_prestige_tier": int((index % 4) + 1),
                    "field_relevance_score": float((index % 5) + 1),
                    "prestige_x_relevance": float((index % 4) + 1) * ((index % 5) + 1),
                    "degree_level": int((index % 4) + 1),
                    "stem_flag": int(index % 2 == 0),
                    "best_degree_prestige": float((index % 6) + 1),
                    "max_seniority_reached": float((index % 5) + 1),
                    "seniority_is_monotone": int(index % 2 == 0),
                    "company_size_is_growing": int(index % 3 == 0),
                    "restlessness_score": float(index % 3),
                    "founding_role_count": float((index % 2) + 1),
                    "longest_founding_tenure": float((index % 7) + 1),
                    "industry_pivot_count": float(index % 4),
                    "industry_alignment": int(index % 2 == 0),
                    "total_inferred_experience": 5.0 + index,
                    "is_serial_founder": int(index % 3 == 0),
                    "exit_x_serial": float(int(index % 5 == 0) * int(index % 3 == 0)),
                    "sacrifice_x_serial": float(index % 5),
                    "industry_prestige_penalty": float(index % 2),
                    "persistence_score": float((index % 7) + 1) / 10.0,
                }
            )
        return pd.DataFrame(rows)

    hq_frame(train_ids, is_test=False).to_csv(repo_root / "hq_baseline" / "features_train.csv", index=False)
    hq_frame(test_ids, is_test=True).to_csv(repo_root / "hq_baseline" / "features_test.csv", index=False)

    def llm_frame(founder_ids: list[str], *, train: bool) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        seed_ids = set(founder_ids[:2]) if train else set()
        for index, founder_id in enumerate(founder_ids):
            is_seed = founder_id in seed_ids
            rows.append(
                {
                    "founder_uuid": founder_id,
                    "le_feature_a": None if is_seed else int(index % 2 == 0),
                    "le_feature_b": None if is_seed else int(index % 3 == 0),
                    "le_feature_c": None if is_seed else int(index % 4 == 0),
                }
            )
        return pd.DataFrame(rows)

    llm_frame(train_ids, train=True).to_csv(repo_root / "llm_engineering" / "features_train.csv", index=False)
    llm_frame(test_ids, train=False).to_csv(repo_root / "llm_engineering" / "features_test.csv", index=False)

    def lambda_frame(founder_ids: list[str]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, founder_id in enumerate(founder_ids):
            rows.append(
                {
                    "founder_uuid": founder_id,
                    "lam_feature_1": int(index % 2 == 0),
                    "lam_feature_2": int(index % 3 == 0),
                    "lam_feature_3": int(index % 4 == 0),
                    "lam_feature_4": int(index % 5 == 0),
                }
            )
        return pd.DataFrame(rows)

    lambda_frame(train_ids).to_csv(repo_root / "lambda_policies" / "predictions_train.csv", index=False)
    lambda_frame(test_ids).to_csv(repo_root / "lambda_policies" / "predictions_test.csv", index=False)

    def policy_frame(founder_ids: list[str]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, founder_id in enumerate(founder_ids):
            rows.append(
                {
                    "founder_uuid": founder_id,
                    "v25_p1": round(0.15 + 0.03 * index, 4),
                    "v25_p11": round(0.25 + 0.02 * index, 4),
                    "taste_a": int(index % 2 == 0),
                    "taste_b": int(index % 3 == 0),
                }
            )
        return pd.DataFrame(rows)

    policy_frame(train_ids).to_csv(repo_root / "policies" / "predictions_train.csv", index=False)
    policy_frame(test_ids).to_csv(repo_root / "policies" / "predictions_test.csv", index=False)

    return repo_root, root / "public.csv", root / "private.csv"


class PipelineSmokeTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        repo_root, public_csv, private_csv = _write_feature_repository(root)
        config = {
            "experiment_id": "smoke_feature_repository_pipeline",
            "description": "Synthetic smoke test for reproduction and distillation",
            "datasets": {
                "public_train_csv": str(public_csv),
                "private_test_csv": str(private_csv),
            },
            "feature_repository": {
                "root_dir": str(repo_root),
                "labels_path": str(repo_root / "splits" / "labels.csv"),
                "train_uuids_path": str(repo_root / "splits" / "train_uuids.txt"),
                "test_uuids_path": str(repo_root / "splits" / "test_uuids.txt"),
            },
            "defaults": {
                "run_mode": "reproduction_mode",
                "target_family": "v25_policies",
                "heldout_evaluation": False,
            },
            "repository_feature_banks": [
                {
                    "feature_bank_id": "hq_baseline",
                    "train_path": str(repo_root / "hq_baseline" / "features_train.csv"),
                    "test_path": str(repo_root / "hq_baseline" / "features_test.csv"),
                    "source_id_column": "founder_uuid",
                    "enabled": True,
                    "feature_prefixes": [],
                    "exclude_columns": [],
                    "label_column": "success",
                    "all_features_binary": False,
                    "binary_feature_columns": [
                        "has_prior_ipo",
                        "has_prior_acquisition",
                        "stem_flag",
                        "seniority_is_monotone",
                        "company_size_is_growing",
                        "industry_alignment",
                        "is_serial_founder",
                    ],
                },
                {
                    "feature_bank_id": "llm_engineering",
                    "train_path": str(repo_root / "llm_engineering" / "features_train.csv"),
                    "test_path": str(repo_root / "llm_engineering" / "features_test.csv"),
                    "source_id_column": "founder_uuid",
                    "enabled": True,
                    "feature_prefixes": ["le_"],
                    "exclude_columns": [],
                    "label_column": None,
                    "all_features_binary": True,
                    "binary_feature_columns": [],
                },
                {
                    "feature_bank_id": "lambda_policies",
                    "train_path": str(repo_root / "lambda_policies" / "predictions_train.csv"),
                    "test_path": str(repo_root / "lambda_policies" / "predictions_test.csv"),
                    "source_id_column": "founder_uuid",
                    "enabled": True,
                    "feature_prefixes": ["lam_"],
                    "exclude_columns": [],
                    "label_column": None,
                    "all_features_binary": True,
                    "binary_feature_columns": [],
                },
                {
                    "feature_bank_id": "policy_v25",
                    "train_path": str(repo_root / "policies" / "predictions_train.csv"),
                    "test_path": str(repo_root / "policies" / "predictions_test.csv"),
                    "source_id_column": "founder_uuid",
                    "enabled": False,
                    "feature_prefixes": ["v25_"],
                    "exclude_columns": [],
                    "label_column": None,
                    "all_features_binary": False,
                    "binary_feature_columns": [],
                },
            ],
            "intermediary_features": [],
            "distillation_feature_sets": [
                {"feature_set_id": "hq_baseline", "feature_bank_ids": ["hq_baseline"]},
                {"feature_set_id": "llm_engineering", "feature_bank_ids": ["llm_engineering"]},
            ],
            "target_families": [
                {
                    "family_id": "v25_policies",
                    "train_path": str(repo_root / "policies" / "predictions_train.csv"),
                    "test_path": str(repo_root / "policies" / "predictions_test.csv"),
                    "source_id_column": "founder_uuid",
                    "target_id_column": "founder_uuid",
                    "target_prefixes": ["v25_"],
                    "task_kind": "regression",
                    "scale_min": 0.0,
                    "scale_max": 1.0,
                    "enabled_by_default": True,
                },
                {
                    "family_id": "taste_policies",
                    "train_path": str(repo_root / "policies" / "predictions_train.csv"),
                    "test_path": str(repo_root / "policies" / "predictions_test.csv"),
                    "source_id_column": "founder_uuid",
                    "target_id_column": "founder_uuid",
                    "target_prefixes": ["taste_"],
                    "task_kind": "classification",
                    "scale_min": None,
                    "scale_max": None,
                    "enabled_by_default": False,
                },
            ],
            "distillation_models": [
                {"model_id": "ridge", "kind": "ridge", "supported_task_kinds": ["regression"]},
                {"model_id": "logreg_classifier", "kind": "logreg_classifier", "supported_task_kinds": ["classification"]},
                {"model_id": "mlp_regressor", "kind": "mlp_regressor", "supported_task_kinds": ["regression"]},
                {"model_id": "mlp_classifier", "kind": "mlp_classifier", "supported_task_kinds": ["classification"]},
            ],
            "reproduction": {
                "outer_cv": {"n_splits": 3, "shuffle": True, "random_state": 42},
                "inner_cv": {"n_splits": 2, "shuffle": True, "random_state": 42},
                "threshold_grid": {"start": 0.2, "stop": 0.8, "step": 0.1},
                "logistic_c_grid": [0.01, 0.1, 1.0],
                "lambda_ranking": {
                    "c": 0.05,
                    "max_iter": 2000,
                    "solver": "liblinear",
                    "class_weight": "balanced",
                    "random_state": 42,
                },
                "experiments": [
                    {
                        "experiment_id": "hq_only",
                        "title": "HQ only",
                        "feature_bank_ids": ["hq_baseline"],
                        "training_pool": "full",
                        "model_kind": "xgb_joel_classifier",
                        "use_exit_override": True,
                        "lambda_top_k": None,
                        "lambda_rank_base_bank_id": None,
                        "standardize": False,
                    },
                    {
                        "experiment_id": "llm_engineering_plus_policy_induction",
                        "title": "LLM Engineering + Policy Induction (v25)",
                        "feature_bank_ids": ["llm_engineering", "policy_v25"],
                        "training_pool": "llm_engineering_non_seed",
                        "model_kind": "nested_l2_logreg",
                        "use_exit_override": False,
                        "lambda_top_k": None,
                        "lambda_rank_base_bank_id": None,
                        "standardize": True,
                    },
                ],
            },
            "distillation_cv": {"n_splits": 3, "shuffle": True, "random_state": 42},
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config_path

    def test_reproduction_mode_writes_benchmark_artifacts(self) -> None:
        if not has_dependency("xgboost"):
            self.skipTest("xgboost is not installed in this environment.")
        root = _workspace_temp_dir()
        try:
            config_path = self._write_config(root)
            config = load_experiment_config(str(config_path))
            overrides = RunOverrides(config_path=str(config_path))

            runs_root = root / "runs"
            with mock.patch("src.pipeline.reproduction.RUNS_DIR", runs_root), mock.patch(
                "src.pipeline.distillation.RUNS_DIR",
                runs_root,
            ):
                run_dir = run_pipeline(config, overrides)

            self.assertTrue((run_dir / "reproduction_results.csv").exists())
            self.assertTrue((run_dir / "reproduction_oof_predictions.csv").exists())
            self.assertTrue((run_dir / "reproduction_test_predictions.csv").exists())
            results = pd.read_csv(run_dir / "reproduction_results.csv")
            self.assertEqual(sorted(results["experiment_id"].tolist()), ["hq_only", "llm_engineering_plus_policy_induction"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_reasoning_distillation_mode_runs_v25_regression(self) -> None:
        root = _workspace_temp_dir()
        try:
            config_path = self._write_config(root)
            config = load_experiment_config(str(config_path))
            overrides = RunOverrides(
                config_path=str(config_path),
                run_mode="reasoning_distillation_mode",
                target_family="v25_policies",
                heldout_evaluation=True,
                active_feature_banks=["hq_baseline"],
                reasoning_models=["ridge"],
            )

            runs_root = root / "runs"
            intermediary_root = root / "intermediary"
            with mock.patch("src.pipeline.distillation.RUNS_DIR", runs_root), mock.patch(
                "src.intermediary_features.storage.INTERMEDIARY_FEATURES_DIR",
                intermediary_root,
            ):
                run_dir = run_pipeline(config, overrides)

            self.assertTrue((run_dir / "reasoning_oof_predictions.csv").exists())
            self.assertTrue((run_dir / "reasoning_metrics.csv").exists())
            self.assertTrue((run_dir / "reasoning_heldout_predictions.csv").exists())
            self.assertTrue((run_dir / "reasoning_heldout_metrics.csv").exists())
            metrics = pd.read_csv(run_dir / "reasoning_metrics.csv")
            self.assertTrue((metrics["feature_set_id"] == "hq_baseline").all())
            self.assertIn("pearson", metrics.columns)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_reasoning_distillation_mode_runs_taste_classification(self) -> None:
        root = _workspace_temp_dir()
        try:
            config_path = self._write_config(root)
            config = load_experiment_config(str(config_path))
            overrides = RunOverrides(
                config_path=str(config_path),
                run_mode="reasoning_distillation_mode",
                target_family="taste_policies",
                heldout_evaluation=True,
                active_feature_banks=["hq_baseline"],
                reasoning_models=["logreg_classifier"],
            )

            runs_root = root / "runs"
            intermediary_root = root / "intermediary"
            with mock.patch("src.pipeline.distillation.RUNS_DIR", runs_root), mock.patch(
                "src.intermediary_features.storage.INTERMEDIARY_FEATURES_DIR",
                intermediary_root,
            ):
                run_dir = run_pipeline(config, overrides)

            self.assertTrue((run_dir / "reasoning_classification_thresholds.json").exists())
            heldout_metrics = pd.read_csv(run_dir / "reasoning_heldout_metrics.csv")
            self.assertIn("f0_5", heldout_metrics.columns)
            self.assertTrue((heldout_metrics["model_id"] == "logreg_classifier").all())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_model_testing_multi_output_parallel_consistency(self) -> None:
        root = _workspace_temp_dir()
        try:
            config_path = self._write_config(root)
            config = load_experiment_config(str(config_path))

            runs_root = root / "runs"
            intermediary_root = root / "intermediary"
            with mock.patch("src.pipeline.distillation.RUNS_DIR", runs_root), mock.patch(
                "src.intermediary_features.storage.INTERMEDIARY_FEATURES_DIR",
                intermediary_root,
            ):
                run_serial = run_pipeline(
                    config,
                    RunOverrides(
                        config_path=str(config_path),
                        run_mode="model_testing_mode",
                        target_family="v25_policies",
                        candidate_feature_sets=["hq_baseline", "llm_engineering"],
                        model_families=["linear_l2"],
                        output_modes=["multi_output"],
                        max_parallel_workers=1,
                    ),
                )
                run_parallel = run_pipeline(
                    config,
                    RunOverrides(
                        config_path=str(config_path),
                        run_mode="model_testing_mode",
                        target_family="v25_policies",
                        candidate_feature_sets=["hq_baseline", "llm_engineering"],
                        model_families=["linear_l2"],
                        output_modes=["multi_output"],
                        max_parallel_workers=2,
                    ),
                )

            serial = pd.read_csv(run_serial / "feature_set_screening.csv").sort_values(
                ["target_family", "output_mode", "feature_set_id", "rank"]
            ).reset_index(drop=True)
            parallel = pd.read_csv(run_parallel / "feature_set_screening.csv").sort_values(
                ["target_family", "output_mode", "feature_set_id", "rank"]
            ).reset_index(drop=True)
            assert_frame_equal(serial, parallel, atol=1e-10, rtol=1e-10)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_mlp_calibration_parallel_consistency(self) -> None:
        root = _workspace_temp_dir()
        try:
            config_path = self._write_config(root)
            config = load_experiment_config(str(config_path))

            runs_root = root / "runs"
            intermediary_root = root / "intermediary"
            with mock.patch("src.pipeline.distillation.RUNS_DIR", runs_root), mock.patch(
                "src.pipeline.mlp_calibration.RUNS_DIR",
                runs_root,
            ), mock.patch(
                "src.intermediary_features.storage.INTERMEDIARY_FEATURES_DIR",
                intermediary_root,
            ):
                run_serial = run_pipeline(
                    config,
                    RunOverrides(
                        config_path=str(config_path),
                        run_mode="mlp_calibration_mode",
                        target_family="v25_policies",
                        candidate_feature_sets=["hq_baseline", "llm_engineering"],
                        mlp_calibration_hidden_layer_sizes=[[8]],
                        mlp_calibration_alpha=[0.1],
                        max_parallel_workers=1,
                    ),
                )
                run_parallel = run_pipeline(
                    config,
                    RunOverrides(
                        config_path=str(config_path),
                        run_mode="mlp_calibration_mode",
                        target_family="v25_policies",
                        candidate_feature_sets=["hq_baseline", "llm_engineering"],
                        mlp_calibration_hidden_layer_sizes=[[8]],
                        mlp_calibration_alpha=[0.1],
                        max_parallel_workers=2,
                    ),
                )

            serial_payload = json.loads((run_serial / "mlp_calibration_recommendations.json").read_text(encoding="utf-8"))
            parallel_payload = json.loads((run_parallel / "mlp_calibration_recommendations.json").read_text(encoding="utf-8"))
            serial_metrics = pd.DataFrame(serial_payload["metrics_table"]).sort_values(
                ["target_family", "feature_set_id", "params_signature"]
            ).reset_index(drop=True)
            parallel_metrics = pd.DataFrame(parallel_payload["metrics_table"]).sort_values(
                ["target_family", "feature_set_id", "params_signature"]
            ).reset_index(drop=True)
            assert_frame_equal(serial_metrics, parallel_metrics, atol=1e-10, rtol=1e-10)
        finally:
            shutil.rmtree(root, ignore_errors=True)
