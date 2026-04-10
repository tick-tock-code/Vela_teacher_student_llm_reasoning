from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_reasoning_reconstruction
from src.pipeline.run_options import RunOverrides


def _workspace_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "tmp" / "ut"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"c_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_public_raw() -> pd.DataFrame:
    rows = []
    industries = ["Tech", "Health", "Finance"]
    for idx in range(12):
        rows.append(
            {
                "founder_uuid": f"train_{idx}",
                "industry": industries[idx % len(industries)],
                "ipos": "[]",
                "acquisitions": "[]",
                "educations_json": json.dumps(
                    [{"degree": "BS", "field": "CS", "qs_ranking": str(5 + idx)}]
                ),
                "jobs_json": json.dumps(
                    [{"role": "Founder" if idx % 2 else "CTO", "company_size": "1001-5000 employees", "duration": "3-5"}]
                ),
                "anonymised_prose": ("founder " * (12 + idx)).strip(),
            }
        )
    return pd.DataFrame(rows)


def _make_private_raw() -> pd.DataFrame:
    rows = []
    industries = ["Tech", "Health", "Finance"]
    for idx in range(4):
        rows.append(
            {
                "founder_uuid": f"test_{idx}",
                "industry": industries[idx % len(industries)],
                "ipos": "[]",
                "acquisitions": "[]",
                "educations_json": json.dumps(
                    [{"degree": "BS", "field": "CS", "qs_ranking": str(8 + idx)}]
                ),
                "jobs_json": json.dumps(
                    [{"role": "Founder" if idx % 2 else "CTO", "company_size": "2-10 employees", "duration": "<2"}]
                ),
                "anonymised_prose": ("private founder " * (10 + idx)).strip(),
            }
        )
    return pd.DataFrame(rows)


def _policy_targets_from_raw(raw_frame: pd.DataFrame, *, include_extra_policy: bool) -> pd.DataFrame:
    rows = []
    for _, row in raw_frame.iterrows():
        jobs = json.loads(row["jobs_json"])
        founder_roles = sum("founder" in item["role"].lower() for item in jobs)
        prose_word_count = len(str(row["anonymised_prose"]).split())
        payload = {
            "uuid": row["founder_uuid"],
            "Policy_1": round(min(1.0, 0.05 + 0.02 * prose_word_count + 0.10 * founder_roles), 6),
            "Policy_2": round(min(1.0, 0.10 + 0.05 * founder_roles), 6),
        }
        if include_extra_policy:
            payload["Policy_0"] = round(min(1.0, 0.15 + 0.01 * prose_word_count), 6)
        rows.append(payload)
    return pd.DataFrame(rows)


class PipelineSmokeTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config = {
            "experiment_id": "smoke_reasoning_reconstruction",
            "description": "Synthetic smoke test",
            "datasets": {
                "public_train_csv": str(root / "public.csv"),
                "private_test_csv": str(root / "private.csv"),
            },
            "reasoning_target_bank": {
                "train_path": str(root / "targets_public.csv"),
                "test_path": str(root / "targets_private.csv"),
                "source_id_column": "uuid",
                "target_id_column": "founder_uuid",
                "target_regex": "^Policy_\\d+$",
                "scale_min": 0.0,
                "scale_max": 1.0,
                "prediction_mode": "regression"
            },
            "reasoning_targets": [
                {"target_id": "Policy_1"},
                {"target_id": "Policy_2"}
            ],
            "reasoning_models": [
                {"model_id": "ridge", "kind": "ridge"}
            ],
            "intermediary_features": [
                {"feature_id": "mirror", "kind": "vcbench_mirror_baseline_v1", "enabled": True}
            ],
            "feature_sets": [
                {"feature_set_id": "mirror", "feature_ids": ["mirror"]}
            ],
            "cv": {
                "n_splits": 3,
                "shuffle": True,
                "random_state": 42
            }
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config_path

    def _write_fixtures(self, root: Path) -> None:
        public_raw = _make_public_raw()
        private_raw = _make_private_raw()
        public_raw.to_csv(root / "public.csv", index=False)
        private_raw.to_csv(root / "private.csv", index=False)
        _policy_targets_from_raw(public_raw, include_extra_policy=True).to_csv(root / "targets_public.csv", index=False)
        _policy_targets_from_raw(private_raw, include_extra_policy=False).to_csv(root / "targets_private.csv", index=False)

    def test_reasoning_only_pipeline_auto_builds_and_reuses_intermediary_bank(self) -> None:
        root = _workspace_temp_dir()
        try:
            self._write_fixtures(root)
            config_path = self._write_config(root)
            config = load_experiment_config(str(config_path))
            overrides = RunOverrides(
                config_path=str(config_path),
                active_intermediary_features=["mirror"],
                reasoning_models=["ridge"],
            )

            runs_root = root / "r"
            intermediary_root = root / "i"
            with mock.patch("src.pipeline.distillation.RUNS_DIR", runs_root), mock.patch(
                "src.intermediary_features.storage.INTERMEDIARY_FEATURES_DIR",
                intermediary_root,
            ):
                first_run_dir = run_reasoning_reconstruction(config, overrides)
                bank_manifest_path = intermediary_root / "mirror" / "v1" / "manifest.json"
                first_mtime = bank_manifest_path.stat().st_mtime

                second_run_dir = run_reasoning_reconstruction(config, overrides)
                second_mtime = bank_manifest_path.stat().st_mtime

            self.assertTrue((first_run_dir / "reasoning_oof_predictions.csv").exists())
            self.assertTrue((first_run_dir / "reasoning_metrics.csv").exists())
            self.assertFalse((first_run_dir / "reasoning_heldout_predictions.csv").exists())
            self.assertFalse((first_run_dir / "reasoning_heldout_metrics.csv").exists())
            self.assertTrue((first_run_dir / "intermediary_feature_manifests.json").exists())
            self.assertTrue((first_run_dir / "feature_set_manifest.json").exists())
            self.assertFalse((first_run_dir / "downstream_public_summary.csv").exists())
            self.assertFalse((first_run_dir / "downstream_private_predictions.csv").exists())

            oof_predictions = pd.read_csv(first_run_dir / "reasoning_oof_predictions.csv")
            self.assertIn("mirror__Policy_1__ridge", oof_predictions.columns)
            self.assertIn("mirror__Policy_2__ridge", oof_predictions.columns)
            self.assertNotIn("mirror__Policy_0__ridge", oof_predictions.columns)

            target_manifest = json.loads((first_run_dir / "reasoning_target_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(target_manifest["available_train_target_count"], 3)
            self.assertEqual(target_manifest["train_target_count"], 2)

            self.assertEqual(first_mtime, second_mtime)
            self.assertTrue(second_run_dir.exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_heldout_prediction_outputs_require_opt_in(self) -> None:
        root = _workspace_temp_dir()
        try:
            self._write_fixtures(root)
            config_path = self._write_config(root)
            config = load_experiment_config(str(config_path))
            overrides = RunOverrides(
                config_path=str(config_path),
                active_intermediary_features=["mirror"],
                reasoning_models=["ridge"],
                run_heldout_reasoning_predictions=True,
            )

            runs_root = root / "r"
            intermediary_root = root / "i"
            with mock.patch("src.pipeline.distillation.RUNS_DIR", runs_root), mock.patch(
                "src.intermediary_features.storage.INTERMEDIARY_FEATURES_DIR",
                intermediary_root,
            ):
                run_dir = run_reasoning_reconstruction(config, overrides)

            self.assertTrue((run_dir / "reasoning_heldout_predictions.csv").exists())
            self.assertTrue((run_dir / "reasoning_heldout_metrics.csv").exists())

            heldout_metrics = pd.read_csv(run_dir / "reasoning_heldout_metrics.csv")
            self.assertEqual(sorted(heldout_metrics["target_id"].tolist()), ["Policy_1", "Policy_2"])
            self.assertTrue((heldout_metrics["feature_set_id"] == "mirror").all())
        finally:
            shutil.rmtree(root, ignore_errors=True)
