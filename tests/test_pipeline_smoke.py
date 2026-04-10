from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_distillation_experiment


def _workspace_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "tmp" / "test_runs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _education_json(count: int, *, top_rank: int) -> str:
    items = [
        {
            "degree": "BS" if idx == 0 else "MS",
            "field": "Computer Science",
            "qs_ranking": str(top_rank + idx * 10),
        }
        for idx in range(count)
    ]
    return json.dumps(items)


def _jobs_json(count: int, *, founder_roles: int, executive_roles: int, large_company_roles: int) -> str:
    items = []
    for idx in range(count):
        role = "Engineer"
        if idx < founder_roles:
            role = "Founder"
        elif idx < founder_roles + executive_roles:
            role = "CTO"
        company_size = "2-10 employees"
        if idx < large_company_roles:
            company_size = "1001-5000 employees"
        items.append(
            {
                "role": role,
                "company_size": company_size,
                "industry": "Tech",
                "duration": "3-5" if idx % 2 == 0 else "<2",
            }
        )
    return json.dumps(items)


def _ipo_json(count: int) -> str:
    return json.dumps([{"amount_raised_usd": ">500M"} for _ in range(count)])


def _acquisition_json(count: int, *, known_count: int) -> str:
    return json.dumps(
        [
            {"acquired_by_well_known": idx < known_count}
            for idx in range(count)
        ]
    )


def _make_raw_public() -> pd.DataFrame:
    rows = []
    industries = ["Tech", "Health", "Finance"]
    for idx in range(12):
        education_count = (idx % 3) + 1
        job_count = (idx % 4) + 1
        founder_roles = 1 if idx % 2 else 0
        executive_roles = 1 + (idx % 2)
        large_company_roles = 1 if idx % 3 == 0 else 0
        ipo_count = idx % 2
        acquisition_count = 1 if idx % 4 == 0 else 0
        prose = ("founder " * (12 + idx)).strip()
        rows.append(
            {
                "founder_uuid": f"train_{idx}",
                "success": idx % 2,
                "industry": industries[idx % len(industries)],
                "ipos": _ipo_json(ipo_count),
                "acquisitions": _acquisition_json(acquisition_count, known_count=acquisition_count),
                "educations_json": _education_json(education_count, top_rank=5 + idx),
                "jobs_json": _jobs_json(
                    job_count,
                    founder_roles=founder_roles,
                    executive_roles=executive_roles,
                    large_company_roles=large_company_roles,
                ),
                "anonymised_prose": prose,
            }
        )
    return pd.DataFrame(rows)


def _make_raw_private() -> pd.DataFrame:
    rows = []
    industries = ["Tech", "Health", "Finance"]
    for idx in range(4):
        education_count = (idx % 3) + 1
        job_count = (idx % 4) + 1
        founder_roles = 1 if idx % 2 else 0
        executive_roles = 1 + (idx % 2)
        large_company_roles = 1 if idx % 3 == 0 else 0
        ipo_count = idx % 2
        acquisition_count = 1 if idx % 3 == 0 else 0
        prose = ("private founder " * (10 + idx)).strip()
        rows.append(
            {
                "founder_uuid": f"test_{idx}",
                "industry": industries[idx % len(industries)],
                "ipos": _ipo_json(ipo_count),
                "acquisitions": _acquisition_json(acquisition_count, known_count=acquisition_count),
                "educations_json": _education_json(education_count, top_rank=8 + idx),
                "jobs_json": _jobs_json(
                    job_count,
                    founder_roles=founder_roles,
                    executive_roles=executive_roles,
                    large_company_roles=large_company_roles,
                ),
                "anonymised_prose": prose,
            }
        )
    return pd.DataFrame(rows)


def _policy_targets_from_raw(raw_frame: pd.DataFrame, *, include_policy_0: bool) -> pd.DataFrame:
    rows = []
    for _, row in raw_frame.iterrows():
        education_count = len(json.loads(row["educations_json"]))
        jobs = json.loads(row["jobs_json"])
        job_count = len(jobs)
        founder_role_count = sum("founder" in item["role"].lower() for item in jobs)
        executive_role_count = sum(
            token in item["role"].lower()
            for item in jobs
            for token in ["cto", "ceo", "cfo", "coo", "vp", "director"]
        )
        large_company_count = sum("1001" in item["company_size"] for item in jobs)
        ipo_count = len(json.loads(row["ipos"]))
        prose_word_count = len(str(row["anonymised_prose"]).split())

        payload = {
            "uuid": row["founder_uuid"],
            "Policy_1": round(min(1.0, 0.08 + 0.02 * prose_word_count + 0.08 * founder_role_count), 6),
            "Policy_2": round(min(1.0, 0.10 + 0.10 * executive_role_count + 0.05 * large_company_count), 6),
        }
        if include_policy_0:
            payload["Policy_0"] = round(min(1.0, 0.05 + 0.18 * education_count + 0.10 * job_count + 0.12 * ipo_count), 6)
        rows.append(payload)
    return pd.DataFrame(rows)


class PipelineSmokeTests(unittest.TestCase):
    def _write_config(self, root: Path, *, approved: bool) -> Path:
        config = {
            "experiment_id": "smoke_distillation",
            "description": "Synthetic smoke test",
            "datasets": {
                "public_train_csv": str(root / "public.csv"),
                "private_test_csv": str(root / "private.csv"),
            },
            "input_features": {
                "kind": "founder_baseline_v1",
            },
            "reasoning_target_bank": {
                "train_path": str(root / "targets_public.csv"),
                "test_path": str(root / "targets_private.csv"),
                "source_id_column": "uuid",
                "target_id_column": "founder_uuid",
                "target_regex": "^Policy_\\d+$",
                "scale_min": 0.0,
                "scale_max": 1.0,
                "prediction_mode": "regression",
            },
            "reasoning_targets": [
                {"target_id": "Policy_1"},
                {"target_id": "Policy_2"}
            ],
            "reasoning_models": [
                {
                    "model_id": "ridge",
                    "kind": "ridge",
                }
            ],
            "downstream_models": [
                {
                    "model_id": "lr_classifier",
                    "kind": "lr_classifier",
                }
            ],
            "cv": {
                "n_splits": 3,
                "shuffle": True,
                "random_state": 42,
            },
            "promotion": {
                "mode": "manual",
                "approved": approved,
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config_path

    def _write_fixtures(self, root: Path) -> None:
        public_raw = _make_raw_public()
        private_raw = _make_raw_private()
        public_raw.to_csv(root / "public.csv", index=False)
        private_raw.to_csv(root / "private.csv", index=False)
        _policy_targets_from_raw(public_raw, include_policy_0=True).to_csv(root / "targets_public.csv", index=False)
        _policy_targets_from_raw(private_raw, include_policy_0=False).to_csv(root / "targets_private.csv", index=False)

    def test_pipeline_runs_public_only_when_promotion_blocked(self) -> None:
        root = _workspace_temp_dir()
        try:
            self._write_fixtures(root)
            config = load_experiment_config(str(self._write_config(root, approved=False)))
            run_dir = run_distillation_experiment(config)

            self.assertTrue((run_dir / "input_feature_manifest.json").exists())
            self.assertTrue((run_dir / "reasoning_target_manifest.json").exists())
            self.assertTrue((run_dir / "reasoning_oof_predictions.csv").exists())
            self.assertTrue((run_dir / "downstream_public_summary.csv").exists())
            self.assertFalse((run_dir / "reasoning_private_predictions.csv").exists())

            manifest = json.loads((run_dir / "reasoning_target_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["available_train_target_count"], 3)
            self.assertEqual(manifest["train_target_count"], 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_pipeline_runs_private_prediction_when_promoted(self) -> None:
        root = _workspace_temp_dir()
        try:
            self._write_fixtures(root)
            config = load_experiment_config(str(self._write_config(root, approved=True)))
            run_dir = run_distillation_experiment(config)

            self.assertTrue((run_dir / "reasoning_private_predictions.csv").exists())
            self.assertTrue((run_dir / "reasoning_private_metrics.csv").exists())
            self.assertTrue((run_dir / "downstream_private_predictions.csv").exists())

            reasoning_predictions = pd.read_csv(run_dir / "reasoning_private_predictions.csv")
            self.assertIn("Policy_1__ridge", reasoning_predictions.columns)
            self.assertIn("Policy_2__ridge", reasoning_predictions.columns)
            self.assertNotIn("Policy_0__ridge", reasoning_predictions.columns)

            reasoning_metrics = pd.read_csv(run_dir / "reasoning_private_metrics.csv")
            self.assertEqual(sorted(reasoning_metrics["target_id"].tolist()), ["Policy_1", "Policy_2"])

            downstream_summary = pd.read_csv(run_dir / "downstream_public_summary.csv")
            self.assertIn("baseline_only", downstream_summary["route_id"].tolist())
            self.assertIn("true_reasoning", downstream_summary["route_id"].tolist())
            self.assertIn("predicted_reasoning__ridge", downstream_summary["route_id"].tolist())
        finally:
            shutil.rmtree(root, ignore_errors=True)
