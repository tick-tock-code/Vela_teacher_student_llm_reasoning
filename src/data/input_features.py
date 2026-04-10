from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.feature_bank import feature_manifest_payload, load_feature_bank
from src.pipeline.config import InputFeatureSpec
from src.utils.placeholders import not_implemented_placeholder


@dataclass(frozen=True)
class ResolvedInputFeatures:
    public_frame: pd.DataFrame
    private_frame: pd.DataFrame
    feature_columns: list[str]
    builder_kind: str
    manifest: dict[str, object]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _parse_sequence(value: Any) -> list[dict[str, Any]]:
    text = _clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _parse_int_like(value: Any) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return None


def _duration_to_years(value: Any) -> float:
    text = _clean_text(value)
    if not text:
        return 0.0
    mapping = {
        "<2": 1.0,
        "2-3": 2.5,
        "3-5": 4.0,
        "6-9": 7.5,
        "10+": 10.0,
    }
    if text in mapping:
        return mapping[text]
    matches = [float(item) for item in re.findall(r"\d+", text)]
    if not matches:
        return 0.0
    if len(matches) == 1:
        return float(matches[0])
    return float(sum(matches) / len(matches))


def _is_large_company(value: Any) -> float:
    text = _clean_text(value).lower()
    if not text:
        return 0.0
    if any(token in text for token in ["5001", "10001", "1001", "enterprise"]):
        return 1.0
    matches = [int(item) for item in re.findall(r"\d+", text)]
    return 1.0 if matches and max(matches) >= 1000 else 0.0


def _count_keyword_matches(values: list[str], keywords: tuple[str, ...]) -> int:
    total = 0
    for value in values:
        lowered = value.lower()
        if any(keyword in lowered for keyword in keywords):
            total += 1
    return total


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "blank"


def _build_founder_baseline_row(row: pd.Series) -> dict[str, float]:
    prose = _clean_text(row.get("anonymised_prose"))
    educations = _parse_sequence(row.get("educations_json"))
    jobs = _parse_sequence(row.get("jobs_json"))
    ipos = _parse_sequence(row.get("ipos"))
    acquisitions = _parse_sequence(row.get("acquisitions"))

    education_ranks = [
        rank
        for rank in (_parse_int_like(item.get("qs_ranking")) for item in educations)
        if rank is not None
    ]
    job_roles = [_clean_text(item.get("role")) for item in jobs]
    job_years = [_duration_to_years(item.get("duration")) for item in jobs]
    large_company_count = sum(_is_large_company(item.get("company_size")) for item in jobs)
    acquired_by_known_count = sum(
        1.0 for item in acquisitions if bool(item.get("acquired_by_well_known", False))
    )

    best_rank = min(education_ranks) if education_ranks else 999.0
    total_job_years = float(sum(job_years))
    job_count = float(len(jobs))

    return {
        "prose_char_count": float(len(prose)),
        "prose_word_count": float(len(prose.split())),
        "prose_line_count": float(prose.count("\n") + 1 if prose else 0),
        "education_count": float(len(educations)),
        "has_education": float(len(educations) > 0),
        "best_qs_rank": float(best_rank),
        "education_top10_count": float(sum(rank <= 10 for rank in education_ranks)),
        "education_top50_count": float(sum(rank <= 50 for rank in education_ranks)),
        "job_count": job_count,
        "has_jobs": float(len(jobs) > 0),
        "executive_role_count": float(
            _count_keyword_matches(job_roles, ("chief", "ceo", "cto", "cfo", "coo", "vp", "president", "director"))
        ),
        "founder_role_count": float(_count_keyword_matches(job_roles, ("founder", "co-founder"))),
        "technical_role_count": float(
            _count_keyword_matches(job_roles, ("engineer", "developer", "scientist", "architect", "technical"))
        ),
        "large_company_job_count": float(large_company_count),
        "total_job_years": total_job_years,
        "avg_job_years": float(total_job_years / job_count) if job_count else 0.0,
        "ipo_count": float(len(ipos)),
        "acquisition_count": float(len(acquisitions)),
        "known_acquirer_count": float(acquired_by_known_count),
    }


def _build_founder_baseline_frames(
    public_raw: pd.DataFrame,
    private_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, object]]:
    public_rows = [
        {"founder_uuid": founder_uuid, **_build_founder_baseline_row(row)}
        for founder_uuid, (_, row) in zip(public_raw["founder_uuid"].astype(str), public_raw.iterrows())
    ]
    private_rows = [
        {"founder_uuid": founder_uuid, **_build_founder_baseline_row(row)}
        for founder_uuid, (_, row) in zip(private_raw["founder_uuid"].astype(str), private_raw.iterrows())
    ]
    public_frame = pd.DataFrame(public_rows)
    private_frame = pd.DataFrame(private_rows)

    public_industry = public_raw["industry"].map(_clean_text).astype(str)
    private_industry = private_raw["industry"].map(_clean_text).astype(str)
    all_industries = sorted(
        {
            value
            for value in pd.concat([public_industry, private_industry], ignore_index=True).tolist()
            if value
        }
    )

    industry_column_map: dict[str, str] = {}
    used_columns: set[str] = set()
    for industry in all_industries:
        base_name = f"industry__{_slugify(industry)}"
        candidate = base_name
        suffix = 2
        while candidate in used_columns:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        used_columns.add(candidate)
        industry_column_map[industry] = candidate

    for industry, column_name in industry_column_map.items():
        public_frame[column_name] = (public_industry == industry).astype(float)
        private_frame[column_name] = (private_industry == industry).astype(float)

    feature_columns = [column for column in public_frame.columns if column != "founder_uuid"]
    manifest = {
        "builder_kind": "founder_baseline_v1",
        "feature_columns": feature_columns,
        "base_feature_count": len(feature_columns) - len(industry_column_map),
        "industry_feature_count": len(industry_column_map),
    }
    return public_frame, private_frame, feature_columns, manifest


def build_input_features(
    *,
    public_raw: pd.DataFrame,
    private_raw: pd.DataFrame,
    spec: InputFeatureSpec,
) -> ResolvedInputFeatures:
    if spec.kind == "founder_baseline_v1":
        public_frame, private_frame, feature_columns, manifest = _build_founder_baseline_frames(
            public_raw,
            private_raw,
        )
        return ResolvedInputFeatures(
            public_frame=public_frame,
            private_frame=private_frame,
            feature_columns=feature_columns,
            builder_kind=spec.kind,
            manifest=manifest,
        )

    if spec.kind == "table_bank":
        if not spec.train_path or not spec.test_path or not spec.feature_regex or spec.expected_feature_count is None:
            raise RuntimeError("table_bank input features require train/test paths and regex metadata.")
        resolved = load_feature_bank(
            Path(spec.train_path),
            Path(spec.test_path),
            source_id_column=spec.source_id_column,
            target_id_column=spec.target_id_column,
            feature_regex=spec.feature_regex,
            expected_feature_count=spec.expected_feature_count,
        )
        manifest = {
            "builder_kind": spec.kind,
            **feature_manifest_payload(resolved),
        }
        return ResolvedInputFeatures(
            public_frame=resolved.train_frame,
            private_frame=resolved.test_frame,
            feature_columns=resolved.feature_columns,
            builder_kind=spec.kind,
            manifest=manifest,
        )

    if spec.kind in {"llm_engineered_rules", "sentence_transformer_embeddings"}:
        not_implemented_placeholder(
            f"build_input_features(kind='{spec.kind}')",
            "Implement this input feature builder before selecting it in the experiment config.",
        )

    raise RuntimeError(f"Unsupported input feature builder '{spec.kind}'.")
