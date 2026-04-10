from __future__ import annotations

import ast
import json
import math
import re
from typing import Any

import pandas as pd


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def parse_sequence(value: Any) -> list[dict[str, Any]]:
    text = clean_text(value)
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


def parse_int_like(value: Any) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return None


def duration_to_years(value: Any) -> float:
    text = clean_text(value)
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


def is_large_company(value: Any) -> float:
    text = clean_text(value).lower()
    if not text:
        return 0.0
    if any(token in text for token in ["5001", "10001", "1001", "enterprise"]):
        return 1.0
    matches = [int(item) for item in re.findall(r"\d+", text)]
    return 1.0 if matches and max(matches) >= 1000 else 0.0


def count_keyword_matches(values: list[str], keywords: tuple[str, ...]) -> int:
    total = 0
    for value in values:
        lowered = value.lower()
        if any(keyword in lowered for keyword in keywords):
            total += 1
    return total


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "blank"


def build_vcbench_mirror_row(row: pd.Series) -> dict[str, float]:
    prose = clean_text(row.get("anonymised_prose"))
    educations = parse_sequence(row.get("educations_json"))
    jobs = parse_sequence(row.get("jobs_json"))
    ipos = parse_sequence(row.get("ipos"))
    acquisitions = parse_sequence(row.get("acquisitions"))

    education_ranks = [
        rank
        for rank in (parse_int_like(item.get("qs_ranking")) for item in educations)
        if rank is not None
    ]
    job_roles = [clean_text(item.get("role")) for item in jobs]
    job_years = [duration_to_years(item.get("duration")) for item in jobs]
    large_company_count = sum(is_large_company(item.get("company_size")) for item in jobs)
    acquired_by_known_count = sum(
        1.0 for item in acquisitions if bool(item.get("acquired_by_well_known", False))
    )

    best_rank = min(education_ranks) if education_ranks else 999.0
    total_job_years = float(sum(job_years))
    job_count = float(len(jobs))

    return {
        "mirror__prose_char_count": float(len(prose)),
        "mirror__prose_word_count": float(len(prose.split())),
        "mirror__prose_line_count": float(prose.count("\n") + 1 if prose else 0),
        "mirror__education_count": float(len(educations)),
        "mirror__has_education": float(len(educations) > 0),
        "mirror__best_qs_rank": float(best_rank),
        "mirror__education_top10_count": float(sum(rank <= 10 for rank in education_ranks)),
        "mirror__education_top50_count": float(sum(rank <= 50 for rank in education_ranks)),
        "mirror__job_count": job_count,
        "mirror__has_jobs": float(len(jobs) > 0),
        "mirror__executive_role_count": float(
            count_keyword_matches(
                job_roles,
                ("chief", "ceo", "cto", "cfo", "coo", "vp", "president", "director"),
            )
        ),
        "mirror__founder_role_count": float(count_keyword_matches(job_roles, ("founder", "co-founder"))),
        "mirror__technical_role_count": float(
            count_keyword_matches(job_roles, ("engineer", "developer", "scientist", "architect", "technical"))
        ),
        "mirror__large_company_job_count": float(large_company_count),
        "mirror__total_job_years": total_job_years,
        "mirror__avg_job_years": float(total_job_years / job_count) if job_count else 0.0,
        "mirror__ipo_count": float(len(ipos)),
        "mirror__acquisition_count": float(len(acquisitions)),
        "mirror__known_acquirer_count": float(acquired_by_known_count),
    }


def build_vcbench_mirror_frames(
    public_raw: pd.DataFrame,
    private_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, object]]:
    public_rows = [
        {"founder_uuid": str(row["founder_uuid"]), **build_vcbench_mirror_row(row)}
        for _, row in public_raw.iterrows()
    ]
    private_rows = [
        {"founder_uuid": str(row["founder_uuid"]), **build_vcbench_mirror_row(row)}
        for _, row in private_raw.iterrows()
    ]
    public_frame = pd.DataFrame(public_rows)
    private_frame = pd.DataFrame(private_rows)

    public_industry = public_raw["industry"].map(clean_text).astype(str)
    private_industry = private_raw["industry"].map(clean_text).astype(str)
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
        base_name = f"mirror__industry__{slugify(industry)}"
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
        "feature_family": "vcbench_mirror_baseline_v1",
        "industry_column_map": industry_column_map,
        "base_feature_count": len(feature_columns) - len(industry_column_map),
        "industry_feature_count": len(industry_column_map),
    }
    return public_frame, private_frame, feature_columns, manifest
