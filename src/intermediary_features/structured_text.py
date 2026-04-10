from __future__ import annotations

from typing import Any

import pandas as pd

from src.intermediary_features.mirror import clean_text, parse_sequence


def _sorted_value_pairs(payload: dict[str, Any], *, exclude: set[str] | None = None) -> list[str]:
    pairs: list[str] = []
    exclude_use = exclude or set()
    for key in sorted(payload):
        if key in exclude_use:
            continue
        value = clean_text(payload.get(key))
        if value:
            pairs.append(f"{key}={value}")
    return pairs


def render_structured_founder_text(row: pd.Series) -> str:
    parts: list[str] = []

    industry = clean_text(row.get("industry"))
    if industry:
        parts.append(f"industry: {industry}")

    educations = parse_sequence(row.get("educations_json"))
    if educations:
        education_parts = []
        for idx, education in enumerate(educations, start=1):
            fields = _sorted_value_pairs(education)
            education_parts.append(f"education_{idx}: {'; '.join(fields)}")
        parts.append("educations: " + " | ".join(education_parts))

    jobs = parse_sequence(row.get("jobs_json"))
    if jobs:
        job_parts = []
        for idx, job in enumerate(jobs, start=1):
            fields = _sorted_value_pairs(job)
            job_parts.append(f"job_{idx}: {'; '.join(fields)}")
        parts.append("jobs: " + " | ".join(job_parts))

    ipos = parse_sequence(row.get("ipos"))
    if ipos:
        ipo_parts = []
        for idx, ipo in enumerate(ipos, start=1):
            fields = _sorted_value_pairs(ipo)
            ipo_parts.append(f"ipo_{idx}: {'; '.join(fields)}")
        parts.append("ipos: " + " | ".join(ipo_parts))

    acquisitions = parse_sequence(row.get("acquisitions"))
    if acquisitions:
        acquisition_parts = []
        for idx, acquisition in enumerate(acquisitions, start=1):
            fields = _sorted_value_pairs(acquisition)
            acquisition_parts.append(f"acquisition_{idx}: {'; '.join(fields)}")
        parts.append("acquisitions: " + " | ".join(acquisition_parts))

    return "\n".join(parts).strip()


def render_structured_text_frames(
    public_raw: pd.DataFrame,
    private_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    public_frame = pd.DataFrame(
        {
            "founder_uuid": public_raw["founder_uuid"].astype(str),
            "rendered_text": [render_structured_founder_text(row) for _, row in public_raw.iterrows()],
        }
    )
    private_frame = pd.DataFrame(
        {
            "founder_uuid": private_raw["founder_uuid"].astype(str),
            "rendered_text": [render_structured_founder_text(row) for _, row in private_raw.iterrows()],
        }
    )
    return public_frame, private_frame
