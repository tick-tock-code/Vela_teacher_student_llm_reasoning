from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
TMP_DIR = BASE_DIR / "tmp"
RUNS_DIR = TMP_DIR / "runs"

VCBENCH_DATA_DIR = DATA_DIR / "VCBench_data"
REASONING_FEATURE_TARGETS_DIR = DATA_DIR / "reasoning_feature_targets"
REASONING_TARGETS_DIR = REASONING_FEATURE_TARGETS_DIR
INTERMEDIARY_FEATURES_DIR = DATA_DIR / "intermediary_features"
SAVED_MODEL_CONFIGS_DIR = DATA_DIR / "saved_model_configs"

LLM_ENGINEERED_DIR = DATA_DIR / "llm_engineered"
LLM_ENGINEERED_FAMILIES_DIR = LLM_ENGINEERED_DIR / "families"
LLM_ENGINEERED_ARCHIVES_DIR = LLM_ENGINEERED_DIR / "archives"

PROJECT_FOLDER_ROOT = BASE_DIR.parents[1] / "Project_folder" / "LLM_Reasoning_Main"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def resolve_existing_path(*paths: str | Path) -> Path:
    if not paths:
        raise ValueError("resolve_existing_path requires at least one candidate.")
    candidates = [resolve_repo_path(path) for path in paths]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_public_csv_path() -> Path:
    return resolve_existing_path(
        DATA_DIR / "vcbench" / "raw" / "vcbench_final_public.csv",
        VCBENCH_DATA_DIR / "vcbench_final_public.csv",
    )


def default_private_csv_path() -> Path:
    return resolve_existing_path(
        DATA_DIR / "vcbench" / "raw" / "vcbench_final_private (success column removed) - vcbench_final_private.csv",
        VCBENCH_DATA_DIR / "vcbench_final_private (success column removed) - vcbench_final_private.csv",
    )


def default_public_sample_csv_path() -> Path:
    return resolve_existing_path(
        DATA_DIR / "vcbench" / "raw" / "vcbench_final_public_sample100.csv",
        VCBENCH_DATA_DIR / "vcbench_final_public_sample100.csv",
    )
