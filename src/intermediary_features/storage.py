from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from src.data.loading import read_table
from src.utils.artifact_io import read_json, write_csv, write_json, write_parquet
from src.utils.paths import INTERMEDIARY_FEATURES_DIR


@dataclass(frozen=True)
class ResolvedIntermediaryBank:
    feature_id: str
    builder_kind: str
    storage_dir: Path
    public_frame: pd.DataFrame
    private_frame: pd.DataFrame
    feature_columns: list[str]
    binary_feature_columns: list[str]
    manifest: dict[str, object]


def model_storage_slug(model_name: str) -> str:
    value = model_name.strip().replace("\\", "/").split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value or "unknown-model"


def mirror_storage_dir() -> Path:
    return INTERMEDIARY_FEATURES_DIR / "mirror" / "v1"


def sentence_transformer_storage_dir(*, variant: str, model_name: str) -> Path:
    return INTERMEDIARY_FEATURES_DIR / "sentence_transformer" / variant / model_storage_slug(model_name)


def llm_engineered_storage_dir() -> Path:
    return INTERMEDIARY_FEATURES_DIR / "llm_engineered" / "pending_prompts"


def _public_path(storage_dir: Path) -> Path:
    return storage_dir / "public.parquet"


def _private_path(storage_dir: Path) -> Path:
    return storage_dir / "private.parquet"


def _manifest_path(storage_dir: Path) -> Path:
    return storage_dir / "manifest.json"


def bank_exists(storage_dir: Path) -> bool:
    return (
        _public_path(storage_dir).exists()
        and _private_path(storage_dir).exists()
        and _manifest_path(storage_dir).exists()
    )


def save_intermediary_bank(
    *,
    feature_id: str,
    builder_kind: str,
    storage_dir: Path,
    public_frame: pd.DataFrame,
    private_frame: pd.DataFrame,
    feature_columns: list[str],
    manifest: dict[str, object],
    binary_feature_columns: list[str] | None = None,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> ResolvedIntermediaryBank:
    write_parquet(_public_path(storage_dir), public_frame)
    write_parquet(_private_path(storage_dir), private_frame)
    for filename, frame in (extra_tables or {}).items():
        write_csv(storage_dir / filename, frame)

    manifest_payload = {
        **manifest,
        "feature_id": feature_id,
        "builder_kind": builder_kind,
        "storage_dir": str(storage_dir),
        "public_path": str(_public_path(storage_dir)),
        "private_path": str(_private_path(storage_dir)),
        "manifest_path": str(_manifest_path(storage_dir)),
        "feature_columns": feature_columns,
        "binary_feature_columns": list(binary_feature_columns or []),
        "public_row_count": int(len(public_frame)),
        "private_row_count": int(len(private_frame)),
        "feature_count": int(len(feature_columns)),
        "extra_tables": sorted((extra_tables or {}).keys()),
    }
    write_json(_manifest_path(storage_dir), manifest_payload)
    return ResolvedIntermediaryBank(
        feature_id=feature_id,
        builder_kind=builder_kind,
        storage_dir=storage_dir,
        public_frame=public_frame,
        private_frame=private_frame,
        feature_columns=feature_columns,
        binary_feature_columns=list(binary_feature_columns or []),
        manifest=manifest_payload,
    )


def load_intermediary_bank(
    *,
    feature_id: str,
    builder_kind: str,
    storage_dir: Path,
) -> ResolvedIntermediaryBank:
    manifest = read_json(_manifest_path(storage_dir))
    public_frame = read_table(_public_path(storage_dir))
    private_frame = read_table(_private_path(storage_dir))
    feature_columns = [str(column) for column in manifest.get("feature_columns", [])]
    binary_feature_columns = [
        str(column) for column in manifest.get("binary_feature_columns", [])
    ]
    return ResolvedIntermediaryBank(
        feature_id=feature_id,
        builder_kind=builder_kind,
        storage_dir=storage_dir,
        public_frame=public_frame,
        private_frame=private_frame,
        feature_columns=feature_columns,
        binary_feature_columns=binary_feature_columns,
        manifest=manifest,
    )
