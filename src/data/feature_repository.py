from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.loading import read_table, rename_identifier_column, select_numeric_feature_columns, validate_unique_ids
from src.intermediary_features.storage import ResolvedIntermediaryBank
from src.pipeline.config import FeatureRepositoryPaths, RepositoryFeatureBankSpec


@dataclass(frozen=True)
class LoadedFeatureRepositorySplits:
    labels_frame: pd.DataFrame
    train_ids: list[str]
    test_ids: list[str]
    train_labels: pd.DataFrame
    test_labels: pd.DataFrame


def load_feature_repository_splits(spec: FeatureRepositoryPaths) -> LoadedFeatureRepositorySplits:
    labels = read_table(Path(spec.labels_path))
    labels = rename_identifier_column(
        labels,
        source_id_column="founder_uuid",
        target_id_column="founder_uuid",
    )
    validate_unique_ids(labels, id_column="founder_uuid")
    labels["split"] = labels["split"].astype(str)
    labels["success"] = labels["success"].astype(int)

    train_ids = [
        line.strip()
        for line in Path(spec.train_uuids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_ids = [
        line.strip()
        for line in Path(spec.test_uuids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return LoadedFeatureRepositorySplits(
        labels_frame=labels,
        train_ids=train_ids,
        test_ids=test_ids,
        train_labels=labels[labels["split"] == "train"].reset_index(drop=True),
        test_labels=labels[labels["split"] == "test"].reset_index(drop=True),
    )


def _reindex_to_canonical(
    frame: pd.DataFrame,
    *,
    canonical_ids: list[str],
    source_name: str,
) -> pd.DataFrame:
    frame_indexed = frame.set_index("founder_uuid")
    missing = [founder_id for founder_id in canonical_ids if founder_id not in frame_indexed.index]
    extra = sorted(set(frame_indexed.index.astype(str)) - set(canonical_ids))
    if missing:
        raise RuntimeError(f"{source_name} is missing {len(missing)} canonical ids. Examples: {missing[:5]}")
    if extra:
        raise RuntimeError(f"{source_name} contains {len(extra)} unexpected ids. Examples: {extra[:5]}")
    return frame_indexed.reindex(canonical_ids).reset_index()


def _select_feature_columns(
    frame: pd.DataFrame,
    *,
    spec: RepositoryFeatureBankSpec,
) -> list[str]:
    # Hard guard: never allow the supervised success label to become a model feature.
    always_exclude = {"success"}
    if spec.feature_prefixes:
        selected = [
            column
            for column in frame.columns
            if column != "founder_uuid"
            and column not in always_exclude
            and (spec.label_column is None or column != spec.label_column)
            and column not in set(spec.exclude_columns)
            and any(column.startswith(prefix) for prefix in spec.feature_prefixes)
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
    else:
        selected = select_numeric_feature_columns(
            frame,
            exclude_columns=(
                list(spec.exclude_columns)
                + ([spec.label_column] if spec.label_column else [])
                + list(always_exclude)
            ),
        )
    if not selected:
        raise RuntimeError(f"Repository feature bank '{spec.feature_bank_id}' resolved zero feature columns.")
    return selected


def _resolve_binary_feature_columns(
    feature_columns: list[str],
    *,
    spec: RepositoryFeatureBankSpec,
) -> list[str]:
    if spec.all_features_binary:
        return feature_columns.copy()
    return [
        column
        for column in feature_columns
        if column in set(spec.binary_feature_columns)
    ]


def load_repository_feature_bank(
    *,
    repository_splits: LoadedFeatureRepositorySplits,
    spec: RepositoryFeatureBankSpec,
) -> ResolvedIntermediaryBank:
    train_frame = read_table(Path(spec.train_path))
    test_frame = read_table(Path(spec.test_path))
    train_frame = rename_identifier_column(
        train_frame,
        source_id_column=spec.source_id_column,
        target_id_column="founder_uuid",
    )
    test_frame = rename_identifier_column(
        test_frame,
        source_id_column=spec.source_id_column,
        target_id_column="founder_uuid",
    )
    validate_unique_ids(train_frame, id_column="founder_uuid")
    validate_unique_ids(test_frame, id_column="founder_uuid")

    train_frame = _reindex_to_canonical(
        train_frame,
        canonical_ids=repository_splits.train_ids,
        source_name=f"repository bank '{spec.feature_bank_id}' train frame",
    )
    test_frame = _reindex_to_canonical(
        test_frame,
        canonical_ids=repository_splits.test_ids,
        source_name=f"repository bank '{spec.feature_bank_id}' test frame",
    )

    feature_columns = _select_feature_columns(train_frame, spec=spec)
    missing_test_columns = [column for column in feature_columns if column not in test_frame.columns]
    if missing_test_columns:
        raise RuntimeError(
            f"Repository feature bank '{spec.feature_bank_id}' is missing test columns: {missing_test_columns}"
        )
    train_use = train_frame[["founder_uuid"] + feature_columns].copy()
    test_use = test_frame[["founder_uuid"] + feature_columns].copy()
    binary_feature_columns = _resolve_binary_feature_columns(feature_columns, spec=spec)

    return ResolvedIntermediaryBank(
        feature_id=spec.feature_bank_id,
        builder_kind="repository_feature_bank",
        storage_dir=Path(spec.train_path).parent,
        public_frame=train_use,
        private_frame=test_use,
        feature_columns=feature_columns,
        binary_feature_columns=binary_feature_columns,
        manifest={
            "feature_bank_id": spec.feature_bank_id,
            "builder_kind": "repository_feature_bank",
            "train_path": spec.train_path,
            "test_path": spec.test_path,
            "feature_columns": feature_columns,
            "binary_feature_columns": binary_feature_columns,
            "public_row_count": len(train_use),
            "private_row_count": len(test_use),
            "feature_count": len(feature_columns),
        },
    )


def load_repository_feature_banks(
    *,
    repository_splits: LoadedFeatureRepositorySplits,
    specs: list[RepositoryFeatureBankSpec],
) -> dict[str, ResolvedIntermediaryBank]:
    return {
        spec.feature_bank_id: load_repository_feature_bank(
            repository_splits=repository_splits,
            spec=spec,
        )
        for spec in specs
    }
