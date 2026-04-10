from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.intermediary_features.mirror import build_vcbench_mirror_frames, clean_text
from src.intermediary_features.sentence_transformer import build_sentence_transformer_frames
from src.intermediary_features.storage import (
    ResolvedIntermediaryBank,
    bank_exists,
    llm_engineered_storage_dir,
    load_intermediary_bank,
    mirror_storage_dir,
    save_intermediary_bank,
    sentence_transformer_storage_dir,
)
from src.intermediary_features.structured_text import render_structured_text_frames
from src.pipeline.config import FeatureSetSpec, IntermediaryFeatureSpec


Logger = Callable[[str], None]


@dataclass(frozen=True)
class AssembledFeatureSet:
    feature_set_id: str
    feature_ids: list[str]
    public_frame: pd.DataFrame
    private_frame: pd.DataFrame
    feature_columns: list[str]
    manifest: dict[str, object]


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _resolve_storage_dir(spec: IntermediaryFeatureSpec):
    if spec.kind == "vcbench_mirror_baseline_v1":
        return mirror_storage_dir()
    if spec.kind == "sentence_transformer_prose_v1":
        return sentence_transformer_storage_dir(variant="prose", model_name=spec.embedding_model_name or "")
    if spec.kind == "sentence_transformer_structured_v1":
        return sentence_transformer_storage_dir(variant="structured", model_name=spec.embedding_model_name or "")
    if spec.kind == "llm_engineered_v1":
        return llm_engineered_storage_dir()
    raise RuntimeError(f"Unsupported intermediary feature kind '{spec.kind}'.")


def _build_bank(
    *,
    spec: IntermediaryFeatureSpec,
    public_raw: pd.DataFrame,
    private_raw: pd.DataFrame,
) -> ResolvedIntermediaryBank:
    storage_dir = _resolve_storage_dir(spec)

    if spec.kind == "vcbench_mirror_baseline_v1":
        public_frame, private_frame, feature_columns, manifest = build_vcbench_mirror_frames(
            public_raw,
            private_raw,
        )
        return save_intermediary_bank(
            feature_id=spec.feature_id,
            builder_kind=spec.kind,
            storage_dir=storage_dir,
            public_frame=public_frame,
            private_frame=private_frame,
            feature_columns=feature_columns,
            manifest=manifest,
        )

    if spec.kind == "sentence_transformer_prose_v1":
        public_text_frame = pd.DataFrame(
            {
                "founder_uuid": public_raw["founder_uuid"].astype(str),
                "rendered_text": public_raw["anonymised_prose"].map(clean_text).astype(str),
            }
        )
        private_text_frame = pd.DataFrame(
            {
                "founder_uuid": private_raw["founder_uuid"].astype(str),
                "rendered_text": private_raw["anonymised_prose"].map(clean_text).astype(str),
            }
        )
        public_frame, private_frame, feature_columns, manifest = build_sentence_transformer_frames(
            public_text_frame=public_text_frame,
            private_text_frame=private_text_frame,
            model_name=spec.embedding_model_name or "",
            feature_prefix="sentence_prose",
        )
        return save_intermediary_bank(
            feature_id=spec.feature_id,
            builder_kind=spec.kind,
            storage_dir=storage_dir,
            public_frame=public_frame,
            private_frame=private_frame,
            feature_columns=feature_columns,
            manifest=manifest,
            extra_tables={
                "public_rendered.csv": public_text_frame,
                "private_rendered.csv": private_text_frame,
            },
        )

    if spec.kind == "sentence_transformer_structured_v1":
        public_text_frame, private_text_frame = render_structured_text_frames(public_raw, private_raw)
        public_frame, private_frame, feature_columns, manifest = build_sentence_transformer_frames(
            public_text_frame=public_text_frame,
            private_text_frame=private_text_frame,
            model_name=spec.embedding_model_name or "",
            feature_prefix="sentence_structured",
        )
        return save_intermediary_bank(
            feature_id=spec.feature_id,
            builder_kind=spec.kind,
            storage_dir=storage_dir,
            public_frame=public_frame,
            private_frame=private_frame,
            feature_columns=feature_columns,
            manifest=manifest,
            extra_tables={
                "public_rendered.csv": public_text_frame,
                "private_rendered.csv": private_text_frame,
            },
        )

    if spec.kind == "llm_engineered_v1":
        raise RuntimeError(
            "The llm_engineered intermediary feature family is scaffolded but inactive. "
            "Provide the custom prompt assets before selecting it."
        )

    raise RuntimeError(f"Unsupported intermediary feature kind '{spec.kind}'.")


def prepare_intermediary_banks(
    *,
    public_raw: pd.DataFrame,
    private_raw: pd.DataFrame,
    feature_specs: list[IntermediaryFeatureSpec],
    force_rebuild: bool,
    logger: Logger | None = None,
) -> dict[str, ResolvedIntermediaryBank]:
    resolved: dict[str, ResolvedIntermediaryBank] = {}
    for spec in feature_specs:
        storage_dir = _resolve_storage_dir(spec)
        if bank_exists(storage_dir) and not force_rebuild:
            _log(logger, f"Reusing intermediary feature bank '{spec.feature_id}' from {storage_dir}.")
            resolved[spec.feature_id] = load_intermediary_bank(
                feature_id=spec.feature_id,
                builder_kind=spec.kind,
                storage_dir=storage_dir,
            )
            continue

        _log(logger, f"Building intermediary feature bank '{spec.feature_id}'.")
        resolved[spec.feature_id] = _build_bank(
            spec=spec,
            public_raw=public_raw,
            private_raw=private_raw,
        )
    return resolved


def _merge_with_full_overlap(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    *,
    on: str,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    merged = left_frame.merge(right_frame, on=on, how="left", validate="one_to_one")
    if merged.isna().any().any():
        missing = sorted(
            set(left_frame[on].astype(str)) - set(right_frame[on].astype(str))
        )
        raise RuntimeError(
            f"{right_name} is missing {len(missing)} ids required by {left_name}. Examples: {missing[:5]}"
        )
    return merged


def assemble_feature_sets(
    *,
    public_founder_ids: pd.Series,
    private_founder_ids: pd.Series,
    banks_by_id: dict[str, ResolvedIntermediaryBank],
    feature_sets: list[FeatureSetSpec],
) -> list[AssembledFeatureSet]:
    public_base = pd.DataFrame({"founder_uuid": public_founder_ids.astype(str).tolist()})
    private_base = pd.DataFrame({"founder_uuid": private_founder_ids.astype(str).tolist()})
    assembled: list[AssembledFeatureSet] = []

    for feature_set in feature_sets:
        public_frame = public_base.copy()
        private_frame = private_base.copy()
        feature_columns: list[str] = []
        component_manifests: list[dict[str, object]] = []

        for feature_id in feature_set.feature_ids:
            bank = banks_by_id[feature_id]
            public_frame = _merge_with_full_overlap(
                public_frame,
                bank.public_frame,
                on="founder_uuid",
                left_name=f"feature set '{feature_set.feature_set_id}' public ids",
                right_name=f"intermediary bank '{feature_id}' public frame",
            )
            private_frame = _merge_with_full_overlap(
                private_frame,
                bank.private_frame,
                on="founder_uuid",
                left_name=f"feature set '{feature_set.feature_set_id}' private ids",
                right_name=f"intermediary bank '{feature_id}' private frame",
            )
            feature_columns.extend(bank.feature_columns)
            component_manifests.append(
                {
                    "feature_id": bank.feature_id,
                    "builder_kind": bank.builder_kind,
                    "storage_dir": str(bank.storage_dir),
                    "feature_count": len(bank.feature_columns),
                }
            )

        assembled.append(
            AssembledFeatureSet(
                feature_set_id=feature_set.feature_set_id,
                feature_ids=feature_set.feature_ids,
                public_frame=public_frame,
                private_frame=private_frame,
                feature_columns=feature_columns,
                manifest={
                    "feature_set_id": feature_set.feature_set_id,
                    "feature_ids": feature_set.feature_ids,
                    "feature_count": len(feature_columns),
                    "component_banks": component_manifests,
                },
            )
        )

    return assembled
