from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass(frozen=True)
class FoldSplit:
    split_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def _build_split_records(splitter, n_rows: int) -> list[FoldSplit]:
    splits: list[FoldSplit] = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(n_rows), np.zeros(n_rows)), start=1):
        splits.append(
            FoldSplit(
                split_id=f"fold_{fold_index:02d}",
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    return splits


def build_public_cv_splits(
    labels: pd.Series | np.ndarray,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[FoldSplit]:
    y = np.asarray(labels, dtype=int)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    splits: list[FoldSplit] = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        splits.append(
            FoldSplit(
                split_id=f"fold_{fold_index:02d}",
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    return splits


def _quantile_buckets(values: np.ndarray, *, max_bins: int, n_splits: int) -> np.ndarray | None:
    series = pd.Series(np.asarray(values, dtype=float))
    max_bins_use = min(max_bins, max(2, len(series) // n_splits))
    for n_bins in range(max_bins_use, 1, -1):
        try:
            buckets = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        if buckets.isna().any():
            continue
        counts = buckets.value_counts()
        if len(counts) < 2:
            continue
        if int(counts.min()) < n_splits:
            continue
        return buckets.to_numpy(dtype=int)
    return None


def build_stratified_reasoning_cv_splits(
    target_frame: pd.DataFrame,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    max_bins: int = 10,
) -> list[FoldSplit]:
    if target_frame.empty:
        raise RuntimeError("Cannot build stratified reasoning CV splits from an empty target frame.")

    stratification_signal = target_frame.astype(float).mean(axis=1).to_numpy(dtype=float)
    buckets = _quantile_buckets(
        stratification_signal,
        max_bins=max_bins,
        n_splits=n_splits,
    )
    if buckets is None:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
        splits: list[FoldSplit] = []
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(target_frame))), start=1):
            splits.append(
                FoldSplit(
                    split_id=f"fold_{fold_index:02d}",
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
            )
        return splits

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    splits: list[FoldSplit] = []
    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(buckets)), buckets),
        start=1,
    ):
        splits.append(
            FoldSplit(
                split_id=f"fold_{fold_index:02d}",
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    return splits
