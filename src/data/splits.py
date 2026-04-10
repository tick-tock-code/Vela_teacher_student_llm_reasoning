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


def build_cv_splits(
    n_rows: int,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[FoldSplit]:
    splitter = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    splits: list[FoldSplit] = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n_rows)), start=1):
        splits.append(
            FoldSplit(
                split_id=f"fold_{fold_index:02d}",
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    return splits
