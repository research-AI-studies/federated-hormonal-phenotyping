"""Feature encoding: one-hot categoricals and scale continuous fields."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class EncodedData:
    matrix: np.ndarray
    feature_names: list[str]
    frame: pd.DataFrame


def encode_features(
    df: pd.DataFrame,
    one_hot: bool = True,
    scale: str = "zscore",
    drop_cols: list[str] | None = None,
) -> EncodedData:
    """One-hot encode categoricals, scale numerics, return a model-ready matrix."""
    work = df.drop(columns=drop_cols or [], errors="ignore").copy()

    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in work.columns if c not in num_cols]

    if scale == "zscore" and num_cols:
        work[num_cols] = StandardScaler().fit_transform(work[num_cols])
    elif scale == "minmax" and num_cols:
        work[num_cols] = MinMaxScaler().fit_transform(work[num_cols])

    if one_hot and cat_cols:
        work = pd.get_dummies(work, columns=cat_cols, drop_first=False)

    work = work.astype(float)
    return EncodedData(
        matrix=work.to_numpy(dtype=np.float32),
        feature_names=work.columns.tolist(),
        frame=work,
    )
