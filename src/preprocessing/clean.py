"""Cleaning: missingness filtering and winsorisation of continuous tails."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    """Drop columns whose missing fraction exceeds ``threshold``."""
    keep = df.columns[df.isna().mean() <= threshold]
    return df[keep].copy()


def winsorize(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    limits: tuple[float, float] = (0.01, 0.01),
) -> pd.DataFrame:
    """Clip the lower/upper tails of numeric columns at the given quantiles."""
    out = df.copy()
    lo_q, hi_q = limits
    numeric = cols or out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric:
        lo = out[col].quantile(lo_q)
        hi = out[col].quantile(1 - hi_q)
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out
