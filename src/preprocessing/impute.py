"""Multivariate imputation by chained equations (MICE).

Uses scikit-learn's IterativeImputer for numeric fields and most-frequent
imputation for categoricals, returning a single completed frame. For multiple
imputation, call ``mice_impute`` with different ``seed`` values and combine
downstream estimates with Rubin's rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer


def mice_impute(
    df: pd.DataFrame,
    max_iter: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a completed copy of ``df`` with numeric and categorical NaNs filled."""
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in out.columns if c not in num_cols]

    if num_cols:
        num_imputer = IterativeImputer(
            max_iter=max_iter, random_state=seed, sample_posterior=True
        )
        out[num_cols] = num_imputer.fit_transform(out[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        out[cat_cols] = cat_imputer.fit_transform(out[cat_cols])

    return out
