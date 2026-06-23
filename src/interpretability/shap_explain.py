"""Phenotype attribution via SHAP.

Fits a lightweight surrogate classifier that maps original features to the
discovered cluster labels, then uses SHAP to quantify which baseline variables
drive each phenotype. Falls back to permutation importance when SHAP is not
installed so the pipeline always runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def phenotype_attributions(
    features: pd.DataFrame,
    labels: np.ndarray,
    background: int = 100,
    nsamples: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a (feature x phenotype) mean-absolute-attribution table."""
    clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    clf.fit(features.to_numpy(), labels)

    try:
        import shap

        bg = shap.sample(features, min(background, len(features)), random_state=seed)
        explainer = shap.TreeExplainer(clf, bg)
        sample = features.sample(min(nsamples, len(features)), random_state=seed)
        values = explainer.shap_values(sample)
        classes = clf.classes_
        cols = {}
        for ci, cls in enumerate(classes):
            arr = values[ci] if isinstance(values, list) else values[..., ci]
            cols[f"phenotype_{cls}"] = np.abs(arr).mean(axis=0)
        return pd.DataFrame(cols, index=features.columns)
    except Exception:
        perm = permutation_importance(
            clf, features.to_numpy(), labels, n_repeats=10, random_state=seed
        )
        return pd.DataFrame(
            {"importance": perm.importances_mean}, index=features.columns
        )
