"""Phenotype-to-outcome mapping statistics for the EORTC scales."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, rankdata


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def cohens_d_rank(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    pooled = np.concatenate([a, b])
    r = rankdata(pooled)
    ra, rb = r[: len(a)], r[len(a):]
    sd = np.sqrt(((len(a) - 1) * ra.std(ddof=1) ** 2 + (len(b) - 1) * rb.std(ddof=1) ** 2)
                 / (len(a) + len(b) - 2))
    return (ra.mean() - rb.mean()) / sd if sd > 0 else np.nan


def holm_adjust(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def map_outcomes(outcomes: pd.DataFrame, labels: np.ndarray, scales: list[str]):
    """Kruskal-Wallis omnibus + Holm across scales; per-scale cluster means."""
    ks = sorted(np.unique(labels))
    results = {}
    raw_p = []
    for s in scales:
        groups = [outcomes[s].values[labels == c] for c in ks]
        groups = [g[~np.isnan(g)] for g in groups]
        try:
            stat, p = kruskal(*groups)
        except ValueError:
            stat, p = np.nan, np.nan
        means = {int(c): float(np.nanmean(outcomes[s].values[labels == c])) for c in ks}
        results[s] = {"H": float(stat), "p": float(p), "means": means}
        raw_p.append(p)
    adj = holm_adjust([r if not np.isnan(r) else 1.0 for r in raw_p])
    for s, a in zip(scales, adj):
        results[s]["p_holm"] = float(a)
    return results


def pairwise_effects(outcomes, labels, scale, c_hi, c_lo):
    """Cliff's delta and Cohen's d (rank) for one cluster pair on one scale."""
    a = outcomes[scale].values[labels == c_hi]
    b = outcomes[scale].values[labels == c_lo]
    p = mannwhitneyu(a[~np.isnan(a)], b[~np.isnan(b)]).pvalue
    return {
        "cliffs_delta": float(cliffs_delta(a, b)),
        "cohens_d": float(cohens_d_rank(a, b)),
        "mwu_p": float(p),
    }


def adjusted_means(outcomes, labels, covariates: pd.DataFrame, scale: str):
    """OLS-adjusted marginal cluster means for one scale (model-based EMM)."""
    import statsmodels.formula.api as smf

    df = covariates.copy()
    df["y"] = outcomes[scale].values
    df["cluster"] = labels.astype(str)
    df = df.dropna(subset=["y"])
    model = smf.ols("y ~ C(cluster) + age + C(diagnosis)", data=df).fit()
    grid = df.copy()
    out = {}
    for c in sorted(df["cluster"].unique()):
        g = grid.copy()
        g["cluster"] = c
        out[int(float(c))] = float(model.predict(g).mean())
    return out
