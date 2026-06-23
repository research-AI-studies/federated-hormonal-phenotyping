"""Consensus-clustering stability assessment.

Repeatedly subsamples the embedding, re-clusters, and measures how often pairs
of samples co-occur in the same cluster. The mean pairwise consensus and the
dispersion of the consensus matrix summarise partition stability.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def consensus_stability(
    x: np.ndarray,
    k: int,
    resamples: int = 100,
    subsample_fraction: float = 0.8,
    seed: int = 42,
) -> dict[str, float]:
    """Return mean consensus and the proportion of ambiguously clustered pairs."""
    rng = np.random.default_rng(seed)
    n = len(x)
    co_assign = np.zeros((n, n))
    co_sample = np.zeros((n, n))

    for r in range(resamples):
        m = max(int(subsample_fraction * n), k + 1)
        idx = rng.choice(n, size=m, replace=False)
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed + r).fit_predict(x[idx])
        for lab in np.unique(labels):
            members = idx[labels == lab]
            co_assign[np.ix_(members, members)] += 1
        co_sample[np.ix_(idx, idx)] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        consensus = np.where(co_sample > 0, co_assign / co_sample, 0.0)

    iu = np.triu_indices(n, k=1)
    vals = consensus[iu]
    ambiguous = float(np.mean((vals > 0.1) & (vals < 0.9)))
    return {
        "mean_consensus": float(np.mean(vals)),
        "proportion_ambiguous_pairs": ambiguous,
    }
