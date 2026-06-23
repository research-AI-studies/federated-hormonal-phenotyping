"""Internal and external cluster-quality metrics."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def internal_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette, Calinski-Harabasz, Davies-Bouldin for a labelling."""
    if len(set(labels)) < 2:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan")}
    return {
        "silhouette": float(silhouette_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
    }


def external_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float]:
    """Adjusted Rand Index and Normalised Mutual Information between labellings."""
    return {
        "ari": float(adjusted_rand_score(labels_a, labels_b)),
        "nmi": float(normalized_mutual_info_score(labels_a, labels_b)),
    }


def sweep_k(x: np.ndarray, k_range: list[int], seed: int = 42) -> dict[int, dict[str, float]]:
    """Evaluate internal metrics across a range of k using k-means on the latent space."""
    out: dict[int, dict[str, float]] = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        out[k] = internal_metrics(x, labels)
    return out
