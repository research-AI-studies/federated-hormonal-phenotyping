"""Cluster-number selection, consensus clustering, and stability."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def gap_statistic(x, labels, k, seed=42, n_ref=10):
    """Gap statistic for a given labelling against a uniform reference."""
    def wk(X, lab):
        tot = 0.0
        for c in np.unique(lab):
            pts = X[lab == c]
            if len(pts) > 1:
                tot += np.sum((pts - pts.mean(0)) ** 2)
        return np.log(tot + 1e-12)
    rng = np.random.default_rng(seed)
    mins, maxs = x.min(0), x.max(0)
    refs = []
    for _ in range(n_ref):
        ref = rng.uniform(mins, maxs, size=x.shape)
        lab = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(ref)
        refs.append(wk(ref, lab))
    return float(np.mean(refs) - wk(x, labels))


def k_sweep(x, k_range, seed=42):
    """Return validity metrics per candidate k using k-means partitions."""
    out = {}
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        out[k] = {
            "silhouette": float(silhouette_score(x, lab)),
            "calinski_harabasz": float(calinski_harabasz_score(x, lab)),
            "davies_bouldin": float(davies_bouldin_score(x, lab)),
            "gap": gap_statistic(x, lab, k, seed),
        }
    return out


def consensus_partition(x, k, seed=42):
    """Co-association consensus across k-means, GMM, and agglomerative."""
    algos = {
        "kmeans": KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x),
        "gmm": GaussianMixture(n_components=k, covariance_type="full",
                               random_state=seed).fit_predict(x),
        "agglo": AgglomerativeClustering(n_clusters=k).fit_predict(x),
    }
    n = len(x)
    co = np.zeros((n, n))
    for lab in algos.values():
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]
            co[np.ix_(idx, idx)] += 1
    co /= len(algos)
    dist = 1.0 - co
    consensus = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(dist)
    pair_ari = {}
    names = list(algos)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair_ari[f"{names[i]}-{names[j]}"] = float(
                adjusted_rand_score(algos[names[i]], algos[names[j]])
            )
    return consensus, algos, pair_ari


def bootstrap_stability(x, k, consensus_labels, n_boot=1000, seed=42):
    """Mean/percentile ARI of bootstrap consensus vs reference partition."""
    rng = np.random.default_rng(seed)
    n = len(x)
    aris = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        lab = KMeans(n_clusters=k, n_init=3, random_state=seed + b).fit_predict(x[idx])
        aris.append(adjusted_rand_score(consensus_labels[idx], lab))
    aris = np.array(aris)
    return {
        "mean_ari": float(aris.mean()),
        "ci_low": float(np.percentile(aris, 2.5)),
        "ci_high": float(np.percentile(aris, 97.5)),
    }


def permutation_silhouette(x, labels, n_perm=1000, seed=42):
    """Empirical p-value and max permuted silhouette vs observed."""
    rng = np.random.default_rng(seed)
    obs = silhouette_score(x, labels)
    perm_max = -1.0
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(labels)
        s = silhouette_score(x, p)
        perm_max = max(perm_max, s)
        if s >= obs:
            ge += 1
    return {"observed": float(obs), "perm_max": float(perm_max),
            "p_value": (ge + 1) / (n_perm + 1)}
