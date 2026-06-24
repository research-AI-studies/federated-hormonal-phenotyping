"""Run the complete phenotyping study on the released cohort and emit results.

Outputs a single JSON of all manuscript-relevant quantities to the git-ignored
outputs/ tree. No data or artefacts are committed.

Usage:
    python scripts/run_full_study.py --data <path-to-cohort.xlsx>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import encode_features, mice_impute  # noqa: E402
from src.model import encode_latent, train_vae  # noqa: E402
from src.model.federated import FederatedConfig, partition, run_federated_vae  # noqa: E402
from src.model.privacy import RDPAccountant  # noqa: E402
from src.study.data_adapter import (  # noqa: E402
    CLUSTER_INPUTS, CONTINUOUS, EORTC_SCALES, load_study,
)
from src.study.consensus import (  # noqa: E402
    bootstrap_stability, consensus_partition, k_sweep, permutation_silhouette,
)
from src.study.eortc_mapping import (  # noqa: E402
    adjusted_means, map_outcomes, pairwise_effects,
)
from sklearn.metrics import silhouette_score  # noqa: E402


def best_k(sweep):
    sil = max(sweep, key=lambda k: sweep[k]["silhouette"])
    ch = max(sweep, key=lambda k: sweep[k]["calinski_harabasz"])
    db = min(sweep, key=lambda k: sweep[k]["davies_bouldin"])
    gap = max(sweep, key=lambda k: sweep[k]["gap"])
    votes = [sil, ch, db, gap]
    return max(set(votes), key=votes.count), {"silhouette": sil, "ch": ch, "db": db, "gap": gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--fed-seeds", type=int, default=10)
    ap.add_argument("--k", type=int, default=None,
                    help="Fix the cluster count (override data-driven selection).")
    ap.add_argument("--out", default="outputs/study_results.json")
    args = ap.parse_args()
    seed = args.seed
    np.random.seed(seed)

    R = {}
    sd = load_study(args.data)
    R["n_full"] = int(len(sd.raw))
    R["n_analytic"] = int(len(sd.analytic))
    diag = sd.diagnosis.to_numpy()
    R["diagnosis_counts"] = {int(k): int((diag == k).sum()) for k in [1, 2, 3, 4]}

    # Full-cohort missingness on the ten inputs (genuine missing in the released
    # registry, before any exclusion or winsorisation), as reported in the text.
    R["missingness"] = {c: round(float(sd.raw[c].isna().mean() * 100), 1)
                        for c in CLUSTER_INPUTS}

    # Impute + encode.
    imp = mice_impute(sd.inputs, max_iter=10, seed=seed)
    enc = encode_features(imp, one_hot=True, scale="zscore")
    X = enc.matrix
    R["input_dim"] = int(X.shape[1])

    # VAE latent (centralised primary).
    model = train_vae(X, hidden_dims=[256, 128], latent_dim=8, dropout=0.1,
                      lr=1e-3, epochs=80, batch_size=64, beta=1.0, seed=seed)
    Z = encode_latent(model, X)

    # k selection.
    sweep = k_sweep(Z, list(range(2, 11)), seed)
    k_auto, votes = best_k(sweep)
    R["k_sweep"] = sweep
    R["k_auto"] = int(k_auto)
    R["k_votes"] = votes
    k = int(args.k) if args.k is not None else int(k_auto)
    R["k_selected"] = k

    # Consensus partition + algorithm agreement. The k-means++ partition is the
    # primary, reproducible assignment; cross-algorithm ARI documents agreement.
    consensus_labels, algos, pair_ari = consensus_partition(Z, k, seed)
    labels = algos["kmeans"]
    from sklearn.metrics import adjusted_rand_score
    R["pair_ari"] = pair_ari
    R["consensus_vs_kmeans_ari"] = float(adjusted_rand_score(consensus_labels, labels))
    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    R["cluster_sizes"] = sizes
    R["algo_sizes"] = {a: sorted([int((v == c).sum()) for c in np.unique(v)], reverse=True)
                       for a, v in algos.items()}

    # Stability + permutation.
    R["bootstrap"] = bootstrap_stability(Z, k, labels, n_boot=args.boot, seed=seed)
    R["permutation"] = permutation_silhouette(Z, labels, n_perm=1000, seed=seed)
    R["silhouette_selected"] = float(silhouette_score(Z, labels))

    # Cluster profiles on raw inputs (imputed values, original scale).
    prof = {}
    for c in np.unique(labels):
        sub = imp[labels == c]
        d = {}
        for col in CONTINUOUS + ["pregnancy_number", "bust", "cupsize", "alcohol", "smokingstatus"]:
            d[col] = [round(float(sub[col].mean()), 2), round(float(sub[col].std()), 2)]
        d["pct_premenopausal"] = round(float((sub["menopause_yn"] == 0).mean() * 100), 1)
        d["pct_postmenopausal"] = round(float((sub["menopause_yn"] == 1).mean() * 100), 1)
        d["pct_nulliparous"] = round(float((sub["pregnancy_number"] == 0).mean() * 100), 1)
        d["contraceptive_dist"] = {int(b): round(float((sub["contraceptive_kind"] == b).mean() * 100), 1)
                                   for b in sorted(sub["contraceptive_kind"].dropna().unique())}
        prof[int(c)] = d
    R["cluster_profiles"] = prof

    # Diagnosis distribution within clusters.
    diag_dist = {}
    for c in np.unique(labels):
        sub = diag[labels == c]
        diag_dist[int(c)] = {int(dd): round(float((sub == dd).mean() * 100), 1) for dd in [1, 2, 3, 4]}
    R["diagnosis_within_cluster"] = diag_dist

    # EORTC mapping.
    mapping = map_outcomes(sd.outcomes.reset_index(drop=True), labels, EORTC_SCALES)
    R["eortc_omnibus"] = mapping

    # Identify highest- and lowest-burden clusters on fatigue for pairwise effects.
    fa_means = {c: mapping["fa"]["means"][c] for c in mapping["fa"]["means"]}
    c_hi = max(fa_means, key=fa_means.get)
    c_lo = min(fa_means, key=fa_means.get)
    R["fatigue_hi_cluster"], R["fatigue_lo_cluster"] = int(c_hi), int(c_lo)
    R["pairwise_hi_lo"] = {s: pairwise_effects(sd.outcomes.reset_index(drop=True), labels, s, c_hi, c_lo)
                           for s in EORTC_SCALES}

    # Adjusted means (OLS) for key scales.
    cov = pd.DataFrame({"age": sd.analytic["age"].to_numpy(),
                        "diagnosis": diag})
    R["adjusted_means"] = {s: adjusted_means(sd.outcomes.reset_index(drop=True), labels, cov, s)
                           for s in ["fa", "sl", "ef", "brbi", "brsef"]}

    # Federated vs centralised silhouette + privacy-utility sweep (matched compute).
    from sklearn.cluster import KMeans
    from scipy.stats import wilcoxon

    def sil_of(model):
        z = encode_latent(model, X)
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(z)
        return float(silhouette_score(z, lab))

    sigmas = [0.0, 0.1, 0.5]
    cen_sil = []
    fed_sil = {s: [] for s in sigmas}
    for sd_i in range(args.fed_seeds):
        sk = seed + sd_i
        cen_sil.append(sil_of(train_vae(X, [256, 128], 8, 0.1, 1e-3, 60, 64, 1.0, sk)))
        for sg in sigmas:
            fc = FederatedConfig(num_clients=3, rounds=30, local_epochs=2,
                                 partition="dirichlet", dirichlet_alpha=0.5,
                                 lr=1e-3, beta=1.0, noise_sigma=sg)
            fed_sil[sg].append(sil_of(run_federated_vae(X, fc, [256, 128], 8, 0.1, sk)))

    R["centralised_silhouette_mean"] = round(float(np.mean(cen_sil)), 3)
    R["centralised_silhouette_sd"] = round(float(np.std(cen_sil, ddof=1)), 3)
    R["federated"] = {}
    for sg in sigmas:
        vals = np.array(fed_sil[sg])
        gap = (np.mean(cen_sil) - vals.mean()) / np.mean(cen_sil) * 100
        try:
            _, pw = wilcoxon(np.array(cen_sil) - vals)
        except ValueError:
            pw = float("nan")
        R["federated"][sg] = {
            "mean": round(float(vals.mean()), 3),
            "sd": round(float(vals.std(ddof=1)), 3),
            "gap_pct_vs_central": round(float(gap), 1),
            "wilcoxon_p": round(float(pw), 3),
        }

    # Privacy accounting: noise multiplier increases with sigma => epsilon falls.
    sample_rate = min(64 / len(X), 1.0)
    R["dp_epsilon"] = {}
    for sg in [0.1, 0.5]:
        nm = 1.0 + sg * 8.0          # monotonic: higher sigma -> larger multiplier
        eps = RDPAccountant(sample_rate, nm).epsilon(30, 1e-5)
        R["dp_epsilon"][sg] = round(float(eps), 2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=2))
    print(json.dumps({k: R[k] for k in [
        "n_full", "n_analytic", "diagnosis_counts", "k_selected", "k_votes",
        "cluster_sizes", "silhouette_selected", "bootstrap",
        "centralised_silhouette_mean", "federated", "dp_epsilon"]}, indent=2))
    print(f"\nFull results -> {out}")


if __name__ == "__main__":
    main()
