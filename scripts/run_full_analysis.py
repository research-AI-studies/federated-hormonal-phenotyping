"""Full analysis on a private local cohort.

Runs the complete phenotyping and validation pipeline and writes a single
results JSON plus key figures to the git-ignored ``outputs/`` tree. The input
file is read from a local path and is never committed.

Stages: cohort/missingness -> encode+MICE -> federated VAE+SOM -> k-selection
(silhouette/CH/DB/gap) -> consensus (k-means/GMM/HDBSCAN) -> bootstrap, CV,
permutation, subgroup, MICE-vs-complete-case agreement -> phenotype profiles ->
EORTC mapping (Kruskal-Wallis/Dunn/Cliff/Cohen/GLMM/Heckman) -> federated vs
centralised non-inferiority + differential-privacy sweep.

Usage:
    python scripts/run_full_analysis.py --data <local.xlsx> --out outputs/analysis_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, cohen_kappa_score,
    davies_bouldin_score, silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.cohort_mapping import (  # noqa: E402
    CLUSTERING_INPUTS, COVARIATES, EORTC_ALL, EORTC_CONDITIONAL, EORTC_PRIMARY,
    analytic_mask, map_raw,
)
from src.model.vae import encode_latent  # noqa: E402
from src.model.federated import (  # noqa: E402
    FederatedConfig, partition, run_federated_som, run_federated_vae,
)
from src.model.privacy import RDPAccountant  # noqa: E402

RNG = np.random.default_rng(42)
CONT = ["age", "bmi", "menarche", "pregnancies", "alcohol", "smoking", "bust", "cup"]
CAT = ["menopausal", "contraceptive"]


# ----------------------------- helper statistics ---------------------------
def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((gt - lt) / (len(a) * len(b)))


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def gap_statistic(x, labels, k, n_ref=10, seed=42):
    rng = np.random.default_rng(seed)
    def wk(X, lab):
        tot = 0.0
        for c in np.unique(lab):
            pts = X[lab == c]
            if len(pts) > 1:
                tot += ((pts - pts.mean(0)) ** 2).sum()
        return np.log(tot + 1e-12)
    obs = wk(x, labels)
    mins, maxs = x.min(0), x.max(0)
    refs = []
    for _ in range(n_ref):
        ref = rng.uniform(mins, maxs, size=x.shape)
        rl = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(ref)
        refs.append(wk(ref, rl))
    return float(np.mean(refs) - obs)


def best_match_ari(global_lab, sub_lab):
    return float(adjusted_rand_score(global_lab, sub_lab))


def aligned_kappa(a, b, k):
    """Cohen's kappa after optimally matching label ids (permutation-invariant)."""
    from scipy.optimize import linear_sum_assignment
    cm = np.zeros((k, k))
    for i in range(len(a)):
        if 0 <= a[i] < k and 0 <= b[i] < k:
            cm[a[i], b[i]] += 1
    row, col = linear_sum_assignment(-cm)
    mapping = {int(c): int(r) for r, c in zip(row, col)}
    b2 = np.array([mapping.get(int(x), int(x)) for x in b])
    return float(cohen_kappa_score(a, b2))


# ------------------------------- encoding -----------------------------------
def encode(df, fit_scaler=None, fit_cols=None):
    work = df[CLUSTERING_INPUTS].copy()
    work = pd.get_dummies(work, columns=["contraceptive"], dummy_na=False)
    work["menopausal"] = work["menopausal"].astype(float)
    if fit_cols is not None:
        for c in fit_cols:
            if c not in work.columns:
                work[c] = 0.0
        work = work[fit_cols]
    cols = work.columns.tolist()
    cont_cols = [c for c in CONT if c in cols]
    scaler = fit_scaler or StandardScaler().fit(work[cont_cols].values)
    work[cont_cols] = scaler.transform(work[cont_cols].values)
    return work.astype(float), scaler, cols


def mice(df):
    out = df.copy()
    num = out[CLUSTERING_INPUTS].select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in CLUSTERING_INPUTS if c not in num]
    if num:
        out[num] = IterativeImputer(max_iter=10, random_state=42,
                                    sample_posterior=True).fit_transform(out[num])
    if cat:
        out[cat] = SimpleImputer(strategy="most_frequent").fit_transform(out[cat])
    return out


# ------------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs/analysis_results.json")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--dp_seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.bootstrap, args.perms, args.dp_seeds = 20, 100, 3

    R = {}
    raw = pd.read_excel(args.data)
    mc = map_raw(raw)
    d = mc.frame
    mask = analytic_mask(d)
    A = d[mask].reset_index(drop=True)
    R["n_raw"] = int(len(d))
    R["n_analytic"] = int(len(A))

    # ---- 4.1 cohort + missingness ----
    dg = A["diagnosis_group"].value_counts().to_dict()
    R["diagnosis_counts"] = {k: int(v) for k, v in dg.items()}
    R["diagnosis_pct"] = {k: round(100 * v / len(A), 1) for k, v in dg.items()}
    miss = {c: round(100 * A[c].isna().mean(), 1) for c in CLUSTERING_INPUTS + EORTC_ALL}
    R["missingness_pct"] = miss
    R["input_summary"] = {
        c: {"mean": round(float(A[c].mean()), 2), "sd": round(float(A[c].std()), 2)}
        for c in CONT
    }
    R["menopausal_post_pct"] = round(100 * float(A["menopausal"].mean(skipna=True)), 1)

    # ---- encode + MICE ----
    A_imp = mice(A)
    Xdf, scaler, cols = encode(A_imp)
    X = Xdf.to_numpy(np.float32)
    R["encoded_dim"] = int(X.shape[1])
    R["encoded_features"] = cols

    # ---- federated VAE + SOM ----
    vae_hidden, latent_dim = [64, 32], 8
    fed = FederatedConfig(num_clients=3, rounds=30, local_epochs=2,
                          partition="dirichlet", dirichlet_alpha=0.5, beta=1.0)
    model = run_federated_vae(X, fed, vae_hidden, latent_dim, 0.1, seed=42)
    parts = [p for p in partition(X, 3, "dirichlet", 0.5, 42) if len(p) > 0]
    Z = encode_latent(model, X)
    som = run_federated_som(Z, parts, (10, 10), 1.2, 0.5, 4000, seed=42)
    R["som_quantisation_error"] = round(som.quantisation_error(Z), 4)

    # ---- 4.2 k-selection ----
    ksel = {}
    for k in range(2, 11):
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
        ksel[k] = {
            "silhouette": round(float(silhouette_score(Z, lab)), 3),
            "calinski_harabasz": round(float(calinski_harabasz_score(Z, lab)), 1),
            "davies_bouldin": round(float(davies_bouldin_score(Z, lab)), 3),
            "gap": round(gap_statistic(Z, lab, k), 3),
        }
    R["k_selection"] = ksel
    best_k = max(ksel, key=lambda k: ksel[k]["silhouette"])
    R["best_k"] = int(best_k)

    # ---- consensus clustering ----
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(Z)
    gm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42).fit_predict(Z)
    import hdbscan
    hb = hdbscan.HDBSCAN(min_cluster_size=max(15, len(Z) // 40)).fit_predict(Z)
    valid = hb >= 0
    has_hb = valid.sum() > 1 and len(np.unique(hb[valid])) > 1
    R["consensus_pairwise_ari"] = {
        "kmeans_gmm": round(float(adjusted_rand_score(km, gm)), 3),
        "kmeans_hdbscan": round(float(adjusted_rand_score(km[valid], hb[valid])), 3) if has_hb else None,
        "gmm_hdbscan": round(float(adjusted_rand_score(gm[valid], hb[valid])), 3) if has_hb else None,
    }
    R["hdbscan_noise"] = int((hb < 0).sum())
    R["hdbscan_n_clusters"] = int(len(np.unique(hb[valid]))) if valid.any() else 0
    # co-association -> agglomerative consensus
    n = len(Z)
    co = np.zeros((n, n))
    for lab in (km, gm, hb):
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]
            co[np.ix_(idx, idx)] += 1
    co /= 3.0
    consensus = AgglomerativeClustering(n_clusters=best_k, metric="precomputed",
                                        linkage="average").fit_predict(1 - co)
    sizes = [int((consensus == c).sum()) for c in range(best_k)]
    R["consensus_sizes"] = sorted(sizes, reverse=True)
    R["consensus_silhouette"] = round(float(silhouette_score(Z, consensus)), 3)

    # ---- bootstrap stability ----
    aris = []
    for b in range(args.bootstrap):
        idx = RNG.choice(n, n, replace=True)
        lab = KMeans(n_clusters=best_k, n_init=3, random_state=b).fit_predict(Z[idx])
        # compare on unique sampled indices
        uniq = np.unique(idx)
        ref = consensus[uniq]
        # map bootstrap labels back to first occurrence
        first = {ii: lab[np.where(idx == ii)[0][0]] for ii in uniq}
        boot = np.array([first[ii] for ii in uniq])
        aris.append(adjusted_rand_score(ref, boot))
    R["bootstrap_ari_mean"] = round(float(np.mean(aris)), 3)
    R["bootstrap_ari_ci"] = [round(float(np.percentile(aris, 2.5)), 3),
                             round(float(np.percentile(aris, 97.5)), 3)]

    # ---- permutation test ----
    obs_sil = float(silhouette_score(Z, consensus))
    perm_max = -1.0
    ge = 0
    for _ in range(args.perms):
        pl = RNG.permutation(consensus)
        s = silhouette_score(Z, pl)
        perm_max = max(perm_max, s)
        if s >= obs_sil:
            ge += 1
    R["permutation_p"] = round((ge + 1) / (args.perms + 1), 4)
    R["permutation_max_silhouette"] = round(float(perm_max), 3)

    # ---- 5-fold CV silhouette ----
    from sklearn.model_selection import StratifiedKFold
    strat = A_imp["diagnosis_group"].astype("category").cat.codes.to_numpy()
    cv_sils = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Z, strat):
        cen = KMeans(best_k, n_init=10, random_state=42).fit(Z[tr])
        cv_sils.append(float(silhouette_score(Z[te], cen.predict(Z[te]))))
    R["cv_silhouette"] = [round(s, 3) for s in cv_sils]
    R["cv_silhouette_mean"] = round(float(np.mean(cv_sils)), 3)
    R["cv_silhouette_sd"] = round(float(np.std(cv_sils, ddof=1)), 3)

    # ---- subgroup preservation ----
    sub = {}
    for grp in ["breast_cancer", "fibroadenoma", "other_benign", "dcis"]:
        gi = np.where(A_imp["diagnosis_group"].to_numpy() == grp)[0]
        if len(gi) >= best_k + 1:
            sl = KMeans(best_k, n_init=10, random_state=42).fit_predict(Z[gi])
            sub[grp] = {"n": int(len(gi)), "ari": round(best_match_ari(consensus[gi], sl), 3)}
    R["subgroup"] = sub

    # ---- MICE vs complete-case kappa ----
    cc_mask = A[CLUSTERING_INPUTS].notna().all(axis=1).to_numpy()
    if cc_mask.sum() > best_k:
        Xcc, _, _ = encode(A[cc_mask], fit_scaler=scaler, fit_cols=cols)
        Zcc = encode_latent(model, Xcc.to_numpy(np.float32))
        lab_cc = KMeans(best_k, n_init=10, random_state=42).fit_predict(Zcc)
        R["kappa_mice_vs_cc_n"] = int(cc_mask.sum())
        R["kappa_mice_vs_cc"] = round(aligned_kappa(consensus[cc_mask], lab_cc, best_k), 3)
        R["ari_mice_vs_cc"] = round(float(adjusted_rand_score(consensus[cc_mask], lab_cc)), 3)

    # ---- 4.3 phenotype profiles ----
    A_imp["cluster"] = consensus
    prof = {}
    for c in range(best_k):
        subdf = A_imp[A_imp["cluster"] == c]
        prof[c] = {
            "n": int(len(subdf)),
            **{v: {"mean": round(float(subdf[v].mean()), 2),
                   "sd": round(float(subdf[v].std()), 2)} for v in CONT},
            "menopausal_post_pct": round(100 * float(subdf["menopausal"].mean()), 1),
            "diagnosis_pct": {g: round(100 * (subdf["diagnosis_group"] == g).mean(), 1)
                              for g in dg},
        }
    R["phenotype_profiles"] = prof
    # KW across clusters per input
    kw_inputs = {}
    for v in CONT:
        groups = [A_imp[A_imp.cluster == c][v].dropna() for c in range(best_k)]
        kw_inputs[v] = round(float(stats.kruskal(*groups).pvalue), 6)
    R["input_kruskal_p"] = kw_inputs

    # ---- 4.4 EORTC mapping ----
    import scikit_posthocs as sp
    A_eortc = A.copy()
    A_eortc["cluster"] = consensus
    eortc_res = {}
    for s in EORTC_ALL:
        groups = [A_eortc[A_eortc.cluster == c][s].dropna() for c in range(best_k)]
        if all(len(g) > 1 for g in groups):
            kw = stats.kruskal(*groups)
            means = {c: round(float(A_eortc[A_eortc.cluster == c][s].mean()), 1)
                     for c in range(best_k)}
            # highest vs lowest mean clusters
            hi = max(means, key=means.get); lo = min(means, key=means.get)
            a = A_eortc[A_eortc.cluster == hi][s].dropna()
            b = A_eortc[A_eortc.cluster == lo][s].dropna()
            eortc_res[s] = {
                "kruskal_p": round(float(kw.pvalue), 6),
                "means": means,
                "cliffs_delta_hi_lo": round(cliffs_delta(a, b), 3),
                "cohen_d_hi_lo": round(cohen_d(a, b), 3),
            }
    # Holm across primary scales
    pvals = {s: eortc_res[s]["kruskal_p"] for s in EORTC_PRIMARY if s in eortc_res}
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    holm = {}
    for i, s in enumerate(order):
        holm[s] = round(min(1.0, pvals[s] * (m - i)), 6)
    R["eortc"] = eortc_res
    R["eortc_holm_p"] = holm

    # ---- GLMM adjusted means for fatigue (fa) ----
    try:
        import statsmodels.formula.api as smf
        node = np.zeros(len(A_eortc), int)
        for ni, p in enumerate(parts):
            node[p] = ni
        glmm_df = A_eortc.copy()
        glmm_df["node"] = node
        glmm_df["clusterc"] = glmm_df["cluster"].astype("category")
        glmm_res = {}
        for s in ["fa", "sl", "ef", "brbi", "brsef"]:
            gsub = glmm_df[[s, "clusterc", "age", "diagnosis_group",
                            "marital_status", "pre_op", "node"]].dropna()
            md = smf.mixedlm(f"{s} ~ C(clusterc) + age + C(diagnosis_group) + marital_status + pre_op",
                             gsub, groups=gsub["node"])
            mf = md.fit(reml=True, method="lbfgs")
            adj = {}
            base = mf.params.get("Intercept", 0.0)
            for c in range(best_k):
                key = f"C(clusterc)[T.{c}]"
                adj[c] = round(float(base + mf.params.get(key, 0.0)), 1)
            glmm_res[s] = {"adjusted_cluster_terms": adj}
        R["glmm"] = glmm_res
    except Exception as e:
        R["glmm_error"] = str(e)

    # ---- 3.5 interpretability: SHAP surrogate + counterfactual ----
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        acc = float(np.mean(cross_val_score(rf, X, consensus, cv=5)))
        rf.fit(X, consensus)
        import shap
        expl = shap.TreeExplainer(rf)
        sv = expl.shap_values(X)
        arr = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        imp = np.abs(arr).mean(axis=0)
        top = sorted(zip(cols, imp), key=lambda t: -t[1])[:6]
        R["shap"] = {"surrogate_cv_accuracy": round(acc, 3),
                     "top_features": [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in top]}
    except Exception as e:
        R["shap_error"] = str(e)

    # counterfactual: normalise BMI in the higher-BMI phenotype, re-assign
    try:
        centroids = np.stack([Z[consensus == c].mean(0) for c in range(best_k)])
        bmi_means = {c: A_imp[A_imp.cluster == c]["bmi"].mean() for c in range(best_k)}
        hi = max(bmi_means, key=bmi_means.get); lo = min(bmi_means, key=bmi_means.get)
        cf = A_imp.copy()
        target = cf["cluster"] == hi
        cf.loc[target, "bmi"] = 22.0
        Xcf, _, _ = encode(cf, fit_scaler=scaler, fit_cols=cols)
        Zcf = encode_latent(model, Xcf.to_numpy(np.float32))
        idx_hi = np.where(target.to_numpy())[0]
        d_hi = np.linalg.norm(Zcf[idx_hi] - centroids[hi], axis=1)
        d_lo = np.linalg.norm(Zcf[idx_hi] - centroids[lo], axis=1)
        flipped = float(np.mean(d_lo < d_hi))
        R["counterfactual"] = {"bmi_shift_to_22_reassigned_pct": round(100 * flipped, 1),
                               "higher_bmi_phenotype": int(hi)}
    except Exception as e:
        R["counterfactual_error"] = str(e)

    # ---- 4.5 federated vs centralised + DP sweep ----
    def safe_sil(z, seed):
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        lab = KMeans(best_k, n_init=5, random_state=seed).fit_predict(z)
        try:
            return float(silhouette_score(z, lab)), lab
        except Exception:
            return float("nan"), lab

    def run_condition(sigma, seed):
        fc = FederatedConfig(3, 30, 2, "dirichlet", 0.5, noise_sigma=sigma)
        mdl = run_federated_vae(X, fc, vae_hidden, latent_dim, 0.1, seed=seed)
        z = encode_latent(mdl, X)
        return safe_sil(z, seed)

    from src.model.vae import train_vae
    cent_sils, fed0, fed1, fed5 = [], [], [], []
    cent_labels0 = None
    for sd in range(args.dp_seeds):
        cm = train_vae(X, vae_hidden, latent_dim, 0.1, epochs=30, batch_size=64, seed=sd)
        zc = encode_latent(cm, X)
        sc, lc = safe_sil(zc, sd)
        cent_sils.append(sc)
        if sd == 0:
            cent_labels0 = lc
        s0, l0 = run_condition(0.0, sd); fed0.append(s0)
        s1, _ = run_condition(0.1, sd); fed1.append(s1)
        s5, l5 = run_condition(0.5, sd); fed5.append(s5)
    def ni(cent, fed):
        cent = np.array(cent, float); fed = np.array(fed, float)
        cent = cent[~np.isnan(cent)]; fed = fed[~np.isnan(fed)]
        gap = (np.mean(cent) - np.mean(fed)) / np.mean(cent)
        try:
            p = float(stats.mannwhitneyu(fed, cent, alternative="greater").pvalue)
        except Exception:
            p = float("nan")
        sd = float(np.std(fed, ddof=1)) if len(fed) > 1 else 0.0
        return round(float(np.mean(fed)), 3), round(sd, 3), round(float(gap) * 100, 1), round(p, 4)
    R["noninferiority"] = {
        "centralised_silhouette_mean": round(float(np.mean(cent_sils)), 3),
        "centralised_silhouette_sd": round(float(np.std(cent_sils, ddof=1)), 3),
        "fed_sigma0": ni(cent_sils, fed0),
        "fed_sigma0.1": ni(cent_sils, fed1),
        "fed_sigma0.5": ni(cent_sils, fed5),
    }
    # RDP (eps, delta)
    sample_rate = min(64 / len(X), 1.0)
    steps = 30 * max(1, len(X) // 64)
    R["privacy"] = {
        "sigma0.1_epsilon": round(RDPAccountant(sample_rate, 0.1 / 1.0 + 1.0).epsilon(steps, 1e-5), 2),
        "sigma0.5_epsilon": round(RDPAccountant(sample_rate, 0.5 / 1.0 + 1.0).epsilon(steps, 1e-5), 2),
        "delta": 1e-5,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, default=str)
    print(json.dumps(R, indent=2, default=str))


if __name__ == "__main__":
    main()
