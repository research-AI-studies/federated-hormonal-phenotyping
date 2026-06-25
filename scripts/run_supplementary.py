"""Supplementary analyses computed under the SAME configuration as the main text.

This script reproduces the primary pipeline exactly (MICE imputation, z-score
one-hot encoding, VAE latent [256,128]->8, k-means at k=4, seed 42) so that every
supplementary quantity is internally consistent with the main manuscript and its
figures/tables. It then computes the analyses that the main results JSON does not
already hold:

  * 5-fold cross-validated silhouette of the k-means partition
  * per-diagnosis subgroup reproduction (ARI vs the global partition)
  * MICE vs complete-case agreement (Cohen's kappa, ARI)
  * SHAP surrogate global feature attribution + surrogate accuracy
  * counterfactual probing (BMI shift) reassignment rate
  * GLMM (mixed-effects) cluster contrasts on EORTC scales, diagnosis as group
  * SOM quantisation error on the latent
  * latent-size grid search across {4, 8, 16, 32}

Usage:
    python scripts/run_supplementary.py --data <path-to-cohort.xlsx>
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import encode_features, mice_impute  # noqa: E402
from src.model import SOM, encode_latent, train_vae  # noqa: E402
from src.study.data_adapter import (  # noqa: E402
    CLUSTER_INPUTS, EORTC_SCALES, load_study,
)
from src.study.consensus import consensus_partition  # noqa: E402

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score, cohen_kappa_score, davies_bouldin_score,
    calinski_harabasz_score, silhouette_score,
)
from sklearn.model_selection import KFold, cross_val_score  # noqa: E402

warnings.filterwarnings("ignore")

CONT10 = ["age", "bmi", "menstruation_firsttime_age", "pregnancy_number",
          "bust", "cupsize", "alcohol", "smokingstatus", "menopause_yn",
          "contraceptive_kind"]
PRETTY = {"age": "Age", "bmi": "Body mass index",
          "menstruation_firsttime_age": "Age at menarche",
          "pregnancy_number": "Pregnancies", "bust": "Bust size",
          "cupsize": "Cup size", "alcohol": "Alcohol", "smokingstatus": "Smoking",
          "menopause_yn": "Menopausal status", "contraceptive_kind": "Contraceptive"}


def build_pipeline(path: str, seed: int, k: int):
    """Reproduce the primary embedding and partition exactly as the main text."""
    sd = load_study(path)
    imp = mice_impute(sd.inputs, max_iter=10, seed=seed)
    enc = encode_features(imp, one_hot=True, scale="zscore")
    X = enc.matrix
    feat = enc.feature_names
    model = train_vae(X, hidden_dims=[256, 128], latent_dim=8, dropout=0.1,
                      lr=1e-3, epochs=80, batch_size=64, beta=1.0, seed=seed)
    Z = encode_latent(model, X)
    _, algos, _ = consensus_partition(Z, k, seed)
    labels = algos["kmeans"]
    return sd, imp, X, feat, model, Z, labels


def cv_silhouette(Z, k, seed, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sils = []
    for tr, te in kf.split(Z):
        km = KMeans(k, n_init=10, random_state=seed).fit(Z[tr])
        pred = km.predict(Z[te])
        if len(np.unique(pred)) > 1:
            sils.append(float(silhouette_score(Z[te], pred)))
    return sils


def subgroup_ari(Z, labels, diag, k, seed):
    out = {}
    for d in [1, 2, 3, 4]:
        m = diag == d
        n = int(m.sum())
        if n < k + 1:
            out[int(d)] = {"n": n, "ari": None}
            continue
        sub = KMeans(k, n_init=10, random_state=seed).fit_predict(Z[m])
        out[int(d)] = {"n": n, "ari": round(float(adjusted_rand_score(labels[m], sub)), 3)}
    return out


def mice_vs_cc(sd, Z, labels, k, seed):
    cc = sd.inputs[CLUSTER_INPUTS].notna().all(axis=1).to_numpy()
    n = int(cc.sum())
    sub = KMeans(k, n_init=10, random_state=seed).fit_predict(Z[cc])
    g = labels[cc]
    # Align label ids via best-overlap mapping before kappa (kappa is label-sensitive).
    mapping = {}
    for c in np.unique(sub):
        overlap = [(int((g[sub == c] == gc).sum()), int(gc)) for gc in np.unique(g)]
        mapping[c] = max(overlap)[1]
    sub_mapped = np.array([mapping[c] for c in sub])
    return {
        "n": n,
        "cohen_kappa": round(float(cohen_kappa_score(g, sub_mapped)), 3),
        "ari": round(float(adjusted_rand_score(g, sub_mapped)), 3),
    }


def shap_surrogate(X, feat, labels, seed):
    import shap
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, random_state=seed, eval_metric="mlogloss",
        tree_method="hist", n_jobs=4,
    )
    acc = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
    clf.fit(X, labels)
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(X)
    arr = np.array(sv)
    # Aggregate |SHAP| across classes and samples -> per-feature global importance.
    if arr.ndim == 3:          # (classes, samples, features) or (samples, features, classes)
        axes = [a for a in range(arr.ndim) if arr.shape[a] == len(feat)]
        fax = axes[0]
        order = list(range(arr.ndim))
        order.remove(fax)
        mean_abs = np.abs(arr).mean(axis=tuple(order))
    else:
        mean_abs = np.abs(arr).mean(axis=0)
    imp = sorted(
        [{"feature": PRETTY.get(feat[i], feat[i]),
          "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in range(len(feat))],
        key=lambda d: d["mean_abs_shap"], reverse=True,
    )
    return {
        "surrogate_cv_accuracy": round(float(acc.mean()), 3),
        "surrogate_cv_accuracy_sd": round(float(acc.std(ddof=1)), 3),
        "feature_importance": imp,
    }


def counterfactual(imp, X, feat, model, Z, labels, k, seed, target_bmi=22.0):
    bmi_raw = imp["bmi"].to_numpy()
    mean_bmi, sd_bmi = float(bmi_raw.mean()), float(bmi_raw.std())
    bmi_col = feat.index("bmi")
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    base = km.predict(Z)
    cluster_bmi = {c: float(imp["bmi"][base == c].mean()) for c in np.unique(base)}
    hi = max(cluster_bmi, key=cluster_bmi.get)
    members = base == hi
    Xmod = X.copy()
    Xmod[members, bmi_col] = (target_bmi - mean_bmi) / sd_bmi
    Zmod = encode_latent(model, Xmod)
    new = km.predict(Zmod[members])
    reassigned = float((new != hi).mean() * 100)
    return {
        "higher_bmi_cluster": int(hi),
        "higher_bmi_cluster_mean_bmi": round(cluster_bmi[hi], 1),
        "target_bmi": target_bmi,
        "n_members": int(members.sum()),
        "reassigned_pct": round(reassigned, 1),
    }


def glmm_contrasts(sd, labels, seed):
    import statsmodels.formula.api as smf

    out = {}
    diag = sd.diagnosis.to_numpy()
    age = sd.analytic["age"].to_numpy()
    ref = int(pd.Series(labels).value_counts().idxmax())   # largest cluster = reference
    for s in ["fa", "sl", "ef", "brbi", "brsef"]:
        y = sd.outcomes[s].reset_index(drop=True).to_numpy()
        df = pd.DataFrame({"y": y, "cluster": labels.astype(int),
                           "age": age, "diagnosis": diag}).dropna()
        df["cluster"] = pd.Categorical(df["cluster"],
                                       categories=[ref] + [c for c in sorted(df["cluster"].unique()) if c != ref])
        try:
            md = smf.mixedlm("y ~ C(cluster) + age", df, groups=df["diagnosis"])
            mf = md.fit(reml=False, method="lbfgs", disp=False)
            terms = {}
            for name in mf.params.index:
                if name.startswith("C(cluster)"):
                    lab = name.split("T.")[-1].rstrip("]")
                    terms[lab] = {"coef": round(float(mf.params[name]), 2),
                                  "p": round(float(mf.pvalues[name]), 4)}
            out[s] = {"reference_cluster": ref, "contrasts": terms,
                      "group_var": round(float(mf.cov_re.iloc[0, 0]), 3)}
        except Exception as e:                     # noqa: BLE001
            out[s] = {"error": str(e)[:120]}
    return out


def latent_grid(X, k, seed, dims=(4, 8, 16, 32)):
    import torch
    out = {}
    for d in dims:
        model = train_vae(X, [256, 128], int(d), 0.1, 1e-3, 80, 64, 1.0, seed)
        Z = encode_latent(model, X)
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            recon, _, _ = model(t)
            mse = float(torch.nn.functional.mse_loss(recon, t).item())
        out[int(d)] = {
            "silhouette": round(float(silhouette_score(Z, lab)), 3),
            "calinski_harabasz": round(float(calinski_harabasz_score(Z, lab)), 1),
            "davies_bouldin": round(float(davies_bouldin_score(Z, lab)), 3),
            "reconstruction_mse": round(mse, 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="outputs/supplementary_results.json")
    args = ap.parse_args()
    seed, k = args.seed, args.k
    np.random.seed(seed)

    print("Reproducing primary pipeline (this trains the VAE)...")
    sd, imp, X, feat, model, Z, labels = build_pipeline(args.data, seed, k)
    diag = sd.diagnosis.to_numpy()

    R = {
        "config": {"seed": seed, "k": k, "input_dim": int(X.shape[1]),
                   "latent_dim": 8, "hidden_dims": [256, 128], "n_analytic": int(len(Z))},
        "primary_silhouette": round(float(silhouette_score(Z, labels)), 3),
        "cluster_sizes": {int(c): int((labels == c).sum()) for c in np.unique(labels)},
    }

    print("  5-fold CV silhouette...")
    sils = cv_silhouette(Z, k, seed)
    R["cv_silhouette"] = [round(s, 3) for s in sils]
    R["cv_silhouette_mean"] = round(float(np.mean(sils)), 3)
    R["cv_silhouette_sd"] = round(float(np.std(sils, ddof=1)), 3)

    print("  Subgroup reproduction ARI...")
    R["subgroup_ari"] = subgroup_ari(Z, labels, diag, k, seed)

    print("  MICE vs complete-case agreement...")
    R["mice_vs_complete_case"] = mice_vs_cc(sd, Z, labels, k, seed)

    print("  SHAP surrogate attribution...")
    R["shap"] = shap_surrogate(X, feat, labels, seed)

    print("  Counterfactual probing...")
    R["counterfactual"] = counterfactual(imp, X, feat, model, Z, labels, k, seed)

    print("  GLMM contrasts...")
    R["glmm"] = glmm_contrasts(sd, labels, seed)

    print("  SOM quantisation error...")
    som = SOM(grid=(8, 8), input_dim=Z.shape[1], seed=seed).train(Z)
    R["som_quantisation_error"] = round(float(som.quantisation_error(Z)), 4)

    print("  Latent-size grid search {4,8,16,32}...")
    R["latent_grid"] = latent_grid(X, k, seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=2))
    print(json.dumps(R, indent=2))
    print(f"\nSupplementary results -> {out}")


if __name__ == "__main__":
    main()
