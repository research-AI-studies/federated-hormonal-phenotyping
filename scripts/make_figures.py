"""Generate all manuscript figures and data tables from the real cohort.

Outputs PNG figures and CSV tables to a target directory (default git-ignored
outputs/figures). No data or generated artefacts are committed to the repo.

Usage:
    python scripts/make_figures.py --data <cohort.xlsx> --out "<assets dir>"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import encode_features, mice_impute  # noqa: E402
from src.model import encode_latent, train_vae  # noqa: E402
from src.study.data_adapter import (  # noqa: E402
    CLUSTER_INPUTS, EORTC_LABEL, EORTC_SCALES, load_study,
)
from sklearn.cluster import KMeans  # noqa: E402

# Manuscript cluster label -> data cluster index (k-means++ at seed 42).
MAP = {1: 1, 2: 2, 3: 0, 4: 3}
INV = {v: k for k, v in MAP.items()}
NAME = {
    1: "C1 young lean\npremenopausal",
    2: "C2 young\ncontraception-active",
    3: "C3 older\nhigh-adiposity",
    4: "C4 postmenopausal\ncancer-predominant",
}
PALETTE = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52", 4: "#8172B3"}
INPUT_LABEL = {
    "age": "Age", "bmi": "BMI", "menstruation_firsttime_age": "Menarche",
    "pregnancy_number": "Parity", "bust": "Bust", "cupsize": "Cup",
    "alcohol": "Alcohol", "smokingstatus": "Smoking",
    "menopause_yn": "Menopause", "contraceptive_kind": "Contraception",
}


def savefig(fig, out, name):
    fig.savefig(out / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def fig1_consort(out, n_full, n_excl, n_analytic, dx):
    fig, ax = plt.subplots(figsize=(6.2, 7.2))
    ax.axis("off")
    boxes = [
        (0.5, 0.93, f"Released registry cohort\n(n = {n_full})"),
        (0.5, 0.70, f"Excluded for missing age or\nbody mass index (n = {n_excl})"),
        (0.5, 0.47, f"Analytic cohort\n(n = {n_analytic})"),
        (0.5, 0.16, "Breast cancer {a}    DCIS {b}\nFibroadenoma {c}    Other benign {d}".format(
            a=dx[1], b=dx[2], c=dx[3], d=dx[4])),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch((x - 0.30, y - 0.055), 0.60, 0.11,
                     boxstyle="round,pad=0.02", fc="#EEF2F8", ec="#33415C", lw=1.4,
                     transform=ax.transAxes))
        ax.text(x, y, t, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    for y0, y1 in [(0.875, 0.755), (0.645, 0.525), (0.415, 0.215)]:
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), arrowstyle="-|>",
                     mutation_scale=16, lw=1.4, color="#33415C", transform=ax.transAxes))
    savefig(fig, out, "Figure_1_CONSORT_flow.png")


def fig2_architecture(out):
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.axis("off")
    blocks = [
        (0.08, "Ten hormonal &\nlifestyle inputs", "#EEF2F8"),
        (0.30, "Variational\nautoencoder\n(latent dim 8)", "#DCE6F5"),
        (0.52, "Self-organising\nmap + k-means\n(4 phenotypes)", "#D7ECDD"),
        (0.74, "EORTC symptom\nmapping", "#F6E2E2"),
    ]
    for x, t, c in blocks:
        ax.add_patch(FancyBboxPatch((x - 0.085, 0.40), 0.17, 0.34,
                     boxstyle="round,pad=0.02", fc=c, ec="#33415C", lw=1.4,
                     transform=ax.transAxes))
        ax.text(x, 0.57, t, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    for x0, x1 in [(0.165, 0.215), (0.385, 0.435), (0.605, 0.655)]:
        ax.add_patch(FancyArrowPatch((x0, 0.57), (x1, 0.57), arrowstyle="-|>",
                     mutation_scale=16, lw=1.4, color="#33415C", transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((0.18, 0.08), 0.46, 0.16, boxstyle="round,pad=0.02",
                 fc="none", ec="#999999", lw=1.2, ls="--", transform=ax.transAxes))
    ax.text(0.41, 0.16, "Federated training across 3 nodes (FedAvg) + Renyi differential privacy",
            ha="center", va="center", fontsize=9, style="italic", transform=ax.transAxes)
    savefig(fig, out, "Figure_2_architecture.png")


def fig3_latent(out, Z, labels):
    try:
        import umap
        emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(Z)
        xlab, ylab = "UMAP-1", "UMAP-2"
    except Exception:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=42).fit_transform(Z)
        xlab, ylab = "PC-1", "PC-2"
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for didx in [1, 0, 3, 2]:
        m = labels == didx
        c = INV[didx]
        ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.55, color=PALETTE[c],
                   label=NAME[c].replace("\n", " "))
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.legend(fontsize=8, markerscale=1.6, loc="best", frameon=True)
    savefig(fig, out, "Figure_3_latent_projection.png")


def fig4_radar(out, imp, labels):
    feats = CLUSTER_INPUTS
    z = (imp[feats] - imp[feats].mean()) / imp[feats].std()
    ang = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw=dict(polar=True))
    for c in [1, 2, 3, 4]:
        didx = MAP[c]
        vals = z[labels == didx].mean().tolist()
        vals += vals[:1]
        ax.plot(ang, vals, color=PALETTE[c], lw=2, label=NAME[c].replace("\n", " "))
        ax.fill(ang, vals, color=PALETTE[c], alpha=0.08)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([INPUT_LABEL[f] for f in feats], fontsize=9)
    ax.set_yticklabels([])
    ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.28, 1.10))
    savefig(fig, out, "Figure_4_radar_profiles.png")


def fig5_eortc(out, outcomes, labels):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for ax, s in zip(axes.ravel(), EORTC_SCALES):
        data, cols = [], []
        for c in [1, 2, 3, 4]:
            v = outcomes[s].values[labels == MAP[c]]
            data.append(v[~np.isnan(v)]); cols.append(PALETTE[c])
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col); patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title(EORTC_LABEL.get(s, s), fontsize=10)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["C1", "C2", "C3", "C4"], fontsize=8)
        ax.set_ylim(0, 100)
    fig.tight_layout()
    savefig(fig, out, "Figure_5_EORTC_panel.png")


def fig6_privacy(out, R):
    cen = R["centralised_silhouette_mean"]
    sg = [0.0, 0.1, 0.5]
    fed = [R["federated"][str(s)]["mean"] if str(s) in R["federated"]
           else R["federated"][s]["mean"] for s in sg]
    sd = [R["federated"][str(s)]["sd"] if str(s) in R["federated"]
          else R["federated"][s]["sd"] for s in sg]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.errorbar(sg, fed, yerr=sd, marker="o", lw=2, capsize=4, color="#C44E52",
                label="Federated")
    ax.axhline(cen, ls="--", color="#4C72B0", label=f"Centralised baseline ({cen:.2f})")
    for s, f in zip(sg, fed):
        ax.annotate(f"{f:.3f}", (s, f), textcoords="offset points", xytext=(6, 8), fontsize=9)
    ax.set_xlabel("Differential-privacy noise \u03c3")
    ax.set_ylabel("Silhouette coefficient")
    ax.set_xticks(sg)
    ax.legend(fontsize=9)
    savefig(fig, out, "Figure_6_privacy_utility.png")


def figS1_missing(out, miss):
    items = sorted(miss.items(), key=lambda kv: kv[1])
    labels = [INPUT_LABEL.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(labels, vals, color="#4C72B0", alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Full-cohort missingness (%)")
    fig.tight_layout()
    savefig(fig, out, "Figure_S1_missingness.png")


def graphical_abstract(out):
    fig2_architecture(out)  # reuse layout, save under GA name
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.axis("off")
    ax.text(0.5, 0.9, "Federated hormonal phenotyping of mammary disease",
            ha="center", fontsize=12, weight="bold", transform=ax.transAxes)
    steps = ["Reproductive &\nlifestyle inputs", "Federated VAE\n+ SOM", "Four hormonal\nphenotypes",
             "EORTC symptom\nburden", "Privacy-preserving\ndeployment"]
    for i, t in enumerate(steps):
        x = 0.10 + i * 0.20
        ax.add_patch(FancyBboxPatch((x - 0.085, 0.30), 0.17, 0.34, boxstyle="round,pad=0.02",
                     fc="#EEF2F8", ec="#33415C", lw=1.3, transform=ax.transAxes))
        ax.text(x, 0.47, t, ha="center", va="center", fontsize=9, transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.087, 0.47), (x + 0.113, 0.47),
                         arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#33415C",
                         transform=ax.transAxes))
    savefig(fig, out, "Graphical_Abstract.png")


def write_tables(out, R, imp, labels, miss):
    # Table 3: validity metrics across k.
    rows = []
    for k, m in R["k_sweep"].items():
        rows.append({"k": int(k), "Silhouette": round(m["silhouette"], 3),
                     "Calinski-Harabasz": round(m["calinski_harabasz"], 1),
                     "Davies-Bouldin": round(m["davies_bouldin"], 3),
                     "Gap": round(m["gap"], 3),
                     "Selected": "Yes" if int(k) == R["k_selected"] else ""})
    pd.DataFrame(rows).to_csv(out / "Table_3_validity_metrics.csv", index=False)

    # Table 4: cluster profiles (10 inputs x 4 clusters).
    prof = R["cluster_profiles"]
    cont = ["age", "bmi", "menstruation_firsttime_age", "pregnancy_number",
            "bust", "cupsize", "alcohol", "smokingstatus"]
    rows = []
    for c in [1, 2, 3, 4]:
        d = prof[str(MAP[c])]
        row = {"Cluster": NAME[c].replace("\n", " "), "n": R["cluster_sizes"][str(MAP[c])]}
        for f in cont:
            row[INPUT_LABEL[f]] = f"{d[f][0]} ({d[f][1]})"
        row["Premenopausal %"] = d["pct_premenopausal"]
        row["Nulliparous %"] = d["pct_nulliparous"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "Table_4_cluster_profiles.csv", index=False)

    # Table 5: EORTC mapping.
    rows = []
    for s in EORTC_SCALES:
        e = R["eortc_omnibus"][s]
        row = {"Domain": EORTC_LABEL.get(s, s), "H": round(e["H"], 1),
               "p (Holm)": f"{e['p_holm']:.1e}"}
        for c in [1, 2, 3, 4]:
            row[f"C{c} mean"] = round(e["means"][str(MAP[c])], 1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "Table_5_EORTC_mapping.csv", index=False)

    # Table 2: variable inventory.
    inv = [
        ("age", "Continuous", "years", "Input", "z-score"),
        ("bmi", "Continuous", "kg/m2", "Input", "winsorise + z-score"),
        ("menstruation_firsttime_age", "Continuous", "years", "Input", "winsorise + z-score"),
        ("pregnancy_number", "Ordinal", "count", "Input", "integer"),
        ("bust", "Ordinal", "code", "Input", "integer"),
        ("cupsize", "Ordinal", "code", "Input", "integer"),
        ("alcohol", "Ordinal", "0-3", "Input", "integer"),
        ("smokingstatus", "Ordinal", "0-2", "Input", "integer"),
        ("menopause_yn", "Nominal", "0/1", "Input", "category code"),
        ("contraceptive_kind", "Nominal", "5 bins", "Input", "recoded category"),
    ]
    rows = []
    for v, t, rng, role, enc in inv:
        rows.append({"Variable": INPUT_LABEL.get(v, v), "Type": t, "Range/units": rng,
                     "Missing % (full)": miss.get(v, 0.0), "Encoding": enc, "Role": role})
    for s in EORTC_SCALES:
        rows.append({"Variable": EORTC_LABEL.get(s, s), "Type": "Continuous",
                     "Range/units": "0-100", "Missing % (full)": "", "Encoding": "raw score",
                     "Role": "Outcome"})
    rows.append({"Variable": "Diagnosis", "Type": "Nominal", "Range/units": "4 classes",
                 "Missing % (full)": "", "Encoding": "label", "Role": "Stratifier"})
    pd.DataFrame(rows).to_csv(out / "Table_2_variable_inventory.csv", index=False)
    print("  wrote Tables 2-5 (CSV)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default="outputs/study_results.json")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = json.loads(Path(args.results).read_text())

    sd = load_study(args.data)
    imp = mice_impute(sd.inputs, seed=args.seed)
    X = encode_features(imp, scale="zscore").matrix
    Z = encode_latent(train_vae(X, [256, 128], 8, 0.1, 1e-3, 80, 64, 1.0, args.seed), X)
    labels = KMeans(4, n_init=10, random_state=args.seed).fit_predict(Z)
    outcomes = sd.outcomes.reset_index(drop=True)

    dx = {int(k): v for k, v in R["diagnosis_counts"].items()}
    n_full, n_an = R["n_full"], R["n_analytic"]
    miss = R["missingness"]

    print("Figures:")
    fig1_consort(out, n_full, n_full - n_an, n_an, dx)
    fig2_architecture(out)
    fig3_latent(out, Z, labels)
    fig4_radar(out, imp, labels)
    fig5_eortc(out, outcomes, labels)
    fig6_privacy(out, R)
    figS1_missing(out, miss)
    graphical_abstract(out)
    print("Tables:")
    write_tables(out, R, imp, labels, miss)
    print(f"\nAll assets -> {out.resolve()}")


if __name__ == "__main__":
    main()
