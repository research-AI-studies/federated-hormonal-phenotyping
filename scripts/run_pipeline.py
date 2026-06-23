"""End-to-end pipeline runner.

Loads a cohort, preprocesses it, trains the (federated) VAE, derives the latent
embedding, fits the (federated) SOM, validates the clustering, computes
phenotype attributions, and writes figures/tables to the git-ignored
``outputs/`` tree.

Usage:
    python scripts/run_pipeline.py --config config/default.yaml \
        --data data/synthetic/cohort.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (  # noqa: E402
    drop_high_missing,
    encode_features,
    load_cohort,
    mice_impute,
    winsorize,
)
from src.model import encode_latent, train_vae  # noqa: E402
from src.model.federated import (  # noqa: E402
    FederatedConfig,
    partition,
    run_federated_som,
    run_federated_vae,
)
from src.model.som import SOM  # noqa: E402
from src.model.privacy import RDPAccountant, calibrate_noise  # noqa: E402
from src.validation import consensus_stability, internal_metrics, sweep_k  # noqa: E402
from src.interpretability import phenotype_attributions  # noqa: E402
from src.figures import (  # noqa: E402
    plot_attribution_heatmap,
    plot_metric_sweep,
    plot_som_hitmap,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the phenotyping pipeline.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--data", default=None, help="Override config data.path")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    np.random.seed(seed)

    data_path = args.data or cfg["data"]["path"]
    out_dir = Path(cfg["outputs"]["dir"])
    fig_dir = Path(cfg["outputs"]["figures_dir"])
    tab_dir = Path(cfg["outputs"]["tables_dir"])
    for d in (out_dir, fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Preprocessing -------------------------------------------------------
    df = load_cohort(data_path, cfg["data"].get("id_col"))
    target_hint = cfg["data"].get("target_hint")
    hint = df[target_hint] if target_hint in df.columns else None
    feat_df = df.drop(columns=[c for c in [target_hint] if c in df.columns])

    feat_df = drop_high_missing(feat_df, cfg["preprocessing"]["missing_threshold"])
    if cfg["preprocessing"]["winsorize"]["enabled"]:
        feat_df = winsorize(feat_df, limits=tuple(cfg["preprocessing"]["winsorize"]["limits"]))
    feat_df = mice_impute(feat_df, cfg["preprocessing"]["impute"]["max_iter"], seed)
    encoded = encode_features(
        feat_df,
        one_hot=cfg["preprocessing"]["one_hot"],
        scale=cfg["preprocessing"]["scale"],
    )
    x = encoded.matrix

    # --- Privacy budget ------------------------------------------------------
    privacy_report = {}
    if cfg["privacy"]["enabled"]:
        sample_rate = min(cfg["model"]["vae"]["batch_size"] / len(x), 1.0)
        steps = cfg["model"]["vae"]["epochs"] * max(1, len(x) // cfg["model"]["vae"]["batch_size"])
        nm = cfg["privacy"]["noise_multiplier"] or calibrate_noise(
            sample_rate, steps, cfg["privacy"]["target_epsilon"], cfg["privacy"]["target_delta"]
        )
        eps = RDPAccountant(sample_rate, nm).epsilon(steps, cfg["privacy"]["target_delta"])
        privacy_report = {"noise_multiplier": nm, "epsilon": eps,
                          "delta": cfg["privacy"]["target_delta"]}

    # --- Representation learning --------------------------------------------
    vcfg = cfg["model"]["vae"]
    if cfg["federated"]["enabled"]:
        fed = FederatedConfig(
            num_clients=cfg["federated"]["num_clients"],
            rounds=cfg["federated"]["rounds"],
            local_epochs=cfg["federated"]["local_epochs"],
            partition=cfg["federated"]["partition"],
            dirichlet_alpha=cfg["federated"]["dirichlet_alpha"],
            lr=vcfg["lr"],
            beta=vcfg["beta"],
        )
        model = run_federated_vae(x, fed, vcfg["hidden_dims"], vcfg["latent_dim"], vcfg["dropout"], seed)
        parts = [p for p in partition(x, fed.num_clients, fed.partition, fed.dirichlet_alpha, seed) if len(p) > 0]
    else:
        model = train_vae(
            x, vcfg["hidden_dims"], vcfg["latent_dim"], vcfg["dropout"],
            vcfg["lr"], vcfg["epochs"], vcfg["batch_size"], vcfg["beta"], seed,
        )
        parts = [np.arange(len(x))]

    latent = encode_latent(model, x)

    # --- SOM clustering ------------------------------------------------------
    scfg = cfg["model"]["som"]
    if cfg["federated"]["enabled"]:
        som = run_federated_som(
            latent, parts, tuple(scfg["grid"]), scfg["sigma"],
            scfg["learning_rate"], scfg["iterations"], seed,
        )
    else:
        som = SOM(tuple(scfg["grid"]), latent.shape[1], scfg["sigma"],
                  scfg["learning_rate"], scfg["iterations"], seed=seed).train(latent)
    nodes = som.predict(latent)

    # --- Validation ----------------------------------------------------------
    sweep = sweep_k(latent, cfg["validation"]["k_range"], seed)
    best_k = min(sweep, key=lambda k: sweep[k]["davies_bouldin"])
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit_predict(latent)
    internal = internal_metrics(latent, labels)
    stability = consensus_stability(
        latent, best_k, cfg["validation"]["consensus_resamples"],
        cfg["validation"]["subsample_fraction"], seed,
    )

    # --- Interpretability ----------------------------------------------------
    attr = phenotype_attributions(
        encoded.frame, labels,
        cfg["interpretability"]["shap_background"],
        cfg["interpretability"]["shap_nsamples"], seed,
    )

    # --- Outputs (git-ignored) ----------------------------------------------
    plot_som_hitmap(nodes, tuple(scfg["grid"]), fig_dir / "som_hitmap.png")
    plot_metric_sweep(sweep, fig_dir / "cluster_sweep.png")
    plot_attribution_heatmap(attr, fig_dir / "attribution_heatmap.png")
    attr.to_csv(tab_dir / "phenotype_attributions.csv")

    summary = {
        "n_samples": int(len(x)),
        "n_features": int(x.shape[1]),
        "best_k": int(best_k),
        "internal_metrics": internal,
        "stability": stability,
        "privacy": privacy_report,
        "target_hint_present": hint is not None,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
