# Federated Hormonal Phenotyping of Breast Cancer Cohorts

A privacy-preserving analytical pipeline for unsupervised phenotyping of
patient-reported and clinical baseline data, combining a **federated
variational autoencoder (VAE)** for representation learning with a
**federated self-organising map (F-SOM)** for topology-preserving cluster
discovery, under **differential-privacy** guarantees.

---

## Pipeline overview

```
raw cohort (local, private)
        │
        ▼
[ preprocessing ]  load → clean → impute (MICE) → encode (one-hot + scale)
        │
        ▼
[ model ]  federated VAE  →  latent embedding  →  federated SOM
        │                                              │
        │                          differential privacy (RDP accountant)
        ▼                                              ▼
[ validation ]  silhouette / CH / DB / gap / ARI / consensus stability
        │
        ▼
[ interpretability ]  SHAP attribution per phenotype
        │
        ▼
[ figures ]  topology maps, stability curves, attribution plots (local only)
```

## Repository layout

| Path | Purpose |
|------|---------|
| `config/` | YAML run configuration (paths, hyper-parameters, privacy budget) |
| `data/` | Schema crosswalk + example-data generator (no real data tracked) |
| `src/preprocessing/` | Loading, cleaning, imputation, encoding |
| `src/model/` | VAE, SOM, federated orchestration, DP accountant |
| `src/validation/` | Internal/external cluster-quality and stability metrics |
| `src/interpretability/` | SHAP-based phenotype attribution |
| `src/figures/` | Plotting utilities (write to git-ignored `outputs/`) |
| `scripts/` | End-to-end pipeline runner and Table 1 builder |
| `tests/` | Smoke tests on example data |
| `notebooks/` | Exploratory notebooks (stripped of outputs before commit) |

## Quick start

```bash
# 1. Create the environment
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate an example cohort (writes to git-ignored data/example/)
python data/example/generate_example.py --n 600 --out data/example/cohort.csv

# 3. Run the full pipeline on the example cohort
python scripts/run_pipeline.py --config config/default.yaml --data data/example/cohort.csv
```

To run on a private cohort, point `--data` at a local file outside the
repository tree (or set `data.path` in the config). Such files are excluded by
`.gitignore` and must never be committed.

## Reproducibility

- Pinned dependencies: [`requirements.txt`](requirements.txt) / [`environment.yml`](environment.yml).
- Containerised runtime: [`Dockerfile`](Dockerfile).
- All stochastic steps accept a `--seed`; defaults live in `config/default.yaml`.

## Licence

Released under the MIT Licence. See [`LICENSE`](LICENSE).
