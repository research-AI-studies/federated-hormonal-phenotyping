"""Generate a synthetic surrogate cohort.

The synthetic data reproduces the schema in ``data/codebook_crosswalk.csv`` and
plausible marginal distributions so the pipeline can be exercised end-to-end
without any real patient records. Output is written to a git-ignored path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(47.0, 11.5, n), 18, 99).round(1)
    bmi = np.clip(rng.normal(25.8, 4.9, n), 12, 60).round(1)
    age_menarche = np.clip(rng.normal(13.0, 1.4, n), 8, 20).round(0)
    parity = rng.poisson(1.6, n).clip(0, 15)
    nulliparous = (parity == 0).astype(int)

    menopausal_status = rng.choice(
        ["premenopausal", "perimenopausal", "postmenopausal"],
        size=n, p=[0.55, 0.12, 0.33],
    )
    education = rng.choice(
        ["primary", "secondary", "tertiary"], size=n, p=[0.18, 0.49, 0.33]
    )
    marital_status = rng.choice(
        ["single", "partnered", "married", "divorced", "widowed"],
        size=n, p=[0.18, 0.14, 0.52, 0.11, 0.05],
    )
    contraceptive_use = rng.choice(
        ["never", "former", "current"], size=n, p=[0.34, 0.46, 0.20]
    )
    alcohol = rng.choice(
        ["never", "occasional", "regular"], size=n, p=[0.41, 0.45, 0.14]
    )
    smoking = rng.choice(
        ["never", "former", "current"], size=n, p=[0.58, 0.24, 0.18]
    )
    bust_size = np.clip(rng.normal(92.0, 9.5, n), 60, 150).round(0)
    cup_size = rng.choice(
        list("ABCDEF"), size=n, p=[0.10, 0.27, 0.30, 0.20, 0.09, 0.04]
    )
    diagnosis = rng.choice(
        ["benign", "DCIS", "invasive"], size=n, p=[0.41, 0.10, 0.49]
    )

    df = pd.DataFrame(
        {
            "patient_id": [f"S{idx:05d}" for idx in range(n)],
            "age": age,
            "bmi": bmi,
            "age_menarche": age_menarche,
            "parity": parity,
            "nulliparous": nulliparous,
            "menopausal_status": menopausal_status,
            "education": education,
            "marital_status": marital_status,
            "contraceptive_use": contraceptive_use,
            "alcohol": alcohol,
            "smoking": smoking,
            "bust_size": bust_size,
            "cup_size": cup_size,
            "diagnosis": diagnosis,
        }
    )

    # Inject realistic missingness (MICE is exercised downstream).
    for col, frac in {"bmi": 0.06, "age_menarche": 0.09, "bust_size": 0.12}.items():
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic cohort.")
    parser.add_argument("--n", type=int, default=600, help="Number of records.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out", type=str, default="data/synthetic/cohort.csv",
        help="Output CSV path (git-ignored).",
    )
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic records to {out}")


if __name__ == "__main__":
    main()
