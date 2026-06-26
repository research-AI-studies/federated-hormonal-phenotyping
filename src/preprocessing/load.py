"""Cohort loading utilities.

Supports CSV and Excel. The loader is format-agnostic about provenance: the
same code path serves the example cohort and a private local cohort.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_cohort(path: str | Path, id_col: str | None = None) -> pd.DataFrame:
    """Load a cohort table from CSV or Excel into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cohort file not found: {path}. Generate an example cohort with "
            f"data/example/generate_example.py or point --data at a local file."
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if id_col and id_col in df.columns:
        df = df.set_index(id_col)

    return df
