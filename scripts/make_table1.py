"""Build a baseline characteristics table (Table 1) from a cohort.

Uses ``tableone`` when available, otherwise falls back to a grouped describe().
The rendered table is written to the git-ignored ``outputs/tables/`` tree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import load_cohort  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a baseline Table 1.")
    ap.add_argument("--data", default="data/example/cohort.csv")
    ap.add_argument("--group", default="diagnosis", help="Stratifying column.")
    ap.add_argument("--out", default="outputs/tables/table1.csv")
    args = ap.parse_args()

    df = load_cohort(args.data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    categorical = [c for c in df.columns if df[c].dtype == "object" and c != args.group]
    try:
        from tableone import TableOne

        groupings = args.group if args.group in df.columns else None
        table = TableOne(df, categorical=categorical, groupby=groupings, pval=bool(groupings))
        table.to_csv(out)
        print(table.tabulate(tablefmt="github"))
    except Exception:
        if args.group in df.columns:
            summary = df.groupby(args.group).describe(include="all").T
        else:
            summary = df.describe(include="all").T
        summary.to_csv(out)
        print(summary.head(40).to_string())

    print(f"\nWrote Table 1 to {out}")


if __name__ == "__main__":
    main()
