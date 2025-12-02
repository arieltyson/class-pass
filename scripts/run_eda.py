from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.classpass.data import load_students, make_binary_target


def summarize(df: pd.DataFrame, target_col: str) -> dict:
    summary: dict = {}

    summary["n_rows"] = int(df.shape[0])
    summary["n_cols"] = int(df.shape[1])

    # missingness per column
    missing = df.isna().mean().to_dict()
    summary["missing_fraction"] = missing

    # target distribution
    value_counts = df[target_col].value_counts(normalize=False)
    value_counts_norm = df[target_col].value_counts(normalize=True)
    summary["target_counts"] = value_counts.to_dict()
    summary["target_fraction"] = value_counts_norm.to_dict()

    # numeric stats
    num_df = df.select_dtypes(include=["number"])
    summary["numeric_summary"] = num_df.describe().transpose().to_dict()

    # top categories for non-numeric
    cat_df = df.select_dtypes(exclude=["number"])
    top_values = {}
    for col in cat_df.columns:
        top_values[col] = cat_df[col].value_counts().head(5).to_dict()
    summary["categorical_top_values"] = top_values

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run EDA on student dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to students.csv")
    parser.add_argument("--target", type=str, default="Target", help="Original target column")
    parser.add_argument("--binary", action="store_true", help="Add binary target column")
    parser.add_argument("--outdir", type=str, default="reports/eda", help="Output directory")
    args = parser.parse_args()

    df = load_students(args.data)

    if args.binary:
        df = make_binary_target(df, original_col=args.target, new_col="BinaryTarget")
        target_col = "BinaryTarget"
    else:
        target_col = args.target

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = summarize(df, target_col=target_col)

    # JSON summary
    json_path = outdir / "eda_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # save simple CSV of target distribution
    target_counts = pd.Series(summary["target_counts"], name="count")
    target_counts.to_csv(outdir / "target_counts.csv")

    print(f"[EDA] Wrote EDA summary to {json_path}")


if __name__ == "__main__":
    main()
