from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

try:
    from classpass.data import load_students, make_binary_target
except ModuleNotFoundError:
    if TYPE_CHECKING:
        raise
    SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from classpass.data import load_students, make_binary_target


def summarize(df: pd.DataFrame, target_col: str) -> dict:
    """Compute basic EDA summaries for the dataset."""
    summary: dict = {}
    summary["n_rows"] = int(df.shape[0])
    summary["n_cols"] = int(df.shape[1])

    # missingness per column
    summary["missing_fraction"] = df.isna().mean().to_dict()

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
    top_values = {col: cat_df[col].value_counts().head(5).to_dict() for col in cat_df.columns}
    summary["categorical_top_values"] = top_values

    return summary


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_target_distribution(df: pd.DataFrame, target_col: str, fig_path: Path) -> None:
    """Plot a bar chart of target class counts and save it."""
    counts = df[target_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    ax.set_title("Target Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run EDA on student dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to students.csv")
    parser.add_argument("--target", type=str, default="Target", help="Original target column")
    parser.add_argument("--binary", action="store_true", help="Add binary target column")
    parser.add_argument("--outdir", type=str, default="reports/eda", help="Output directory")
    parser.add_argument(
        "--figdir",
        type=str,
        default="reports/figures",
        help="Directory to store generated plots.",
    )
    args = parser.parse_args()

    df = load_students(args.data)

    if args.binary:
        df = make_binary_target(df, original_col=args.target, new_col="BinaryTarget")
        target_col = "BinaryTarget"
    else:
        target_col = args.target

    outdir = ensure_dir(args.outdir)
    figdir = ensure_dir(args.figdir)

    summary = summarize(df, target_col=target_col)

    # JSON summary
    json_path = outdir / "eda_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # save simple CSV of target distribution
    target_counts = pd.Series(summary["target_counts"], name="count")
    target_counts.to_csv(outdir / "target_counts.csv")

    # save class distribution plot
    plot_target_distribution(df, target_col, figdir / "target_distribution.png")

    print(f"[EDA] Wrote EDA summary to {json_path}")
    print(f"[EDA] Saved target distribution plot to {figdir / 'target_distribution.png'}")


if __name__ == "__main__":
    main()
