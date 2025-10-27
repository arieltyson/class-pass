"""Generate exploratory data analysis and preprocessing verification artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

try:
    from classpass.eda import (
        categorical_cardinality,
        compute_missingness,
        numerical_summary,
        summarize_dataframe,
    )
    from classpass.preprocess import (
        apply_transformers,
        fit_transformers,
        load_dataset,
        train_val_test_split,
    )
except ModuleNotFoundError:
    if TYPE_CHECKING:
        raise
    SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from classpass.eda import (
        categorical_cardinality,
        compute_missingness,
        numerical_summary,
        summarize_dataframe,
    )
    from classpass.preprocess import (
        apply_transformers,
        fit_transformers,
        load_dataset,
        train_val_test_split,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EDA and preprocessing sanity checks for the ClassPass dataset."
    )
    parser.add_argument("--data", required=True, help="Path to the raw CSV dataset.")
    parser.add_argument("--target", default="Target", help="Target column name.")
    parser.add_argument(
        "--outdir",
        default="reports/eda",
        help="Directory to store generated EDA artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top categories to record for categorical features.",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Scaling strategy to validate during preprocessing.",
    )
    return parser


def ensure_outdir(path: str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def run_preprocessing_checks(df: pd.DataFrame, target: str, scaler: str) -> dict[str, str]:
    """Attempt to fit preprocessing pipeline and report derived metadata."""
    train_df, val_df, test_df = train_val_test_split(df, target=target)
    X_train, _, state = fit_transformers(train_df, target=target, scaler=scaler)
    X_val, _ = apply_transformers(val_df, state, target=target)
    X_test, _ = apply_transformers(test_df, state, target=target)
    return {
        "train_shape": f"{X_train.shape[0]} x {X_train.shape[1]}",
        "val_shape": f"{X_val.shape[0]} x {X_val.shape[1]}",
        "test_shape": f"{X_test.shape[0]} x {X_test.shape[1]}",
        "n_numeric": str(len(state["num_cols"])),
        "n_categorical": str(len(state["cat_cols"])),
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    df = load_dataset(args.data)
    outdir = ensure_outdir(args.outdir)

    high_level = summarize_dataframe(df, args.target)
    missingness = compute_missingness(df)
    numeric_stats = numerical_summary(df)
    categorical_stats = categorical_cardinality(df, top_k=args.top_k)
    preprocessing = run_preprocessing_checks(df, args.target, args.scaler)

    with open(outdir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "overview": high_level,
                "preprocessing": preprocessing,
            },
            fh,
            indent=2,
        )

    missingness.to_csv(outdir / "missingness.csv", index=False)
    numeric_stats.to_csv(outdir / "numeric_summary.csv", index=False)
    with open(outdir / "categorical_top_values.json", "w", encoding="utf-8") as fh:
        json.dump(categorical_stats, fh, indent=2)

    print("EDA artifacts written to", outdir.resolve())


if __name__ == "__main__":
    main()
