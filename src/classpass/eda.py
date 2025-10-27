"""Exploratory data analysis helpers for ClassPass."""

from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_dataframe(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Return high-level dataset metadata and target distribution."""
    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "feature_types": df.dtypes.astype(str).to_dict(),
    }
    if target in df.columns:
        target_counts = (
            df[target]
            .value_counts(dropna=False)
            .rename_axis("label")
            .to_frame("count")
            .reset_index()
        )
        target_counts["percentage"] = target_counts["count"] / float(df.shape[0])
        summary["target_distribution"] = target_counts.to_dict(orient="records")
    return summary


def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-column missing counts and percentages."""
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_percent"])
    missing = df.isna().sum()
    percent = missing / float(df.shape[0])
    result = (
        pd.DataFrame(
            {
                "column": missing.index,
                "missing_count": missing.values,
                "missing_percent": percent.values,
            }
        )
        .sort_values(by="missing_count", ascending=False)
        .reset_index(drop=True)
    )
    return result


def numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return pd.DataFrame(
            columns=["feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        )
    desc = df[num_cols].describe().transpose().reset_index().rename(columns={"index": "feature"})
    return desc


def categorical_cardinality(df: pd.DataFrame, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Return unique counts and top categories for non-numeric features."""
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    report: dict[str, list[dict[str, Any]]] = {}
    for col in cat_cols:
        counts = (
            df[col]
            .value_counts(dropna=False)
            .head(top_k)
            .rename_axis("value")
            .reset_index(name="count")
        )
        counts["percentage"] = counts["count"] / float(df.shape[0])
        report[col] = counts.to_dict(orient="records")
    return report
