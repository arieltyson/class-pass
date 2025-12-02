from __future__ import annotations

from pathlib import Path

import pandas as pd


# Load UCI data
def load_students(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path.resolve()}")

    # try semicolon first
    df = pd.read_csv(path, sep=";")

    return df


# Map multiclass target to a binary target
def make_binary_target(
    df: pd.DataFrame,
    original_col: str = "Target",
    new_col: str = "BinaryTarget",
) -> pd.DataFrame:
    if original_col not in df.columns:
        raise KeyError(f"Expected column '{original_col}' in DataFrame.")

    mapping = {
        "Dropout": "At Risk",
        # Any other label gets mapped to 'Continue' by default below
    }

    bin_series = df[original_col].map(mapping).fillna("Continue")
    df = df.copy()
    df[new_col] = bin_series
    return df
