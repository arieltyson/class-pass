"""Dataset loading and preprocessing helpers for ClassPass."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_NUM_SPLIT_SEGMENTS = 3


def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV dataset and validate the target column."""
    df = pd.read_csv(path)
    if "Target" not in df.columns:
        msg = "Expected a 'Target' column with values: Dropout/Enrolled/Graduate."
        raise ValueError(msg)
    return df


def train_val_test_split(
    df: pd.DataFrame,
    *,
    target: str = "Target",
    splits: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation/test splits."""
    if len(splits) != _NUM_SPLIT_SEGMENTS:
        msg = "splits must contain exactly three proportions (train, val, test)."
        raise ValueError(msg)
    if not np.isclose(sum(splits), 1.0):
        msg = "train/val/test proportions must sum to 1.0."
        raise ValueError(msg)
    train, val, test = splits
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train),
        stratify=df[target],
        random_state=seed,
    )
    rel = test / (val + test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel,
        stratify=temp_df[target],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def fit_transformers(
    train_df: pd.DataFrame,
    target: str = "Target",
    scaler: str = "standard",
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Fit preprocessing transformers on the training split."""
    y_train = train_df[target].astype(str)
    X_train = train_df.drop(columns=[target])

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in X_train.columns if col not in num_cols]

    X_cat = (
        pd.get_dummies(X_train[cat_cols], drop_first=False)
        if cat_cols
        else pd.DataFrame(index=X_train.index)
    )
    X_num = X_train[num_cols].copy()

    if scaler == "standard":
        mu = X_num.mean()
        sigma = X_num.std().replace(0, 1.0)
        X_num = (X_num - mu) / sigma
        scaler_state = ("standard", mu, sigma)
    elif scaler == "minmax":
        mn = X_num.min()
        mx = X_num.max().replace(0, 1.0)
        X_num = (X_num - mn) / (mx - mn).replace(0, 1.0)
        scaler_state = ("minmax", mn, mx)
    else:
        scaler_state = ("none", None, None)

    X_proc = pd.concat([X_num, X_cat], axis=1)
    state = {
        "scaler_state": scaler_state,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "ohe_cols": X_cat.columns.tolist(),
    }
    return X_proc, y_train, state


def apply_transformers(
    df: pd.DataFrame,
    state: dict,
    target: str = "Target",
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply stored transformers to a new split."""
    y = df[target].astype(str)
    X = df.drop(columns=[target])

    num_cols = state["num_cols"]
    cat_cols = state["cat_cols"]
    ohe_cols = state["ohe_cols"]

    X_num = X[num_cols].copy()
    mode, a, b = state["scaler_state"]
    if mode == "standard":
        X_num = (X_num - a) / b.replace(0, 1.0)
    elif mode == "minmax":
        X_num = (X_num - a) / (b - a).replace(0, 1.0)

    X_cat = (
        pd.get_dummies(X[cat_cols], drop_first=False) if cat_cols else pd.DataFrame(index=X.index)
    )
    for col in ohe_cols:
        if col not in X_cat.columns:
            X_cat[col] = 0
    X_cat = X_cat[ohe_cols] if ohe_cols else X_cat

    return pd.concat([X_num, X_cat], axis=1), y
