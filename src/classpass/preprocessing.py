from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


@dataclass
class PreprocessedSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: List[str]


def _infer_feature_types(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in drop_cols + [target_col]]
    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]
    return num_cols, cat_cols

# one hot encode categories and scale nums
def build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str] | None = None,
    scaler: str = "standard",
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    if drop_cols is None:
        drop_cols = []

    num_cols, cat_cols = _infer_feature_types(df, target_col, drop_cols)

    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    else:
        raise ValueError("scaler must be 'standard' or 'minmax'")

    transformers = []
    if num_cols:
        transformers.append(("num", scaler_obj, num_cols))
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor, num_cols, cat_cols

# split in train / val / test, then preprocess
def preprocess_and_split(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str] | None = None,
    scaler: str = "standard",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessedSplits:
    if drop_cols is None:
        drop_cols = []

    y = df[target_col].values
    X = df.drop(columns=drop_cols + [target_col])

    # first split off test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # now split train vs val
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_train_val,
        random_state=random_state,
        stratify=y_train_val,
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(
        df=X_train_val.assign(**{target_col: y_train_val}),  # for type inference
        target_col=target_col,
        drop_cols=drop_cols,
        scaler=scaler,
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # build feature names for later plots / explanations
    feature_names: List[str] = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_feature_names)

    return PreprocessedSplits(
        X_train=X_train_proc,
        y_train=y_train,
        X_val=X_val_proc,
        y_val=y_val,
        X_test=X_test_proc,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )