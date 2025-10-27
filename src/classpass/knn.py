"""Lightweight, from-scratch k-Nearest Neighbors classifier."""

from __future__ import annotations

from collections import Counter
from typing import Literal, Tuple

import numpy as np


DistanceMetric = Literal["euclidean", "manhattan"]


class KNNClassifier:
    """Custom kNN classifier for tabular data.

    Parameters
    ----------
    k:
        Number of neighbors to consult (k >= 1).
    distance:
        Distance metric to use: ``"euclidean"`` or ``"manhattan"``.
    """

    def __init__(self, k: int = 7, distance: DistanceMetric = "euclidean") -> None:
        if k < 1:
            msg = "k must be at least 1."
            raise ValueError(msg)
        if distance not in {"euclidean", "manhattan"}:
            msg = "distance must be either 'euclidean' or 'manhattan'."
            raise ValueError(msg)

        self.k = k
        self.distance = distance
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    # --------------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Memorize the training data."""
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=str)
        self.classes_ = np.unique(self._y)
        return self

    # --------------------------------------------------------------------- #
    def _dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between rows of A and B."""
        if self.distance == "euclidean":
            return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
        return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)

    # --------------------------------------------------------------------- #
    def kneighbors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return distances and indices for k nearest neighbors."""
        if self._X is None:
            msg = "Model must be fit before calling kneighbors."
            raise RuntimeError(msg)
        X = np.asarray(X, dtype=float)
        distances = self._dist(X, self._X)
        indices = np.argsort(distances, axis=1)[:, : self.k]
        d_sorted = np.take_along_axis(distances, indices, axis=1)
        return d_sorted, indices

    # --------------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input rows."""
        if self._y is None:
            msg = "Model must be fit before calling predict."
            raise RuntimeError(msg)
        _, idx = self.kneighbors(X)
        preds = []
        for row in idx:
            counts = Counter(self._y[row])
            max_votes = max(counts.values())
            winners = sorted([cls for cls, votes in counts.items() if votes == max_votes])
            preds.append(winners[0])  # stable tie-break
        return np.asarray(preds, dtype=str)

    # --------------------------------------------------------------------- #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class probabilities for each input row."""
        if self.classes_ is None:
            msg = "Model must be fit before calling predict_proba."
            raise RuntimeError(msg)
        _, idx = self.kneighbors(X)
        proba = np.zeros((idx.shape[0], len(self.classes_)), dtype=float)
        for row_idx, row in enumerate(idx):
            counts = Counter(self._y[row])
            for class_idx, cls in enumerate(self.classes_):
                proba[row_idx, class_idx] = counts.get(cls, 0) / float(self.k)
        return proba

    # --------------------------------------------------------------------- #
    def neighbor_explanations(self, X: np.ndarray) -> list[list[int]]:
        """Return neighbor indices for explainability purposes."""
        _, idx = self.kneighbors(X)
        return idx.tolist()

