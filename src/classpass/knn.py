from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


DistanceName = Literal["euclidean", "manhattan"]

# compute pairwise distances between rows of X and Y
def _pairwise_distance(X: np.ndarray, Y: np.ndarray, metric: DistanceName) -> np.ndarray:
    if metric == "euclidean":
        # (x - y)^2 = x^2 + y^2 - 2xy
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
        cross = X @ Y.T
        dists = np.sqrt(np.maximum(X_sq + Y_sq - 2 * cross, 0.0))
    elif metric == "manhattan":
        # Broadcast subtraction, then sum abs along feature axis
        dists = np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    else:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")
    return dists


@dataclass
class NeighborExplanation:
    indices: np.ndarray       # shape (n_samples, k)
    distances: np.ndarray     # shape (n_samples, k)
    neighbor_labels: np.ndarray  # shape (n_samples, k)

# simple knn classifier from scratch (will be tested against sci kit later)
class KNNClassifier:

    def __init__(self, k: int = 5, distance: DistanceName = "euclidean"):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.distance = distance

        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self._X_train = np.asarray(X, dtype=float)
        self._y_train = np.asarray(y)
        self.classes_ = np.unique(self._y_train)
        return self

    def _check_is_fitted(self):
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("KNNClassifier has not been fitted yet.")

    # return (distances, indices) of knns in the training set for each row in X
    def _kneighbors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        dists = _pairwise_distance(X, self._X_train, metric=self.distance)
        # argsort along train_samples axis
        neighbor_idx = np.argpartition(dists, self.k - 1, axis=1)[:, : self.k]
        # sort the first k neighbors by distance for nicer explanations
        row_indices = np.arange(X.shape[0])[:, None]
        sorted_order = np.argsort(dists[row_indices, neighbor_idx], axis=1)
        neighbor_idx = neighbor_idx[row_indices, sorted_order]
        neighbor_dists = dists[row_indices, neighbor_idx]
        return neighbor_dists, neighbor_idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        assert self.classes_ is not None
        return self.classes_[class_indices]

    # probability estimate = fraction of neighbours in each class
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        assert self.classes_ is not None

        neighbor_dists, neighbor_idx = self._kneighbors(X)
        neighbor_labels = self._y_train[neighbor_idx]

        n_samples = neighbor_labels.shape[0]
        n_classes = self.classes_.shape[0]
        proba = np.zeros((n_samples, n_classes), dtype=float)

        for i, c in enumerate(self.classes_):
            proba[:, i] = np.mean(neighbor_labels == c, axis=1)

        return proba

    # essentially returns explanation for classification
    def explain(self, X: np.ndarray) -> NeighborExplanation:
        self._check_is_fitted()
        neighbor_dists, neighbor_idx = self._kneighbors(X)
        neighbor_labels = self._y_train[neighbor_idx]
        return NeighborExplanation(
            indices=neighbor_idx,
            distances=neighbor_dists,
            neighbor_labels=neighbor_labels,
        )