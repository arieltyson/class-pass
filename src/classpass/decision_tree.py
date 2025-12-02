from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Impurity functions
def entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))


def gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p**2)


# Tree Node
@dataclass
class TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None
    prediction: Any | None = None  # leaf class
    depth: int = 0


# simple decision tree classifier from scratch w/ entropy or gini split
class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "entropy",
        max_depth: int = 5,
        min_samples_split: int = 2,
    ):
        assert criterion in ("entropy", "gini")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: TreeNode | None = None
        self.classes_: np.ndarray | None = None

    # Fitting
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth) -> TreeNode:
        node = TreeNode(depth=depth)

        # stopping conditions
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            node.prediction = self._majority_class(y)
            return node

        # find best split
        feat, thresh = self._best_split(X, y)
        if feat is None:  # no valid split
            node.prediction = self._majority_class(y)
            return node

        node.feature_index = feat
        node.threshold = thresh

        left_idx = X[:, feat] <= thresh
        right_idx = ~left_idx

        node.left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        node.right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return node

    # Utilities
    def _impurity(self, y: np.ndarray) -> float:
        return entropy(y) if self.criterion == "entropy" else gini(y)

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        parent_imp = self._impurity(y)

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for thresh in thresholds:
                left_idx = X[:, feature] <= thresh
                right_idx = ~left_idx

                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                left_imp = self._impurity(y[left_idx])
                right_imp = self._impurity(y[right_idx])

                child_imp = left_idx.mean() * left_imp + right_idx.mean() * right_imp

                gain = parent_imp - child_imp
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = thresh

        return best_feature, best_threshold

    # Prediction
    def predict(self, X: np.ndarray):
        return np.array([self._predict_row(row, self.root) for row in X])

    def _predict_row(self, row, node: TreeNode):
        if node.prediction is not None:
            return node.prediction

        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)

    def predict_proba(self, X: np.ndarray):
        preds = self.predict(X)
        probs = []

        for p in preds:
            vec = np.zeros(len(self.classes_))
            vec[self.classes_ == p] = 1.0
            probs.append(vec)

        return np.array(probs)

    # Rule Extraction
    def extract_rules(self) -> list[str]:
        """Return human-readable decision paths."""
        rules = []
        self._collect_rules(self.root, path=[], rules=rules)
        return rules

    def _collect_rules(self, node: TreeNode, path: list[str], rules: list[str]):
        if node.prediction is not None:
            rule = " AND ".join(path) + f" => Predict {node.prediction}"
            rules.append(rule)
            return

        left_rule = f"Feature[{node.feature_index}] <= {node.threshold:.4f}"
        right_rule = f"Feature[{node.feature_index}] > {node.threshold:.4f}"

        self._collect_rules(node.left, path + [left_rule], rules)
        self._collect_rules(node.right, path + [right_rule], rules)
