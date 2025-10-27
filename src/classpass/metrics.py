"""Evaluation metrics for ClassPass models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)


def macro_f1(y_true, y_pred) -> float:
    """Compute macro-averaged F1 score."""
    return f1_score(y_true, y_pred, average="macro")


def confusion(y_true, y_pred):
    """Return confusion matrix with consistent label ordering."""
    return confusion_matrix(y_true, y_pred, labels=np.unique(y_true))


def pr_curves(y_true, y_score, classes):
    """Generate precision-recall curves and average precision per class."""
    curves = {}
    for i, cls in enumerate(classes):
        y_bin = (np.asarray(y_true) == cls).astype(int)
        precision, recall, thresholds = precision_recall_curve(y_bin, y_score[:, i])
        ap = average_precision_score(y_bin, y_score[:, i])
        curves[cls] = (precision, recall, thresholds, ap)
    return curves


def brier(y_true, y_proba, classes):
    """Compute mean Brier score across classes."""
    y_true = np.asarray(y_true, dtype=str)
    total = 0.0
    for i, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        total += brier_score_loss(y_bin, y_proba[:, i])
    return total / len(classes)
