from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true,
    y_pred,
    *,
    pos_label: str | None = "At Risk",
) -> dict[str, float]:
    """Compute accuracy and F1 scores with safe handling of binary vs. multiclass."""
    classes = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    BINARY_CLASS_COUNT = 2
    result = {"accuracy": float(accuracy_score(y_true, y_pred))}

    # Always provide macro F1 for multiclass comparability
    result["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))

    is_binary = len(classes) == BINARY_CLASS_COUNT and (
        pos_label in classes if pos_label is not None else True
    )
    if is_binary:
        positive = pos_label if pos_label is not None else classes[1]
        result["f1_binary"] = float(f1_score(y_true, y_pred, average="binary", pos_label=positive))
    else:
        # Fallback: mirror macro when binary assumptions do not hold
        result["f1_binary"] = result["f1_macro"]

    return result


def plot_confusion(y_true, y_pred, labels: Sequence[str], outpath: str | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d", cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.close()


def plot_f1_vs_k(ks: list[int], f1_scores: list[float], outpath: str | None = None):
    plt.figure()
    plt.plot(ks, f1_scores, marker="o")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Validation F1 (At Risk)")
    plt.title("F1 vs k (Validation)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.close()
