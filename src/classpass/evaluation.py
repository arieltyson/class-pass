from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# evaluate everything

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", pos_label="At Risk")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def plot_confusion(y_true, y_pred, labels: Sequence[str], outpath: str | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d", cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.close()


def plot_f1_vs_k(ks: List[int], f1_scores: List[float], outpath: str | None = None):
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