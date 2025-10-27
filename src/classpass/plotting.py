"""Plotting utilities for evaluation artifacts."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, out_path):
    """Render and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr_curves(curves, out_path):
    """Plot per-class precision-recall curves."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for cls, (precision, recall, _, ap) in curves.items():
        ax.plot(recall, precision, label=f"{cls} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.set_title("Precision-Recall Curves")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reliability(y_true, y_proba, classes, bins, out_path):
    """Plot reliability diagram illustrating calibration."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for i, cls in enumerate(classes):
        conf = y_proba[:, i]
        y_bin = (np.asarray(y_true) == cls).astype(int)
        edges = np.linspace(0, 1, bins + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        acc, conf_bin = [], []
        for lo, hi, mid in zip(edges[:-1], edges[1:], mids):
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() == 0:
                continue
            acc.append(y_bin[mask].mean())
            conf_bin.append(mid)
        if conf_bin:
            ax.plot(conf_bin, acc, marker="o", linestyle="-", label=cls)
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

