"""Command-line entry point to train and evaluate the kNN baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.preprocessing import LabelEncoder

try:
    from classpass.knn import KNNClassifier
    from classpass.metrics import brier, confusion, macro_f1, pr_curves
    from classpass.plotting import plot_confusion_matrix, plot_pr_curves, plot_reliability
    from classpass.preprocess import (
        apply_transformers,
        fit_transformers,
        load_dataset,
        train_val_test_split,
    )
except ModuleNotFoundError:
    if TYPE_CHECKING:
        raise
    SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from classpass.knn import KNNClassifier
    from classpass.metrics import brier, confusion, macro_f1, pr_curves
    from classpass.plotting import plot_confusion_matrix, plot_pr_curves, plot_reliability
    from classpass.preprocess import (
        apply_transformers,
        fit_transformers,
        load_dataset,
        train_val_test_split,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ClassPass kNN baseline.")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--target", default="Target", help="Target column name.")
    parser.add_argument("--k", type=int, default=7, help="Number of neighbors.")
    parser.add_argument(
        "--distance",
        choices=["euclidean", "manhattan"],
        default="euclidean",
        help="Distance metric for kNN.",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Numerical scaling strategy.",
    )
    parser.add_argument(
        "--outdir",
        default="reports/figures",
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins for reliability diagrams.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    df = load_dataset(args.data)
    train_df, val_df, test_df = train_val_test_split(df, target=args.target)

    Xtr, ytr, state = fit_transformers(train_df, target=args.target, scaler=args.scaler)
    Xva, yva = apply_transformers(val_df, state, target=args.target)
    Xte, yte = apply_transformers(test_df, state, target=args.target)

    le = LabelEncoder().fit(ytr)
    classes = le.classes_.tolist()

    clf = KNNClassifier(k=args.k, distance=args.distance).fit(Xtr.values, ytr.values)
    y_pred = clf.predict(Xte.values)

    f1 = macro_f1(yte.values, y_pred)
    cm = confusion(yte.values, y_pred)

    proba = clf.predict_proba(Xte.values)
    curves = pr_curves(yte.values, proba, classes)
    brier_mc = brier(yte.values, proba, classes)

    print(f"Macro-F1: {f1:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(f"Brier (multiclass avg): {brier_mc:.4f}")
    for cls, (_, _, _, ap) in curves.items():
        print(f"AP({cls}): {ap:.3f}")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, classes, os.path.join(args.outdir, "cm_baseline.png"))
    plot_pr_curves(curves, os.path.join(args.outdir, "pr_curves_baseline.png"))
    plot_reliability(
        yte.values,
        proba,
        classes,
        args.bins,
        os.path.join(args.outdir, "reliability_baseline.png"),
    )

    with open("artifacts.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "macro_f1": float(f1),
                "brier": float(brier_mc),
                "classes": classes,
                "params": {
                    "k": args.k,
                    "distance": args.distance,
                    "scaler": args.scaler,
                },
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
