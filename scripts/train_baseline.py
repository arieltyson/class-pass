from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from src.classpass.data import load_students, make_binary_target
from src.classpass.evaluation import compute_metrics, plot_confusion, plot_f1_vs_k
from src.classpass.knn import KNNClassifier
from src.classpass.preprocessing import preprocess_and_split


def parse_k_grid(k_arg: str) -> List[int]:
    return [int(x) for x in k_arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline kNN for student risk prediction."
    )
    parser.add_argument("--data", type=str, required=True, help="Path to students.csv")
    parser.add_argument(
        "--target",
        type=str,
        default="Target",
        help="Original target column (before binary mapping)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use binary target At Risk vs Continue.",
    )
    parser.add_argument(
        "--drop-cols",
        type=str,
        default="",
        help="Comma-separated list of columns to drop before training.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="standard",
        choices=["standard", "minmax"],
        help="Scaler for numeric features.",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan"],
        help="Distance metric for kNN.",
    )
    parser.add_argument(
        "--k-grid",
        type=str,
        default="3,5,7,9,11",
        help="Grid of k values, e.g. '3,5,7,9'.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test split.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of full data reserved for validation (inside train+val).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="reports/figures",
        help="Directory for plots and metrics artifact.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )
    args = parser.parse_args()

    df = load_students(args.data)

    if args.binary:
        df = make_binary_target(df, original_col=args.target, new_col="BinaryTarget")
        target_col = "BinaryTarget"
    else:
        target_col = args.target

    drop_cols = [c for c in args.drop_cols.split(",") if c.strip()]

    splits = preprocess_and_split(
        df=df,
        target_col=target_col,
        drop_cols=drop_cols,
        scaler=args.scaler,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    k_values = parse_k_grid(args.k_grid)

    best_k = None
    best_f1 = -np.inf
    f1_per_k = []

    print("[Train] Tuning k on validation set...")

    for k in k_values:
        clf = KNNClassifier(k=k, distance=args.distance)
        clf.fit(splits.X_train, splits.y_train)
        y_val_pred = clf.predict(splits.X_val)

        metrics = compute_metrics(splits.y_val, y_val_pred)
        f1 = metrics["f1_binary"]
        f1_per_k.append(f1)

        print(f"  k={k:2d} -> val F1(At Risk) = {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    assert best_k is not None
    print(f"[Train] Best k on validation: {best_k} (F1={best_f1:.4f})")

    # retrain on train + val using best k, then evaluate on test
    X_train_full = np.vstack([splits.X_train, splits.X_val])
    y_train_full = np.concatenate([splits.y_train, splits.y_val])

    final_clf = KNNClassifier(k=best_k, distance=args.distance)
    final_clf.fit(X_train_full, y_train_full)

    y_test_pred = final_clf.predict(splits.X_test)
    test_metrics = compute_metrics(splits.y_test, y_test_pred)

    print("\n[Test metrics]")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = sorted(np.unique(splits.y_test).tolist())
    plot_confusion(
        splits.y_test,
        y_test_pred,
        labels=labels,
        outpath=outdir / "cm_knn.png",
    )

    plot_f1_vs_k(
        k_values,
        f1_per_k,
        outpath=outdir / "f1_vs_k.png",
    )

    # save a JSON summarizing the run
    artifact = {
        "data_path": str(Path(args.data).resolve()),
        "target_col": target_col,
        "binary": args.binary,
        "drop_cols": drop_cols,
        "scaler": args.scaler,
        "distance": args.distance,
        "k_values": k_values,
        "best_k": best_k,
        "val_f1_per_k": f1_per_k,
        "test_metrics": test_metrics,
    }
    with (outdir / "artifacts_knn.json").open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"\n[Done] Saved confusion matrix, F1-vs-k plot, and artifacts to {outdir}")


if __name__ == "__main__":
    main()