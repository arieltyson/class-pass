from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.classpass.data import load_students, make_binary_target
from src.classpass.decision_tree import DecisionTreeClassifier
from src.classpass.evaluation import compute_metrics, plot_confusion
from src.classpass.preprocessing import preprocess_and_split


def parse_depth_grid(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train custom Decision Tree classifier.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, default="Target")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--drop-cols", type=str, default="")
    parser.add_argument("--depth-grid", type=str, default="3,5,7,9")
    parser.add_argument("--criterion", type=str, default="entropy", choices=["entropy", "gini"])
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--outdir", type=str, default="reports/figures_tree")
    args = parser.parse_args()

    df = load_students(args.data)

    if args.binary:
        df = make_binary_target(df, args.target, "BinaryTarget")
        df = df.drop(columns=[args.target])
        target_col = "BinaryTarget"
    else:
        target_col = args.target

    drop_cols = [c for c in args.drop_cols.split(",") if c.strip()]
    splits = preprocess_and_split(df, target_col=target_col, drop_cols=drop_cols)

    depths = parse_depth_grid(args.depth_grid)

    best_depth = None
    best_f1 = -1
    f1_map = {}

    print("[Tree] Tuning max_depth on validation set...")

    for d in depths:
        model = DecisionTreeClassifier(
            criterion=args.criterion,
            max_depth=d,
            min_samples_split=args.min_samples_split,
        ).fit(splits.X_train, splits.y_train)

        preds = model.predict(splits.X_val)
        f1 = compute_metrics(splits.y_val, preds)["f1_binary"]
        f1_map[d] = f1
        print(f"  depth={d} -> val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_depth = d

    print(f"[Tree] Best max_depth={best_depth} (F1={best_f1:.4f})")

    # train final model
    X_full = np.vstack([splits.X_train, splits.X_val])
    y_full = np.concatenate([splits.y_train, splits.y_val])

    model = DecisionTreeClassifier(
        criterion=args.criterion,
        max_depth=best_depth,
        min_samples_split=args.min_samples_split,
    ).fit(X_full, y_full)

    test_preds = model.predict(splits.X_test)
    metrics = compute_metrics(splits.y_test, test_preds)

    print("\n[Test Results]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_confusion(
        splits.y_test,
        test_preds,
        labels=sorted(np.unique(splits.y_test)),
        outpath=outdir / "cm_tree.png",
    )

    # save rules
    rules = model.extract_rules()
    with (outdir / "tree_rules.txt").open("w") as f:
        f.write("\n".join(rules))

    # save artifact
    with (outdir / "tree_artifacts.json").open("w") as f:
        json.dump(
            {
                "best_depth": best_depth,
                "f1_val_map": f1_map,
                "test_metrics": metrics,
            },
            f,
            indent=2,
        )

    print(f"[Tree] Saved confusion matrix, rules, and artifacts to {outdir}")


if __name__ == "__main__":
    main()
