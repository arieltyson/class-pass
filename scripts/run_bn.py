from __future__ import annotations

import argparse

from src.classpass.bayesian_network import BayesianNetwork
from src.classpass.data import load_students, make_binary_target
from src.classpass.evaluation import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, default="Target")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--outdir", type=str, default="reports/bn")
    args = parser.parse_args()

    df = load_students(args.data)
    if args.binary:
        df = make_binary_target(df, args.target, "BinaryTarget")
        target_col = "BinaryTarget"
    else:
        target_col = args.target

    # Fit Bayesian Network
    bn = BayesianNetwork()
    bn.fit(df, target_col=target_col)

    # Predictions
    preds = df.apply(bn.predict, axis=1)
    metrics = compute_metrics(df[target_col], preds)

    print("[Bayesian Network Results]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
