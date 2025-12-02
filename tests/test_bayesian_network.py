import pandas as pd

from src.classpass.bayesian_network import PREDICTION_THRESHOLD, BayesianNetwork


def test_bn_basic():
    df = pd.DataFrame(
        {
            "Curricular units 1st sem (grade)": [5, 15],
            "Debtor": [1, 0],
            "Curricular units 1st sem (approved)": [1, 6],
            "BinaryTarget": ["At Risk", "Continue"],
        }
    )

    bn = BayesianNetwork().fit(df, target_col="BinaryTarget")

    row = df.iloc[0]
    prob = bn.predict_proba(row)
    assert prob[1] > PREDICTION_THRESHOLD

    assert bn.predict(df.iloc[1]) in ("At Risk", "Continue")
