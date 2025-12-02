import pandas as pd

from scripts.run_eda import summarize
from src.classpass.data import make_binary_target

ABSENCES_MEAN_EXPECTED = 5


def test_eda_summary_keys():
    # create a tiny dummy dataset
    df = pd.DataFrame(
        {
            "Target": ["Dropout", "Graduate", "Enrolled"],
            "Age": [20, 22, 19],
            "Gender": ["M", "F", "F"],
            "Absences": [5, 0, 2],
        }
    )

    # add binary target (At Risk vs Continue)
    df = make_binary_target(df, "Target", "BinaryTarget")

    summary = summarize(df, target_col="BinaryTarget")

    # check required keys exist
    expected_keys = {
        "n_rows",
        "n_cols",
        "missing_fraction",
        "target_counts",
        "target_fraction",
        "numeric_summary",
        "categorical_top_values",
    }

    assert expected_keys.issubset(summary.keys())


def test_eda_missing_fraction():
    df = pd.DataFrame(
        {
            "Target": ["Dropout", "Graduate", None],
            "Age": [20, None, 19],
            "Gender": ["M", "F", "F"],
        }
    )

    df = make_binary_target(df, "Target", "BinaryTarget")
    summary = summarize(df, target_col="BinaryTarget")

    missing = summary["missing_fraction"]

    # target column had 1 missing → mapped to Continue, so original missing still counted
    assert missing["Target"] == 1 / 3
    assert missing["Age"] == 1 / 3
    assert missing["Gender"] == 0.0


def test_eda_numeric_summary():
    df = pd.DataFrame(
        {
            "Target": ["Dropout", "Graduate", "Enrolled"],
            "Absences": [10, 0, 5],
        }
    )

    df = make_binary_target(df, "Target", "BinaryTarget")
    summary = summarize(df, target_col="BinaryTarget")

    numeric = summary["numeric_summary"]
    assert "Absences" in numeric["mean"]
    assert numeric["mean"]["Absences"] == ABSENCES_MEAN_EXPECTED  # mean of [10,0,5]
