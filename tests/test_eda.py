import pandas as pd
import pytest

from classpass import eda

EXPECTED_ROWS = 4
EXPECTED_COLUMNS = 3
EXPECTED_DROPOUT = 2
EXPECTED_GRADUATE = 1
EXPECTED_HIGH_ATTENDANCE = 2


def sample_dataframe():
    return pd.DataFrame(
        {
            "Target": ["Dropout", "Enrolled", "Graduate", "Dropout"],
            "GPA": [3.0, 3.5, None, 2.8],
            "Attendance": ["High", "Medium", "High", None],
        }
    )


def test_summarize_dataframe_counts_target_distribution():
    df = sample_dataframe()
    summary = eda.summarize_dataframe(df, "Target")
    assert summary["n_rows"] == EXPECTED_ROWS
    assert summary["n_columns"] == EXPECTED_COLUMNS
    dist = {entry["label"]: entry["count"] for entry in summary["target_distribution"]}
    assert dist["Dropout"] == EXPECTED_DROPOUT
    assert dist["Graduate"] == EXPECTED_GRADUATE


def test_missingness_reports_expected_columns():
    df = sample_dataframe()
    missing = eda.compute_missingness(df)
    assert missing.loc[missing["column"] == "GPA", "missing_count"].iloc[0] == 1
    assert missing.loc[missing["column"] == "Attendance", "missing_count"].iloc[0] == 1


def test_numerical_summary_contains_gpa_stats():
    df = sample_dataframe()
    stats = eda.numerical_summary(df)
    gpa_row = stats.loc[stats["feature"] == "GPA"].iloc[0]
    expected_mean = (3.0 + 3.5 + 2.8) / 3
    assert gpa_row["count"] == pytest.approx(3.0)
    assert gpa_row["mean"] == pytest.approx(expected_mean, abs=1e-6)


def test_categorical_cardinality_reports_top_values():
    df = sample_dataframe()
    report = eda.categorical_cardinality(df, top_k=2)
    assert "Attendance" in report
    values = {item["value"]: item["count"] for item in report["Attendance"]}
    assert values["High"] == EXPECTED_HIGH_ATTENDANCE
