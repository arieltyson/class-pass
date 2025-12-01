from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CPT:
    """Conditional Probability Table.
    Stores P(node=1 | parents). Parents are binary.
    """
    table: dict  # key: tuple of parent values, value: probability of node=1


class BayesianNetwork:
    """
    A simple manually-structured Bayesian Network for dropout risk inference.

    Nodes:
        LowGrades      (no parents)
        FinancialRisk  (no parents)
        LowEngagement  (no parents)
        DropoutRisk    (parents = [LowGrades, FinancialRisk, LowEngagement])
    """

    def __init__(self, grade_thresh=10.0, engagement_thresh=3):
        self.grade_thresh = grade_thresh
        self.engagement_thresh = engagement_thresh

        self.low_grades_cpt = None
        self.fin_risk_cpt = None
        self.low_engage_cpt = None
        self.dropout_cpt = None

    # Feature extraction from raw row
    def extract_indicators(self, row: pd.Series):
        low_grades = int(row["Curricular units 1st sem (grade)"] < self.grade_thresh)
        financial_risk = int(row["Debtor"] == 1)
        low_engagement = int(row["Curricular units 1st sem (approved)"] < self.engagement_thresh)

        return low_grades, financial_risk, low_engagement

    # Training = estimating CPTs from data
    def fit(self, df: pd.DataFrame, target_col="BinaryTarget"):
        # Indicator variables
        LG = (df["Curricular units 1st sem (grade)"] < self.grade_thresh).astype(int)
        FR = (df["Debtor"] == 1).astype(int)
        LE = (df["Curricular units 1st sem (approved)"] < self.engagement_thresh).astype(int)
        Y = (df[target_col] == "At Risk").astype(int)

        # Priors
        self.low_grades_cpt = CPT({
            (): LG.mean()
        })
        self.fin_risk_cpt = CPT({
            (): FR.mean()
        })
        self.low_engage_cpt = CPT({
            (): LE.mean()
        })

        table = {}
        for lg in [0, 1]:
            for fr in [0, 1]:
                for le in [0, 1]:
                    mask = (LG == lg) & (FR == fr) & (LE == le)
                    if mask.sum() == 0:
                        p = 0.5
                    else:
                        p = Y[mask].mean()
                        if np.isnan(p):
                            p = 0.5
                    table[(lg, fr, le)] = p

        self.dropout_cpt = CPT(table)
        return self

    # Inference via enumeration
    def predict_proba(self, row: pd.Series):
        LG, FR, LE = self.extract_indicators(row)

        p_dropout = self.dropout_cpt.table[(LG, FR, LE)]
        return np.array([1 - p_dropout, p_dropout])

    def predict(self, row: pd.Series):
        probs = self.predict_proba(row)
        return "At Risk" if probs[1] > 0.5 else "Continue"
