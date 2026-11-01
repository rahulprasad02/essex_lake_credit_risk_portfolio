"""
src/anomaly_detector.py
Mandatory AI Feature 1: Anomaly Detection System.
Implements 3 clear behavioral & cohort-outlier rules to flag unusual accounts.
"""

import numpy as np
import pandas as pd


def detect_portfolio_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate 3 transparent anomaly rules against each loan account:
      - Rule 1: Sudden Delinquency Jump (clean on-us history, sudden DPD spike > 30)
      - Rule 2: Payment Shortfall Outlier (> 2.5 SD of loan_type peer cohort)
      - Rule 3: External Bureau Divergence (clean on-us repayment, but severe bureau distress)
    """
    df = df.copy()

    # Calculate peer cohort benchmarks for Rule 2
    cohort_stats = (
        df.groupby("loan_type")["repay_avg_shortfall"]
        .agg(["mean", "std"])
        .reset_index()
    )
    cohort_stats.rename(
        columns={
            "mean": "cohort_mean_shortfall",
            "std": "cohort_std_shortfall",
        },
        inplace=True,
    )
    df = pd.merge(df, cohort_stats, on="loan_type", how="left")

    # Initialize anomaly columns
    df["is_anomaly"] = False
    df["anomaly_rule"] = "None"
    df["anomaly_reason"] = "Normal behavior"
    df["actual_value"] = ""
    df["expected_value"] = ""
    df["review_status"] = "Pending Review"

    # --- Rule 1: Sudden Delinquency Jump ---
    # Zero missed payments in the past 9 months, but sudden DPD spike in recent months
    rule1_mask = (df["repay_missed_count"] == 0) & (df["repay_max_dpd"] >= 30)
    df.loc[rule1_mask, "is_anomaly"] = True
    df.loc[rule1_mask, "anomaly_rule"] = "Rule 1: Sudden Delinquency Jump"
    df.loc[rule1_mask, "anomaly_reason"] = (
        "Account had 0 missed payments historically, but recent DPD spiked suddenly."
    )
    df.loc[rule1_mask, "actual_value"] = df.loc[
        rule1_mask, "repay_max_dpd"
    ].apply(lambda v: f"Max DPD: {int(v)} days")
    df.loc[rule1_mask, "expected_value"] = "Historical baseline: 0 DPD"

    # --- Rule 2: Cohort Payment Shortfall Outlier (> 2.5 Std Dev above peer mean) ---
    threshold = df["cohort_mean_shortfall"] + (
        2.5 * df["cohort_std_shortfall"]
    )
    rule2_mask = (
        (~df["is_anomaly"])
        & (df["repay_avg_shortfall"] > threshold)
        & (df["repay_avg_shortfall"] > 50)
    )
    df.loc[rule2_mask, "is_anomaly"] = True
    df.loc[rule2_mask, "anomaly_rule"] = "Rule 2: Peer Shortfall Outlier"
    df.loc[rule2_mask, "anomaly_reason"] = (
        "Monthly payment shortfall is > 2.5 standard deviations above loan-type peers."
    )
    df.loc[rule2_mask, "actual_value"] = df.loc[
        rule2_mask, "repay_avg_shortfall"
    ].apply(lambda v: f"Shortfall: ${v:,.2f}")
    df.loc[rule2_mask, "expected_value"] = df.loc[
        rule2_mask, "cohort_mean_shortfall"
    ].apply(lambda v: f"Peer Avg: ${v:,.2f}")

    # --- Rule 3: External Bureau Divergence ---
    # Currently paying normally on this loan (0 missed payments), but has severe external write-offs or >60 DPD
    rule3_mask = (
        (~df["is_anomaly"])
        & (df["repay_missed_count"] == 0)
        & (
            (df["bureau_written_off_count"] >= 1)
            | (df["bureau_max_dpd"] >= 60)
        )
    )
    df.loc[rule3_mask, "is_anomaly"] = True
    df.loc[rule3_mask, "anomaly_rule"] = "Rule 3: Hidden Bureau Distress"
    df.loc[rule3_mask, "anomaly_reason"] = (
        "Performing normally on internal loan, but has severe distress reported on external credit."
    )
    df.loc[rule3_mask, "actual_value"] = df.loc[rule3_mask].apply(
        lambda r: f"Ext DPD: {int(r['bureau_max_dpd'])}, Write-offs: {int(r['bureau_written_off_count'])}",
        axis=1,
    )
    df.loc[rule3_mask, "expected_value"] = (
        "Expected: Clean external bureau standing"
    )

    return df


def get_anomaly_documentation():
    """Return formal governance documentation required for defense."""
    return {
        "Trigger": "Breach of any of 3 statistical criteria (sudden DPD spike >30, Z-score shortfall >2.5, or cross-institution divergence).",
        "False Positive": "A borrower experiencing an administrative payroll delay or banking holiday who quickly cures the overdue amount within 5 days.",
        "False Negative": "A strategically delinquent borrower who consistently pays small, partial amounts just high enough to avoid breaching the 2.5 SD shortfall threshold.",
        "Human Review Step": "Risk analysts inspect the review queue, view peer benchmarks, and click 'Confirm Anomaly' (routes to collections) or 'Dismiss Alert' (logs explanatory note).",
    }
