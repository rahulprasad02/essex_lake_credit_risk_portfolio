"""
src/data_pipeline.py
End-to-end data ingestion, cleaning, and aggregation pipeline.
Produces a clean dataset with grain: 1 row per account_id (~14,685 rows).
"""

import os

import numpy as np
import pandas as pd


def load_raw_data(data_dir: str = os.path.join("data", "raw")):
    """Load all 5 raw CSV files from data/raw/."""
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    loan_accounts = pd.read_csv(os.path.join(data_dir, "loan_accounts.csv"))
    bureau_credit = pd.read_csv(os.path.join(data_dir, "bureau_credit.csv"))
    repayment_history = pd.read_csv(
        os.path.join(data_dir, "repayment_history.csv")
    )
    collections_events = pd.read_csv(
        os.path.join(data_dir, "collections_events.csv")
    )
    return (
        customers,
        loan_accounts,
        bureau_credit,
        repayment_history,
        collections_events,
    )


def clean_and_impute(
    customers: pd.DataFrame,
    loan_accounts: pd.DataFrame,
    bureau_credit: pd.DataFrame,
):
    """Clean categorical values and impute skewed attributes with robust medians."""
    # 1. Standardize inconsistent gender codes
    customers["gender"] = customers["gender"].replace(
        {"M": "Male", "F": "Female"}
    )

    # 2. Impute skewed customer demographics using median
    customers["years_employed"] = customers["years_employed"].fillna(
        customers["years_employed"].median()
    )
    customers["num_dependents"] = customers["num_dependents"].fillna(
        customers["num_dependents"].median()
    )
    customers["years_at_residence"] = customers["years_at_residence"].fillna(
        customers["years_at_residence"].median()
    )

    # 3. Impute loan interest rate by cohort median (per loan_type)
    loan_accounts["interest_rate_pct"] = loan_accounts.groupby("loan_type")[
        "interest_rate_pct"
    ].transform(lambda grp: grp.fillna(grp.median()))

    # 4. In bureau_credit, missing DPD means performing line (0 DPD)
    bureau_credit["days_past_due"] = bureau_credit["days_past_due"].fillna(0)

    return customers, loan_accounts, bureau_credit


def aggregate_child_tables(
    bureau_credit: pd.DataFrame,
    repayment_history: pd.DataFrame,
    collections_events: pd.DataFrame,
):
    """Aggregate one-to-many child tables to prevent Cartesian explosion."""
    # 1. Bureau summary by customer_id
    bureau_agg = (
        bureau_credit.groupby("customer_id")
        .agg(
            bureau_total_lines=("bureau_id", "count"),
            bureau_active_lines=(
                "account_status",
                lambda s: (s == "Active").sum(),
            ),
            bureau_written_off_count=(
                "account_status",
                lambda s: (s == "Written-Off").sum(),
            ),
            bureau_total_sanctioned=("sanctioned_amount", "sum"),
            bureau_total_outstanding=("outstanding_amount", "sum"),
            bureau_max_dpd=("days_past_due", "max"),
        )
        .reset_index()
    )

    # 2. Repayment history summary by account_id
    repayment_history["shortfall"] = (
        repayment_history["emi_due"] - repayment_history["emi_paid"]
    )
    repayment_history["is_recent_missed"] = (
        (repayment_history["payment_month"] >= 10)
        & (repayment_history["payment_status"] == "Missed")
    ).astype(int)

    repay_agg = (
        repayment_history.groupby("account_id")
        .agg(
            repay_total_months=("payment_month", "count"),
            repay_missed_count=(
                "payment_status",
                lambda s: (s == "Missed").sum(),
            ),
            repay_recent_missed=("is_recent_missed", "sum"),
            repay_late_count=("payment_status", lambda s: (s == "Late").sum()),
            repay_max_dpd=("days_past_due", "max"),
            repay_avg_shortfall=("shortfall", "mean"),
        )
        .reset_index()
    )
    repay_agg["repay_missed_pct"] = (
        repay_agg["repay_missed_count"] / repay_agg["repay_total_months"]
    )

    # 3. Collections events summary by account_id
    collections_agg = (
        collections_events.groupby("account_id")
        .agg(
            collections_actions_count=("collection_event_id", "count"),
            total_amount_recovered=("amount_recovered", "sum"),
        )
        .reset_index()
    )

    return bureau_agg, repay_agg, collections_agg


def build_master_dataset(
    data_dir: str = os.path.join("data", "raw")
) -> pd.DataFrame:
    """Execute complete data preparation sequence and return master DataFrame."""
    cust, loans, bureau, repay, coll = load_raw_data(data_dir)
    cust, loans, bureau = clean_and_impute(cust, loans, bureau)
    bureau_agg, repay_agg, coll_agg = aggregate_child_tables(
        bureau, repay, coll
    )

    # Left-join customers with bureau summary
    cust_merged = pd.merge(cust, bureau_agg, on="customer_id", how="left")
    bureau_cols = [
        "bureau_total_lines",
        "bureau_active_lines",
        "bureau_written_off_count",
        "bureau_total_sanctioned",
        "bureau_total_outstanding",
        "bureau_max_dpd",
    ]
    cust_merged[bureau_cols] = cust_merged[bureau_cols].fillna(0)

    # Merge loan_accounts with customer profiles
    master_df = pd.merge(loans, cust_merged, on="customer_id", how="inner")

    # Merge account-level repayment and collections summaries
    master_df = pd.merge(master_df, repay_agg, on="account_id", how="left")
    master_df = pd.merge(master_df, coll_agg, on="account_id", how="left")

    # Fill sparse events with zero
    master_df["collections_actions_count"] = master_df[
        "collections_actions_count"
    ].fillna(0)
    master_df["total_amount_recovered"] = master_df[
        "total_amount_recovered"
    ].fillna(0)

    # Derived categorical segments for portfolio analysis
    master_df["age_group"] = pd.cut(
        master_df["age"],
        bins=[18, 30, 45, 60, 100],
        labels=[
            "18-30 (Young)",
            "31-45 (Prime)",
            "46-60 (Mature)",
            "60+ (Senior)",
        ],
    )
    master_df["income_tier"] = pd.qcut(
        master_df["annual_income"],
        q=4,
        labels=[
            "Low (<25%)",
            "Medium-Low (25-50%)",
            "Medium-High (50-75%)",
            "High (>75%)",
        ],
    )
    master_df["has_bureau_writeoff"] = (
        master_df["bureau_written_off_count"] > 0
    ).astype(int)

    return master_df
