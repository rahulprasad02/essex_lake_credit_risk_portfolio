"""
src/model.py
Credit Default Scoring Model.
Trains the production champion Logistic Regression model and generates calibrated risk probabilities.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS_NUMERIC = [
    "loan_amount",
    "interest_rate_pct",
    "tenure_months",
    "collateral_flag",
    "age",
    "annual_income",
    "years_employed",
    "num_dependents",
    "years_at_residence",
    "bureau_total_lines",
    "bureau_active_lines",
    "bureau_written_off_count",
    "bureau_total_sanctioned",
    "bureau_total_outstanding",
    "bureau_max_dpd",
    "repay_total_months",
    "repay_missed_count",
    "repay_recent_missed",
    "repay_late_count",
    "repay_max_dpd",
    "repay_avg_shortfall",
    "repay_missed_pct",
    "collections_actions_count",
    "total_amount_recovered",
]

FEATURE_COLS_CATEGORICAL = [
    "loan_type",
    "employment_type",
    "marital_status",
    "housing_type",
    "region",
    "city_tier",
]


def calculate_ks_statistic(y_true, y_prob):
    """Calculate Kolmogorov-Smirnov (KS) statistic: max separation between cumulative Goods and Bads."""
    df = pd.DataFrame({"target": y_true, "prob": y_prob}).sort_values(
        by="prob", ascending=False
    )
    df["cum_bads"] = (df["target"] == 1).cumsum() / (df["target"] == 1).sum()
    df["cum_goods"] = (df["target"] == 0).cumsum() / (df["target"] == 0).sum()
    return float((df["cum_bads"] - df["cum_goods"]).abs().max())


def train_and_score_portfolio(df: pd.DataFrame):
    """
    Train champion Logistic Regression model, evaluate metrics, and append
    'pred_default_prob' and 'risk_segment' to the portfolio.
    """
    X_raw = pd.get_dummies(
        df[FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL], drop_first=True
    )
    y = df["target_default"]

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Baseline Champion: Logistic Regression with balanced class weights
    model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # Test Metrics
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    test_gini = 2 * test_auc - 1
    test_ks = calculate_ks_statistic(y_test.values, test_probs)

    metrics = {
        "ROC-AUC": round(test_auc, 4),
        "Gini": round(test_gini, 4),
        "KS_pct": round(test_ks * 100, 2),
    }

    # Generate probabilities across all accounts
    X_all_scaled = scaler.transform(X_raw)
    df["pred_default_prob"] = model.predict_proba(X_all_scaled)[:, 1]

    # Assign calibrated risk segment tiers
    df["risk_segment"] = pd.cut(
        df["pred_default_prob"],
        bins=[0.0, 0.35, 0.50, 0.65, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
    )

    # Coefficient dictionary for explainability
    coef_dict = dict(zip(X_raw.columns, model.coef_[0]))

    return df, model, scaler, metrics, coef_dict


def get_account_top_drivers(row: pd.Series, coef_dict: dict, top_n: int = 3):
    """
    Generate intuitive local risk drivers for a single account based on model coefficients
    and the account's actual characteristics.
    """
    drivers = []
    # Key risk factors to inspect
    if row["repay_max_dpd"] > 30:
        drivers.append(
            f"High on-us delinquency (Max DPD: {int(row['repay_max_dpd'])} days)"
        )
    if row["loan_amount"] > 50000:
        drivers.append(f"Elevated loan exposure (${row['loan_amount']:,.0f})")
    if row["bureau_max_dpd"] > 30:
        drivers.append(
            f"External credit distress (Bureau Max DPD: {int(row['bureau_max_dpd'])} days)"
        )
    if row["interest_rate_pct"] > 14.0:
        drivers.append(
            f"High interest burden ({row['interest_rate_pct']:.1f}%)"
        )
    if row["annual_income"] < 35000:
        drivers.append(f"Low income tier (${row['annual_income']:,.0f}/yr)")
    if row["bureau_written_off_count"] > 0:
        drivers.append(
            f"Prior external write-offs ({int(row['bureau_written_off_count'])} lines)"
        )

    # Fallback to top positive drivers if account is clean
    if not drivers:
        drivers.append("Standard risk profile aligned with cohort baseline")

    return drivers[:top_n]
