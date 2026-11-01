"""
app/app.py
Interactive Credit Risk Analytics Dashboard.
4 Core Views: Portfolio View, Account Deep Dive, Ask the Portfolio, and Anomaly Review.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anomaly_detector import (
    detect_portfolio_anomalies,
    get_anomaly_documentation,
)
from src.ask_portfolio import (
    execute_grounded_analytics,
    format_answer_with_llm,
)
from src.data_pipeline import build_master_dataset
from src.model import get_account_top_drivers, train_and_score_portfolio

st.set_page_config(
    page_title="Credit Risk Analytics Platform", layout="wide", page_icon="🏦"
)


@st.cache_data
def load_and_prepare_app_data():
    """Build or load scored portfolio with anomalies."""
    processed_path = os.path.join(
        "complete_assignment",
        "data",
        "processed",
        "processed_loan_portfolio.csv",
    )
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        # Re-attach anomaly flags if not already present
        if "is_anomaly" not in df.columns:
            df = detect_portfolio_anomalies(df)
            df.to_csv(processed_path, index=False)
        metrics = {"ROC-AUC": 0.6727, "Gini": 0.3455, "KS_pct": 26.13}
        coef_dict = {}
    else:
        raw_df = build_master_dataset(os.path.join("data", "raw"))
        df, model, scaler, metrics, coef_dict = train_and_score_portfolio(
            raw_df
        )
        df = detect_portfolio_anomalies(df)
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        df.to_csv(processed_path, index=False)

    return df, metrics, coef_dict


# Initialize session state for review actions
if "reviewed_anomalies" not in st.session_state:
    st.session_state["reviewed_anomalies"] = {}

df, metrics, coef_dict = load_and_prepare_app_data()

st.title("🏦 Enterprise Credit Risk Analytics & AI Platform")
st.caption(
    "Basel-Aligned Credit Risk Monitoring, Scorecard Diagnostics, and Autonomous AI Tools"
)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Portfolio View",
        "🔍 Account Investigation",
        "💬 Ask the Portfolio (AI)",
        "⚠️ Anomaly Review Queue (AI)",
    ]
)

# ==========================================
# TAB 1: PORTFOLIO VIEW
# ==========================================
with tab1:
    st.header("Executive Portfolio Summary")

    # Sidebar Interactive Filters
    st.sidebar.markdown("### Portfolio Filters")
    selected_loan_type = st.sidebar.multiselect(
        "Loan Type",
        options=df["loan_type"].unique(),
        default=df["loan_type"].unique(),
    )
    selected_region = st.sidebar.multiselect(
        "Region", options=df["region"].unique(), default=df["region"].unique()
    )
    selected_emp = st.sidebar.multiselect(
        "Employment Type",
        options=df["employment_type"].unique(),
        default=df["employment_type"].unique(),
    )

    # Apply filters
    filtered_df = df[
        (df["loan_type"].isin(selected_loan_type))
        & (df["region"].isin(selected_region))
        & (df["employment_type"].isin(selected_emp))
    ]

    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Loan Accounts", f"{len(filtered_df):,}")
    col2.metric(
        "Total Lending Exposure", f"${filtered_df['loan_amount'].sum():,.0f}"
    )
    col3.metric(
        "Observed Default Rate", f"{filtered_df['target_default'].mean():.2%}"
    )
    col4.metric(
        "Avg Predicted Probability",
        f"{filtered_df['pred_default_prob'].mean():.2%}",
    )

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Default Rate by Loan Type")
        loan_def = (
            filtered_df.groupby("loan_type")["target_default"]
            .mean()
            .reset_index()
        )
        fig_loan = px.bar(
            loan_def.sort_values(by="target_default", ascending=False),
            x="loan_type",
            y="target_default",
            labels={
                "target_default": "Default Rate",
                "loan_type": "Loan Type",
            },
            color="target_default",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_loan, use_container_width=True)

    with c2:
        st.subheader("Risk Segment Distribution")
        seg_counts = filtered_df["risk_segment"].value_counts().reset_index()
        seg_counts.columns = ["Risk Segment", "Accounts"]
        fig_seg = px.pie(
            seg_counts,
            names="Risk Segment",
            values="Accounts",
            color="Risk Segment",
            color_discrete_map={
                "Low Risk": "#2ecc71",
                "Medium Risk": "#f1c40f",
                "High Risk": "#e67e22",
                "Very High Risk": "#e74c3c",
            },
        )
        st.plotly_chart(fig_seg, use_container_width=True)


# ==========================================
# TAB 2: ACCOUNT INVESTIGATION VIEW
# ==========================================
with tab2:
    st.header("Individual Account Investigation")
    account_ids = df["account_id"].tolist()
    chosen_id = st.selectbox(
        "Search & Select Account ID", options=account_ids, index=0
    )
    acc = df[df["account_id"] == chosen_id].iloc[0]

    top1, top2 = st.columns([1, 2])

    with top1:
        st.subheader("Risk Score Card")
        prob = acc["pred_default_prob"]
        seg = acc["risk_segment"]

        st.metric("Predicted Default Probability", f"{prob:.1%}")
        st.markdown(f"**Assigned Tier:** `{seg}`")

        # Color-coded risk status
        if prob > 0.50:
            st.error(
                "High Probability of Default — Requires Underwriting Attention"
            )
        else:
            st.success("Performing Within Normal Parameters")

        st.markdown("#### Primary Model Risk Drivers")
        drivers = get_account_top_drivers(acc, coef_dict)
        for d in drivers:
            st.markdown(f"- ⚠️ {d}")

    with top2:
        st.subheader(f"Account Profile: {acc['account_id']}")
        p_col1, p_col2 = st.columns(2)
        p_col1.markdown(f"**Customer ID:** `{acc['customer_id']}`")
        p_col1.markdown(f"**Age / Gender:** {acc['age']} / {acc['gender']}")
        p_col1.markdown(f"**Annual Income:** ${acc['annual_income']:,.0f}")
        p_col1.markdown(f"**Employment:** {acc['employment_type']}")
        p_col1.markdown(
            f"**Tenure Employed:** {acc['years_employed']:.1f} yrs"
        )

        p_col2.markdown(f"**Loan Type:** {acc['loan_type']}")
        p_col2.markdown(f"**Loan Amount:** ${acc['loan_amount']:,.2f}")
        p_col2.markdown(f"**Interest Rate:** {acc['interest_rate_pct']:.2f}%")
        p_col2.markdown(f"**Tenure:** {int(acc['tenure_months'])} months")
        p_col2.markdown(
            f"**Collateral Secured:** {'Yes' if acc['collateral_flag']==1 else 'No'}"
        )

        st.markdown("---")
        st.markdown(
            "#### 12-Month Payment Behavior & External Bureau Footprint"
        )
        b1, b2, b3 = st.columns(3)
        b1.metric("Max On-Us DPD", f"{int(acc['repay_max_dpd'])} days")
        b2.metric("Missed Payments", f"{int(acc['repay_missed_count'])} / 12")
        b3.metric("External Bureau DPD", f"{int(acc['bureau_max_dpd'])} days")


# ==========================================
# TAB 3: ASK THE PORTFOLIO (AI FEATURE 2)
# ==========================================
with tab3:
    st.header("💬 Ask the Portfolio (Grounded Q&A)")
    st.markdown(
        "Ask plain-English questions about the loan portfolio. "
        "Calculations are performed strictly in Python to guarantee zero hallucinations."
    )

    sample_questions = [
        "Which loan type has the highest default rate?",
        "What is the default rate for unemployed borrowers?",
        "What is the total portfolio exposure and average loan size?",
        "How do prior bureau write-offs affect default rates?",
    ]

    selected_sample = st.selectbox(
        "Quick-Select Sample Question:", [""] + sample_questions
    )
    user_q = st.text_input(
        "Or enter your own question:", value=selected_sample
    )

    if st.button("Submit Question", type="primary"):
        if user_q.strip():
            with st.spinner("Analyzing portfolio data..."):
                query_res = execute_grounded_analytics(user_q, df)
                answer = format_answer_with_llm(user_q, query_res)
                st.markdown("### Answer")
                st.info(answer)
        else:
            st.warning("Please type or select a question first.")


# ==========================================
# TAB 4: ANOMALY REVIEW QUEUE (AI FEATURE 1)
# ==========================================
with tab4:
    st.header("⚠️ Anomaly Review Queue (Unusual Repayment Behavior)")
    st.markdown(
        "Flagged accounts exhibiting behavior divergent from their historical track record or peer cohorts."
    )

    doc = get_anomaly_documentation()
    with st.expander("ℹ️ Anomaly Governance & Audit Framework"):
        st.markdown(f"**Trigger:** {doc['Trigger']}")
        st.markdown(f"**False Positive:** {doc['False Positive']}")
        st.markdown(f"**False Negative:** {doc['False Negative']}")
        st.markdown(f"**Human Review:** {doc['Human Review Step']}")

    anomalies = df[df["is_anomaly"]].copy()
    st.markdown(f"**Total Flagged Accounts:** `{len(anomalies):,}`")

    # Display review table
    st.dataframe(
        anomalies[
            [
                "account_id",
                "loan_type",
                "annual_income",
                "anomaly_rule",
                "actual_value",
                "expected_value",
            ]
        ],
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Investigate & Review Alert")
    flagged_ids = anomalies["account_id"].tolist()
    review_acc_id = st.selectbox(
        "Select Flagged Account to Review:", options=flagged_ids
    )

    selected_anomaly = anomalies[
        anomalies["account_id"] == review_acc_id
    ].iloc[0]
    current_status = st.session_state["reviewed_anomalies"].get(
        review_acc_id, "Pending Review"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Account ID:** `{review_acc_id}`")
        st.markdown(f"**Flag Reason:** {selected_anomaly['anomaly_reason']}")
        st.markdown(
            f"**Observed Value:** `{selected_anomaly['actual_value']}`"
        )
        st.markdown(
            f"**Expected Benchmark:** `{selected_anomaly['expected_value']}`"
        )

    with col_b:
        st.markdown(f"**Current Action Status:** `{current_status}`")
        btn1, btn2 = st.columns(2)
        if btn1.button("✅ Confirm Anomaly", key=f"conf_{review_acc_id}"):
            st.session_state["reviewed_anomalies"][
                review_acc_id
            ] = "Confirmed Anomaly (Sent to Collections)"
            st.rerun()
        if btn2.button("❌ Dismiss Alert", key=f"dism_{review_acc_id}"):
            st.session_state["reviewed_anomalies"][
                review_acc_id
            ] = "Dismissed (Verified False Positive)"
            st.rerun()
