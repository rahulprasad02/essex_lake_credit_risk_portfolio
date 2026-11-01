"""
src/ask_portfolio.py
Mandatory AI Feature 2: Grounded "Ask the Portfolio" Engine.
Executes deterministic analytics in Python first, then formats the answer via LLM.
Prevents hallucination on numerical figures.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from src.llm_provider import generate_llm_response, get_llm_client

load_dotenv()


def execute_grounded_analytics(query: str, df: pd.DataFrame):
    """
    Deterministic query handler: extracts intent, executes Python calculations
    directly against the DataFrame, and produces verified metrics.
    """
    q = query.lower().strip()

    # Query 1: Highest default rate by loan type
    if "loan type" in q and ("default" in q or "highest" in q or "risk" in q):
        stats = (
            df.groupby("loan_type")["target_default"]
            .mean()
            .sort_values(ascending=False)
        )
        highest_type = stats.index[0]
        highest_rate = stats.iloc[0] * 100
        lowest_type = stats.index[-1]
        lowest_rate = stats.iloc[-1] * 100
        details = ", ".join([f"{k}: {v*100:.1f}%" for k, v in stats.items()])
        return {
            "valid": True,
            "topic": "Default Rate by Loan Type",
            "source_columns": ["loan_type", "target_default"],
            "data_summary": (
                f"{highest_type} has the highest default rate at {highest_rate:.1f}%, "
                f"while {lowest_type} has the lowest at {lowest_rate:.1f}%. Full breakdown: {details}."
            ),
        }

    # Query 2: Unemployed or employment risk
    elif "unemployed" in q or "employment" in q:
        emp_stats = (
            df.groupby("employment_type")["target_default"]
            .mean()
            .sort_values(ascending=False)
        )
        unemployed_rate = emp_stats.get("Unemployed", 0.0) * 100
        salaried_rate = emp_stats.get("Salaried", 0.0) * 100
        details = ", ".join(
            [f"{k}: {v*100:.1f}%" for k, v in emp_stats.items()]
        )
        return {
            "valid": True,
            "topic": "Default Rate by Employment Type",
            "source_columns": ["employment_type", "target_default"],
            "data_summary": (
                f"Unemployed borrowers exhibit the highest default rate at {unemployed_rate:.1f}%, "
                f"compared to Salaried borrowers at {salaried_rate:.1f}%. Full breakdown: {details}."
            ),
        }

    # Query 3: Portfolio totals / exposure / overall default
    elif any(
        k in q
        for k in [
            "total exposure",
            "total loan",
            "overall default",
            "portfolio size",
            "how many loans",
        ]
    ):
        total_loans = len(df)
        total_exp = df["loan_amount"].sum()
        avg_loan = df["loan_amount"].mean()
        overall_def = df["target_default"].mean() * 100
        return {
            "valid": True,
            "topic": "Overall Portfolio Statistics",
            "source_columns": ["account_id", "loan_amount", "target_default"],
            "data_summary": (
                f"The portfolio contains exactly {total_loans:,} loans with a total exposure of "
                f"${total_exp:,.2f} (average loan size: ${avg_loan:,.2f}). The overall portfolio default rate is {overall_def:.2f}%."
            ),
        }

    # Query 4: External bureau write-offs impact
    elif "bureau" in q or "write-off" in q or "external" in q:
        bureau_stats = df.groupby("has_bureau_writeoff")[
            "target_default"
        ].mean()
        clean_rate = bureau_stats.get(0, 0.0) * 100
        wo_rate = bureau_stats.get(1, 0.0) * 100
        return {
            "valid": True,
            "topic": "Prior Bureau Write-Off Impact",
            "source_columns": ["bureau_written_off_count", "target_default"],
            "data_summary": (
                f"Borrowers with prior external bureau write-offs default at {wo_rate:.1f}%, "
                f"compared to {clean_rate:.1f}% for clean borrowers (a relative increase of {wo_rate/clean_rate - 1:.1%})."
            ),
        }

    # Query 5: Region or geographic distribution
    elif "region" in q or "geography" in q:
        reg_stats = (
            df.groupby("region")["target_default"]
            .mean()
            .sort_values(ascending=False)
        )
        details = ", ".join(
            [f"{k}: {v*100:.1f}%" for k, v in reg_stats.items()]
        )
        return {
            "valid": True,
            "topic": "Default Rate by Geographic Region",
            "source_columns": ["region", "target_default"],
            "data_summary": f"Default rates across regions: {details}.",
        }

    # Out of scope / Unsupported questions
    else:
        return {
            "valid": False,
            "topic": "Unsupported Query",
            "source_columns": [],
            "data_summary": (
                "This question cannot be answered from the available loan portfolio data. "
                "Supported topics include: default rates by loan type, employment risk, "
                "total exposure, bureau write-off impact, and regional breakdowns."
            ),
        }


def format_answer_with_llm(user_question: str, query_result: dict) -> str:
    """
    Format the deterministic calculation into plain English via LLM if provider is active;
    otherwise use the grounded template directly.
    """
    if not query_result["valid"]:
        return query_result["data_summary"]

    _, provider_type = get_llm_client()

    # If offline, use deterministic fallback directly
    if provider_type == "offline":
        return (
            f"**Answer:** {query_result['data_summary']}\n\n"
            f"*Data Provenance: Calculated directly from columns `{', '.join(query_result['source_columns'])}`.*"
        )

    prompt = (
        f"You are a credit risk portfolio analyst. A user asked: '{user_question}'.\n"
        f"The underlying data analysis returned these exact facts: '{query_result['data_summary']}'.\n"
        f"Sources: {', '.join(query_result['source_columns'])}.\n"
        f"Write a concise, professional 2-3 sentence answer citing these exact figures. Do not invent numbers."
    )

    try:
        response_text = generate_llm_response(prompt)
        if response_text:
            return (
                response_text
                + f"\n\n*(Source: {', '.join(query_result['source_columns'])}) [Provider: {provider_type}]*"
            )
        else:
            return (
                f"**Answer:** {query_result['data_summary']}\n\n"
                f"*Data Provenance: Calculated directly from columns `{', '.join(query_result['source_columns'])}`.*"
            )
    except Exception as e:
        return (
            f"**Answer:** {query_result['data_summary']}\n\n"
            f"*(Source: {', '.join(query_result['source_columns'])}) [Fallback Mode: {str(e)}]*"
        )
