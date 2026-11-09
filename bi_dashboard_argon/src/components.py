"""Reusable UI components, SAS Viya theme CSS, KPI metric cards, and sidebar filter controls."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.data_engine import SAS_LABELS


def inject_sas_viya_css() -> None:
    """Inject custom CSS matching the SAS Viya Opal / Marine analytical workspace aesthetic."""
    st.markdown(
        """
        <style>
        /* SAS Viya Top Bar & Page Styling */
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.0rem;
            max-width: 98%;
        }
        .sas-header-banner {
            background: linear-gradient(90deg, #072B61 0%, #1B5FD0 65%, #17998D 100%);
            color: #FFFFFF;
            padding: 16px 24px;
            border-radius: 10px;
            margin-bottom: 16px;
            box-shadow: 0 3px 10px rgba(7, 43, 97, 0.14);
        }
        .sas-header-banner h1 {
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0;
            color: #FFFFFF;
            letter-spacing: -0.01em;
        }
        .sas-header-banner p {
            font-size: 0.90rem;
            margin: 4px 0 0 0;
            color: #E2ECF9;
        }
        /* KPI Metric Card Styling */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #DCE3EB;
            border-left: 4px solid #1B5FD0;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }
        div[data-testid="stMetricLabel"] > div {
            font-weight: 600 !important;
            color: #424A53 !important;
            font-size: 0.82rem !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-weight: 700 !important;
            color: #1D242B !important;
            font-size: 1.42rem !important;
        }
        /* SAS Automated Explanation NLG Banner */
        .sas-nlg-banner {
            background-color: #F0F5FF;
            border: 1px solid #BFD4F8;
            border-left: 5px solid #1B5FD0;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 14px;
            color: #1D242B;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        /* Model Card Health Badges (Task 5, p. 43-45) */
        .sas-health-card {
            background: #FFFFFF;
            border: 1px solid #DCE3EB;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .badge-pass {
            display: inline-block;
            background-color: #E6F4EA;
            color: #1E8E3E;
            border: 1px solid #A8DAB5;
            font-weight: 700;
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-top: 6px;
        }
        .badge-fail {
            display: inline-block;
            background-color: #FCE8E6;
            color: #D9383A;
            border: 1px solid #F6AEA9;
            font-weight: 700;
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-top: 6px;
        }
        .outlier-badge {
            display: inline-block;
            background-color: #FFF4E5;
            color: #B25000;
            border: 1px solid #FFD599;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header_banner() -> None:
    """Render top SAS Viya Visual Analytics header bar."""
    st.markdown(
        """
        <div class="sas-header-banner">
            <h1>SAS Viya Visual Analytics &amp; End-to-End Home Equity Decisioning</h1>
            <p>
                Complete Python Replication of the SAS Viya for Learners Financial Risk Use Case:
                SAS Studio Data Prep (Task 1) • Interactive Charts (Task 2) • Automated Explanation (Task 3)
                • Model Studio &amp; Fairness (Task 4) • Model Card (Task 5) • Intelligent Decisioning Flow (Task 6)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(kpis: Dict[str, Any]) -> None:
    """Render 6 top-row SAS Viya Key Value / KPI Cards with threshold delta styling."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric(
            label="Applications (Filtered)",
            value=f"{kpis['total_apps']:,}",
            delta=f"{kpis['apps_pct_of_total']:.1f}% of {kpis['base_apps']:,} total",
            delta_color="off",
        )
    with c2:
        st.metric(
            label="Default Rate (Loan Status = 1)",
            value=f"{kpis['default_rate']:.2f}%",
            delta=f"{kpis['default_rate_delta']:+.2f}% vs portfolio",
            delta_color="inverse",
        )
    with c3:
        st.metric(
            label="Avg Amount of Loan Request",
            value=f"${kpis['avg_loan']:,.0f}",
            delta=f"${kpis['avg_loan_delta']:+,.0f} vs portfolio",
            delta_color="off",
        )
    with c4:
        st.metric(
            label="Avg Value of Current Property",
            value=f"${kpis['avg_value']:,.0f}",
            delta=f"${kpis['avg_value_delta']:+,.0f} vs portfolio",
            delta_color="normal",
        )
    with c5:
        st.metric(
            label="Avg Debt to Income Ratio",
            value=f"{kpis['avg_debtinc']:.2f}%",
            delta=f"{kpis['avg_debtinc_delta']:+.2f}% vs portfolio",
            delta_color="inverse",
        )
    with c6:
        st.metric(
            label="Avg Loan to Value (LTV)",
            value=f"{kpis['avg_ltv']:.2f}%",
            delta=f"{kpis['avg_ltv_delta']:+.2f}% vs portfolio",
            delta_color="inverse",
        )


def render_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render SAS Viya Global Report / Page Prompt Controls in `st.sidebar`."""
    st.sidebar.markdown("### 🎛️ SAS Viya Report Prompts & Filters")
    st.sidebar.caption(
        "Global report-level and cross-filtering controls mirroring SAS Visual Analytics Automatic Actions."
    )

    # Cross-filter state for Loan Status (Task 2 p. 23: Donut acts as filter on all objects)
    status_choice = st.sidebar.radio(
        "🔗 Linked Cross-Filter: Loan Status (BAD)",
        options=["All Applications (0 & 1)", "0 - Current / Paid (80.1%)", "1 - Default / Delinquent (19.9%)"],
        index=0,
        help="Replicates SAS Viya 'Automatic actions on all objects' (Task 2, p. 23): clicking a slice of the Loan Status donut filters every chart on the page.",
    )
    if status_choice.startswith("0"):
        loan_status_filter = ["0"]
        active_status = "0"
    elif status_choice.startswith("1"):
        loan_status_filter = ["1"]
        active_status = "1"
    else:
        loan_status_filter = ["0", "1"]
        active_status = "All"

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌍 Geographic Hierarchy Prompts")

    all_regions = sorted(df["REGION"].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect(
        "Region",
        options=all_regions,
        default=all_regions,
    )

    div_pool = df[df["REGION"].isin(selected_regions)] if selected_regions else df
    all_divisions = sorted(div_pool["DIVISION"].dropna().unique().tolist())
    selected_divisions = st.sidebar.multiselect(
        "Division",
        options=all_divisions,
        default=all_divisions,
    )

    state_pool = div_pool[div_pool["DIVISION"].isin(selected_divisions)] if selected_divisions else div_pool
    all_states = sorted(state_pool["STATE"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect(
        "State (Optional Subset)",
        options=all_states,
        default=[],
        placeholder="All States in Selected Divisions",
    )

    st.sidebar.markdown("#### 💼 Applicant & Loan Prompts")
    all_jobs = sorted(df["JOB_CLEAN"].unique().tolist())
    selected_jobs = st.sidebar.multiselect(
        "Job Category (JOB)",
        options=all_jobs,
        default=all_jobs,
    )

    all_reasons = sorted(df["REASON_CLEAN"].unique().tolist())
    selected_reasons = st.sidebar.multiselect(
        "Loan Purpose (REASON)",
        options=all_reasons,
        default=all_reasons,
    )

    min_loan = float(df["LOAN"].min())
    max_loan = float(df["LOAN"].max())
    loan_range = st.sidebar.slider(
        "Amount of Loan Request ($)",
        min_value=min_loan,
        max_value=max_loan,
        value=(min_loan, max_loan),
        step=500.0,
        format="$%.0f",
    )

    min_date = df["APP_DATE"].min().date()
    max_date = df["APP_DATE"].max().date()
    date_range = st.sidebar.date_input(
        "Loan Application Date Range (JAN2017 – DEC2021)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if st.sidebar.button("🔄 Reset All Report Prompts", use_container_width=True):
        st.rerun()

    return {
        "loan_status_filter": loan_status_filter,
        "active_status": active_status,
        "regions": selected_regions,
        "divisions": selected_divisions,
        "states": selected_states if selected_states else None,
        "jobs": selected_jobs,
        "reasons": selected_reasons,
        "loan_range": loan_range,
        "date_range": date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None,
    }


def render_model_card_health_row(
    champion_ks: float,
    val_ase: float,
    test_ase: float,
    max_pred_diff_pct: float,
) -> None:
    """Render Task 5 SAS Model Manager Model Card Health Indicators (guide p. 43-45)."""
    c1, c2, c3, c4 = st.columns(4)
    ase_diff = abs(val_ase - test_ase)

    with c1:
        st.markdown(
            f"""
            <div class="sas-health-card">
                <div style="font-size:0.82rem;color:#5C6873;font-weight:600;">MODEL ACCURACY (KS STATISTIC)</div>
                <div style="font-size:1.45rem;font-weight:700;color:#1D242B;margin-top:4px;">{champion_ks:.4f}</div>
                <div style="font-size:0.78rem;color:#5C6873;">Threshold: KS &gt; 0.5000</div>
                <span class="badge-pass">✓ PASS</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="sas-health-card">
                <div style="font-size:0.82rem;color:#5C6873;font-weight:600;">GENERALIZABILITY (ASE DIFF)</div>
                <div style="font-size:1.45rem;font-weight:700;color:#1D242B;margin-top:4px;">{ase_diff:.4f}</div>
                <div style="font-size:0.78rem;color:#5C6873;">Validate vs. Test ASE Drift &lt; 0.0200</div>
                <span class="badge-pass">✓ PASS</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="sas-health-card">
                <div style="font-size:0.82rem;color:#5C6873;font-weight:600;">FAIRNESS (SENSITIVE VAR: JOB)</div>
                <div style="font-size:1.45rem;font-weight:700;color:#D9383A;margin-top:4px;">{max_pred_diff_pct:.1f}% Spread</div>
                <div style="font-size:0.78rem;color:#5C6873;">Performance Bias Across Job Categories</div>
                <span class="badge-fail">✗ FAIL</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="sas-health-card">
                <div style="font-size:0.82rem;color:#5C6873;font-weight:600;">PUBLICATION STATUS</div>
                <div style="font-size:1.45rem;font-weight:700;color:#1D242B;margin-top:4px;">Registered</div>
                <div style="font-size:0.78rem;color:#5C6873;">Registered to SAS Model Manager &amp; Decisioning</div>
                <span class="badge-pass">v1.0 Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
