"""Plotly figure generators mirroring SAS Viya Visual Analytics objects and themes.

Every chart replicates a specific visual object documented in `guide.pdf`:
- Task 2 (Page 1 `Interactive charts`):
  1. Donut chart (`Frequency of Loan Status`) with `Percent of total` outside labels
     styled in 16pt Bold Deep Purple (`#4B1E78`) and center total KPI.
  2. Histogram (`Frequency of Amount of Loan Request`) with SAS VA binning and hover tooltips.
  3. Correlation Matrix (`Correlation of Selected Measures`) across all 11 continuous
     measures with Weak-to-Strong blue gradient and outlier flags.
  4. Time Series Line Chart (`Frequency of Loan Application Date`) formatted by `MMMYYYY`.
  5. Grouped Bar Chart (`Amount Due on Existing Mortgage, Value of Current Property by Job Category`, Fig. 23 p. 16).
  6. Geographic US State Choropleth & Regional Hierarchy Sunburst (`Region -> Division -> State -> City`).
- Task 3 (Page 2 `Explain Loan Status`):
  7. Relative Importance Horizontal Bar Chart (`What factors are most related to Loan Status?`).
  8. Factor Relationship Stacked Histogram / Grouped Bar Chart (`What is the relationship between Loan Status and [Factor]?`).
- Task 4 & 5 (Page 3 `Model Studio & Fairness`):
  9. ROC Curves & Cumulative Lift Charts across Gradient Boosting, Random Forest, and Logistic Regression.
  10. Sensitive Variable `JOB` Fairness & Bias Multi-Bar and Parity Charts.
- Task 6 (Page 4 `Intelligent Decisioning Flow`):
  11. Sankey / Funnel Flow Chart of the `Loan Decision (Final)` pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_engine import CONTINUOUS_MEASURES, SAS_LABELS

# SAS Viya Visual Analytics Opal / Marine Palette
SAS_PRIMARY_BLUE = "#1B5FD0"
SAS_TEAL = "#17998D"
SAS_ORANGE = "#E66C37"
SAS_PURPLE_LABEL = "#4B1E78"  # Exact 16pt bold deep purple requested on guide p. 22
SAS_RED_ALERT = "#D9383A"
SAS_GREEN_OK = "#1E8E3E"
SAS_SLATE_GRAY = "#5C6873"
SAS_GRID_COLOR = "#E6EAF0"

STATUS_COLORS = {
    "0": "#2A6FDB",  # SAS Viya Blue for Loan Status = 0
    "1": "#E06030",  # SAS Viya Coral/Orange for Loan Status = 1
}


def _apply_sas_viya_layout(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    height: int = 380,
    show_legend: bool = True,
) -> go.Figure:
    """Apply clean SAS Viya Visual Analytics typography, margins, and container styling."""
    full_title = (
        f"<b>{title}</b><br><span style='font-size:12px;color:#5C6873;font-weight:normal'>{subtitle}</span>"
        if subtitle
        else f"<b>{title}</b>"
    )
    fig.update_layout(
        title=dict(
            text=full_title,
            x=0.01,
            y=0.96,
            xanchor="left",
            yanchor="top",
            font=dict(size=15, color="#1D242B", family="Inter, Segoe UI, sans-serif"),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#1D242B"),
        height=height,
        margin=dict(l=48, r=28, t=64 if subtitle else 52, b=46),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#1D242B",
            font_size=12,
            font_family="Inter, Segoe UI, sans-serif",
            font_color="#FFFFFF",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#D0D7DE",
        tickfont=dict(size=11, color="#424A53"),
        title_font=dict(size=12, color="#1D242B"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=SAS_GRID_COLOR,
        zeroline=False,
        linecolor="#D0D7DE",
        tickfont=dict(size=11, color="#424A53"),
        title_font=dict(size=12, color="#1D242B"),
    )
    return fig


def create_loan_status_donut(df: pd.DataFrame, active_status: str = "All") -> go.Figure:
    """Task 2 Object 1 (Top-Left, guide p. 20-23): Donut chart `Frequency of Loan Status`.

    Replicates the exact SAS Viya customizations on p. 22:
    - Donut style with center KPI (`6K` / exact filtered total).
    - Data labels showing `Actual values` and `Percentage of total` (`80.1%` vs `19.9%`).
    - Data label font styled in **16pt Bold Deep Purple** (`#4B1E78`).
    """
    counts = (
        df.groupby("LOAN_STATUS", as_index=False)
        .size()
        .rename(columns={"size": "Frequency"})
        .sort_values("LOAN_STATUS")
    )
    total_n = int(counts["Frequency"].sum())
    counts["Percent"] = (counts["Frequency"] / total_n * 100.0) if total_n else 0.0
    counts["DisplayLabel"] = counts["LOAN_STATUS"].map(
        {"0": "0 (Current / Paid)", "1": "1 (Default / Delinquent)"}
    )

    pull_vals = []
    for s in counts["LOAN_STATUS"]:
        if active_status in ("0", "1") and s == active_status:
            pull_vals.append(0.06)
        else:
            pull_vals.append(0.0)

    colors = [STATUS_COLORS.get(s, SAS_PRIMARY_BLUE) for s in counts["LOAN_STATUS"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["DisplayLabel"],
                values=counts["Frequency"],
                hole=0.58,
                pull=pull_vals,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2.5)),
                textinfo="percent+value",
                textposition="outside",
                textfont=dict(
                    size=16,
                    color=SAS_PURPLE_LABEL,
                    family="Inter, Segoe UI, Arial Black, sans-serif",
                ),
                hovertemplate=(
                    "<b>Loan Status: %{label}</b><br>"
                    "Frequency: %{value:,}<br>"
                    "Share of Total: %{percent}<extra></extra>"
                ),
                sort=False,
            )
        ]
    )

    center_text = f"{total_n / 1000:.1f}K" if total_n >= 1000 else f"{total_n:,}"
    fig.add_annotation(
        text=f"<b>{center_text}</b><br><span style='font-size:11px;color:#5C6873'>Applications<br>({total_n:,})</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="#1D242B"),
    )

    _apply_sas_viya_layout(
        fig,
        title="Frequency of Loan Status",
        subtitle="Donut Chart • Outside Labels: 16pt Bold Deep Purple (SAS VA Task 2, p. 22)",
        height=390,
        show_legend=True,
    )
    return fig


def create_loan_request_histogram(df: pd.DataFrame, color_by_status: bool = False) -> go.Figure:
    """Task 2 Object 2 (Top-Right, guide p. 20-21): Histogram `Frequency of Amount of Loan Request`."""
    if color_by_status:
        fig = px.histogram(
            df,
            x="LOAN",
            color="LOAN_STATUS",
            nbins=36,
            barmode="stack",
            color_discrete_map=STATUS_COLORS,
            labels={"LOAN": "Amount of Loan Request", "LOAN_STATUS": "Loan Status"},
        )
    else:
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=df["LOAN"],
                    nbinsx=36,
                    marker=dict(
                        color=SAS_PRIMARY_BLUE,
                        line=dict(color="#FFFFFF", width=1),
                    ),
                    hovertemplate=(
                        "<b>Amount of Loan Request:</b> $%{x:,.0f}<br>"
                        "<b>Frequency:</b> %{y:,} applications<extra></extra>"
                    ),
                    name="Loan Applications",
                )
            ]
        )

    mean_loan = float(df["LOAN"].mean()) if len(df) else 0.0
    median_loan = float(df["LOAN"].median()) if len(df) else 0.0

    fig.add_vline(
        x=mean_loan,
        line_dash="dash",
        line_color=SAS_ORANGE,
        line_width=2,
        annotation_text=f"Avg: ${mean_loan:,.0f}",
        annotation_position="top right",
        annotation_font=dict(size=11, color=SAS_ORANGE),
    )

    _apply_sas_viya_layout(
        fig,
        title="Frequency of Amount of Loan Request",
        subtitle=f"Distribution ($1,100 – $89,900) • Mean: ${mean_loan:,.0f} • Median: ${median_loan:,.0f}",
        height=390,
        show_legend=color_by_status,
    )
    fig.update_xaxes(title_text="Amount of Loan Request ($)", tickprefix="$", tickformat=",")
    fig.update_yaxes(title_text="Frequency (Count of Applications)", tickformat=",")
    return fig


def create_correlation_matrix_heatmap(
    corr_raw: pd.DataFrame,
    corr_abs: pd.DataFrame,
    show_signed: bool = False,
) -> go.Figure:
    """Task 2 Object 3 (Bottom-Left, guide p. 20-21): `Correlation of Selected Measures` across 11 measures."""
    mat = corr_raw if show_signed else corr_abs
    colorscale = (
        "RdBu_r"
        if show_signed
        else [
            [0.0, "#F2F5FA"],
            [0.25, "#C6D9F1"],
            [0.5, "#6FA0E3"],
            [0.75, "#2869C7"],
            [1.0, "#0B357A"],
        ]
    )
    zmin = -1.0 if show_signed else 0.0
    zmax = 1.0

    # Short axis labels with full names in hover
    short_map = {
        "Age of Oldest Credit Line (months)": "CLAGE (Credit Age)",
        "Amount Due on Existing Mortgage": "MORTDUE (Mort. Due)",
        "Amount of Loan Request": "LOAN (Loan Req.)",
        "Debt to Income Ratio": "DEBTINC (DTI Ratio)",
        "Loan to Value ratio": "LTV (Loan/Value)",
        "Number of Credit Lines": "CLNO (Credit Lines)",
        "Number of Delinquent Credit Lines": "DELINQ (Delinq.)",
        "Number of Derogatory Reports": "DEROG (Derog.)",
        "Number of Recent Credit Inquiries": "NINQ (Inquiries)",
        "Value of Current Property": "VALUE (Prop. Value)",
        "Years at Present Job": "YOJ (Years on Job)",
    }
    short_labels = [short_map.get(idx, idx) for idx in mat.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=mat.values,
            x=short_labels,
            y=short_labels,
            customdata=corr_raw.values,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            text=np.round(corr_raw.values, 2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=10),
            colorbar=dict(
                title="Pearson r" if show_signed else "|r| Strength",
                thickness=14,
                len=0.85,
            ),
            hovertemplate=(
                "<b>%{y}</b> vs <b>%{x}</b><br>"
                "Signed Correlation (r): %{customdata:.4f}<br>"
                "Absolute Strength (|r|): %{z:.4f}<extra></extra>"
            ),
        )
    )

    _apply_sas_viya_layout(
        fig,
        title="Correlation of Selected Measures (11 Continuous Items)",
        subtitle="Strongest correlation: Value of Current Property vs. Amount Due on Existing Mortgage (r = 0.88)",
        height=430,
        show_legend=False,
    )
    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def create_application_date_timeseries(df: pd.DataFrame, show_default_rate: bool = True) -> go.Figure:
    """Task 2 Object 4 (Bottom-Right, guide p. 20-21): `Frequency of Loan Application Date` (`MMMYYYY`)."""
    monthly = (
        df.groupby("APP_MONTH", as_index=False)
        .agg(
            Frequency=("BAD_NUM", "count"),
            Defaults=("BAD_NUM", "sum"),
            Avg_Loan=("LOAN", "mean"),
        )
        .sort_values("APP_MONTH")
    )
    monthly["MONYY7"] = monthly["APP_MONTH"].dt.strftime("%b%Y").str.upper()
    monthly["Default_Rate"] = np.where(
        monthly["Frequency"] > 0,
        monthly["Defaults"] / monthly["Frequency"] * 100.0,
        0.0,
    )

    if show_default_rate:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=monthly["APP_MONTH"],
                y=monthly["Frequency"],
                mode="lines+markers",
                name="Application Frequency",
                line=dict(color=SAS_PRIMARY_BLUE, width=2.5),
                marker=dict(size=5, color=SAS_PRIMARY_BLUE),
                customdata=monthly[["MONYY7", "Avg_Loan"]],
                hovertemplate=(
                    "<b>Loan Application Date: %{customdata[0]}</b><br>"
                    "Frequency: %{y:,} applications<br>"
                    "Avg Loan Request: $%{customdata[1]:,.0f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["APP_MONTH"],
                y=monthly["Default_Rate"],
                mode="lines",
                name="Monthly Default Rate (%)",
                line=dict(color=SAS_ORANGE, width=1.8, dash="dot"),
                hovertemplate="<b>Default Rate:</b> %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Frequency (Applications)", secondary_y=False)
        fig.update_yaxes(
            title_text="Default Rate (%)",
            secondary_y=True,
            showgrid=False,
            ticksuffix="%",
            range=[0, max(50.0, float(monthly["Default_Rate"].max() * 1.15)) if len(monthly) else 50.0],
        )
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=monthly["APP_MONTH"],
                    y=monthly["Frequency"],
                    mode="lines+markers",
                    name="Application Frequency",
                    line=dict(color=SAS_PRIMARY_BLUE, width=2.5),
                    marker=dict(size=5),
                    customdata=monthly["MONYY7"],
                    hovertemplate=(
                        "<b>Loan Application Date: %{customdata}</b><br>"
                        "Frequency: %{y:,} applications<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_yaxes(title_text="Frequency (Applications)")

    _apply_sas_viya_layout(
        fig,
        title="Frequency of Loan Application Date (MMMYYYY / MONYY7)",
        subtitle="Monthly Application Volume (JAN2017 – DEC2021) with Optional Default Rate Overlay",
        height=430,
        show_legend=show_default_rate,
    )
    fig.update_xaxes(title_text="Loan Application Date (MMMYYYY)", tickformat="%b %Y")
    return fig


def create_job_mortgage_value_chart(df: pd.DataFrame) -> go.Figure:
    """Replicate Figure 23 (guide p. 16): `Amount Due on Existing Mortgage, Value of Current Property by Job Category`."""
    grp = (
        df.groupby("JOB_CLEAN", as_index=False)
        .agg(
            Avg_MORTDUE=("MORTDUE", "mean"),
            Avg_VALUE=("VALUE", "mean"),
            Avg_LOAN=("LOAN", "mean"),
            Default_Rate=("BAD_NUM", lambda s: s.mean() * 100.0),
            Count=("BAD_NUM", "count"),
        )
        .sort_values("Avg_VALUE", ascending=False)
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=grp["JOB_CLEAN"],
            y=grp["Avg_MORTDUE"],
            name="Average Amount Due on Existing Mortgage",
            marker_color=SAS_PRIMARY_BLUE,
            hovertemplate="<b>%{x}</b><br>Avg Mortgage Due: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=grp["JOB_CLEAN"],
            y=grp["Avg_VALUE"],
            name="Average Value of Current Property",
            marker_color=SAS_TEAL,
            hovertemplate="<b>%{x}</b><br>Avg Property Value: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=grp["JOB_CLEAN"],
            y=grp["Default_Rate"],
            name="Default Rate (%)",
            mode="lines+markers",
            line=dict(color=SAS_ORANGE, width=2.5),
            marker=dict(size=8, color=SAS_ORANGE),
            hovertemplate="<b>%{x}</b><br>Default Rate: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(barmode="group")
    _apply_sas_viya_layout(
        fig,
        title="Amount Due on Existing Mortgage & Value of Current Property by Job Category",
        subtitle="SAS Visual Analytics Dual-Axis Bar-Line Chart (Guide Figure 23, p. 16)",
        height=390,
        show_legend=True,
    )
    fig.update_xaxes(title_text="Job Category (JOB)")
    fig.update_yaxes(title_text="Average Dollar Amount ($)", tickprefix="$", tickformat=",", secondary_y=False)
    fig.update_yaxes(title_text="Default Rate (%)", ticksuffix="%", showgrid=False, secondary_y=True)
    return fig


def create_us_state_choropleth(df: pd.DataFrame, metric: str = "Default Rate (%)") -> go.Figure:
    """Geographic US State Choropleth map for Workbook Geographic Exploration (guide p. 55)."""
    state_grp = (
        df.groupby(["STATE", "STATE_ABBR", "DIVISION", "REGION"], as_index=False)
        .agg(
            Applications=("BAD_NUM", "count"),
            Defaults=("BAD_NUM", "sum"),
            Avg_Loan=("LOAN", "mean"),
            Avg_Property_Value=("VALUE", "mean"),
            Avg_LTV_Pct=("LTV_PCT", "mean"),
        )
    )
    state_grp["Default Rate (%)"] = np.where(
        state_grp["Applications"] > 0,
        state_grp["Defaults"] / state_grp["Applications"] * 100.0,
        0.0,
    )
    state_grp["Average Loan Request ($)"] = state_grp["Avg_Loan"]
    state_grp["Application Volume"] = state_grp["Applications"]
    state_grp["Average LTV (%)"] = state_grp["Avg_LTV_Pct"]

    color_scale = "Blues" if metric != "Default Rate (%)" else "YlOrRd"

    fig = px.choropleth(
        state_grp,
        locations="STATE_ABBR",
        locationmode="USA-states",
        color=metric,
        scope="usa",
        color_continuous_scale=color_scale,
        hover_name="STATE",
        hover_data={
            "STATE_ABBR": False,
            "REGION": True,
            "DIVISION": True,
            "Applications": ":,",
            "Default Rate (%)": ":.1f",
            "Average Loan Request ($)": ":$,.0f",
            "Average LTV (%)": ":.1f",
        },
    )
    _apply_sas_viya_layout(
        fig,
        title=f"US Geographic Distribution by State — {metric}",
        subtitle="Hierarchy: Region → Division → State → City (51 States/DC, 858 Proper-Cased Cities)",
        height=400,
        show_legend=False,
    )
    fig.update_geos(bgcolor="#FFFFFF", lakecolor="#FFFFFF", landcolor="#F4F6F9")
    return fig


def create_regional_hierarchy_treemap(df: pd.DataFrame) -> go.Figure:
    """Interactive Region -> Division -> State drill-down Treemap colored by Default Rate (%)."""
    grp = (
        df.groupby(["REGION", "DIVISION", "STATE"], as_index=False)
        .agg(
            Applications=("BAD_NUM", "count"),
            Defaults=("BAD_NUM", "sum"),
            Avg_Loan=("LOAN", "mean"),
        )
    )
    grp["Default Rate (%)"] = np.where(
        grp["Applications"] > 0,
        grp["Defaults"] / grp["Applications"] * 100.0,
        0.0,
    )

    fig = px.treemap(
        grp,
        path=[px.Constant("All US Regions"), "REGION", "DIVISION", "STATE"],
        values="Applications",
        color="Default Rate (%)",
        color_continuous_scale="RdYlBu_r",
        hover_data={"Applications": ":,", "Default Rate (%)": ":.1f", "Avg_Loan": ":$,.0f"},
    )
    _apply_sas_viya_layout(
        fig,
        title="Geographic Hierarchy Drill-Down: Region → Division → State",
        subtitle="Click any Region or Division tile to drill down • Tile Size = Applications • Color = Default Rate (%)",
        height=400,
        show_legend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Task 3 Visuals: Automated Explanation (guide p. 25-28)
# ---------------------------------------------------------------------------

def create_relative_importance_chart(imp_df: pd.DataFrame, selected_factor: str) -> go.Figure:
    """Task 3 Left Panel (guide p. 26-27): `What factors are most related to Loan Status?`."""
    plot_df = imp_df.sort_values("Relative Importance", ascending=True).copy()
    colors = [
        SAS_ORANGE if f == selected_factor else SAS_PRIMARY_BLUE
        for f in plot_df["Factor"]
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["Relative Importance"],
                y=plot_df["Factor"],
                orientation="h",
                marker=dict(color=colors),
                text=plot_df["Relative Importance"].map(lambda v: f"{v:.2f}"),
                textposition="outside",
                customdata=plot_df[["Role", "Missing %"]],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Relative Importance: %{x:.2f}<br>"
                    "Data Role: %{customdata[0]}<br>"
                    "Missing Values: %{customdata[1]:.1f}%<extra></extra>"
                ),
            )
        ]
    )
    _apply_sas_viya_layout(
        fig,
        title="What factors are most related to Loan Status?",
        subtitle="Relative Importance Score (Normalized 0.00 – 1.00 • Active Factor Highlighted in Orange)",
        height=470,
        show_legend=False,
    )
    fig.update_xaxes(title_text="Relative Importance", range=[0, 1.14], showgrid=True, gridcolor=SAS_GRID_COLOR)
    fig.update_yaxes(title_text="", showgrid=False)
    return fig


def create_factor_relationship_chart(df: pd.DataFrame, factor_col: str) -> go.Figure:
    """Task 3 Right Panel (guide p. 26): `What is the relationship between Loan Status and [Factor]?`."""
    label = SAS_LABELS.get(factor_col, factor_col)

    if factor_col in CONTINUOUS_MEASURES:
        plot_df = df[[factor_col, "LOAN_STATUS"]].dropna(subset=[factor_col]).copy()
        fig = px.histogram(
            plot_df,
            x=factor_col,
            color="LOAN_STATUS",
            nbins=32,
            barmode="stack",
            color_discrete_map=STATUS_COLORS,
            category_orders={"LOAN_STATUS": ["0", "1"]},
            labels={factor_col: label, "LOAN_STATUS": "Loan Status"},
        )
        _apply_sas_viya_layout(
            fig,
            title=f"What is the relationship between Loan Status and {label}?",
            subtitle="Stacked Frequency Histogram by Loan Status (0 = Current vs 1 = Default)",
            height=470,
            show_legend=True,
        )
        fig.update_xaxes(title_text=label)
        fig.update_yaxes(title_text="Frequency (Observations)", tickformat=",")
        return fig

    # Categorical predictor -> Grouped/Stacked Bar with Default Rate line
    cat_df = df.assign(_cat=df[factor_col].fillna("Missing").astype(str))
    grp = (
        cat_df.groupby(["_cat", "LOAN_STATUS"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    fig = px.bar(
        grp,
        x="_cat",
        y="Count",
        color="LOAN_STATUS",
        barmode="group",
        color_discrete_map=STATUS_COLORS,
        category_orders={"LOAN_STATUS": ["0", "1"]},
        labels={"_cat": label, "LOAN_STATUS": "Loan Status", "Count": "Frequency"},
    )
    _apply_sas_viya_layout(
        fig,
        title=f"What is the relationship between Loan Status and {label}?",
        subtitle="Grouped Frequency Breakdown by Loan Status (0 vs 1)",
        height=470,
        show_legend=True,
    )
    fig.update_xaxes(title_text=label)
    fig.update_yaxes(title_text="Frequency (Observations)", tickformat=",")
    return fig


# ---------------------------------------------------------------------------
# Task 4 & 5 Visuals: Model Studio Assessment & Fairness on JOB (p. 28-45)
# ---------------------------------------------------------------------------

def create_roc_comparison_chart(roc_curves: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Task 4 Model Comparison ROC Chart across Pipelines 1 & 2 (guide p. 33 & 40)."""
    palette = {
        "Gradient Boosting (Champion)": SAS_PRIMARY_BLUE,
        "Open Source Python (Random Forest)": SAS_TEAL,
        "Logistic Regression": SAS_ORANGE,
    }
    fig = go.Figure()
    for model_name, data in roc_curves.items():
        fig.add_trace(
            go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                mode="lines",
                name=f"{model_name} (AUC={data['auc']:.3f}, KS={data['ks']:.3f})",
                line=dict(color=palette.get(model_name, SAS_PRIMARY_BLUE), width=2.6),
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "False Positive Rate (1 - Specificity): %{x:.3f}<br>"
                    "True Positive Rate (Sensitivity): %{y:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline (AUC=0.500)",
            line=dict(color="#9AA5B1", width=1.5, dash="dash"),
            showlegend=True,
        )
    )
    _apply_sas_viya_layout(
        fig,
        title="ROC Curve Comparison (Validate Partition)",
        subtitle="Champion: Gradient Boosting (Pipeline 1) vs. Python Random Forest (Pipeline 2) & Logistic Regression",
        height=390,
        show_legend=True,
    )
    fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)", range=[0, 1])
    fig.update_yaxes(title_text="True Positive Rate (Sensitivity)", range=[0, 1.02])
    return fig


def create_cumulative_lift_chart(lift_curves: Dict[str, pd.DataFrame]) -> go.Figure:
    """Task 4 Cumulative Lift Chart across Pipelines (guide p. 33 & 40)."""
    palette = {
        "Gradient Boosting (Champion)": SAS_PRIMARY_BLUE,
        "Open Source Python (Random Forest)": SAS_TEAL,
        "Logistic Regression": SAS_ORANGE,
    }
    fig = go.Figure()
    for model_name, ldf in lift_curves.items():
        fig.add_trace(
            go.Scatter(
                x=ldf["percentile"],
                y=ldf["cumulative_lift"],
                mode="lines+markers",
                name=model_name,
                line=dict(color=palette.get(model_name, SAS_PRIMARY_BLUE), width=2.5),
                marker=dict(size=6),
                customdata=ldf["captured_response_pct"],
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "Top %{x:.0f}% of Scored Population<br>"
                    "Cumulative Lift: %{y:.2f}x<br>"
                    "Captured Defaults: %{customdata:.1f}%<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#9AA5B1", annotation_text="Baseline Lift = 1.0x")
    _apply_sas_viya_layout(
        fig,
        title="Cumulative Lift by Depth of File (Validate Partition)",
        subtitle="Top Decile Lift ~ 4.4x for Champion Gradient Boosting Model",
        height=390,
        show_legend=True,
    )
    fig.update_xaxes(title_text="Depth of File (Percentile %)", ticksuffix="%")
    fig.update_yaxes(title_text="Cumulative Lift (x Baseline)")
    return fig


def create_job_fairness_chart(job_fairness_df: pd.DataFrame) -> go.Figure:
    """Task 4 & 5 Sensitive Variable `JOB` Fairness & Prediction Bias Chart (guide p. 34-36, 43-45)."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=job_fairness_df["Job Category (JOB)"],
            y=job_fairness_df["Average Predicted P(BAD=1) (%)"],
            name="Avg Predicted Default Prob P(BAD=1) (%)",
            marker_color=SAS_PRIMARY_BLUE,
            text=job_fairness_df["Average Predicted P(BAD=1) (%)"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=job_fairness_df["Job Category (JOB)"],
            y=job_fairness_df["Actual Default Rate (%)"],
            name="Observed Default Rate (%)",
            marker_color=SAS_TEAL,
            text=job_fairness_df["Actual Default Rate (%)"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=job_fairness_df["Job Category (JOB)"],
            y=job_fairness_df["True Positive Rate (Equal Opportunity %)"],
            mode="lines+markers",
            name="True Positive Rate / Equal Opportunity (%)",
            line=dict(color=SAS_ORANGE, width=2.5),
            marker=dict(size=8),
        )
    )
    fig.update_layout(barmode="group")
    _apply_sas_viya_layout(
        fig,
        title="Fairness & Bias Assessment Across Sensitive Variable: Job Category (JOB)",
        subtitle="Model Card Status: FAIRNESS = FAIL due to disparity between Sales (34.9% pred. risk) and Executive/Office",
        height=410,
        show_legend=True,
    )
    fig.update_xaxes(title_text="Sensitive Attribute: Job Category (JOB)")
    fig.update_yaxes(title_text="Rate / Probability (%)", ticksuffix="%", range=[0, 105])
    return fig


# ---------------------------------------------------------------------------
# Task 6 Visual: Intelligent Decisioning Flow Sankey (guide p. 45-54)
# ---------------------------------------------------------------------------

def create_decision_flow_sankey(batch_res: Dict[str, Any], prob_cutoff: float, debtinc_cutoff: float) -> go.Figure:
    """Visualize the two-stage `Loan Decision (Final)` flow across all applications."""
    total = batch_res["total"]
    declined_ml = batch_res["declined_ml_count"]
    passed_ml = total - declined_ml
    declined_dti = batch_res["declined_dti_count"]
    approved = batch_res["approved_count"]

    labels = [
        f"Start: All Applications ({total:,})",
        f"Gradient Boosting: P_BAD1 < {prob_cutoff:.2f} ({passed_ml:,})",
        f"Declined Applications (Final): P_BAD1 >= {prob_cutoff:.2f} ({declined_ml:,})",
        f"Approved: DTI < {debtinc_cutoff:.0f}% ({approved:,})",
        f"Declined: DTI >= {debtinc_cutoff:.0f}% or Unverified ({declined_dti:,})",
        f"Create Email Message (Final): Notifications ({total:,})",
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=20,
                    line=dict(color="#FFFFFF", width=1.5),
                    label=labels,
                    color=[
                        SAS_PRIMARY_BLUE,
                        SAS_TEAL,
                        SAS_RED_ALERT,
                        SAS_GREEN_OK,
                        SAS_ORANGE,
                        "#4B1E78",
                    ],
                ),
                link=dict(
                    source=[0, 0, 1, 1, 2, 3, 4],
                    target=[1, 2, 3, 4, 5, 5, 5],
                    value=[
                        passed_ml,
                        declined_ml,
                        approved,
                        declined_dti,
                        declined_ml,
                        approved,
                        declined_dti,
                    ],
                    color=[
                        "rgba(23, 153, 141, 0.30)",
                        "rgba(217, 56, 58, 0.30)",
                        "rgba(30, 142, 62, 0.35)",
                        "rgba(230, 108, 55, 0.35)",
                        "rgba(217, 56, 58, 0.20)",
                        "rgba(30, 142, 62, 0.25)",
                        "rgba(230, 108, 55, 0.20)",
                    ],
                ),
            )
        ]
    )
    _apply_sas_viya_layout(
        fig,
        title="SAS Intelligent Decisioning Pipeline — Loan Decision (Final) Flow",
        subtitle="Start → Gradient Boosting (Pipeline 1) → P_BAD1 < Cutoff → Check Debt to Income → Create Email Message → End",
        height=380,
        show_legend=False,
    )
    return fig
