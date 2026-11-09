import streamlit as st

def kpi_card(title, value, delta=None):
    """Render a KPI card with custom styling."""
    st.metric(label=title, value=value, delta=delta)

def render_filters(df):
    """Render sidebar filters and return selections."""
    st.sidebar.header("Filters")

    regions = ["All"] + list(df['REGION'].dropna().unique())
    selected_region = st.sidebar.multiselect("Select Region", regions, default=["All"])

    jobs = ["All"] + list(df['JOB'].dropna().unique())
    selected_job = st.sidebar.multiselect("Select Job", jobs, default=["All"])

    return selected_region, selected_job
