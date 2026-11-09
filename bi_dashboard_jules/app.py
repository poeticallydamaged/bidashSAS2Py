import streamlit as st
from src.data_engine import load_data, filter_data
from src.visuals import (
    plot_loan_status_histogram,
    plot_loan_status_pie,
    plot_credit_age_bar,
    plot_value_vs_mortdue
)
from src.components import kpi_card, render_filters

st.set_page_config(page_title="Home Equity Analysis Report", layout="wide")

# Load and process data
df = load_data()

# Render Sidebar
st.sidebar.image("https://www.sas.com/content/dam/SAS/brand/logo/logo.svg", width=100) # Optional SAS branding representation
selected_region, selected_job = render_filters(df)
filtered_df = filter_data(df, selected_region, selected_job)

# Header
st.title("Home Equity Analysis Report")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_card("Total Applicants", f"{len(filtered_df):,}")
with col2:
    default_rate = (filtered_df['Loan Status'] == 'Defaulted').mean() * 100
    kpi_card("Default Rate", f"{default_rate:.1f}%")
with col3:
    avg_loan = filtered_df['LOAN'].mean()
    kpi_card("Avg Loan Request", f"${avg_loan:,.0f}")
with col4:
    avg_value = filtered_df['VALUE'].mean()
    kpi_card("Avg Property Value", f"${avg_value:,.0f}")

st.markdown("---")

# Tabs matching the guide
tab1, tab2 = st.tabs(["Interactive charts", "Explain Loan Status"])

with tab1:
    # First row of charts
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(plot_loan_status_histogram(filtered_df), use_container_width=True)
    with row1_col2:
        st.plotly_chart(plot_loan_status_pie(filtered_df), use_container_width=True)

    # Second row of charts
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(plot_credit_age_bar(filtered_df), use_container_width=True)
    with row2_col2:
        st.plotly_chart(plot_value_vs_mortdue(filtered_df), use_container_width=True)

with tab2:
    st.header("Loan Status Drivers")
    st.markdown("""
    This section simulates the **Automated Explanation** object from SAS Viya.
    Below are the key factors influencing whether an applicant defaults on a loan.
    """)

    # Feature Importance Mockup
    importance_data = {
        'Factor': ['Debt-to-Income Ratio (DEBTINC)', 'Number of Delinquent Lines (DELINQ)', 'Number of Derogatory Reports (DEROG)', 'Job Category (JOB)'],
        'Relative Importance': [0.85, 0.72, 0.65, 0.45]
    }
    import pandas as pd
    imp_df = pd.DataFrame(importance_data)
    import plotly.express as px
    fig = px.bar(imp_df, x='Relative Importance', y='Factor', orientation='h', title='Relative Importance for Default Risk')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Explanation Summary
    - **Debt-to-Income Ratio** is the most significant driver. Higher ratios strongly correlate with defaults.
    - **Delinquency and Derogatory reports** also play a crucial role.
    """)
