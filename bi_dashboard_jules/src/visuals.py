import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# SAS Viya Theme Colors
THEME_COLORS = ['#0055A4', '#FF8000', '#00A9E0', '#4CAF50', '#E53935']

def update_layout_theme(fig):
    """Apply consistent SAS Viya styling to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=40, l=40, r=20, b=40),
        font=dict(family="sans-serif", color="#333"),
        title_font=dict(size=16, color="#0055A4"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig.update_xaxes(showgrid=False, linecolor='lightgray', tickcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='whitesmoke', linecolor='lightgray', tickcolor='lightgray')
    return fig

def plot_loan_status_histogram(df):
    """Histogram of Amount of Loan Request by Loan Status."""
    fig = px.histogram(
        df,
        x="LOAN",
        color="Loan Status",
        barmode="overlay",
        color_discrete_sequence=['#0055A4', '#FF8000'],
        title="Amount of Loan Request by Loan Status",
        labels={'LOAN': 'Amount of Loan Request', 'count': 'Frequency'}
    )
    fig.update_traces(opacity=0.7)
    return update_layout_theme(fig)

def plot_loan_status_pie(df):
    """Pie chart for Loan Status distribution."""
    status_counts = df['Loan Status'].value_counts().reset_index()
    status_counts.columns = ['Loan Status', 'Count']

    fig = px.pie(
        status_counts,
        names='Loan Status',
        values='Count',
        color='Loan Status',
        color_discrete_map={'Paid': '#0055A4', 'Defaulted': '#FF8000'},
        title="Loan Status Distribution",
        hole=0.4 # Donut chart style often used in dashboards
    )
    return update_layout_theme(fig)

def plot_credit_age_bar(df):
    """Bar chart for Average Age of Oldest Credit Line by Job."""
    agg_df = df.groupby('JOB')['CLAGE'].mean().reset_index()
    agg_df = agg_df.sort_values('CLAGE', ascending=False)

    fig = px.bar(
        agg_df,
        x='JOB',
        y='CLAGE',
        title="Average Age of Oldest Credit Line by Job",
        labels={'CLAGE': 'Average Credit Line Age (Months)', 'JOB': 'Job Category'},
        color_discrete_sequence=['#00A9E0']
    )
    return update_layout_theme(fig)

def plot_value_vs_mortdue(df):
    """Scatter plot: Property Value vs. Mortgage Due."""
    # Take a sample if dataset is too large for responsive scatter
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    fig = px.scatter(
        sample_df,
        x='VALUE',
        y='MORTDUE',
        color='Loan Status',
        color_discrete_map={'Paid': '#0055A4', 'Defaulted': '#FF8000'},
        title="Property Value vs. Mortgage Due (Sample)",
        labels={'VALUE': 'Property Value', 'MORTDUE': 'Mortgage Due'}
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    return update_layout_theme(fig)
