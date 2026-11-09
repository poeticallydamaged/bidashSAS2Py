import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """Load the dataset and perform initial preprocessing."""
    df = pd.read_csv('data/dataset.csv')

    # Fill missing values for numerical columns with median or 0
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical with "Unknown"
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    # Convert BAD to categorical for descriptive labeling
    if 'BAD' in df.columns:
        df['Loan Status'] = df['BAD'].map({1: 'Defaulted', 0: 'Paid'})

    return df

def filter_data(df, region=None, job=None):
    """Filter the dataset based on selections."""
    filtered_df = df.copy()
    if region and "All" not in region:
        filtered_df = filtered_df[filtered_df['REGION'].isin(region)]
    if job and "All" not in job:
        filtered_df = filtered_df[filtered_df['JOB'].isin(job)]
    return filtered_df
