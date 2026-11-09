# Home Equity Analysis Dashboard

This is a Streamlit-based Python application that replicates the SAS Viya Visual Analytics dashboard for Home Equity Analysis, based on the provided tutorial guide.

## Setup Instructions

1. Install Python 3.9+
2. Create and activate a virtual environment (optional but recommended):
   python3 -m venv venv
   source venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Run the dashboard:
   ./run.sh

## Architecture Map (SAS Viya to Python)
* **Data Prep**: src/data_engine.py replicates Data preparations using pandas cache.
* **Calculated Items**: Processed dynamically inside Python functions.
* **Layouts**: Handled via Streamlit columns, tabs, and metrics.
* **Visual Theme**: Replicated via .streamlit/config.toml and Plotly theme overrides in src/visuals.py.
